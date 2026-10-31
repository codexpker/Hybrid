"""Independent hybrid predictor; official TimeRadar file imported read-only.

No checkpoints loaded. Original TF structure is instantiated from scratch.
"""
import hashlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

OFFICIAL_SHA='881e8912e4c5b298b202e5b7ac2fe58107c7dcf1b56acf733e28c99f2ffc0317'

class CausalBlock(nn.Module):
    def __init__(self,d=128,heads=4):
        super().__init__();self.d=d;self.heads=heads
        self.ln1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.out=nn.Linear(d,d)
        self.ln2=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def rotary(self,x):
        n=x.shape[-2];hd=x.shape[-1];half=hd//2
        ang=torch.arange(n,device=x.device,dtype=x.dtype)[:,None]*torch.exp(
            -math.log(10000)*torch.arange(half,device=x.device,dtype=x.dtype)/half)[None,:]
        a,b=x[...,:half],x[...,half:]
        return torch.cat([a*ang.cos()-b*ang.sin(),a*ang.sin()+b*ang.cos()],-1)
    def forward(self,x):
        b,n,d=x.shape;q,k,v=self.qkv(self.ln1(x)).chunk(3,-1)
        q,k,v=[z.reshape(b,n,self.heads,d//self.heads).transpose(1,2) for z in (q,k,v)]
        z=F.scaled_dot_product_attention(self.rotary(q),self.rotary(k),v,is_causal=True)
        x=x+self.out(z.transpose(1,2).reshape(b,n,d))
        return x+self.ff(self.ln2(x))

class HybridPredictor(nn.Module):
    def __init__(self,official_file,history=400,copies=2,mode='hybrid',temporal_depth=4):
        super().__init__()
        assert history>=100 and history%5==0 and copies>0 and copies%2==0
        assert mode in ('hybrid','temporal','tf')
        p=Path(official_file);assert hashlib.sha256(p.read_bytes()).hexdigest()==OFFICIAL_SHA
        spec=importlib.util.spec_from_file_location('pinned_timeradar',p)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.tf=module.Model(SimpleNamespace(seq_len=100,hidden_dim=64,d_model=256,depth=10,patch_len=5,mask_mode='c'))
        self.history=history;self.copies=copies;self.mode=mode
        self.embed=nn.Linear(10,128);self.query=nn.Parameter(torch.zeros(1,1,128))
        if temporal_depth<1:raise ValueError('temporal_depth must be positive')
        self.blocks=nn.ModuleList([CausalBlock() for _ in range(temporal_depth)])
        self.tfproj=nn.Linear(512,128)
        self.cross=nn.MultiheadAttention(128,4,batch_first=True)
        self.head=nn.Linear(128,5)
        # Stored inactive modules preserve initialization parity, but cannot be
        # counted as effective capacity or passed to the optimizer.
        inactive=('tf.','tfproj.','cross.') if mode=='temporal' else (('embed.','query','blocks.','cross.') if mode=='tf' else ())
        for name,p in self.named_parameters():
            if any(name.startswith(prefix) for prefix in inactive):p.requires_grad=False

    def parameter_counts(self):
        return {'stored':sum(p.numel() for p in self.parameters()),
                'active':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'inactive':sum(p.numel() for p in self.parameters() if not p.requires_grad)}

    def preprocess(self,history):
        assert history.ndim==3 and history.shape[1]==self.history
        valid=torch.isfinite(history);count=valid.sum(1,keepdim=True)
        if not bool((count>0).all()):raise ValueError('Entire missing history channel')
        raw=history.nan_to_num(nan=0.,posinf=0.,neginf=0.)
        mean=(raw*valid).sum(1,keepdim=True)/count
        var=((raw-mean).square()*valid).sum(1,keepdim=True)/count
        scale=(var+1e-5).sqrt()
        # Only past-value forward fill; initial unavailable positions filled mean,
        # with explicit missing mask. TF policy rejects unavailable prefix.
        if not bool(valid[:,0,:].all()):raise ValueError('Missing initial history value; no past fill available')
        idx=torch.arange(self.history,device=history.device)[None,:,None].expand_as(raw)
        idx=torch.where(valid,idx,torch.zeros_like(idx)).cummax(1).values
        filled=raw.gather(1,idx)
        return (filled-mean)/scale,valid,mean,scale

    def forward(self,history):
        z,valid,mean,scale=self.preprocess(history);b,l,c=z.shape
        x=z.permute(0,2,1).reshape(b*c,l)
        mask=valid.permute(0,2,1).reshape(b*c,l)
        # No target argument in prediction API; no future observations available.
        tf_repr=tf_rec=None
        if self.mode!='temporal':
            tf_repr,tf_rec=self.tf(z[:,-100:],norm=0,copies=self.copies)
            h=tf_repr.reshape(self.copies,b*c,20,256).mean(0)
            kv=self.tfproj(torch.cat([h.real,h.imag],-1))
        if self.mode!='tf':
            tokens=self.embed(torch.cat([x.reshape(b*c,-1,5),mask.reshape(b*c,-1,5).float()],-1))
            tokens=torch.cat([tokens,self.query.expand(b*c,-1,-1)],1)
            for blk in self.blocks:tokens=blk(tokens)
            q=tokens[:,-1:]
            if self.mode=='hybrid':q=q+self.cross(q,kv,kv,need_weights=False)[0]
        else:q=kv.mean(1,keepdim=True)
        pred=self.head(q[:,0]).reshape(b,c,5).permute(0,2,1)
        return {'prediction_normalized':pred,'prediction':pred*scale+mean,
                'mean':mean,'scale':scale,'tf_reconstruction':tf_rec,
                'tf_history_normalized':z[:,-100:],'tf_valid':valid[:,-100:]}

def objective(output,target,future_weight=1.,tf_weight=.1):
    assert target.shape==output['prediction'].shape
    valid=torch.isfinite(target);assert bool(valid.any())
    y=(target-output['mean'])/output['scale']
    future=F.huber_loss(output['prediction_normalized'][valid],y[valid],delta=1.)
    rec=output['tf_reconstruction']
    auxiliary=target.new_zeros(()) if rec is None else (
        (rec-output['tf_history_normalized'].unsqueeze(0)).square()*output['tf_valid'].unsqueeze(0)
    ).sum()/(output['tf_valid'].sum()*len(rec))
    return future_weight*future+tf_weight*auxiliary,{'future':future,'tf':auxiliary}
