"""Fixed native-V conditional quantiles and a separate raw forecast control.

No old generator, Energy objective, adapter, lag search or score fusion.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .hybrid_v1.provenance import OfficialSources
from .hybrid_v1.temporal import TemporalEncoder

SOURCE_SHA='881e8912e4c5b298b202e5b7ac2fe58107c7dcf1b56acf733e28c99f2ffc0317'
WEIGHT_SHA='528aea6e803f9e6b1ea5d07e64e44162309bdb016cc7721a8ac3753d942e6c7c'
QUANTILES=(.05,.5,.95)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state_hashes(model):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()for k,v in model.state_dict().items()}


@contextmanager
def fixed_native_rng(device):
    device=torch.device(device);ids=[device.index or 0]if device.type=='cuda'else[]
    with torch.random.fork_rng(devices=ids):
        torch.random.default_generator.manual_seed(84551)
        if ids:torch.cuda.default_generators[ids[0]].manual_seed(84551)
        yield


def load_native(workspace,device):
    root=Path(workspace)/'preparations/timeradar_cross_variable_20261116_v1/runtime/official'
    source=root/'TimeRadar.py';checkpoint=root/'checkpoint.pth'
    assert sha(source)==SOURCE_SHA and sha(checkpoint)==WEIGHT_SHA
    spec=importlib.util.spec_from_file_location('_native_v_conditioning_official',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    native=module.Model(SimpleNamespace(seq_len=100,hidden_dim=64,d_model=256,depth=10,patch_len=5,mask_mode='c'))
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if 'state_dict'in saved:saved=saved['state_dict']
    saved={k.removeprefix('module.'):v for k,v in saved.items()}
    extras=set(saved)-set(native.state_dict());assert extras=={'tau_n','tau_a','tau_inited'}
    for key in sorted(extras):native.register_buffer(key,saved[key].clone())
    native.load_state_dict(saved,strict=True);native.requires_grad_(False);native.to(device).eval()
    return native


def actual_mask_bits(channels,device):
    with fixed_native_rng(device):
        first=torch.rand(5*channels,20,device=device)<.5
        return torch.cat((first,~first),0).reshape(10,channels,20).cpu().numpy()


class SimpleHistoryQuantiles(nn.Module):
    def __init__(self,channels,bias):
        super().__init__();self.channels=channels
        self.body=nn.Sequential(nn.Flatten(1),nn.Linear(512*channels,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
        self.head=nn.Linear(32,3);zero_quantile_head(self.head,bias)
    def forward(self,history):
        if history.shape[1:]!=(512,self.channels):raise ValueError('Strictly past512 x fixedC history required')
        return torch.sort(self.head(self.body(history)),dim=-1,stable=True).values


class StructuredHistory(nn.Module):
    def __init__(self,channels,sources,method,bias=None):
        super().__init__();self.channels=channels;self.method=method
        if method not in ('C','D'):raise ValueError(method)
        cfg=SimpleNamespace(width=128,temporal_patch=16,heads=4,temporal_layers=2,use_variate_attention=True,temporal_enabled=True)
        self.encoder=TemporalEncoder(cfg,sources)
        self.head=nn.Linear(channels*128,3 if method=='C'else channels)
        if method=='C':zero_quantile_head(self.head,bias)
    def forward(self,history):
        if history.shape[1:]!=(512,self.channels):raise ValueError('Strictly past512 x fixedC history required')
        tokens,_,_=self.encoder(history,torch.ones_like(history,dtype=torch.bool))
        values=self.head(tokens[:,:,-1,:].reshape(history.shape[0],self.channels*128))
        return torch.sort(values,dim=-1,stable=True).values if self.method=='C'else values


def zero_quantile_head(head,bias):
    bias=torch.as_tensor(bias,dtype=head.bias.dtype)
    if bias.shape!=(3,)or not torch.isfinite(bias).all():raise ValueError('Three finite FIT global quantiles required')
    with torch.no_grad():head.weight.zero_();head.bias.copy_(bias)


def build_predictor(method,channels,workspace,bias):
    if method=='B':return SimpleHistoryQuantiles(channels,bias)
    return StructuredHistory(channels,OfficialSources.from_workspace(workspace),method,bias)


def pinball_per_sample(prediction,target):
    if prediction.ndim!=2 or prediction.shape[1]!=3 or target.shape!=(len(prediction),):raise ValueError('Quantile/target shape')
    tau=prediction.new_tensor(QUANTILES);error=target[:,None]-prediction
    return torch.maximum(tau*error,(tau-1)*error).mean(-1)


def conditional_score(V,q_normalized,s_V):
    V=np.asarray(V,dtype=np.float64);q=np.asarray(q_normalized,dtype=np.float64)
    if q.shape!=(len(V),3)or not np.isfinite(q).all()or not np.all(q[:,1:]>=q[:,:-1]):raise ValueError('Ordered finite quantiles')
    if not np.isfinite(s_V)or s_V<=0:raise ValueError('Positive finite FIT-only source scale')
    width=q[:,2]-q[:,0];denominator=np.maximum(width,1e-3)
    return dict(canonical_score=(V/s_V-q[:,2])/denominator,q_normalized=q,
        q_physical=s_V*q,width_normalized=width,width_physical=s_V*width,
        denominator_normalized=denominator,denominator_physical=s_V*denominator,floor_active=width<1e-3)


def raw_residual_score(prediction,actual):
    p=np.asarray(prediction,dtype=np.float64);y=np.asarray(actual,dtype=np.float64)
    if p.shape!=y.shape or p.ndim!=2:raise ValueError('Raw mean and actual shape mismatch')
    residual=(p-y)**2
    return residual,residual.mean(-1)
