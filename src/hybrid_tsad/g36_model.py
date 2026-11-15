"""Fixed G36 conditional response density; no query-time model updates.

Self-supervised target is four response observations after a strictly earlier
context. This is not the historical whole-window density or switching HMM.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .normality_density import haar,NormalityDensity


class PastConv(nn.Conv1d):
    def forward(self,x):return super().forward(F.pad(x,(self.dilation[0]*(self.kernel_size[0]-1),0)))


class ConditionalCoupling(nn.Module):
    def __init__(self,parity,hidden=64):
        super().__init__();mask=((torch.arange(4)[:,None]+torch.arange(4)[None,:]+parity)%2).float().flatten()
        self.register_buffer('mask',mask[None])
        self.net=nn.Sequential(nn.Linear(16+16+hidden,hidden),nn.GELU(),nn.Linear(hidden,32))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,context,inverse=False):
        passive=x*self.mask
        logs,shift=self.net(torch.cat((passive,self.mask.expand_as(x),context),-1)).chunk(2,-1)
        logs=2*torch.tanh(logs/2)*(1-self.mask);shift=shift*(1-self.mask)
        if inverse:return passive+(1-self.mask)*(x-shift)*torch.exp(-logs),-logs.sum(-1)
        return passive+(1-self.mask)*(x*torch.exp(logs)+shift),logs.sum(-1)


class ConditionalResponseFlow(nn.Module):
    def __init__(self,variant='full',hidden=64,kernel=3):
        super().__init__()
        if variant not in ('full','history_only'):raise ValueError(variant)
        self.variant=variant;self.history=48;self.block=4;self.channels=9
        self.input=nn.Conv1d(9,hidden,1)
        self.layers=nn.ModuleList(PastConv(hidden,hidden,kernel,dilation=d)for d in (1,2,4,8))
        self.steps=nn.ModuleList(ConditionalCoupling(i,hidden)for i in range(4))

    def split(self,arrived):
        if arrived.ndim!=3 or arrived.shape[1:]!=(52,9):raise ValueError('Exactly48 past +4 arrived rows, nine channels')
        if not bool(torch.isfinite(arrived).all()):raise ValueError('Finite FIT-scaled values required')
        return arrived[:,:48],arrived[:,48:,:4]

    def context(self,past):
        if past.shape[1:]!=(48,9):raise ValueError('Wrong past context shape')
        if self.variant=='history_only':past=torch.cat((past[:,:,:4],torch.zeros_like(past[:,:,4:])),dim=-1)
        h=F.gelu(self.input(past.transpose(1,2)))
        for layer in self.layers:h=h+F.gelu(layer(h))/2
        return h[:,:,-1]

    def transform(self,target,context,inverse=False):
        if target.shape[1:]!=(4,4):raise ValueError('Response block must be four by four')
        z=target;ld=target.new_zeros(target.shape[0])
        for i in (reversed(range(4))if inverse else range(4)):
            if i%2:z=haar(z)
            flat,change=self.steps[i](z.flatten(1),context,inverse=inverse)
            z=flat.reshape(-1,4,4);ld=ld+change
            if i%2:z=haar(z,inverse=True)
        return z,ld

    def components(self,arrived):
        past,target=self.split(arrived);context=self.context(past);z,ld=self.transform(target,context)
        base=NormalityDensity.prior_nll(z).sum((1,2))
        return dict(score=(base-ld)/16,base_nll=base/16,log_abs_det=ld,
                    latent=z,target_response=target)

    def score(self,arrived):return self.components(arrived)['score']
    def loss(self,arrived):return self.score(arrived).mean()
    def forward(self,arrived):return self.loss(arrived)
