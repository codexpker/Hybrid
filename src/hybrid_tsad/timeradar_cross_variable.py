"""A decoder-input adapter around the unchanged official TimeRadar forward.

Only the adapter groups K,B,C,P correctly. Native FRFT order still aggregates
batch/channel input and therefore the full model is not sample independent.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ComplexChannelAdapter(nn.Module):
    def __init__(self,dimension=256,rank=16,mode='cross'):
        super().__init__()
        if mode not in ('self','cross'):raise ValueError(mode)
        self.dimension,self.rank,self.mode=dimension,rank,mode
        self.down=nn.Linear(2*dimension,rank)
        self.up=nn.Linear(rank,2*dimension)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)

    def forward(self,representation,copies,batch,channels):
        if not torch.is_complex(representation)or representation.ndim!=3:
            raise ValueError('Native complex [K*B*C,P,D] required')
        n,patches,dim=representation.shape
        if n!=copies*batch*channels or dim!=self.dimension:
            raise ValueError('Explicit original K,B,C layout does not match representation')
        z=representation.reshape(copies,batch,channels,patches,dim)
        h=F.gelu(self.down(torch.cat((z.real,z.imag),-1)))
        if self.mode=='cross'and channels>1:m=(h.sum(2,keepdim=True)-h)/(channels-1)
        else:m=h
        real,imag=self.up(m).chunk(2,-1)
        delta=torch.complex(real,imag)
        return (z+delta).reshape_as(representation)


class TimeRadarCrossVariable(nn.Module):
    def __init__(self,native,mode='cross',rank=16):
        super().__init__()
        if mode not in ('decoder_head','self','cross'):raise ValueError(mode)
        self.native=native;self.mode=mode;self._layout=None;self.adapter_enabled=True
        self.adapter=None if mode=='decoder_head'else ComplexChannelAdapter(native.repr_dim,rank,mode)
        for p in self.native.parameters():p.requires_grad_(False)
        for p in self.native.decoder.parameters():p.requires_grad_(True)
        self._hook=None if self.adapter is None else self.native.decoder.register_forward_pre_hook(self._adapt)
        self.train(False)

    def _adapt(self,module,args):
        if self._layout is None:raise RuntimeError('Adapter requires wrapper forward layout context')
        if not self.adapter_enabled:return args
        return(self.adapter(args[0],*self._layout),)+args[1:]

    def forward(self,x,norm=0,mask_mode=None,copies=10):
        if self._layout is not None:raise RuntimeError('Reentrant native forward is not supported')
        actual_mode=self.native.mask_mode if mask_mode is None else mask_mode
        if actual_mode not in ('c','nomask'):raise ValueError('Only official complementary or no-mask modes are supported')
        actual_copies=1 if actual_mode=='nomask'else copies
        if actual_mode=='c'and(copies<2 or copies%2):raise ValueError('Complementary masks need positive even copies')
        self._layout=(actual_copies,x.shape[0],x.shape[2])
        try:return self.native(x,norm=norm,mask_mode=mask_mode,copies=copies)
        finally:self._layout=None

    def cal_anomaly_score(self,*args,**kwargs):
        return self.native.cal_anomaly_score(*args,**kwargs)

    def train(self,mode=True):
        self.training=mode
        self.native.eval()
        self.native.decoder.train(mode)
        if self.adapter is not None:self.adapter.train(mode)
        return self

    def close(self):
        if self._hook is not None:self._hook.remove();self._hook=None
        self._layout=None

    def trainable_names(self):
        return[n for n,p in self.named_parameters()if p.requires_grad]
