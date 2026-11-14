"""LEFT-native history/lag candidate. Independently implemented causal operators.

LEFT's native forward/loss/score are inherited unchanged. These operators are
not TimesFM or the former V1 Q/K routing implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('L','LS','LC','LSC')

class CausalBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.dilation=dilation
        self.conv=nn.Conv1d(8,8,2,dilation=dilation)
        self.norm=nn.LayerNorm(8)
    def forward(self,x):
        y=self.conv(F.pad(x,(self.dilation,0)))
        return x+F.gelu(self.norm(y.transpose(1,2)).transpose(1,2))

class HistoryContext(nn.Module):
    def __init__(self,channels,width,seed=2025):
        super().__init__();self.channels=channels
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+20001)
            self.input=nn.Conv1d(2,8,1)
            self.blocks=nn.Sequential(*(CausalBlock(d) for d in (1,2,4,8,16,32,64,128,256)))
            self.long_out=nn.Linear(channels*8,width)
            torch.random.default_generator.manual_seed(seed+20002)
            self.self_kernel=nn.Parameter(torch.randn(channels,1,256)/(256**.5))
            self.self_out=nn.Linear(channels,width)
            torch.random.default_generator.manual_seed(seed+20003)
            self.cross_kernel=nn.Parameter(torch.randn(channels,channels,256)/(max(1,(channels-1)*256)**.5))
            self.cross_out=nn.Linear(channels,width)
        self.register_buffer('off_diagonal',1-torch.eye(channels)[:,:,None])
        with torch.no_grad():
            for layer in (self.long_out,self.self_out,self.cross_out):
                layer.weight.zero_();layer.bias.zero_()
    def components(self,history,observed):
        assert history.shape[1:]==(512,self.channels) and observed.shape==history.shape
        x=torch.where(observed,history,torch.zeros_like(history));B,T,C=x.shape
        ci=torch.stack((x,observed.to(x.dtype)),dim=-1).permute(0,2,3,1).reshape(B*C,2,T)
        h=self.blocks(self.input(ci)).reshape(B,C,8,T).permute(0,3,1,2).reshape(B,T,C*8)
        present=observed.any(-1).to(torch.int32).cumsum(1).gt(0).to(x.dtype)[:,:,None]
        long=self.long_out(h)*present
        # Output at history index h corresponds to target h+1; weights store
        # lag1..256, hence reverse before cross-correlation.
        padded=F.pad(x.transpose(1,2),(255,0))
        self_value=F.conv1d(padded,self.self_kernel.flip(-1),groups=C).transpose(1,2)
        cross_value=F.conv1d(padded,(self.cross_kernel*self.off_diagonal).flip(-1)).transpose(1,2)
        return dict(long=long[:,-192:],self_value=self_value[:,-192:],cross_value=cross_value[:,-192:],
                    self_token=self.self_out(self_value)[:,-192:]*present[:,-192:],
                    cross_token=self.cross_out(cross_value)[:,-192:]*present[:,-192:])
    def forward(self,history,observed,arm,capture_rows=None):
        if arm not in ARMS:raise ValueError(arm)
        parts=self.components(history,observed);token=parts['long']
        if arm in ('LS','LSC'):token=token+parts['self_token']
        if arm in ('LC','LSC'):token=token+parts['cross_token']
        self.captured=None if capture_rows is None else {k:v[capture_rows].detach() for k,v in parts.items()}
        return token

def candidate_class(native_class,arm):
    if arm not in ARMS:raise ValueError(arm)
    class HybridNativeHistoryLag(native_class):
        def __init__(self,configs):
            super().__init__(configs)
            self.history_context=HistoryContext(self.enc_in,self.d_model)
            self.context_arm=arm;self._pending_history_token=None
            self.capture_rows=None;self.captured=None
            self.allow_basic_padding=False
            def add_context(module,args,output):
                if self._pending_history_token is None:raise RuntimeError('Native Et used outside complete candidate forward')
                return output+self._pending_history_token
            self.Et.register_forward_hook(add_context)
        def _forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec):
            if x_enc.shape[1]==192:
                if not self.allow_basic_padding:raise ValueError('192 input is permitted only in explicit official _basic context')
                # Original _basic consumes the identical 192 random draws;
                # padding carries no fabricated native targets or extra RNG.
                x_enc=F.pad(x_enc,(0,0,321,0),value=float('nan'))
            if x_enc.shape[1]!=513:raise ValueError('Exactly 513 arrived observations required')
            observed=torch.isfinite(x_enc)
            if not bool(observed[:,-192:].all()):raise ValueError('Native 192 targets must be genuinely observed')
            self._pending_history_token=self.history_context(x_enc[:,:512],observed[:,:512],self.context_arm,self.capture_rows)
            try:
                result=super()._forward(x_enc[:,-192:],x_mark_enc,x_dec,x_mark_dec)
                self.captured=None if self.capture_rows is None else dict(self.history_context.captured,
                    native_freq_to_time=result[3][self.capture_rows].detach(),native_time=result[4][self.capture_rows].detach(),
                    native_target=result[5][self.capture_rows].detach(),native_MS=result[9][self.capture_rows].detach())
                return result
            finally:self._pending_history_token=None
    return HybridNativeHistoryLag
