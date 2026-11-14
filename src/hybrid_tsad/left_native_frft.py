"""LEFT-native candidate: invertible FRFT basis on the STFT frequency axis.

This is a project candidate, not TimeRadar FTFRecon or official LEFT.
"""
def wrap_transform(native):
    import torch
    from torch import nn
    from torch_frft.dfrft_module import dfrft,idfrft
    Native=type(native)
    class FractionalTF(Native):
        def forward(self,x):
            ri=super().forward(x)
            z=torch.complex(ri[:,:,0],ri[:,:,1])
            z=dfrft(z,2*torch.tanh(self.raw_order),dim=-2)
            return torch.stack((z.real,z.imag),dim=2)
        def inverse(self,ri,length):
            z=torch.complex(ri[:,:,0],ri[:,:,1])
            z=idfrft(z,2*torch.tanh(self.raw_order),dim=-2)
            return super().inverse(torch.stack((z.real,z.imag),dim=2),length)
    tf=FractionalTF(native.n_fft,native.hop_length,native.win_length,native.window_eps)
    tf.window_param=native.window_param
    tf.raw_order=nn.Parameter(torch.zeros((),device=native.window_param.device,dtype=native.window_param.dtype))
    return tf

def candidate_class(native_class):
    class HybridNativeFRFT(native_class):
        def __init__(self,configs):
            super().__init__(configs)
            self.tf=wrap_transform(self.tf)
    return HybridNativeFRFT
