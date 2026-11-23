"""Frozen official features and a supervised copy/channel MLP baseline."""
from pathlib import Path
import torch
from torch import nn
from .native_v_conditioning import fixed_native_rng, load_native

OUT = Path('results/native_supervised_mlp_20261123_v1')

class SupervisedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(10240,64),nn.GELU(),nn.Linear(64,1))

    def forward(self, representation):
        if representation.ndim != 5 or representation.shape[0:2] != (10,1) or representation.shape[-2:] != (20,256):
            raise ValueError('Require complex [10,1,C,20,256] native representation')
        features=torch.cat((representation.real,representation.imag),dim=-1).flatten(-2)
        logits=self.layers(features)
        return logits.mean(dim=(0,2,3)).squeeze(0)

class _RepresentationReady(Exception):
    def __init__(self, value):
        self.value=value

@torch.no_grad()
def encode(model, window):
    if window.shape[0]!=1 or window.shape[1]!=100:
        raise ValueError('B1 independent native window required')
    def capture(_module,args):
        raise _RepresentationReady(args[0])
    hook=model.decoder.register_forward_pre_hook(capture)
    try:
        with fixed_native_rng(window.device):
            model(window,norm=0,mask_mode='c',copies=10)
    except _RepresentationReady as found:
        h=found.value
    finally:
        hook.remove()
    return h.reshape(10,1,window.shape[2],20,256)


def load_heads(workspace, device):
    import json
    import hashlib
    root=Path(workspace)/OUT
    seal=json.loads((root/'ALL_SELECTION.json').read_text())
    heads=[]
    for kind in ('formal','aligned240'):
        r=seal[kind];path=root/r['relative']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=True)
        head=SupervisedMLP().to(device);head.load_state_dict(saved['head']);head.eval().requires_grad_(False)
        heads.append(head)
    return heads,seal
