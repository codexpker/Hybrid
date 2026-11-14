"""One parameter-preserving T-anchor cross-view completion computation graph.

Original model/fusion code remains unchanged. Candidate checkpoints require an
explicit architecture receipt because their state keys intentionally equal V1.
"""
import hashlib
import json
from pathlib import Path
import torch
from .model import HybridV1
from .multiview import SelectedFusion

ARCHITECTURE='HYBRID_V1_T_ANCHOR_COMPLETION_V1'
INITIAL_SHA256='9a8f18ba57f64bc0ecbf17cab20d4fc1bc9758b433667f135de6eaa27ace6071'

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def positional_queries(like):
    """Parameter/buffer-free standard sinusoidal positions, distinct per slot."""
    n,d=like.shape[-2:]
    position=torch.arange(n,device=like.device,dtype=torch.float32)[:,None]
    frequency=torch.exp(torch.arange(0,d,2,device=like.device,dtype=torch.float32)*(-torch.log(torch.tensor(10000.,device=like.device))/d))
    phases=position*frequency[None]
    values=torch.zeros(n,d,device=like.device,dtype=torch.float32)
    values[:,0::2]=phases.sin();values[:,1::2]=phases[:,:values[:,1::2].shape[1]].cos()
    return values.to(like.dtype).expand(like.shape[0],-1,-1)

class TAnchorFusion(SelectedFusion):
    def forward(self,time,fractional,multiscale,time_valid,fractional_valid,ms_valid):
        if self.mode!='left_interaction' or self.author.n_layers!=1:
            raise ValueError('Only the original one-layer LEFT interaction recipe')
        # No in-place updates: all three reads use the same prefusion snapshots.
        raw_t,raw_f,raw_m=time,fractional,multiscale
        context=torch.cat((raw_f,raw_m),dim=1)
        context_valid=torch.cat((fractional_valid,ms_valid),dim=1)
        t=self._block(self.author.t_from_f[0],raw_t,context,time_valid,context_valid)
        f=self._block(self.author.f_from_t[0],positional_queries(raw_f),raw_t,fractional_valid,time_valid)
        m=self._block(self.author.m_from_t[0],positional_queries(raw_m),raw_t,ms_valid,time_valid)
        return t,f,m

class TAnchorHybridV1(HybridV1):
    architecture_identity=ARCHITECTURE
    def __init__(self,config,sources):
        super().__init__(config,sources)
        if self.config.fusion_layers!=1 or self.config.fusion!='left_interaction':
            raise ValueError('One fixed original fusion layer required')
        # Reuse the actual existing author modules, without init/RNG/new state.
        self.fusion.__class__=TAnchorFusion

    def _encode(self,*args,**kwargs):
        encoded=super()._encode(*args,**kwargs)
        # Reuse the exact original decoder while removing both own-view skips.
        encoded['fractional']=dict(encoded['fractional'])
        encoded['fractional']['native']=torch.zeros_like(encoded['fractional']['native'])
        encoded['f_base']=torch.zeros_like(encoded['f_base'])
        return encoded

def identity(model):
    if type(model) is not TAnchorHybridV1 or type(model.fusion) is not TAnchorFusion:
        raise PermissionError('Wrong model/fusion loader, even if keys match')
    return dict(architecture=ARCHITECTURE,implementation_sha256=file_sha(__file__),
                fixed_query='standard_sinusoidal_position_0_to_Nminus1_per_branch',
                same_original_parameter_keys=True,no_extra_trainable_parameters=True)

def checkpoint_identity(model,path):
    return dict(identity(model),checkpoint_sha256=file_sha(path))

def verify_checkpoint_metadata(path,metadata,implementation_sha256):
    if (metadata.get('architecture')!=ARCHITECTURE or
        metadata.get('implementation_sha256')!=implementation_sha256 or
        implementation_sha256!=file_sha(__file__) or
        metadata.get('checkpoint_sha256')!=file_sha(path)):
        raise PermissionError('Explicit candidate checkpoint/class/source identity required')

def construct_candidate(config,*,initial_path=None):
    from .config import HybridV1Config
    from .provenance import OfficialSources
    # Matches original cli.construct; worker exposes only its owned physical GPU.
    torch.manual_seed(config['seed'])
    model=TAnchorHybridV1(HybridV1Config(**config['model']),OfficialSources(**config['sources'])).to(config.get('device','cpu'))
    if initial_path is not None:
        if file_sha(initial_path)!=INITIAL_SHA256:raise PermissionError('Only original source initial9a8')
        model.load_state_dict(torch.load(initial_path,map_location=config.get('device','cpu'),weights_only=True),strict=True)
    identity(model)
    return model

def load_candidate(config,path,metadata,*,implementation_sha256):
    verify_checkpoint_metadata(path,metadata,implementation_sha256)
    model=construct_candidate(config)
    model.load_state_dict(torch.load(path,map_location=config.get('device','cpu'),weights_only=True),strict=True)
    identity(model)
    return model
