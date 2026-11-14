"""Independent LEFT-inspired joint/dual-cycle learning core; V1 is untouched.

Official TimeDecoder/FreqDecoder structures are instantiated per channel (V=1).
Frequency axes are raw complex FRFT coefficients [100,1], not LEFT STFT bins.
No prototype memory, EMA, new scoring weights, or target-trained LEFT weights.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import torch
from torch import nn
from .model import HybridV1
from .multiview import masked_pool,decode_scales
from .objectives import ssl_objective as original_ssl,elementwise

ARCHITECTURE='HYBRID_LEFT_JOINT_FRFT_CYCLE_V2'
U_MODULES=('fusion','joint_fusion','time_cycle_decoder','fractional_cycle_decoder','ms_decoder','forecast_head')

class JointCycleHybrid(HybridV1):
    architecture_identity=ARCHITECTURE
    def __init__(self,config,sources):
        super().__init__(config,sources)
        cfg=self.config
        if cfg.fusion!='left_interaction' or cfg.fusion_layers!=1 or cfg.reconstruction_projection!='real':
            raise ValueError('Only original one-layer interaction and signed-real inverse')
        tools,_=self.sources.left()
        # Remove the actual superseded reconstruction heads, never leave them
        # registered as if the new objective trained them.
        del self.tf_fusion_delta
        del self.tf_backbone.author.decoder
        self.joint_fusion=nn.Sequential(nn.LayerNorm(2*cfg.width),nn.Linear(2*cfg.width,2*cfg.width),nn.GELU(),nn.Linear(2*cfg.width,cfg.width))
        self.time_cycle_decoder=tools.TimeDecoder(ts=cfg.local_window,vs=1,z_dim=cfg.width,d_model=cfg.width)
        self.fractional_cycle_decoder=tools.FreqDecoder(vs=1,Freq=cfg.local_window,TT=1,z_dim=cfg.width,d_model=cfg.width)
        self._adaptation_route='STAGE1'

    def train(self,mode=True):
        super().train(mode)
        if mode and getattr(self,'_adaptation_route',None) in ('U','F'):
            allowed=U_MODULES+(('event_head',) if self._adaptation_route=='F' else ())
            for name,child in self.named_children():child.train(name in allowed)
        return self

    def _encode(self,*args,**kwargs):
        # Lexically scoped capture of this exact invocation, with no persistent
        # tensors/cache and no modification to original encoder/fusion code.
        captured=[]
        def capture(module,inputs,outputs):
            t,f,_=outputs;captured.append((masked_pool(t,inputs[3]),masked_pool(f,inputs[4])))
        handle=self.fusion.register_forward_hook(capture)
        try:e=super()._encode(*args,**kwargs)
        finally:handle.remove()
        if len(captured)!=1:raise RuntimeError('One fusion call per encode required')
        t,f=captured[0];b,k,c,d=e['t_features'].shape
        e['joint_latent']=self.joint_fusion(torch.cat((t,f),-1)).reshape(b,k,c,d)
        return e

    def reconstruct(self,context,observed_mask=None,hide_mask=None,mask_seed=0,*,copy_masks=None):
        observed=self._validate(context,observed_mask);hidden=self.make_hidden_masks(observed,hide_mask,mask_seed=mask_seed)
        cfg=self.config;b,h,c=context.shape;rows=[]
        if copy_masks is not None and copy_masks.shape!=(b,hidden.shape[1],cfg.copies,c,cfg.local_window//cfg.tf_patch):raise ValueError('Copy mask axes')
        for group in range(hidden.shape[1]):
            e=self._encode(context,observed,hidden[:,group],mask_seed=int(mask_seed)+group,copy_masks=None if copy_masks is None else copy_masks[:,group])
            z=e['joint_latent'].reshape(b*cfg.copies*c,cfg.width)
            dt=self.time_cycle_decoder(z).reshape(b,cfg.copies,c,cfg.local_window).permute(0,1,3,2)
            df=self.fractional_cycle_decoder(z).reshape(b,cfg.copies,c,2,cfg.local_window)
            spectrum=torch.complex(df[:,:,:,0],df[:,:,:,1])
            synth=[];analysis=[]
            for i in range(b):
                order=e['fractional']['order'][i]
                synth.append(self.tf_backbone.ifrft(spectrum[i],order,dim=-1).real.transpose(-1,-2))
                analysis.append(self.tf_backbone.frft(dt[i].transpose(-1,-2).to(torch.complex64),order,dim=-1))
            tf_norm=torch.stack(synth);time_analysis=torch.stack(analysis)
            ms=decode_scales(self.ms_decoder,e['fused_ms'],e['ms_axes'],cfg)
            ms=ms.reshape(b,cfg.copies,c,len(cfg.ms_factors),h).permute(0,1,3,4,2)
            mean,scale=e['stats']['mean'],e['stats']['scale']
            tf_ok=e['fractional']['valid'].any(-1)[:,:,None,:].expand(-1,-1,cfg.local_window,-1)
            ms_ok=e['stats']['channel_valid'][:,None,None].expand(-1,cfg.copies,len(cfg.ms_factors),h,-1)
            rows.append(dict(tf_reconstruction=tf_norm*scale[:,None]+mean[:,None],tf_normalized=tf_norm,
                ms_reconstruction_by_scale=ms*scale[:,None,None]+mean[:,None,None],ms_normalized_by_scale=ms,
                tf_valid=tf_ok,ms_valid_by_scale=ms_ok,t_features=e['t_features'],f_features=e['f_features'],
                visible_mask=e['visible'],order=e['fractional']['order'],transform_hidden_mask=e['fractional']['transform_hidden_mask'],
                stats=e['stats'],axes=e['axes'],joint_latent=e['joint_latent'],time_cycle_normalized=dt,
                cycle_analysis=time_analysis,cycle_spectrum=spectrum))
        keys=('tf_reconstruction','tf_normalized','ms_reconstruction_by_scale','ms_normalized_by_scale','tf_valid','ms_valid_by_scale',
              't_features','f_features','visible_mask','order','transform_hidden_mask','joint_latent','time_cycle_normalized','cycle_analysis','cycle_spectrum')
        result={key:torch.stack([r[key] for r in rows],1) for key in keys}
        result['stats']={key:torch.stack([r['stats'][key] for r in rows],1) for key in rows[0]['stats']}
        result['ms_reconstruction']=result['ms_reconstruction_by_scale'].mean(3)
        result['ms_normalized']=result['ms_normalized_by_scale'].mean(3)
        result['ms_valid']=result['ms_valid_by_scale'].all(3)
        own=hidden[:,:,-1].to(context.dtype)
        for key in ('t_features','f_features'):result['endpoint_'+key]=(result[key].mean(2)*own[...,None]).sum(1)
        result.update(hidden_mask=hidden,observed_mask=observed,axes=[r['axes'] for r in rows],pass_name='current_arrived_endpoint_hidden_reconstruction',
            group_count=len(rows),normalization_scope='visible_student_only',reconstruction_projection='real',architecture=ARCHITECTURE)
        return result

def joint_objective(forecast,reconstruction,batch,kind='huber',active_components=None,*,frft):
    if active_components is not None and set(active_components)!={'r_tf','r_ms','r_pred'}:raise ValueError('No unregistered component ablations')
    base=original_ssl(forecast,reconstruction,batch,kind,active_components)
    obs=batch['reconstruction_observed'];w=reconstruction['time_cycle_normalized'].shape[-2]
    gt=torch.where(obs,batch['reconstruction_context'],0)
    norm=(gt[:,None]-reconstruction['stats']['mean'])/reconstruction['stats']['scale']
    # One complete observed local channel is a legitimate spectral target.
    # Natural missingness is never imputed into a pseudo-ground-truth spectrum.
    support=obs[:,-w:].all(1);pred=reconstruction['cycle_analysis'];targets=[]
    model_frft=frft
    for b in range(len(gt)):
        targets.append(torch.stack([model_frft(norm[b,g,-w:].transpose(0,1).to(torch.complex64),reconstruction['order'][b,g],dim=-1) for g in range(norm.shape[1])]))
    target=torch.stack(targets)[:,:,None].expand_as(pred)
    mask=support[:,None,None,:,None].expand_as(pred)
    error=torch.where(mask,pred-target,torch.zeros_like(pred))
    count=mask.flatten(1).sum(1)*2
    values=(elementwise(error.real,kind)+elementwise(error.imag,kind))*mask
    per=values.flatten(1).sum(1)/count.clamp_min(1)
    analysis=per[count>0].mean() if bool((count>0).any()) else pred.real.sum()*0
    parts=dict(base['parts'],analysis=analysis,synthesis=base['parts']['tf'])
    base['valid_counts']['analysis']=count
    base['per_window']['analysis']=per
    # An unavailable optional spectral target never invalidates the original
    # observed time/MS/forecast supervision or changes the frozen sample plan.
    # Analysis is exactly zero above with count=0; no imputed pseudo-spectrum.
    base.update(total=.5*(analysis+parts['synthesis'])+.5*parts['ms']+.1*parts['consistency']+parts['pred'],parts=parts,
        spectral_target_complete_channel_count=support.sum(),cycle_weights=dict(analysis=.5,synthesis=.5,ms=.5,consistency=.1,pred=1.))
    return base

def update_policy(model,route):
    if type(model) is not JointCycleHybrid or route not in ('STAGE1','Z','U','F'):raise ValueError('Explicit candidate/route required')
    model._adaptation_route=route;allowed=U_MODULES+(('event_head',) if route=='F' else ())
    for name,p in model.named_parameters():p.requires_grad_(route=='STAGE1' or (route in ('U','F') and name.split('.')[0] in allowed))
    if route=='Z':model.eval()
    return [n for n,p in model.named_parameters() if p.requires_grad]

@contextmanager
def training_context(model):
    from . import training
    from functools import partial
    old_policy,old_objective=training.update_policy,training.ssl_objective
    training.update_policy=update_policy
    training.ssl_objective=partial(joint_objective,frft=model.tf_backbone.frft)
    try:yield training
    finally:
        training.update_policy,training.ssl_objective=old_policy,old_objective

def fit(model,fit_runs,select_runs,**kwargs):
    with training_context(model) as training:return training.fit(model,fit_runs,select_runs,**kwargs)

def adapt(source,route,fit_runs=None,select_runs=None,**kwargs):
    import copy
    model=copy.deepcopy(source)
    if route=='F' and kwargs.get('support',{}).get('K')==0:route='Z'
    if route=='Z':
        update_policy(model,'Z')
        return model,dict(route='Z',updates=0,target_fit=False,architecture=ARCHITECTURE)
    if route not in ('U','F'):raise ValueError('Adaptation is only Z/U/F')
    return fit(model,fit_runs,select_runs,route=route,**kwargs)

def construct(config,*,shared_initial=None,expected_initial_sha256=None):
    """Seeded new heads, explicit exact shared-9a8 restoration; never full-V1 load."""
    from .config import HybridV1Config
    from .provenance import OfficialSources
    torch.manual_seed(config['seed'])
    model=JointCycleHybrid(HybridV1Config(**config['model']),OfficialSources(**config['sources']))
    mapping=None
    if shared_initial is not None:
        if hashlib.sha256(Path(shared_initial).read_bytes()).hexdigest()!=expected_initial_sha256:raise ValueError('Shared initial bytes changed')
        old=torch.load(shared_initial,map_location='cpu',weights_only=True);state=model.state_dict()
        common=sorted(set(state)&set(old));new=sorted(set(state)-set(old));removed=sorted(set(old)-set(state))
        if not all(n.startswith(('joint_fusion.','time_cycle_decoder.','fractional_cycle_decoder.')) for n in new):raise ValueError('Unexpected new tensors')
        if not all(n.startswith(('tf_fusion_delta.','tf_backbone.author.decoder.')) for n in removed):raise ValueError('Unexpected removal')
        for n in common:
            if state[n].shape!=old[n].shape or state[n].dtype!=old[n].dtype:raise ValueError('Shared tensor shape/dtype changed')
            state[n]=old[n]
        model.load_state_dict(state,strict=True)
        if not all(torch.equal(model.state_dict()[n],old[n]) for n in common):raise AssertionError('Shared init not exact')
        mapping=dict(shared_initial_sha256=expected_initial_sha256,shared_names=common,new_names=new,removed_names=removed,shared_exact=True,
                     complete_old_state_exact=False,new_heads_seed=config['seed'])
    return model.to(config.get('device','cpu')),mapping

def save_checkpoint(model,path,metadata_path):
    import json
    torch.save(model.state_dict(),path)
    meta={**identity(model),'checkpoint_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'model_config':model.config.to_dict()}
    with Path(metadata_path).open('x',encoding='utf-8') as f:json.dump(meta,f,indent=2)
    return meta

def load_checkpoint(config,path,metadata_path):
    import json
    from .config import HybridV1Config
    meta=json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    if meta.get('architecture')!=ARCHITECTURE or meta.get('implementation_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise PermissionError('Wrong architecture/implementation')
    canonical=json.loads(json.dumps(HybridV1Config(**config['model']).to_dict()))
    if meta.get('checkpoint_sha256')!=hashlib.sha256(Path(path).read_bytes()).hexdigest() or meta.get('model_config')!=canonical:raise PermissionError('Checkpoint/config identity mismatch')
    devices=[torch.cuda.current_device()] if str(config.get('device','cpu')).startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):model,_=construct(config)
    model.load_state_dict(torch.load(path,map_location=config.get('device','cpu'),weights_only=True),strict=True)
    return model.eval()

def identity(model):
    if type(model) is not JointCycleHybrid:raise PermissionError('Wrong architecture despite overlapping keys')
    return dict(architecture=ARCHITECTURE,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        r_tf='inverse_FRFT_joint_frequency_decoder_signed_real',r_ms='original_multiscale_reconstruction',r_pred='original_causal_P32_forecast_h1',
        per_channel_decoder=True,prototype_memory=False,source_weight_initialization='explicit shared9a8 mapping; new joint/decoder seed2025')
