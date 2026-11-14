"""Paired ordinary native-LEFT sensitivity utilities; no robustness mechanism."""
from contextlib import contextmanager
import random
import numpy as np

def paired_plan(length, positive_variance_channels):
    high=length-192+1
    if high!=14032:raise ValueError('fixed SMD FIT geometry mismatch')
    base=np.random.default_rng(2025).integers(0,high,size=(200,128))
    rng=np.random.default_rng(84551);selected=rng.choice(25600,size=2560,replace=False)
    donors=np.full((200,128),-1,np.int64);mask=np.zeros((200,128),bool)
    for flat in selected:
        row,col=divmod(int(flat),128);candidate=int(rng.integers(0,high))
        while abs(candidate-int(base[row,col]))<192:candidate=int(rng.integers(0,high))
        donors[row,col]=candidate;mask[row,col]=True
    channels=np.asarray(sorted(positive_variance_channels)[:4],np.int64)
    if len(channels)!=4:raise ValueError('four positive FIT variance channels required')
    return dict(base_starts=base,donor_starts=donors,replacement_mask=mask,selected_flat_order=selected,corruption_channels=channels)

def native_batch(values,plan,update,condition):
    if condition not in ['A','B','C']:raise ValueError('exact A/B/C only')
    starts=plan['base_starts'][update]
    if condition!='A':starts=np.where(plan['replacement_mask'][update],plan['donor_starts'][update],starts)
    x=np.stack([values[int(s):int(s)+192]for s in starts]).astype(np.float32)
    if condition=='B':
        slots=np.flatnonzero(plan['replacement_mask'][update])
        for channel in plan['corruption_channels']:x[slots,48:144,int(channel)]+=np.float32(3.)
    return x

@contextmanager
def diagnostic_eval(model,torch):
    """No update/selection: restore modes, step, transient attrs and all RNGs."""
    modes={name:module.training for name,module in model.named_modules()};step=model.step
    debug=model.last_x_res;cache=model._ms_cache.copy()
    cpu=torch.get_rng_state().clone();cuda=[s.clone()for s in torch.cuda.get_rng_state_all()]
    py=random.getstate();numpy=np.random.get_state()
    try:
        model.eval()
        with torch.no_grad():yield
    finally:
        for name,module in model.named_modules():module.training=modes[name]
        model.step=step;model.last_x_res=debug;model._ms_cache=cache
        torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda);random.setstate(py);np.random.set_state(numpy)
