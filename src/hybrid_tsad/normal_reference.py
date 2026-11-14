"""Fixed source-FIT state retrieval and native-LEFT decoder bottleneck.

Normal-memory encoder/decoder bottlenecks are established mechanisms (MemAE).
This controlled reuse is not a new memory-learning method or official MemAE.
"""
import types
import numpy as np

def state_features(values,ends):
    ends=np.asarray(ends,np.int64)
    if len(ends)==0 or ends.min()<191 or ends.max()>=len(values):raise ValueError('full causal192 context required')
    cs=np.concatenate([np.zeros((1,values.shape[1])),np.cumsum(values,axis=0)])
    ss=np.concatenate([np.zeros((1,values.shape[1])),np.cumsum(values*values,axis=0)])
    lo=ends-191;mu=(cs[ends+1]-cs[lo])/192
    std=np.sqrt(np.maximum((ss[ends+1]-ss[lo])/192-mu*mu,0))
    return np.concatenate([mu,std,values[ends]],axis=1)

def nearest_reference(features,bank,*,query_starts=None,reference_starts=None,guard=544,block=512):
    """Float64 RMS-L2, first chronological tie, optional leave-block-out."""
    x=np.asarray(features,np.float64);b=np.asarray(bank,np.float64)
    if x.shape[1]!=b.shape[1]:raise ValueError('feature dimensions differ')
    if (query_starts is None)!=(reference_starts is None):raise ValueError('both footprint arrays required')
    norm=(b*b).sum(axis=1);indices=np.empty(len(x),np.int64);dist=np.empty(len(x))
    for a in range(0,len(x),block):
        q=x[a:a+block];squared=(q*q).sum(axis=1)[:,None]+norm[None,:]-2*q@b.T
        if query_starts is not None:
            starts=np.asarray(query_starts[a:a+len(q)],np.int64)[:,None];refs=np.asarray(reference_starts,np.int64)[None,:]
            legal=(refs+192<=starts-guard)|(refs>=starts+192+guard)
            squared=np.where(legal,squared,np.inf)
        chosen=squared.argmin(axis=1);minimum=squared[np.arange(len(q)),chosen]
        if not np.isfinite(minimum).all():raise ValueError('no legal normal reference')
        indices[a:a+len(q)]=chosen;dist[a:a+len(q)]=np.sqrt(np.maximum(minimum,0)/x.shape[1])
    return indices,dist

def freeze_to_decoders(model):
    """Global eval disables native explicit EMA; only three heads train."""
    for p in model.parameters():p.requires_grad_(False)
    for name in ['ms_decoder','Dt','Df']:
        for p in getattr(model,name).parameters():p.requires_grad_(True)
    decoder_mode(model,False)

def decoder_mode(model,training):
    model.eval()
    for name in ['ms_decoder','Dt','Df']:getattr(model,name).train(training)
    assert not model.training
    for name,sub in model.named_modules():
        allowed=any(name==head or name.startswith(head+'.')for head in ['ms_decoder','Dt','Df'])
        if sub.training and not allowed:raise AssertionError('frozen module entered train mode: '+name)

def native_raw_reference_score(model,current,reference):
    """Zero-trained comparator, same complete domain-native scoring incl gate."""
    result=model._forward(current,None,None,None)
    current_gt,_,lengths,_,_,_,_,_,_,_,cache=result
    ref_scales,_=model.ms_utils(reference,return_residual=True)
    ref_ms=model.ms_utils.concat_sampling_list(ref_scales[1:]+[reference])
    return model._ms_anomaly_score(ref_ms,current_gt,lengths,reference,reference,current,reference,cache)

class NativeReferenceHook:
    """Intercept exactly one pre-up fused-MS value and both joint decoder inputs."""
    def __init__(self,model,*,mode,bank_ms=None,bank_joint=None):
        if mode not in ['capture','control','bottleneck']:raise ValueError(mode)
        self.model=model;self.mode=mode;self.bank_ms=bank_ms;self.bank_joint=bank_joint;self.references=None;self.counts={};self.captured={}
        original=model.ms_utils.split_2_list;self.original_split=original
        def split(module,x,ms_t_lens,mode='encoder'):
            if mode=='encoder':
                self.counts['MS']=self.counts.get('MS',0)+1
                if self.counts['MS']!=1:raise AssertionError('unexpected additional encoder split')
                channels=model.enc_in;batch=x.shape[0]//channels
                if x.shape[0]!=batch*channels:raise AssertionError('MS batch/channel layout')
                if self.mode=='capture':self.captured['MS']=x.detach().reshape(batch,channels,x.shape[1],x.shape[2]).cpu().clone()
                elif self.mode=='bottleneck':
                    self._check_batch(batch);x=self.bank_ms[self.references].reshape_as(x)
            return original(x,ms_t_lens,mode=mode)
        model.ms_utils.split_2_list=types.MethodType(split,model.ms_utils)
        self.handles=[model.Dt.register_forward_pre_hook(self._joint('Dt')),model.Df.register_forward_pre_hook(self._joint('Df'))]
    def _check_batch(self,batch):
        if self.references is None or len(self.references)!=batch:raise AssertionError('set immutable reference IDs before forward')
    def _joint(self,name):
        def hook(module,args):
            self.counts[name]=self.counts.get(name,0)+1
            if self.counts[name]!=1:raise AssertionError('unexpected repeated joint decoder')
            z=args[0]
            if self.mode=='capture':
                self.captured[name]=z.detach().cpu().clone()
                if name=='Df'and not self.captured['Dt'].equal(self.captured['Df']):raise AssertionError('native joint paths differ')
            if self.mode=='bottleneck':self._check_batch(len(z));return(self.bank_joint[self.references],)+args[1:]
        return hook
    def begin(self,references=None):
        self.references=references;self.counts={};self.captured={}
    def finish(self):
        if self.counts!={'MS':1,'Dt':1,'Df':1}:raise AssertionError(self.counts)
        return self.captured
    def remove(self):
        self.model.ms_utils.split_2_list=self.original_split
        for h in self.handles:h.remove()
