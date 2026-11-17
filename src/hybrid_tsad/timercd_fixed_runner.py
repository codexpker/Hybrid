"""CAL and QUERY run separately, original official model/head, zero updates."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from .timercd_fixed import ARMS,sha,detector_from_config,predict_with_cache


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):
    with Path(path).open('x',encoding='utf-8')as f:json.dump(value,f,indent=2,allow_nan=False)
def resolve(w,path):
    p=Path(path);return p if p.is_absolute()else w/p


class RoleReader:
    def __init__(self,w,cfg,rid,allowed):
        p=resolve(w,cfg['data_binding']);assert sha(p)==cfg['data_binding_sha256']
        self.record=next(x for x in read(p)['records']if x['id']==rid);self.w=w;self.allowed=set(allowed);self.access=[]
    def load(self,role):
        if role not in self.allowed:raise PermissionError('Role not authorized before decoding')
        entry=next(x for x in self.record['roles']if x['role']==role);p=resolve(self.w,entry['path']);assert sha(p)==entry['sha256']
        with np.load(p,allow_pickle=False)as z:x,index=z['x'].copy(),z['index'].copy()
        assert x.dtype==np.float64 and list(x.shape)==entry['shape']and np.isfinite(x).all()
        self.access.append(dict(role=role,path=entry['path'],sha256=entry['sha256'],fields=['x','index']))
        return x,index
    def fit_statistics(self):
        entry=self.record['fit_statistics'];p=resolve(self.w,entry['path']);assert sha(p)==entry['sha256']
        with np.load(p,allow_pickle=False)as z:mean,std=z['mean'].copy(),z['official_safe_std'].copy()
        self.access.append(dict(role='FIT_STATISTICS',path=entry['path'],sha256=entry['sha256'],fields=['mean','official_safe_std']))
        return mean,std


def state_hash(model):
    import hashlib
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()for k,v in model.state_dict().items()}


def parity(w,cfg,out,rid,device):
    import torch
    reader=RoleReader(w,cfg,rid,['FIT']);raw,index=reader.load('FIT');raw,index=raw[:5017],index[:5017]
    detector=detector_from_config(w,cfg,raw.shape[1],device)
    before=state_hash(detector._tester.model)
    # Witness executes the actual original API, not a replacement fixture.
    witness_prob,witness_diff=detector.predict(raw,return_logits=True)
    actual,normalization=predict_with_cache(detector,raw,index,'A_official_role')
    assert np.array_equal(actual['score'],witness_prob)and np.array_equal(actual['logit'],witness_diff)
    assert len(actual['score'])==len(raw)and len(actual['window_starts_offset'])==2
    assert int(actual['padding_points'])==4983
    mean,std=reader.fit_statistics();fit_bank,fit_norm=predict_with_cache(detector,raw,index,'B_fit_only',mean,std)
    assert fit_norm['normalization_count']==1
    assert state_hash(detector._tester.model)==before
    import time_rcd, time_rcd._inference, time_rcd._core.ts_encoder_bi_bias as encoder
    dump(out/'PARITY_RECEIPT.json',dict(status='PASS_ACTUAL_OFFICIAL_API_BITEXACT',record=rid,
        official_module_file=time_rcd.__file__,inference_file=time_rcd._inference.__file__,encoder_file=encoder.__file__,
        encoder_sha256=sha(encoder.__file__),checkpoint_sha256=cfg['checkpoint_sha256'],roles=reader.access,
        prob_bitexact=True,raw_head_classdiff_bitexact=True,model_unchanged=True,
        outer_padding=4983,internal_patch_padding=int(actual['internal_patch_padding']),B_norm_count=1,
        A_normalization=normalization,B_normalization=fit_norm))
    np.savez_compressed(out/'PARITY.npz',**actual,B_prob=fit_bank['score'])


def score(w,cfg,out,rid,arm,device):
    reader=RoleReader(w,cfg,rid,['CAL','QUERY']);mean,std=reader.fit_statistics()if arm=='B_fit_only'else(None,None)
    detector=detector_from_config(w,cfg,reader.record['channels'],device);before=state_hash(detector._tester.model);banks=[];norms={}
    for role in ['CAL','QUERY']:
        values,index=reader.load(role)
        bank,norm=predict_with_cache(detector,values,index,arm,mean,std);norms[role]=norm
        path=out/(role+'.npz');np.savez_compressed(path,**bank)
        banks.append(dict(record=rid,arm=arm,role=role,path=str(path),sha256=sha(path),points=len(index),
                          padding_points=int(bank['padding_points']),effective_window=int(bank['effective_window'])))
    assert state_hash(detector._tester.model)==before
    dump(out/'SCORE_RECEIPT.json',dict(status='COMPLETED',record=rid,arm=arm,banks=banks,roles=reader.access,
        checkpoint_sha256=cfg['checkpoint_sha256'],normalization=norms,model_unchanged=True,optimizer_updates=0,
        window=5000,batch_size=1,official_forward_and_head=True,primary_score='original softmax(mean_channel(head_logits))[class1]',
        context='separate complete CAL and QUERY roles; no concatenation',timing='offline whole-role statistics for A; offline full-window for B',
        no_labels_decoded=True,no_threshold_fitting=True))


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['parity','score']);p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--record',required=True);p.add_argument('--arm',choices=ARMS,default=ARMS[0]);p.add_argument('--device',default='cpu')
    a=p.parse_args();w=a.workspace.resolve();cfg=read(a.config);device=torch.device(a.device)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    out=w/cfg['output']/a.record/(a.arm if a.phase=='score'else'parity');out.mkdir(parents=True,exist_ok=False)
    begin=time.time();rc=0
    if device.type=='cuda':
        torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(30*2**30/torch.cuda.get_device_properties(device).total_memory,device)
        torch.cuda.reset_peak_memory_stats(device)
    dump(out/'START.json',dict(pid=os.getpid(),phase=a.phase,record=a.record,arm=a.arm,config_sha256=sha(a.config),runner_sha256=sha(__file__),
        inference_sha256=sha(Path(__file__).with_name('timercd_fixed.py')),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        if a.phase=='parity':parity(w,cfg,out,a.record,device)
        else:score(w,cfg,out,a.record,a.arm,device)
    except BaseException:
        rc=1;dump(out/'FAILED.json',dict(traceback=traceback.format_exc()));raise
    finally:
        end=time.time();dump(out/'OS.json',dict(pid=os.getpid(),exit_code=rc,start_unix=begin,end_unix=end,seconds=end-begin,
            GPU_hours=(end-begin)/3600 if device.type=='cuda'else 0,peak_allocated_bytes=torch.cuda.max_memory_allocated(device)if device.type=='cuda'else 0))


if __name__=='__main__':main()
