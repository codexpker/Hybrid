"""DONOR-only normal qualification and frozen scoring; no calibration or training."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):
    with Path(path).open('x',encoding='utf-8')as f:json.dump(value,f,indent=2,allow_nan=False)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def normal_intervals(lo,hi,starts,ends):
    """Return half-open normal runs inside this role; all rules precede scores."""
    cursor=lo;normal=[]
    for start,end in zip(starts,ends):
        if end<=lo or start>=hi:continue
        if start>cursor:normal.append((cursor,min(int(start),hi)))
        cursor=max(cursor,min(int(end),hi))
    if cursor<hi:normal.append((cursor,hi))
    return normal


def prepare(workspace,out):
    old=workspace/'preparations/left_multievent_mixed_u_20261111_v1'
    meta=read(old/'INPUT_QUALIFICATION.json');rows={x['role']:x for x in meta['roles']}
    assert(rows['DONOR']['original_start'],rows['DONOR']['end'])==(495000,607479)
    assert rows['FIT']['end']<495000 and rows['SELECT']['end']<495000
    plan=old/'PAIRED_PLAN.npz';gate=read(old/'DATA_PLAN_GATE.json')
    assert sha(plan)==gate['plan_sha256']
    # Read only the thirteen frozen DONOR event bounds, never the full label array.
    with np.load(plan,allow_pickle=False)as z:
        starts=z['donor_event_starts'].copy();ends=z['donor_event_ends'].copy()
    assert len(starts)==len(ends)==13 and np.all(starts>=495000)and np.all(ends<=607479)
    assert np.all(starts[1:]>=ends[:-1])
    index=[];roles=[];sections=[];detail=[];section_id=0
    for role,(lo,hi)in [('CAL',(495000,550695)),('CHECK',(551783,607479))]:
        for start,end in normal_intervals(lo,hi,starts,ends):
            eligible_start,eligible_end=start+512,end-512
            n=max(0,eligible_end-eligible_start)
            detail.append(dict(role=role,normal_start=start,normal_end=end,eligible_start=eligible_start,
                eligible_end=eligible_end,points=n,eligible=n>0,section_id=section_id))
            if n:
                ix=np.arange(eligible_start,eligible_end,dtype=np.int64)
                assert ix[0]-191>=start and ix[-1]<end
                index.append(ix);roles.append(np.full(n,0 if role=='CAL'else 1,np.int8));sections.append(np.full(n,section_id,np.int32))
            section_id+=1
    ix=np.concatenate(index);rr=np.concatenate(roles);ss=np.concatenate(sections)
    assert len(ix)>0 and np.all(np.diff(ix)>0)
    assert all(np.sum(rr==r)>=512 for r in(0,1))
    oldtrain=workspace/'results/left_multievent_mixed_u_20261111_v1/left_train'
    native=read(oldtrain/'RUNTIME_STATE.json');receipt=read(oldtrain/'RECEIPT.json')
    assert [x['role']for x in receipt['roles']]==['FIT','SELECT']and receipt['CAL_QUERY_not_accessed']
    assert receipt['checkpoint_sha256']==native['checkpoint_sha256']
    cfg=read(old/'configs/SWAT/model_configs_Left.json');assert cfg['seq_len']==192 and cfg['enc_in']==31
    densitytrain=workspace/'results/hybrid_normality_density_20261115_v1/SWAT/full/train'
    d=read(densitytrain/'TRAIN_RECEIPT.json')
    assert[x['role']for x in d['roles']]==['FIT','SELECT']and d['CAL_QUERY_unopened']
    assert sha(old/'FIT_ONLY_SCALER.json')==d['scaler_sha256']
    assert sha(old/'data/DONOR.npz')==rows['DONOR']['sha256']
    f=out/'SWAT_ALLOWED_ENDPOINTS.npz';np.savez_compressed(f,index=ix,role=rr,section_id=ss)
    dump(out/'SWAT_INDEX_MANIFEST.json',dict(status='QUALIFIED_AND_FROZEN_BEFORE_NEW_SCORES',
         index_path=str(f),index_sha256=sha(f),donor_event_plan_sha256=sha(plan),
         donor_role=rows['DONOR'],normal_labels_source='frozen donor_event_starts/ends only; full QUERY label array never loaded',
         raw_roles=dict(CAL=[495000,550695],CHECK=[551783,607479]),
         midpoint=551239,split_guard_each_side=544,normal_run_boundary_erosion_each_side=512,
         intervals=detail,counts={role:int(np.sum(rr==i))for i,role in enumerate(['CAL','CHECK'])},
         all_eligible_points_retained=True,selection_uses_no_scores=True,history=192,
         density_checkpoint_sha256=d['selected_sha256'],native_checkpoint_sha256=native['checkpoint_sha256'],
         native_final_python_step=native['final_python_step'],scaler_sha256=d['scaler_sha256'],
         native_model_config_sha256=sha(old/'configs/SWAT/model_configs_Left.json'),
         source_density_commit='d77252e17509d263d6d87a64c98b79516a935fd3',
         QUERY_values_or_labels_never_accessed=True,threshold_not_fitted=True))


def score(workspace,prep,out,method,device_name):
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    device=torch.device(device_name)
    if device.type=='cuda':
        torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(30*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
        torch.cuda.reset_peak_memory_stats(device)
    seed=2025;torch.manual_seed(seed);np.random.seed(seed)
    manifest=read(prep/'SWAT_INDEX_MANIFEST.json');f=prep/'SWAT_ALLOWED_ENDPOINTS.npz'
    assert sha(f)==manifest['index_sha256']
    sources=read(prep/'SWAT_SOURCE_IDENTITIES.json')
    for name,row in sources.items():
        path=workspace/row['relative'];assert sha(path)==row['sha256'],name
    with np.load(f,allow_pickle=False)as z:index=z['index'].copy();roles=z['role'].copy();section=z['section_id'].copy()
    old=workspace/'preparations/left_multievent_mixed_u_20261111_v1'
    raw=old/'data/DONOR.npz';assert sha(raw)==manifest['donor_role']['sha256']
    scaler=read(old/'FIT_ONLY_SCALER.json');assert sha(old/'FIT_ONLY_SCALER.json')==manifest['scaler_sha256']
    with np.load(raw,allow_pickle=False)as z:
        assert z['observed'].all();values=np.float32((z['values'].astype(np.float64)-np.asarray(scaler['mean']))/np.asarray(scaler['scale']))
    from hybrid_tsad.normality_density import NormalityDensity
    checkpoint=None;before=None
    if method=='LEFT_native':
        base=module('swat_frozen_native',old/'native_baseline.py');model=base.native(torch).to(device)
        checkpoint=workspace/'results/left_multievent_mixed_u_20261111_v1/left_train/model/SWAT/Left/checkpoints_0/checkpoint.pth'
        assert sha(checkpoint)==manifest['native_checkpoint_sha256']
        loaded=model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=False)
        assert not loaded.missing_keys and all(k.endswith(('total_ops','total_params'))for k in loaded.unexpected_keys)
        model.step=manifest['native_final_python_step']
    else:
        model=NormalityDensity(31,192,'full',hidden=64).to(device)
        if method=='Hybrid_full':
            checkpoint=workspace/'results/hybrid_normality_density_20261115_v1/SWAT/full/train/selected.pth'
            assert sha(checkpoint)==manifest['density_checkpoint_sha256']
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert state['domain']=='SWAT'and state['variant']=='full'
            model.load_state_dict(state['state'],strict=True)
        else:assert method=='Student_identity'
    model.eval();before={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    # Reproduce the original first mature CAL batch, without querying QUERY.
    qualification=read(old/'INPUT_QUALIFICATION.json');calrow=next(r for r in qualification['roles']if r['role']=='CAL')
    calfile=old/'data/CAL.npz';assert sha(calfile)==calrow['sha256']
    with np.load(calfile,allow_pickle=False)as z:
        assert z['observed'].all();cal=np.float32((z['values'].astype(np.float64)-np.asarray(scaler['mean']))/np.asarray(scaler['scale']))
    if method=='LEFT_native':
        cached=workspace/'results/left_multievent_mixed_u_20261111_v1/left_score/CAL.npz'
        with np.load(cached,allow_pickle=False)as z:cache_index=z['index'][:256].copy();expected=z['score'][:256].copy()
    else:
        variant='full'if method=='Hybrid_full'else'identity'
        cached=workspace/f'results/hybrid_normality_density_20261115_v1/SWAT/{variant}/score/CAL.npz'
        with np.load(cached,allow_pickle=False)as z:cache_index=z['index'][0,:128].copy();expected=z['raw_score'][0,:128].copy()
    end=cache_index-calrow['original_start']
    x=torch.from_numpy(np.stack([cal[t-192:t+1]for t in end])).to(device)
    with torch.no_grad():
        if method=='LEFT_native':
            raw_score=model.infer(x[:,-192:],None,None,None)[0].cpu().numpy();_,flat=base.native_score.native_numpy_scores([raw_score],31)
            reproduced=flat.reshape(len(end),192)[:,-1]
        else:reproduced=(model.identity_score(x)if method=='Student_identity'else model.score(x)).cpu().numpy().mean(-1)
    max_error=float(np.max(np.abs(reproduced-expected)))
    assert np.allclose(reproduced,expected,rtol=1e-5,atol=1e-6),(method,max_error)
    parts=[];channels=[];start=time.time()
    with torch.no_grad():
        for begin in range(0,len(index),128):
            ix=index[begin:begin+128];local=ix-495000
            assert np.all(local>=192)and np.all(local<len(values))
            # Density's accepted H+1 interface discards first point internally.
            x=torch.from_numpy(np.stack([values[t-192:t+1]for t in local])).to(device)
            if method=='LEFT_native':
                raw_score=model.infer(x[:,-192:],None,None,None)[0].cpu().numpy()
                _,flat=base.native_score.native_numpy_scores([raw_score],31)
                channel=raw_score[:,-1];s=flat.reshape(len(ix),192)[:,-1]
            else:
                channel=(model.identity_score(x)if method=='Student_identity'else model.score(x)).cpu().numpy();s=channel.mean(-1)
            assert s.shape==(len(ix),)and np.isfinite(s).all()
            parts.append(s);channels.append(channel)
    if device.type=='cuda':torch.cuda.synchronize(device)
    elapsed=time.time()-start
    assert all(torch.equal(before[k],v.detach().cpu())for k,v in model.state_dict().items())
    if method=='LEFT_native':assert model.step==manifest['native_final_python_step']
    if checkpoint:assert sha(checkpoint)==manifest['native_checkpoint_sha256'if method=='LEFT_native'else'density_checkpoint_sha256']
    output=out/method;output.mkdir(parents=True,exist_ok=False)
    bank=output/'SWAT_NORMAL.npz';np.savez_compressed(bank,index=index,score=np.concatenate(parts),per_channel=np.concatenate(channels),role=roles,section_id=section,max_input_index=index,min_input_index=index-191)
    fullscore=np.concatenate(parts);normal_bank=output/'NORMAL_BANK.npz'
    np.savez_compressed(normal_bank,CAL_index=index[roles==0],CAL_score=fullscore[roles==0],CHECK_index=index[roles==1],CHECK_score=fullscore[roles==1])
    dump(output/'RECEIPT.json',dict(status='COMPLETED_FROZEN_FORWARD_ONLY',method=method,
         index_manifest_sha256=sha(prep/'SWAT_INDEX_MANIFEST.json'),bank_sha256=sha(normal_bank),raw_bank_sha256=sha(bank),
         points=len(index),checkpoint_sha256=sha(checkpoint)if checkpoint else None,
         scaler_sha256=manifest['scaler_sha256'],seed=seed,inference_seconds=elapsed,
         device=device_name,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
         source_identities=sources,worker_sha256=sha(__file__),first_CAL_cache_compatibility=dict(cache_sha256=sha(cached),points=len(end),max_abs_error=max_error,rtol=1e-5,atol=1e-6),
         effective_footprint='[t-191,t] all observations arrived; no future; native smooth3 remains inside window',
         peak_allocated_bytes=torch.cuda.max_memory_allocated(device)if device.type=='cuda'else 0,
         no_parameter_or_buffer_update=True,threshold_not_fitted=True,QUERY_values_or_labels_never_accessed=True,
         score_unchanged='native original terminal output'if method=='LEFT_native'else'original fixed joint channel-mean NLL'if method=='Hybrid_full'else'original raw-time fixed Student5 prior'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','score'])
    parser.add_argument('--workspace',type=Path,required=True);parser.add_argument('--prep',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=['LEFT_native','Hybrid_full','Student_identity']);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();args.prep.mkdir(parents=True,exist_ok=True);args.output.mkdir(parents=True,exist_ok=True)
    if args.phase=='prepare':prepare(args.workspace,args.prep)
    else:score(args.workspace,args.prep,args.output,args.method,args.device)
