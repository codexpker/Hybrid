"""Frozen official TimesFM3 endpoint forecast baseline; no adaptation or anomaly head."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from .native_v_conditioning import sha,state_hashes
from .native_v_conditioning_runner import read,dump,role_values,role_index,setup

EXPERIMENT='timesfm3_frozen_20261121_v1'
SOURCE_COMMIT='918df02c9d366b2145bc5876135fa2131d3ed2a6'
WEIGHT_REVISION='43046b85ec22d584a13f8098c2ed39c889e129c2'
WEIGHT_SHA='a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da'
PREDICT_OPTIONS=dict(horizon=1,return_quantiles=True,use_symmetric_averaging=False,make_positive=False,sort_quantiles=True,use_znorm=False,padding_mode='none')
BATCH_TOLERANCE=dict(rtol=1e-4,atol=1e-4)
OUTER_CHUNK=64


def asset_root(w):return w/'preparations'/EXPERIMENT/'assets'


def prepare(w,root):
    root.mkdir(parents=True,exist_ok=False)
    for domain in ['SWAT','PSM']:
        old=w/'results/native_v_conditioning_20261119_v1'/domain;out=root/domain;out.mkdir()
        source=read(old/'TASK.json')
        task={k:source[k]for k in ['domain','entity','channels','data','roles','channel_order','channel_order_source','scaler_sha256','cache_identity','role_is_permission_not_identity','labels_read']}
        task.update(source_TASK_relative=(old/'TASK.json').relative_to(w).as_posix(),source_TASK_sha256=sha(old/'TASK.json'),experiment=EXPERIMENT,history=512,horizon=1,score='meanC((official_sorted_median_prediction-current_SOURCEFIT_normalized)^2)',
            external_normalization='original SOURCEFIT scaler exactly once; official use_znorm=False; internal original CPM/RevIN retained',
            source_commit=SOURCE_COMMIT,weight_revision=WEIGHT_REVISION,weight_sha256=WEIGHT_SHA,predict_options=PREDICT_OPTIONS,indexes={})
        shutil.copyfile(old/'SCALER.npz',out/'SCALER.npz');shutil.copyfile(old/'TASK.json',out/'ORIGINAL_TASK.json')
        for role in ['CAL','QUERY']:
            p=w/source['indexes'][role]['relative'];assert sha(p)==source['indexes'][role]['sha256']
            target=out/f'{role}_INDEX.npy';shutil.copyfile(p,target)
            task['indexes'][role]=dict(source['indexes'][role],relative=target.relative_to(w).as_posix(),sha256=sha(target))
        dump(out/'TASK.json',task)
    dump(root/'CONFIG.json',dict(experiment=EXPERIMENT,domains=['SWAT','PSM'],method='TimesFM3_frozen',source_commit=SOURCE_COMMIT,
        weight_revision=WEIGHT_REVISION,weight_sha256=WEIGHT_SHA,predict_options=PREDICT_OPTIONS,batch_tolerance=BATCH_TOLERANCE,
        official_per_core_batch_size=4,host_outer_chunk=OUTER_CHUNK,training=False,adaptation=False,labels_read=False,score_precision='savedFP32 median/actual promotedFP64 before square and meanC'))


def load_forecaster(w,device):
    a=asset_root(w);checkpoint=a/'model.safetensors';assert sha(checkpoint)==WEIGHT_SHA
    source=a/f'timesfm-{SOURCE_COMMIT}'/'src';sys.path.insert(0,str(source))
    from timesfm3.torch import timesfm3_forecaster as official
    assert Path(official.__file__).resolve()==(source/'timesfm3/torch/timesfm3_forecaster.py').resolve()
    TimesFM3Forecaster=official.TimesFM3Forecaster
    forecaster=TimesFM3Forecaster.from_pretrained(str(a),device=str(device),local_files_only=True)
    model=forecaster.model;model.requires_grad_(False);model.eval()
    assert all(not m.training for m in model.modules())and all(not p.requires_grad for p in model.parameters())
    return forecaster


def history_context(values,lo,t):
    stop=int(t)-int(lo)
    if stop<512 or stop>=len(values):raise PermissionError('Prediction requires full legal past512 and arrived target')
    result=np.ascontiguousarray(values[stop-512:stop].T,dtype=np.float32)
    assert result.shape==(values.shape[1],512)and np.isfinite(result).all()
    return result


def prediction_arrays(outputs,channels):
    med=np.stack([np.asarray(o.forecast,dtype=np.float32)for o in outputs])
    quant=np.stack([np.asarray(o.quantiles,dtype=np.float32)for o in outputs])
    assert med.shape==(len(outputs),channels,1)and quant.shape==(len(outputs),channels,1,9)
    assert np.array_equal(med,quant[...,4])and np.isfinite(quant).all()
    return med[:,:,0],quant[:,:,0,:]


def residual_score(median,actual):
    if median.shape!=actual.shape or median.ndim!=2:raise ValueError('Aligned [point,channel] arrays required')
    err=(median.astype(np.float64)-actual.astype(np.float64))**2
    return err,err.mean(axis=1)


def run_predictions(forecaster,contexts):
    return prediction_arrays(list(forecaster.predict_batch(contexts=contexts,**PREDICT_OPTIONS)),contexts[0].shape[0])


def gate(w,root):
    out=root/'engineering_gate';out.mkdir(parents=True,exist_ok=False);started=time.monotonic();device=setup(30)
    f=load_forecaster(w,device);model=f.model;before=state_hashes(model)
    from safetensors import safe_open
    with safe_open(asset_root(w)/'model.safetensors',framework='pt',device='cpu')as sf:
        assert set(sf.keys())==set(before)
        for name in sf.keys():assert hashlib.sha256(sf.get_tensor(name).contiguous().numpy().tobytes()).hexdigest()==before[name]
    rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone())
    shapes=[];native_decode=model.decode
    def capture(*args,**kwargs):
        target=kwargs['target'];shapes.append(dict(shape=list(target.shape),horizon=kwargs['horizon']))
        return native_decode(*args,**kwargs)
    model.decode=capture;domains=[]
    for domain in ['SWAT','PSM']:
        task=read(root/domain/'TASK.json');values,actual,lo,rec=role_values(w,root/domain,task,'SOURCE_FIT')
        times=[lo+512,lo+1024,lo+2048];hist=[history_context(values,lo,t)for t in times]
        single=[prediction_arrays([f.predict(x,**PREDICT_OPTIONS)],task['channels'])for x in hist]
        pred,quant=run_predictions(f,hist)
        expected=np.concatenate([x[0]for x in single]);np.testing.assert_allclose(pred,expected,**BATCH_TOLERANCE)
        maxdiff=float(np.max(np.abs(pred.astype(np.float64)-expected)))
        reverse,_=run_predictions(f,list(reversed(hist)));np.testing.assert_allclose(reverse[::-1],pred,**BATCH_TOLERANCE)
        # The wrapper never includes target t. Mutating t and all future leaves the input byte-identical.
        changed=values.copy();changed[times[0]-lo:]=12345;assert np.array_equal(history_context(changed,lo,times[0]),hist[0])
        # One core-batch warmup, one fixed64-context host call; the official API internally uses B4.
        run_predictions(f,[hist[0]]*4);torch.cuda.synchronize();durations=[]
        tic=time.monotonic();run_predictions(f,[hist[i%3]for i in range(OUTER_CHUNK)]);torch.cuda.synchronize();durations.append(time.monotonic()-tic)
        seconds_per_point=sum(durations)/OUTER_CHUNK
        actual_y=np.stack([values[t-lo]for t in times]);err,score=residual_score(pred,actual_y)
        np.savez_compressed(out/f'{domain}_SOURCE_FIXTURE.npz',absolute_time_index=times,history=np.stack(hist),median=pred,quantiles=quant,actual_normalized=actual_y,squared_error_per_channel=err,canonical_score=score)
        domains.append(dict(domain=domain,channels=task['channels'],times=times,record_id=rec['record_id'],history_start=[t-512 for t in times],history_end_exclusive=times,
            median_quantile_index=f.config.median_quantile_index,quantile_levels=f.config.quantiles,median_equals_quantile4_exact=True,sample_batch_max_abs=maxdiff,batch_tolerance=BATCH_TOLERANCE,
            warmup_samples=4,timed_samples=OUTER_CHUNK,timed_outer_chunk=OUTER_CHUNK,timed_internal_batch_size=4,timed_batch_seconds=durations,seconds_per_point=seconds_per_point,
            fixed_total_points=task['indexes']['CAL']['count']+task['indexes']['QUERY']['count'],
            estimated_complete_GPU_seconds=seconds_per_point*(task['indexes']['CAL']['count']+task['indexes']['QUERY']['count']),
            current_future_mutation_input_invariant=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    model.decode=native_decode
    after=state_hashes(model);assert before==after and all(p.grad is None for p in model.parameters())
    assert torch.equal(rng[0],torch.get_rng_state())and torch.equal(rng[1],torch.cuda.get_rng_state(device))
    dump(out/'GATE.json',dict(status='PASS_ENGINEERING_ONLY',domains=domains,actual_config=dataclasses.asdict(f.config),decode_shapes=shapes,
        parameters=sum(p.numel()for p in model.parameters()),state_tensor_count=len(model.state_dict()),state_hashes=before,frozen_eval=True,no_grad=True,
        model_state_unchanged=True,CPU_CUDA_RNG_restored=True,labels_read=False,QUERY_opened=False,CAL_opened=False,elapsed_GPU_seconds=time.monotonic()-started,
        source_commit=SOURCE_COMMIT,weight_revision=WEIGHT_REVISION,weight_sha256=WEIGHT_SHA))
    print(json.dumps(dict(event='STAGE_COMPLETED',stage='SOURCE_FIT_GATE',domains=domains)),flush=True)


def score(w,root,domain,role,part,parts):
    assert read(root/'engineering_gate/GATE.json')['status']=='PASS_ENGINEERING_ONLY'
    out=root/'score'/domain/role/f'part{part}';out.mkdir(parents=True,exist_ok=False);started=time.monotonic();device=setup(30)
    task=read(root/domain/'TASK.json');f=load_forecaster(w,device);before=state_hashes(f.model)
    values,actual,lo,rec=role_values(w,root/domain,task,role);index=np.array_split(role_index(w,root/domain,task,role),parts)[part]
    n=len(index);c=task['channels'];median=np.empty((n,c),np.float32);quantiles=np.empty((n,c,9),np.float32);observed=np.empty((n,c),np.float32);raw=np.empty((n,c),np.float64)
    rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone())
    for start in range(0,n,OUTER_CHUNK):
        times=index[start:start+OUTER_CHUNK];contexts=[history_context(values,lo,int(t))for t in times]
        med,q=run_predictions(f,contexts)
        # Arrived current observations are first indexed after official forecasts return.
        target=np.stack([values[int(t)-lo]for t in times]);target_raw=np.stack([actual[int(t)-lo]for t in times])
        median[start:start+len(times)]=med;quantiles[start:start+len(times)]=q;observed[start:start+len(times)]=target;raw[start:start+len(times)]=target_raw
        if(start+len(times))%4096==0:print(json.dumps(dict(event='SCORE_PROGRESS',domain=domain,role=role,part=part,done=start+len(times),total=n)),flush=True)
    err,scores=residual_score(median,observed)
    assert before==state_hashes(f.model)and all(not m.training for m in f.model.modules())and all(p.grad is None and not p.requires_grad for p in f.model.parameters())
    assert torch.equal(rng[0],torch.get_rng_state())and torch.equal(rng[1],torch.cuda.get_rng_state(device))
    bank=out/'BANK.npz';np.savez_compressed(bank,absolute_time_index=index,available_index=index,history_start=index-512,history_end_exclusive=index,
        prediction_normalized=median,quantiles_normalized=quantiles,quantile_levels=np.asarray(f.config.quantiles),actual_normalized=observed,actual_raw=raw,
        squared_error_per_channel=err,canonical_score=scores)
    dump(out/'RECEIPT.json',dict(status='COMPLETED',domain=domain,entity=task['entity'],record_id=rec['record_id'],role=role,part=part,parts=parts,count=n,
        relative=bank.relative_to(w).as_posix(),sha256=sha(bank),source_commit=SOURCE_COMMIT,weight_revision=WEIGHT_REVISION,weight_sha256=WEIGHT_SHA,
        scaler_sha256=task['scaler_sha256'],TASK_sha256=sha(root/domain/'TASK.json'),predict_options=PREDICT_OPTIONS,actual_config=dataclasses.asdict(f.config),
        input_unit='SOURCE_FIT normalized',output_unit='same SOURCE_FIT normalized; official internal inverse retained',
        current_target_in_model_input=False,all_parameters_frozen=True,model_state_RNG_unchanged=True,labels_read=False,
        elapsed_GPU_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    print(json.dumps(dict(event='SCORE_COMPLETED',domain=domain,role=role,part=part)),flush=True)


def merge(w,root):
    banks=[]
    for domain in ['SWAT','PSM']:
        task=read(root/domain/'TASK.json')
        for role in ['CAL','QUERY']:
            folder=root/'score'/domain/role;receipts=[read(p)for p in sorted(folder.glob('part*/RECEIPT.json'),key=lambda p:int(p.parent.name[4:]))]
            assert receipts and len(receipts)==receipts[0]['parts'];arrays={};levels=None
            for r in receipts:
                path=w/r['relative'];assert sha(path)==r['sha256']
                with np.load(path,allow_pickle=False)as z:
                    for key in z.files:
                        if key=='quantile_levels':
                            if levels is not None:assert np.array_equal(levels,z[key])
                            levels=z[key].copy()
                        else:arrays.setdefault(key,[]).append(z[key].copy())
            data={k:np.concatenate(v)for k,v in arrays.items()};data['quantile_levels']=levels
            assert np.array_equal(data['absolute_time_index'],role_index(w,root/domain,task,role))
            path=folder/'BANK.npz';assert not path.exists();np.savez_compressed(path,**data)
            r=dict(status='COMPLETED',domain=domain,entity=task['entity'],role=role,record_id=receipts[0]['record_id'],count=len(data['canonical_score']),relative=path.relative_to(w).as_posix(),sha256=sha(path),parts=receipts,labels_read=False)
            dump(folder/'SEALED.json',r);banks.append(dict(domain=domain,method='TimesFM3_frozen',role=role,relative=r['relative'],sha256=r['sha256'],receipt_relative=(folder/'SEALED.json').relative_to(w).as_posix(),receipt_sha256=sha(folder/'SEALED.json')))
    return banks


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','gate','score','queue','merge']);p.add_argument('--workspace',required=True,type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--domain',choices=['SWAT','PSM']);p.add_argument('--role',choices=['CAL','QUERY']);p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1)
    a=p.parse_args();root=a.output or a.workspace/'results'/EXPERIMENT
    if a.action=='prepare':prepare(a.workspace,root)
    elif a.action=='gate':gate(a.workspace,root)
    elif a.action=='score':score(a.workspace,root,a.domain,a.role,a.part,a.parts)
    elif a.action=='merge':merge(a.workspace,root)
    else:
        from .native_v_conditioning_queue import run_jobs
        if (root/'GATE_COMPLETED.json').exists():gatejobs=read(root/'GATE_COMPLETED.json')['jobs']
        else:
            gatejobs=run_jobs(a.workspace,root,'source_gate',[('TimesFM3_SOURCEFIT',['gate'],30,900)],'hybrid_tsad.timesfm3_frozen')
            dump(root/'GATE_COMPLETED.json',dict(status='COMPLETED',jobs=gatejobs))
        # Fixed indices only. Shards affect resource routing, not each independent horizon1 call.
        speeds={r['domain']:r['seconds_per_point']for r in read(root/'engineering_gate/GATE.json')['domains']};tasks=[]
        for domain,role,parts in [('SWAT','QUERY',16),('PSM','QUERY',4),('SWAT','CAL',1),('PSM','CAL',1)]:
            n=read(root/domain/'TASK.json')['indexes'][role]['count']
            for part in range(parts):
                size=(n+parts-1)//parts
                timeout=max(900,int(speeds[domain]*size*3+180))
                tasks.append((domain+'_'+role+'_'+str(part),['score','--domain',domain,'--role',role,'--part',str(part),'--parts',str(parts)],30,timeout))
        jobs=run_jobs(a.workspace,root,'frozen_readout',tasks,'hybrid_tsad.timesfm3_frozen');banks=merge(a.workspace,root)
        dump(root/'GLOBAL_RAW_BANK_SEAL.json',dict(status='COMPLETED',banks=banks,jobs=gatejobs+jobs,labels_read=False,no_labels_decoded=True,
            actual_GPU_hours=sum(j['actual_GPU_seconds']for j in gatejobs+jobs)/3600,training=False,adaptation=False,weight_sha256=WEIGHT_SHA,source_commit=SOURCE_COMMIT))
        print(json.dumps(dict(event='STAGE_COMPLETED',stage='GLOBAL_RAW_BANK_SEAL',banks=4)),flush=True)


if __name__=='__main__':main()
