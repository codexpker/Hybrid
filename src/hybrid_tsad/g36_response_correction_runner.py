"""Freeze official LEFT history features; fit one fixed pair of linear corrections."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .g36_runner import Data,batch,dump,hashes,read,seed,sha,make
from .g36_response_correction import frozen_features,feature_stats,design,fit_head,ridge_predict


def load_left(w,cfg,device):
    path=w/cfg['LEFT_checkpoint'];receipt=read(w/cfg['LEFT_train_receipt'])
    assert receipt['selected_update']==2100 and receipt['selected_sha256']==sha(path)
    provenance=read(w/cfg['LEFT_provenance_manifest'])
    assert provenance['source_LEFT_sha256']==sha(w/cfg['upstream']/'models/Left/LEFT.py')
    assert provenance['source_model_config_sha256']==sha(w/cfg['source_model_config'])
    seed(cfg['seed'],device);model=make(w,cfg,'LEFT_native',device)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['method']=='LEFT_native'and saved['update']==2100
    model.load_state_dict(saved['state'],strict=True);model.step=saved['native_step']
    for p in model.parameters():p.requires_grad_(False)
    model.eval();assert not any(m.training for m in model.modules())
    return model,dict(checkpoint_sha256=sha(path),native_step=model.step,
        source_LEFT_sha256=sha(w/cfg['upstream']/'models/Left/LEFT.py'),
        source_Tools_sha256=sha(w/cfg['upstream']/'models/Left/Tools.py'),
        source_model_config_sha256=sha(w/cfg['source_model_config']))


def extract(w,config,cfg,out,device,query=False,gate=None):
    if query:
        assert gate is not None;release=read(gate)
        assert release['status']=='RESPONSE_CORRECTION_NORMAL_GATE_PASS_QUERY_RELEASED'
        assert release['config_sha256']==sha(config)
        assert release['head_sha256']==sha(w/cfg['output']/'heads/HEADS.npz')
        assert release['fit_receipt_sha256']==sha(w/cfg['output']/'heads/FIT_RECEIPT.json')
        assert read(w/cfg['output']/'heads/FIT_RECEIPT.json')['normal_gate_pass']
    roles=['CAL','QUERY']if query else['FIT','SELECT']
    model,identity=load_left(w,cfg,device);before=hashes(model);native_step=model.step
    data=Data(w,cfg,roles)
    with np.load(w/cfg['ridge_model'])as z:coef=z['coef'].copy();intercept=z['intercept'].copy()
    tracked=[w/cfg['LEFT_checkpoint'],w/cfg['LEFT_train_receipt'],w/cfg['ridge_model'],data.root/'manifest.json',data.root/'scaler.npz']
    originals={str(p):sha(p)for p in tracked};banks=[];replays=[]
    for record in data.manifest['records']:
        if record['role']not in roles:continue
        values,index=data.load(record);ends=np.arange(51,len(values));features=[]
        for begin in range(0,len(ends),cfg['batch_size']):
            e=ends[begin:begin+cfg['batch_size']]
            f=frozen_features(model,batch(values,e,device))
            assert f.ndim==2 and torch.isfinite(f).all();features.append(f.cpu().numpy())
        phi=np.concatenate(features);ix=index[ends]
        h=np.stack([values[t-51:t-3]for t in ends]);target=np.stack([values[t-3:t+1,:4]for t in ends])
        pred=ridge_predict(h,coef,intercept)
        if record['role']=='SELECT':
            with np.load(w/cfg['ridge_banks']/(record['run']+'__SELECT.npz'))as z:
                assert np.array_equal(ix,z['index'])and np.array_equal(target,z['target'])
                error=float(np.max(np.abs(pred-z['prediction'])))
                assert np.allclose(pred,z['prediction'],atol=1e-10,rtol=1e-10)
                replays.append(dict(role='SELECT',max_absolute_prediction_error=error))
        file=out/(record['run']+'__'+record['role']+'.npz')
        np.savez_compressed(file,index=ix,feature=phi,target=target,ridge_prediction=pred,
                            max_feature_index=ix-4,min_feature_index=ix-51,target_start_index=ix-3,target_end_index=ix)
        banks.append(dict(run=record['run'],role=record['role'],path=str(file),sha256=sha(file),points=len(ix),feature_dim=phi.shape[1]))
    assert hashes(model)==before and model.step==native_step
    assert not any(m.training for m in model.modules())and not any(p.requires_grad for p in model.parameters())
    assert all(sha(Path(p))==v for p,v in originals.items())
    dump(out/'EXTRACT_RECEIPT.json',dict(status='COMPLETED',banks=banks,identity=identity,roles=data.access,
        config_sha256=sha(config),immutable_state_hashes=before,model_tensor_state_proto_step_unchanged=True,
        cache_debug_fields='non-parameter ms cache and last_x_res may update; no learned state changes',
        all_modules_eval=True,all_parameters_frozen=True,SELECT_ridge_replay=replays,
        original_sha256=originals,no_labels_decoded=True,no_threshold_fitting=True,
        legal_history='only first48 of arrived52; latest feature observation t-4',
        feature='original fusion pooled T,F before memory; T is already cross-view fused'))


def fit(w,config,cfg,out):
    folder=w/cfg['output']/'extract_fit_select';meta=read(folder/'EXTRACT_RECEIPT.json')
    assert meta['config_sha256']==sha(config)
    arrays={}
    for record in meta['banks']:
        path=folder/Path(record['path']).name;assert sha(path)==record['sha256']
        with np.load(path)as z:arrays[record['role']]={k:z[k].copy()for k in z.files}
    assert set(arrays)=={'FIT','SELECT'}
    train=arrays['FIT'];val=arrays['SELECT'];N=len(train['index']);assert N==1101
    mean,scale=feature_stats(train['feature']);weights={};fitted={};metrics={}
    for name in cfg['methods']:
        W,stats=fit_head(train['feature'],train['target']-train['ridge_prediction'],mean,scale,name,cfg['lambda_mean_scalar'])
        assert stats['alpha_sumloss']==17616.;weights[name]=W;fitted[name]=stats
    for role,a in arrays.items():
        rows={}
        for name in ['Ridge']+cfg['methods']:
            pred=a['ridge_prediction'].copy()
            if name!='Ridge':pred+= (design(a['feature'],mean,scale,name)@weights[name]).reshape(-1,4,4)
            error=pred-a['target'];rows[name]=dict(scalar_MSE=float(np.mean(error**2)),scalar_MAE=float(np.mean(np.abs(error))))
            np.savez_compressed(out/(role+'__'+name+'.npz'),index=a['index'],prediction=pred,target=a['target'],residual=error,evidence=error**2,score=(error**2).mean((1,2)))
        metrics[role]=rows
    np.savez_compressed(out/'HEADS.npz',feature_mean=mean,feature_scale=scale,W_T_only=weights['T_only'],W_TF=weights['TF'])
    gate=metrics['SELECT']['TF']['scalar_MSE']<min(metrics['SELECT'][k]['scalar_MSE']for k in ['Ridge','T_only'])
    with np.load(w/cfg['data']/'scaler.npz')as z:response_scale=z['scale'][:4]
    native={}
    for name in ['Ridge']+cfg['methods']:
        with np.load(out/('SELECT__'+name+'.npz'))as z:e=z['residual']*response_scale
        native[name]=dict(MAE=np.abs(e).mean((0,1)).tolist(),RMSE=np.sqrt((e**2).mean((0,1))).tolist())
    dump(out/'FIT_RECEIPT.json',dict(status='NORMAL_GATE_READY',normal_gate_pass=bool(gate),config_sha256=sha(config),
         source_extract_sha256=sha(folder/'EXTRACT_RECEIPT.json'),head_sha256=sha(out/'HEADS.npz'),FIT_N=N,
         loss='1/(N*16) squared residual sum +1*||W||F^2, no bias',fit_objectives=fitted,normal_metrics=metrics,
         SELECT_native_channel_errors=native,feature_stats_fit_only=True,no_CAL_QUERY_access=True,
         methods='T_only masks fused F-token slot, not all frequency information',no_model_training=True))


def score(w,config,cfg,out,gate):
    release=read(gate);root=w/cfg['output'];receipt=read(root/'heads/FIT_RECEIPT.json')
    assert release['status']=='RESPONSE_CORRECTION_NORMAL_GATE_PASS_QUERY_RELEASED'and receipt['normal_gate_pass']
    assert release['config_sha256']==sha(config)
    assert release['fit_receipt_sha256']==sha(root/'heads/FIT_RECEIPT.json')
    assert release['head_sha256']==receipt['head_sha256']==sha(root/'heads/HEADS.npz')
    with np.load(root/'heads/HEADS.npz')as z:heads={k:z[k].copy()for k in z.files}
    meta=read(root/'extract_query/EXTRACT_RECEIPT.json');banks=[]
    for row in meta['banks']:
        p=root/'extract_query'/Path(row['path']).name;assert sha(p)==row['sha256']
        with np.load(p)as z:a={k:z[k].copy()for k in z.files}
        for method in cfg['methods']:
            pred=a['ridge_prediction']+(design(a['feature'],heads['feature_mean'],heads['feature_scale'],method)@heads['W_'+method]).reshape(-1,4,4)
            residual=pred-a['target'];folder=out/method;folder.mkdir(exist_ok=True)
            file=folder/(row['run']+'__'+row['role']+'.npz')
            np.savez_compressed(file,index=a['index'],score=(residual**2).mean((1,2)),prediction=pred,target=a['target'],residual=residual,evidence=residual**2,max_feature_index=a['max_feature_index'],min_feature_index=a['min_feature_index'],target_start_index=a['target_start_index'],target_end_index=a['target_end_index'])
            banks.append(dict(method=method,path=str(file),sha256=sha(file)))
    dump(out/'SCORE_RECEIPT.json',dict(status='COMPLETED',banks=banks,no_threshold_fitting=True,no_labels_decoded=True,head_sha256=receipt['head_sha256']))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['extract_fit_select','fit','extract_query','score']);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--gate',type=Path)
    a=p.parse_args();w=a.workspace.resolve();config=a.config.resolve();cfg=read(config)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    out=w/cfg['output']/('heads'if a.phase=='fit'else a.phase);out.mkdir(parents=True,exist_ok=False)
    device=torch.device(a.device);start=time.time();rc=0
    if device.type=='cuda':
        torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(30*1024**3/torch.cuda.get_device_properties(device).total_memory,device);torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dump(out/'START.json',dict(pid=os.getpid(),phase=a.phase,start_unix=start,config_sha256=sha(config),source_sha256=sha(__file__),core_sha256=sha(Path(__file__).with_name('g36_response_correction.py')),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        if a.phase in ['extract_fit_select','extract_query']:extract(w,config,cfg,out,device,a.phase=='extract_query',a.gate)
        elif a.phase=='fit':fit(w,config,cfg,out)
        else:score(w,config,cfg,out,a.gate)
    except BaseException:
        rc=1;dump(out/'FAILED.json',dict(traceback=traceback.format_exc()));raise
    finally:
        end=time.time();dump(out/'OS.json',dict(pid=os.getpid(),start_unix=start,end_unix=end,seconds=end-start,exit_code=rc,GPU_hours=(end-start)/3600 if device.type=='cuda'else 0,peak_allocated_bytes=torch.cuda.max_memory_allocated(device)if device.type=='cuda'else 0))


if __name__=='__main__':main()
