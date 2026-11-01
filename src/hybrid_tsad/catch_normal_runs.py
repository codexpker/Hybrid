"""Independent normal-run data injection into unchanged external CATCH.detect_fit.

Only FIT4/SELECT2, fixed C2/H192, and seed2025 are admitted. Native optimization
is not reimplemented. The paired normal-training release is separate from L0.
"""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import time

from . import catch_baseline as base

PARAMS=dict(base.PARAMS,num_epochs=3)
SOURCE_MANIFEST='07287e30bb63dd7dae5f785955a370c4ee7420b6d0cbeb6363324d906f631bb7'


def observed_native_step(opt,native_step,*args,**kwargs):
    """Read-only gradient/update evidence around exactly one original Adam step."""
    import torch
    params=[p for group in opt.param_groups for p in group['params']]
    grads=[p.grad.detach() for p in params if p.grad is not None]
    assert all(torch.isfinite(g).all().item() for g in grads)
    stats={'gradient_tensors':len(grads),
           'gradient_abs_max':max((float(g.abs().max()) for g in grads),default=0.),
           'gradient_l2_norm':math.sqrt(sum(float(g.double().square().sum()) for g in grads))}
    assert math.isfinite(stats['gradient_abs_max']) and math.isfinite(stats['gradient_l2_norm'])
    before=[p.detach().clone() for p in params]
    result=native_step(opt,*args,**kwargs)
    assert all(torch.isfinite(p).all().item() for p in params)
    stats['parameter_delta_abs_max']=max((float((p.detach()-v).abs().max()) for p,v in zip(params,before)),default=0.)
    assert math.isfinite(stats['parameter_delta_abs_max'])
    return result,stats


def production_layout():
    return dict(fit_runs=4,select_runs=2,points_per_run=24576,window=192,
                fit_stride=1,select_stride=192,fit_windows=97540,fit_batches=763,
                fit_last_batch=4,select_windows=256,select_batches=2,
                main_steps_if_complete=2289,mask_interval=76,mask_calls_if_complete=30,
                epochs=3,LEFT_reference_epochs=20,equal_optimization_budget=False)


@contextmanager
def independent_data(api,detector,fit_runs,select_runs,trace):
    """Temporarily replace only split/provider globals; restore even on failure.

    Existing IndependentNormalWindows supplies all window/role/index semantics.
    Native StandardScaler.fit/transform runs first. Its exact mean/scale feed the
    shared dataset; provider verifies transformed full values bitwise equal.
    """
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader
    from .weak_lag import check_normal_runs
    from .left_normal_runs import IndependentNormalWindows
    class ObservedWindows(IndependentNormalWindows):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.access=[]
        def __getitem__(self,index):
            self.access.append(int(index));return super().__getitem__(index)
    check_normal_runs(fit_runs,'FIT');check_normal_runs(select_runs,'SELECT')
    if fit_runs[0].scenario!=select_runs[0].scenario:
        raise PermissionError('One observation scene per fit/select pair')
    if getattr(api,'_independent_normal_data_active',False):
        raise RuntimeError('Global data hooks require an isolated non-nested process')
    def frame(runs,role):
        values=np.concatenate([r.values for r in runs],axis=0)
        return pd.DataFrame(values,columns=['u','y'],index=pd.RangeIndex(len(values),name=role+'_ROW'))
    fit_frame=frame(fit_runs,'FIT');select_frame=frame(select_runs,'SELECT')
    native_split,native_provider=api.train_val_split,api.anomaly_detection_data_provider
    datasets={}
    def split(data,ratio,seq_len):
        if data is not fit_frame or ratio!=.8 or seq_len is not None:
            raise PermissionError('Only native detect_fit first-frame split call admitted')
        trace.append({'event':'split_override','fit_points':len(fit_frame),
                      'select_points':len(select_frame),'index':'role-local row placeholder, not physical clock'})
        return fit_frame,select_frame
    def provider(data,batch_size,win_size=100,step=100,mode='train'):
        if mode not in ('train','val') or (batch_size,win_size,step)!=(128,192,1):
            raise PermissionError('Training data hook forbids score/test/labels or changed native call')
        role='FIT' if mode=='train' else 'SELECT'
        expected=fit_frame if mode=='train' else select_frame
        runs=fit_runs if mode=='train' else select_runs
        if not data.index.equals(expected.index) or data.index.name!=expected.index.name or list(data.columns)!=['u','y']:
            raise PermissionError('Native transformed frame identity/layout differs')
        scaler={'mean':detector.scaler.mean_.copy(),'scale':detector.scaler.scale_.copy(),
                'fit_ids':[(r.scenario,r.role,r.run_id) for r in fit_runs]}
        ds=ObservedWindows(runs,scaler,mode)
        if not np.array_equal(np.concatenate(ds.normalized),data.values):
            raise ValueError('Shared windows do not reproduce native float64 scaler transform')
        datasets[mode]=ds
        loader=DataLoader(ds,batch_size=128,shuffle=True,num_workers=0,drop_last=False)
        trace.append({'event':'provider_override','mode':mode,'role':role,'windows':len(ds),
                      'batches':len(loader),'stride':ds.step,'shuffle':True,'drop_last':False,
                      'native_scaler_full_value_parity':True,'run_ids':[r.run_id for r in runs]})
        return loader
    api.train_val_split=split;api.anomaly_detection_data_provider=provider
    api._independent_normal_data_active=True
    try:yield fit_frame,datasets
    finally:
        api.train_val_split=native_split;api.anomaly_detection_data_provider=native_provider
        del api._independent_normal_data_active


def validate(path):
    cfg=json.loads(Path(path).read_text())
    if (cfg.get('operation')!='CATCH_C2_NORMAL_TRAIN' or cfg.get('scenario') not in ('LIN','WEAK')
        or cfg.get('seed')!=2025 or cfg.get('params')!=PARAMS or cfg.get('upstream_commit')!=base.COMMIT
        or cfg.get('data') is not None or cfg.get('checkpoint') is not None
        or cfg.get('max_seconds')!=600 or cfg.get('max_allocated_GiB')!=20
        or cfg.get('physical_gpu')!=({'LIN':0,'WEAK':1}[cfg['scenario']])):
        raise PermissionError('Only the fixed independent-normal CATCH pair may be prepared')
    if {r['path'] for r in cfg['adapter_hashes']}!={'__init__.py','catch_normal_runs.py','catch_baseline.py',
                                                             'left_normal_runs.py','left_baseline.py','weak_lag.py'}:
        raise ValueError('Complete fixed own-source helper identity required')
    for ref in cfg['adapter_hashes']:
        if base.sha(Path(__file__).parent/ref['path'])!=ref['sha256']:
            raise ValueError('Frozen adapter helper differs: '+ref['path'])
    root=Path(cfg['source_root'])
    actual={str(p.relative_to(root)).replace('\\','/') for p in (root/'ts_benchmark/baselines/catch').rglob('*.py')}
    if set(cfg['source_hashes'])!=actual|set(base.SUPPORT):raise ValueError('Complete external source identity required')
    for rel,digest in cfg['source_hashes'].items():
        if base.sha(root/rel)!=digest:raise ValueError('External author source differs: '+rel)
    ref=cfg['normal_inputs']
    if base.sha(ref['path'])!=ref['sha256']:raise ValueError('Existing normal manifest differs')
    accepted=cfg['cpu_source_acceptance']
    if (accepted['sha256']!='5482634b720b12e9e57996bc097e1668a90c86bfa375629e0b592d4e615bf129'
        or base.sha(accepted['path'])!=accepted['sha256']):
        raise ValueError('Previously accepted FIT/SELECT data lineage differs')
    inputs=json.loads(Path(ref['path']).read_text())
    if (inputs['scenario']!=cfg['scenario'] or inputs['source_manifest_sha256']!=SOURCE_MANIFEST
        or set(inputs['roles'])!={'FIT','SELECT'}):
        raise PermissionError('Only previously fixed normal role references permitted')
    for role,count in [('FIT',4),('SELECT',2)]:
        rows=inputs['roles'][role]
        if len(rows)!=count:raise ValueError('Normal run count differs')
        for i,row in enumerate(rows):
            p=Path(row['path'])
            if ((row['scenario'],row['role'],row['run_id'])!=(cfg['scenario'],role,i)
                or list(p.parts[-3:-1])!=[cfg['scenario'],role] or p.name!=f'run{i:02d}_N.npy'
                or row['shape']!=[24576,2] or row['dtype']!='float64' or not p.is_file()):
                raise PermissionError('Not an exact existing normal source reference')
    if Path(cfg['output']).exists():raise FileExistsError(cfg['output'])
    return cfg,root,inputs


def require_release(path,cfg):
    a=json.loads(Path(cfg['approval_path']).read_text())
    if (a.get('status')!='PRIMARY_RELEASED_CATCH_C2_NORMAL_TRAIN' or a.get('approved_by')!='root'
        or a.get('config_sha256')!=base.sha(path) or a.get('adapter_sha256')!=base.sha(__file__)
        or a.get('allow_CAL_DEV_CONFIRM') is not False or a.get('allow_labels') is not False
        or a.get('allow_score') is not False):
        raise PermissionError('Separate root normal-training release required before importing torch')


def execute(cfg,root,inputs):
    import hashlib
    import os
    import subprocess
    import traceback
    import numpy as np
    import torch
    from .left_normal_runs import load_normal_assets
    output=Path(cfg['output']);output.mkdir(exist_ok=False,parents=True)
    start=time.monotonic();trace=[];updates=[];validation=[];initial={}
    report={'status':'STARTED','scope':'official CATCH optimizer loop with independent FIT/SELECT data hooks',
            'layout':production_layout(),'query_read':False,'labels_read':False}
    native_step=torch.optim.Adam.step;native_backward=torch.autograd.backward;handle=None
    try:
        if not torch.cuda.is_available():raise RuntimeError('Approved normal training requires A40')
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(20*2**30/torch.cuda.get_device_properties(0).total_memory,0)
        torch.cuda.reset_peak_memory_stats();torch.set_num_threads(1)
        uuid=str(torch.cuda.get_device_properties(0).uuid)
        gpu=subprocess.check_output(['nvidia-smi','-i',str(cfg['physical_gpu']),
              '--query-gpu=uuid,name','--format=csv,noheader'],text=True).strip()
        if uuid not in gpu:raise RuntimeError('Actual GPU UUID differs from physical mapping')
        report.update(GPU_UUID=uuid,GPU_inventory=gpu,torch=torch.__version__,cuda=torch.version.cuda,
                      numpy=np.__version__,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
        random.seed(2025);np.random.seed(2025);torch.manual_seed(2025);torch.cuda.manual_seed_all(2025)
        report['RNG_before_data_fit']={'cpu':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            'cuda':hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
        fit=load_normal_assets(inputs['roles']['FIT'],cfg['scenario'],'FIT')
        select=load_normal_assets(inputs['roles']['SELECT'],cfg['scenario'],'SELECT')
        api=base.external_api(root);detector=api.CATCH(**cfg['params'])
        def digest(state):
            h=hashlib.sha256()
            for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        def pre(module,args):
            if isinstance(module,api.CATCHModel) and not initial:
                initial.update({k:v.detach().cpu().clone() for k,v in module.state_dict().items()})
        handle=torch.nn.modules.module.register_module_forward_pre_hook(pre)
        def step(opt,*a,**kw):
            event={'event':'optimizer_step','optimizer':'main' if opt is detector.optimizer else 'mask',
                   'lr':[g['lr'] for g in opt.param_groups]}
            result,stats=observed_native_step(opt,native_step,*a,**kw)
            event.update(stats);updates.append(event)
            assert torch.cuda.max_memory_allocated()<=20*2**30
            return result
        def backward(tensors,*a,**kw):
            value=tensors if torch.is_tensor(tensors) else tensors[0]
            assert torch.isfinite(value).all().item()
            updates.append({'event':'backward','loss':float(value.detach().cpu())})
            return native_backward(tensors,*a,**kw)
        torch.optim.Adam.step=step;torch.autograd.backward=backward
        native_validate=detector.detect_validate
        def validated(*a,**kw):
            value=native_validate(*a,**kw);assert np.isfinite(value)
            validation.append({'epoch':len(validation)+1,'loss':float(value),
                'main_lr':detector.optimizer.param_groups[0]['lr'],'mask_lr':detector.optimizerM.param_groups[0]['lr']})
            return value
        detector.detect_validate=validated
        with independent_data(api,detector,fit,select,trace) as (frame,datasets):
            detector.detect_fit(frame,None)
            assert len(datasets['train'])==97540 and len(datasets['val'])==256
        np.savez_compressed(output/'window_order.npz',
            fit=np.asarray(datasets['train'].access,dtype=np.int32),
            select=np.asarray(datasets['val'].access,dtype=np.int32))
        torch.cuda.synchronize()
        best={k:v.detach().cpu() for k,v in detector.early_stopping.check_point.items()}
        last=detector.model.state_dict()
        assert all(torch.isfinite(v).all().item() for v in last.values())
        torch.save(best,output/'best_checkpoint.pth')
        np.savez(output/'scaler.npz',mean=detector.scaler.mean_,scale=detector.scaler.scale_,
                 var=detector.scaler.var_,n_samples_seen=detector.scaler.n_samples_seen_)
        from .weak_lag import fit_scaler
        left_ref=fit_scaler(fit)
        counts={n:sum(e.get('optimizer')==n for e in updates) for n in ('main','mask')}
        assert len(validation)==3 and counts=={'main':2289,'mask':30}
        effective={n:sum(e.get('optimizer')==n and e['parameter_delta_abs_max']>0 for e in updates if e['event']=='optimizer_step') for n in ('main','mask')}
        assert all(effective[n]>0 for n in ('main','mask'))
        final_delta={k:float((v.detach().cpu()-initial[k]).abs().max()) for k,v in last.items()}
        assert any(v>0 for k,v in final_delta.items() if 'mask_generator' not in k)
        assert any(v>0 for k,v in final_delta.items() if 'mask_generator' in k)
        report.update(status='COMPLETED_PENDING_REVIEW',validation=validation,optimizer_steps=counts,
            effective_nonzero_parameter_updates=effective,
            best_epoch=max(v['epoch'] for v in validation if v['loss']==detector.early_stopping.val_loss_min),
            best_validation_loss=float(detector.early_stopping.val_loss_min),
            initial_state_sha256=digest(initial),best_state_sha256=digest(best),last_state_sha256=digest(last),
            parameter_delta_abs_max=final_delta,
            best_checkpoint_sha256=base.sha(output/'best_checkpoint.pth'),scaler_sha256=base.sha(output/'scaler.npz'),
            source_unchanged=all(base.sha(root/p)==h for p,h in cfg['source_hashes'].items()),
            formal_data_hashes=inputs,params=cfg['params'],seed=2025,
            source_commit=base.COMMIT,source_hashes=cfg['source_hashes'],adapter_hashes=cfg['adapter_hashes'],
            imported_source={name:{'path':str(Path(m.__file__).resolve()),'sha256':base.sha(m.__file__)}
                for name,m in list(__import__('sys').modules.items()) if name.startswith('ts_benchmark') and getattr(m,'__file__',None)},
            LEFT_fit_scaler_reference={'mean_max_abs_difference':float(np.max(np.abs(detector.scaler.mean_-left_ref['mean']))),
                'scale_max_abs_difference':float(np.max(np.abs(detector.scaler.scale_-left_ref['scale']))),
                'description':'same FIT arrays; original sklearn scaler retained; no select/query fit'},
            final_lr={'main':detector.optimizer.param_groups[0]['lr'],'mask':detector.optimizerM.param_groups[0]['lr']})
        assert report['source_unchanged']
    except BaseException:
        report.update(status='FAILED_NO_AUTOMATIC_RETRY',traceback=traceback.format_exc());raise
    finally:
        torch.optim.Adam.step=native_step;torch.autograd.backward=native_backward
        if handle:handle.remove()
        report['whole_seconds']=time.monotonic()-start
        if torch.cuda.is_available():report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        (output/'optimizer_events.json').write_text(json.dumps(updates))
        (output/'data_hook_trace.json').write_text(json.dumps(trace,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    cfg,root,inputs=validate(args.config)
    if args.execute:
        require_release(args.config,cfg);print(json.dumps(execute(cfg,root,inputs)),flush=True)
    else:print(json.dumps({'status':'PLAN_ONLY_NO_TORCH_OR_ARRAY_READ','layout':production_layout(),
                          'scenario':cfg['scenario'],'config_sha256':base.sha(args.config)},indent=2))


if __name__=='__main__':main()
