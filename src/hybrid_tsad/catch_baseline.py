"""Thin, synthetic-only admission adapter for the external official CATCH API.

The author model, optimization loop, native split, and scorer are not copied or
changed. Preview/tests do not import torch. No real-data operation is admitted.
"""
import argparse
import hashlib
import importlib
import json
import math
import random
import sys
import time
import types
from pathlib import Path

COMMIT = '3647c69be5eb56649b072596cf89098e689e20c3'
PARAMS = dict(seq_len=192,batch_size=128,num_epochs=1,lr=.005,Mlr=.001,
              cf_dim=16,d_model=16,d_ff=32,e_layers=1,n_heads=4,head_dim=32,
              dropout=.3,patch_size=16,patch_stride=8,inference_patch_size=96,
              auxi_lambda=.01,dc_lambda=.05,score_lambda=.5)
SUPPORT = ['ts_benchmark/__init__.py','ts_benchmark/baselines/__init__.py',
           'ts_benchmark/baselines/utils.py','ts_benchmark/utils/__init__.py',
           'ts_benchmark/utils/data_processing.py',
           'ts_benchmark/baselines/time_series_library/utils/timefeatures.py',
           'ts_benchmark/baselines/time_series_library/utils/__init__.py',
           'scripts/multivariate_detection/detect_score/PSM_script/CATCH.sh']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_layout(n=2048,score_n=401):
    """Official split floor, stride1 FIT/val, no drop_last, stride192 thre."""
    cut=int(.8*n)
    fit,select=cut-192+1,n-cut-192+1
    if min(fit,select)<=0 or math.ceil(fit/128)<10:
        raise ValueError('Native loader must contain >=10 training batches')
    return dict(train_points=n,fit_points=cut,select_points=n-cut,
                fit_windows=fit,select_windows=select,
                fit_batches=math.ceil(fit/128),select_batches=math.ceil(select/128),
                fit_last_batch=fit%128 or 128,select_last_batch=select%128 or 128,
                mask_interval=min(math.ceil(fit/128)//10,100),
                score_windows=score_n//192,scored_points=(score_n//192)*192,
                unscored_tail=score_n%192,score_points=score_n)


def synthetic_sequences(seed=2025,train_points=2048,score_points=401):
    """Separate explicit RNG streams, no labels or weak-lag DEV assets."""
    import numpy as np
    result=[]
    for stream,n in enumerate((train_points,score_points)):
        rng=np.random.default_rng(np.random.SeedSequence([seed,stream]))
        t=np.arange(n,dtype=np.float64)
        u=np.sin(t/17)+.3*np.cos(t/43)+rng.normal(0,.1,n)
        v=.7*np.sin((t-7)/17)+.2*np.cos(t/43)+rng.normal(0,.1,n)
        result.append(np.column_stack((u,v)))
    return result


def validate(path):
    cfg=json.loads(Path(path).read_text(encoding='utf-8'))
    if (cfg.get('operation')!='CATCH_C2_NATIVE_L0' or cfg.get('upstream_commit')!=COMMIT
        or cfg.get('seed')!=2025 or cfg.get('params')!=PARAMS
        or cfg.get('train_points')!=2048 or cfg.get('score_points')!=401
        or cfg.get('data') is not None or cfg.get('checkpoint') is not None):
        raise PermissionError('Only frozen synthetic native CATCH L0 is supported')
    if cfg.get('max_seconds')!=180 or cfg.get('max_allocated_GiB')!=20 or cfg.get('physical_gpu')!=0:
        raise PermissionError('L0 budget/device differs')
    root=Path(cfg['source_root']).resolve()
    expected=cfg['source_hashes']
    actual={str(p.relative_to(root)).replace('\\','/') for p in
            (root/'ts_benchmark/baselines/catch').rglob('*.py')}
    if set(expected)!=actual|set(SUPPORT):
        raise ValueError('Expected complete CATCH and explicit imported support identity')
    for rel,digest in expected.items():
        if sha(root/rel)!=digest:raise ValueError('External source mismatch: '+rel)
    if sha(__file__)!=cfg['adapter_sha256']:
        raise ValueError('Adapter differs from frozen execution snapshot')
    if Path(cfg['output']).exists():raise FileExistsError(cfg['output'])
    return cfg,root,native_layout(cfg['train_points'],cfg['score_points'])


def require_release(path,cfg):
    release=json.loads(Path(cfg['approval_path']).read_text())
    if (release.get('status')!='PRIMARY_RELEASED_CATCH_C2_NATIVE_L0'
        or release.get('approved_by')!='root' or release.get('config_sha256')!=sha(path)
        or release.get('adapter_sha256')!=sha(__file__)
        or release.get('allow_real_data') is not False
        or release.get('allow_labels') is not False):
        raise PermissionError('Exact root synthetic L0 release required before importing torch')


def external_api(root):
    """Same narrow namespace isolation as existing CATCH baseline scripts."""
    sys.path.insert(0,str(root))
    name='ts_benchmark.baselines.time_series_library'
    proxy=types.ModuleType(name)
    proxy.__path__=[str(root/'ts_benchmark/baselines/time_series_library')]
    sys.modules[name]=proxy
    return importlib.import_module('ts_benchmark.baselines.catch.CATCH')


def execute(cfg,root,layout):
    import os
    import subprocess
    import traceback
    import numpy as np
    import pandas as pd
    import torch
    out=Path(cfg['output']);out.mkdir(exist_ok=False,parents=True)
    started=time.monotonic()
    report={'status':'STARTED','weights_saved':False,'labels_read':False,
            'scope':'one-epoch engineering L0; not official three-epoch replication',
            'layout':layout,'seed':2025,'source_commit':COMMIT,
            'source_hashes':cfg['source_hashes'],'adapter_sha256':sha(__file__)}
    events=[];forwards=[];phase=['fit'];initial={};native_step=None;native_backward=None
    handles=[];access=[];native_getitem=None
    def write():
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        (out/'optimizer_events.json').write_text(json.dumps(events,indent=2,allow_nan=False))
        (out/'forward_trace.json').write_text(json.dumps(forwards,indent=2,allow_nan=False))
        (out/'loader_access.json').write_text(json.dumps(access))
    try:
        if not torch.cuda.is_available():raise RuntimeError('Approved L0 requires one A40')
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(20*2**30/torch.cuda.get_device_properties(0).total_memory,0)
        torch.cuda.reset_peak_memory_stats()
        uuid=str(torch.cuda.get_device_properties(0).uuid)
        gpu=subprocess.check_output(['nvidia-smi','-i',str(cfg['physical_gpu']),
              '--query-gpu=uuid,name','--format=csv,noheader'],text=True).strip()
        if uuid not in gpu:raise RuntimeError('Logical/physical GPU UUID mismatch')
        report.update(torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,
                      pandas=pd.__version__,GPU_UUID=uuid,GPU_inventory=gpu,
                      CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
        torch.set_num_threads(1)
        random.seed(2025);np.random.seed(2025);torch.manual_seed(2025);torch.cuda.manual_seed_all(2025)
        api=external_api(root)
        model=api.CATCH(**cfg['params'])
        from ts_benchmark.baselines.utils import SegLoader
        native_getitem=SegLoader.__getitem__
        def observed_getitem(loader,index):
            access.append({'mode':loader.mode,'window_index':int(index),
                           'start':int(index)*(192 if loader.mode=='thre' else 1)})
            return native_getitem(loader,index)
        SegLoader.__getitem__=observed_getitem
        train,score=synthetic_sequences()
        np.savez(out/'synthetic_inputs.npz',train=train,score=score)
        report['synthetic_inputs_sha256']=sha(out/'synthetic_inputs.npz')
        def frame(x):return pd.DataFrame(x,columns=['synthetic_u','synthetic_v'],
                                         index=pd.date_range('2000-01-01',periods=len(x),freq='min'))
        def state_digest():
            h=hashlib.sha256()
            for k,v in model.model.state_dict().items():
                h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        def rng():
            return {'cpu':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                    'cuda':hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
        def before(module,args):
            if isinstance(module,api.CATCHModel) and not initial:
                initial.update({k:v.detach().cpu().clone() for k,v in module.named_parameters()})
        def after(module,args,output):
            if not isinstance(module,api.CATCHModel):return
            assert all(torch.isfinite(v).all().item() for v in output)
            forwards.append({'phase':phase[0],'training':module.training,
                             'input_shape':list(args[0].shape),'output_shape':list(output[0].shape),
                             'input_sha256':hashlib.sha256(args[0].detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                             'dc_loss':float(output[2].detach().cpu())})
        handles=[torch.nn.modules.module.register_module_forward_pre_hook(before),
                 torch.nn.modules.module.register_module_forward_hook(after)]
        native_step=torch.optim.Adam.step;native_backward=torch.autograd.backward
        def observed_step(opt,*a,**kw):
            params=[p for g in opt.param_groups for p in g['params']]
            grads=[p.grad for p in params if p.grad is not None]
            assert all(torch.isfinite(g).all().item() for g in grads)
            previous=[p.detach().clone() for p in params]
            event={'event':'optimizer_step','optimizer':'main' if opt is model.optimizer else 'mask',
                   'lr':[g['lr'] for g in opt.param_groups],'gradient_tensors':len(grads),
                   'gradient_abs_max':max((float(g.detach().abs().max()) for g in grads),default=0.)}
            result=native_step(opt,*a,**kw)
            assert all(torch.isfinite(p).all().item() for p in params)
            event['parameter_delta_abs_max']=max(float((p.detach()-v).abs().max()) for p,v in zip(params,previous))
            events.append(event)
            assert torch.cuda.max_memory_allocated()<=20*2**30
            return result
        def observed_backward(tensors,*a,**kw):
            t=tensors if torch.is_tensor(tensors) else tensors[0]
            assert torch.isfinite(t).all().item()
            events.append({'event':'backward','loss':float(t.detach().cpu())})
            return native_backward(tensors,*a,**kw)
        torch.optim.Adam.step=observed_step;torch.autograd.backward=observed_backward
        t=time.monotonic()
        try:model.detect_fit(frame(train),None)
        finally:
            torch.optim.Adam.step=native_step;torch.autograd.backward=native_backward
        torch.cuda.synchronize();report['fit_seconds']=time.monotonic()-t
        assert len(model.train_data_loader)==layout['fit_batches']>=10
        assert len(model.valid_data_loader)==layout['select_batches']
        counts={name:sum(e.get('optimizer')==name for e in events) for name in ('main','mask')}
        assert counts=={'main':12,'mask':12}
        assert [e['event'] if e['event']=='backward' else e['optimizer'] for e in events]==['mask','backward','main']*12
        assert events[0]['gradient_tensors']==0
        assert any(e.get('parameter_delta_abs_max',0)>0 for e in events if e.get('optimizer')=='mask')
        delta={k:float((p.detach().cpu()-initial[k]).abs().max()) for k,p in model.model.named_parameters()}
        assert any(v>0 for k,v in delta.items() if 'mask_generator' not in k)
        assert np.isfinite(model.early_stopping.val_loss_min)
        fit_state=state_digest();phase[0]='score';before_rng=rng()
        t=time.monotonic();scores,alias=model.detect_score(frame(score));torch.cuda.synchronize()
        report['score_seconds']=time.monotonic()-t
        scores=np.asarray(scores)
        assert scores.shape==(layout['scored_points'],) and np.isfinite(scores).all()
        assert np.array_equal(scores,alias) and state_digest()==fit_state
        assert len(model.thre_loader.dataset)==layout['score_windows']
        idx=np.arange(layout['scored_points'])
        np.savez(out/'scores.npz',score=scores,index=idx,window_id=idx//192,
                 window_end=(idx//192+1)*192-1,valid_mask=np.arange(401)<layout['scored_points'],
                 missing_indices=np.arange(layout['scored_points'],401))
        report.update(status='COMPLETED_PENDING_REVIEW',optimizer_steps=counts,
            first_mask_step_no_gradient=True,mask_before_current_backward=True,
            parameter_delta_abs_max=delta,validation_loss=float(model.early_stopping.val_loss_min),
            final_learning_rates={'main':model.optimizer.param_groups[0]['lr'],'mask':model.optimizerM.param_groups[0]['lr']},
            score_shape=list(scores.shape),score_sha256=sha(out/'scores.npz'),
            state_before_score=fit_state,state_after_score=state_digest(),
            score_rng_before=before_rng,score_rng_after=rng(),eval_gumbel_not_disabled=True,
            parameters=sum(p.numel() for p in model.model.parameters()),
            imported_source={name:{'path':str(Path(m.__file__).resolve()),'sha256':sha(m.__file__)}
                 for name,m in list(sys.modules.items()) if name.startswith('ts_benchmark') and getattr(m,'__file__',None)},
            source_unchanged=all(sha(root/p)==v for p,v in cfg['source_hashes'].items()),
            normal_run_protocol='L0 only: native contiguous 80/20; independent FIT/SELECT not implemented',
            timing_semantics='retrospective reconstruction; points visible only after window end',
            scheduler_behavior='native OneCycle constructed; no training scheduler.step; native type1 for both optimizers')
        assert report['source_unchanged']
    except BaseException:
        report.update(status='FAILED_NO_AUTOMATIC_RETRY',traceback=traceback.format_exc())
        raise
    finally:
        if native_step is not None:torch.optim.Adam.step=native_step
        if native_backward is not None:torch.autograd.backward=native_backward
        if native_getitem is not None:SegLoader.__getitem__=native_getitem
        for handle in handles:handle.remove()
        report['whole_seconds']=time.monotonic()-started
        if torch.cuda.is_available():
            report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        write()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    cfg,root,layout=validate(a.config)
    if a.execute:
        require_release(a.config,cfg)
        print(json.dumps(execute(cfg,root,layout)),flush=True)
    else:print(json.dumps({'status':'PREVIEW_ONLY_NO_TORCH','layout':layout,
                          'source_files':len(cfg['source_hashes']),'config_sha256':sha(a.config)},indent=2))


if __name__=='__main__':main()
