"""LEFT normal-run adapter: native training loop, separately released operations.

No original model/Exp.train loop is copied or modified. CAL, fault query and
evaluation have no CLI execution path.
"""
import argparse
from contextlib import contextmanager
import json
import math
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

from . import left_baseline as baseline
from .normal_run_data import IndependentNormalWindows

COLUMNS=('xmeas_9','xmeas_21','xmv_10')
CURRICULUM=dict(warmup_steps=123,ramp_steps=246,lambda_max=.3)
TRAIN_CONFIG=dict(train_epochs=20,batch_size=128,optim='adam',learning_rate=1e-4,lradj='type2',patience=3)
TRAIN_SEED_DEVICES={2025:(0,'895053ba-a4d6-49a6-d831-865d5982f009'),
    2026:(1,'514a18de-bcd9-8992-c47f-775f98aed4b6'),
    2027:(2,'a3b69732-11e4-3841-7a95-735c2ac27e47')}


def validate_training_identity(cfg):
    """Only preregistered seeds/cards; logical cuda0 is initialized before Exp."""
    seed=cfg.get('seed')
    if (seed not in TRAIN_SEED_DEVICES
        or (cfg.get('physical_gpu_index'),cfg.get('expected_gpu_uuid'))!=TRAIN_SEED_DEVICES[seed]
        or cfg.get('job_id')!=f'LEFT_TEP_C3_H100_NORMAL_SEED{seed}'):
        raise PermissionError('Predeclared seed/job/physical-GPU identity required')


def native_infer_score(model,values):
    """Official LEFT infer returns (score,None), never a bare Tensor."""
    result=model.infer(values,None,None,None)
    if not isinstance(result,tuple) or len(result)!=2:raise TypeError('Official infer tuple required')
    score,auxiliary=result
    if auxiliary is not None:raise ValueError('Unexpected official infer auxiliary output')
    return score


def effective_configs(model_config,data_config,train_config):
    if (model_config.get('seq_len'),model_config.get('enc_in'),data_config.get('seq_len'),data_config.get('enc_in'))!=(192,25,192,25):
        raise ValueError('Expected unchanged author PSM configs as source')
    if train_config!=TRAIN_CONFIG:raise ValueError('Original Adam/lr/epoch/patience config changed')
    if any(model_config.get(k)!=v for k,v in dict(warmup_steps=1000,ramp_steps=2000,lambda_max=.3).items()):
        raise ValueError('Expected source curriculum 1000/2000/.3')
    model=dict(model_config,seq_len=100,enc_in=3,**CURRICULUM)
    data=dict(data_config,seq_len=100,enc_in=3)
    return dict(model=model,data=data,train=dict(train_config),changes={
        'model':{k:[model_config[k],model[k]] for k in ('seq_len','enc_in','warmup_steps','ramp_steps')},
        'data':{k:[data_config[k],data[k]] for k in ('seq_len','enc_in')},
        'selection':'only normal validation calls at model.step>=369 enter native EarlyStopping',
        'unchanged_model_keys':[k for k in model_config if model_config[k]==model[k]],
        'not_author_default_curriculum':True})


def normal_exp_class(official,fit_runs,select_runs,scaler,events):
    """Factory only. Instantiation/forward is prohibited during CPU preparation."""
    train=IndependentNormalWindows(fit_runs,scaler,'train',window=100)
    select=IndependentNormalWindows(select_runs,scaler,'val',window=100)
    class PublicNormalOnly(official.Exp_Anomaly_Detection):
        def _build_model(self):
            # Native adds a runtime torch.device to data configs; keep it on
            # the instance but do not pretend it is an author JSON field.
            data_json={k:v for k,v in vars(self.model_configs).items() if k!='device'}
            parsed=effective_configs(vars(self.hyper_args),data_json,vars(self.train_configs))
            self.hyper_args.__dict__.update(parsed['model']);self.model_configs.__dict__.update(parsed['data'])
            if (self.hyper_args.seq_len,self.hyper_args.enc_in,self.model_configs.seq_len,self.model_configs.enc_in)!=(100,3,100,3):
                raise ValueError('Model and loader shapes must agree')
            events.append({'kind':'effective_configs','value':parsed})
            return super()._build_model()

        def _get_data(self,flag,step=None):
            if flag not in ('train','val'):raise PermissionError('No CAL/query/test loader')
            expected=1 if flag=='train' else 100
            if step is not None and step!=expected:raise ValueError('Fixed within-run strides only')
            from torch.utils.data import DataLoader
            ds=train if flag=='train' else select
            return ds,DataLoader(ds,batch_size=self.train_configs.batch_size,shuffle=flag=='train',num_workers=0,drop_last=False)

        def test(self,**kwargs):raise PermissionError('No automatic query evaluation')
        def evaluate_spot(self,**kwargs):raise PermissionError('No automatic calibration/evaluation')
    return PublicNormalOnly


@contextmanager
def eligible_checkpoints(official,events,minimum_step=369):
    """Narrow module-global EarlyStopping wrapper; eligible calls delegate exactly.

    All validation still occurs in original Exp.train. Pre-course calls are
    logged but cannot save, update best/counter or trigger early stopping.
    """
    if minimum_step!=369:raise ValueError('Predeclared curriculum-completion boundary required')
    if getattr(official,'_eligible_normal_selection_active',False):raise RuntimeError('No nested global controller')
    native=official.EarlyStopping
    class EligibleEarlyStopping(native):
        def __call__(self,val_loss,model,path):
            step=model.step
            if not isinstance(step,int) or step<0 or not math.isfinite(float(val_loss)):
                raise ValueError('Finite validation loss and actual nonnegative Python step required')
            event={'kind':'normal_select_eligibility','model_step':step,'loss':float(val_loss),'eligible':step>=minimum_step}
            events.append(event)
            if step<minimum_step:
                event.update(native_called=False,counter_after=self.counter,early_stop_after=self.early_stop)
                return None
            result=super().__call__(val_loss,model,path)
            event.update(native_called=True,counter_after=self.counter,early_stop_after=self.early_stop,
                         best_validation_after=float(self.val_loss_min))
            return result
    official.EarlyStopping=EligibleEarlyStopping;official._eligible_normal_selection_active=True
    try:yield
    finally:
        official.EarlyStopping=native;del official._eligible_normal_selection_active


def validate_normal_metadata(inputs):
    """Only the existing 50 normal metadata references, no array/header/body IO."""
    expected={'FIT':list(range(1,21))+list(range(51,61)),'SELECT':list(range(61,81))}
    if tuple(inputs['columns'])!=COLUMNS or set(inputs['roles'])!=set(expected):
        raise PermissionError('Fixed three-variable FIT/SELECT only; no CAL or fault query')
    for role,ids in expected.items():
        rows=inputs['roles'][role]
        if [x['simulationRun'] for x in rows]!=ids:raise PermissionError('Original normal run roles cannot be reallocated')
        for row in rows:
            p=Path(row['array_path'])
            if (row['role']!=role.lower() or row['faultNumber']!=0 or tuple(row['column_names'])!=COLUMNS
                or row['shape']!=[500,3] or row['dtype']!='float64'
                or p.parts[-2]!=role.lower() or p.name!=f"run{row['simulationRun']:03d}.npy"
                or row['raw_sha256']!='12d1055b852763fac09ef36bd7c9612a800c8bf7b6f75f344d12119206f9e940'
                or Path(row['raw_source']).name!='TEP_FaultFree_Training.RData'
                or row['imputation'] is not False or row['standardization'] is not False):
                raise PermissionError('Normal original-float64 lineage/column order differs')
    return inputs


def load_normal_arrays(inputs,*,released=False):
    """Prepared future normal-training IO; never called by current preview/profile."""
    if released is not True:raise PermissionError('Actual normal-array body reads need separate training release')
    validate_normal_metadata(inputs)
    import numpy as np
    from .normal_run_data import NormalRun
    result={}
    for role in ('FIT','SELECT'):
        result[role]=[]
        for ref in inputs['roles'][role]:
            if baseline.file_sha(ref['array_path'])!=ref['sha256']:raise ValueError('Accepted normal bytes changed')
            values=np.load(ref['array_path'],allow_pickle=False)
            if values.shape!=(500,3) or values.dtype!=np.float64 or not np.isfinite(values).all():
                raise ValueError('Original finite float64 normal shape required')
            result[role].append(NormalRun('TEP_REACTOR',role,ref['simulationRun'],COLUMNS,values))
    return result


def preview(path):
    cfg=baseline.read_json(path)
    if (cfg.get('operation')!='LEFT_TEP_C3_H100_SYNTHETIC_PROFILE' or cfg.get('upstream_commit')!=baseline.COMMIT
        or cfg.get('seed')!=2025 or cfg.get('steps')!=3 or cfg.get('batch_size')!=128
        or cfg.get('inference_batch_size')!=1
        or cfg.get('num_optimizer_updates')!=3 or cfg.get('forced_backward_probes')!=2 or cfg.get('infer_calls')!=10
        or cfg.get('max_seconds')!=180 or cfg.get('max_allocated_GiB')!=20
        or cfg.get('physical_gpu_index')!=0 or not isinstance(cfg.get('expected_gpu_uuid'),str)
        or cfg.get('data') is not None or cfg.get('checkpoint') is not None):
        raise PermissionError('Only fixed synthetic profile preparation is available')
    return validate_common(cfg)


def validate_common(cfg):
    """Exact implementation/source and FIT/SELECT metadata only; no arrays."""
    root=Path(cfg['source_root'])
    required=set(baseline.MODEL_FILES+['exp/exp_anomaly_detection.py',
        'configs/PSM/model_configs_Left.json','configs/PSM/model_configs_0.json','configs/PSM/train_configs.json']+
        ['models/CrossAD/'+n+'.py' for n in ('Basic_CrossAD','Attention_Blocks','EncDec','Context_Blocks')])
    if set(cfg['source_hashes'])!=required:raise ValueError('Exact fixed official source closure required')
    for name,digest in cfg['source_hashes'].items():
        if baseline.file_sha(root/name)!=digest:raise ValueError('Fixed external source differs: '+name)
    if set(cfg['adapter_hashes'])!={'__init__.py','left_baseline.py','normal_run_data.py','left_normal_public.py'}:
        raise ValueError('Exact own-source helper identity required')
    for name,digest in cfg['adapter_hashes'].items():
        if baseline.file_sha(Path(__file__).parent/name)!=digest:raise ValueError('Own adapter differs: '+name)
    ref=cfg['normal_metadata']
    if baseline.file_sha(ref['path'])!=ref['sha256']:raise ValueError('Frozen metadata differs')
    validate_normal_metadata(baseline.read_json(ref['path']))
    accepted=cfg['normal_source_acceptance']
    if baseline.file_sha(accepted['path'])!=accepted['sha256']:raise ValueError('Normal source acceptance differs')
    receipt=baseline.read_json(accepted['path'])
    if (receipt.get('status')!='PRIMARY_ACCEPTED_LIMITED_NORMAL_ASSETS'
        or receipt.get('source_manifest_sha256')!='1e2cca586b9632cfac6b5cf8d4a73e9af486a490173753611796e8ee51958d1e'):
        raise PermissionError('Only the accepted 70-normal-run source is available')
    parsed=effective_configs(baseline.read_json(root/'configs/PSM/model_configs_Left.json'),
        baseline.read_json(root/'configs/PSM/model_configs_0.json'),baseline.read_json(root/'configs/PSM/train_configs.json'))
    if parsed!=cfg['effective_configs']:raise ValueError('Declared effective model/data/train configs differ')
    return cfg,root,parsed


def preview_train(path):
    cfg=baseline.read_json(path)
    validate_training_identity(cfg)
    if (cfg.get('operation')!='LEFT_TEP_C3_H100_NORMAL_TRAIN' or cfg.get('upstream_commit')!=baseline.COMMIT
        or cfg.get('batch_size')!=128 or cfg.get('inference_batch_size')!=1
        or cfg.get('max_seconds')!=900 or cfg.get('max_allocated_GiB')!=20
        or cfg.get('initialization')!='scratch' or cfg.get('checkpoint') is not None
        or cfg.get('max_epochs')!=20 or cfg.get('max_optimizer_updates')!=1880
        or cfg.get('eligibility_step')!=369):
        raise PermissionError('Only predeclared normal FIT/SELECT training candidates')
    if Path(cfg['output']).exists():raise FileExistsError(cfg['output'])
    contract=cfg['normal_train_contract']
    if baseline.file_sha(contract['path'])!=contract['sha256']:raise ValueError('Normal-training contract bytes differ')
    return validate_common(cfg)


def require_train_release(config_path,release_path,cfg):
    approved=baseline.read_json(release_path)
    exact={'status':'PRIMARY_RELEASED_LEFT_TEP_C3_H100_NORMAL_TRAIN','approved_by':'root',
        'operation':cfg['operation'],'config_sha256':baseline.file_sha(config_path),
        'adapter_hashes':cfg['adapter_hashes'],'seed':cfg['seed'],'job_id':cfg['job_id'],
        'allowed_jobs':[cfg['job_id']],'physical_gpu_index':cfg['physical_gpu_index'],
        'expected_gpu_uuid':cfg['expected_gpu_uuid'],'allow_normal_FIT_SELECT':True,
        'allow_CAL':False,'allow_Faulty_Testing':False,'allow_query_labels':False,
        'allow_checkpoint_initialization':False}
    if any(approved.get(k)!=v for k,v in exact.items()):
        raise PermissionError('Exact separate normal-training root release required before array/Torch/model access')
    return approved


def train_normal(config_path,release_path):
    cfg,root,parsed=preview_train(config_path)
    require_train_release(config_path,release_path,cfg)
    import os
    import sys
    import time
    import traceback
    from .normal_run_data import fit_scaler
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();events=[];torch=None
    report={'status':'RUNNING','operation':cfg['operation'],'seed':cfg['seed'],'job_id':cfg['job_id'],
        'config_sha256':baseline.file_sha(config_path),'release_sha256':baseline.file_sha(release_path),
        'adapter_sha256':baseline.file_sha(__file__),'official_train_loop_inherited':True,
        'read_roles':['FIT','SELECT'],'CAL_or_fault_or_query_or_labels_read':False,
        'profile_weights_reused':False,'effective_configs':parsed}
    try:
        runs=load_normal_arrays(baseline.read_json(cfg['normal_metadata']['path']),released=True)
        scaler=fit_scaler(runs['FIT'])
        serial={k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in scaler.items()}
        serial.update(SELECT_used_for_fit=False,normal_metadata=cfg['normal_metadata'])
        (out/'SCALER_TEP.json').write_text(json.dumps(serial,indent=2,allow_nan=False))
        before_environment=os.environ.get('CUDA_VISIBLE_DEVICES')
        if before_environment!=str(cfg['physical_gpu_index']):raise RuntimeError('Exact physical GPU launch mapping required')
        torch,_,loaded=baseline.model_runtime(cfg,root)
        def uuid():return str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')
        if uuid()!=cfg['expected_gpu_uuid']:raise RuntimeError('Actual initialized GPU UUID differs')
        report['loaded_modules']=loaded
        official=baseline.import_official(root,'exp.exp_anomaly_detection')
        Base=normal_exp_class(official,runs['FIT'],runs['SELECT'],scaler,events)

        def state_summary(model):
            return {'step':model.step,'modes':{k:v.training for k,v in model.named_modules()},
                'python_rng':baseline.hashlib.sha256(pickle.dumps(random.getstate(),protocol=4)).hexdigest(),
                'numpy_rng':baseline.hashlib.sha256(pickle.dumps(sys.modules['numpy'].random.get_state(),protocol=4)).hexdigest(),
                'torch_rng':baseline.hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                'cuda_rng':baseline.hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                'state':{k:baseline.hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                         for k,v in model.state_dict().items()}}

        class ObservedNormal(Base):
            def _basic(self):
                before=state_summary(self.model)
                try:return super()._basic()
                finally:
                    events.append({'kind':'official_basic','before':before,'after':state_summary(self.model)})
                    thop=sys.modules.get('thop')
                    if thop is not None:
                        report['actual_thop']={'path':thop.__file__,'version':str(getattr(thop,'__version__','UNVERIFIED')),
                            'sha256':baseline.file_sha(thop.__file__)}

            def _get_data(self,flag,step=None):
                ds,loader=super()._get_data(flag,step)
                events.append({'kind':'loader','flag':flag,'windows':len(ds),'batches':len(loader),
                    'first':ds.footprint(0),'last':ds.footprint(len(ds)-1),'stride':ds.step,
                    'shuffle':flag=='train','drop_last':False})
                return ds,loader

            def _select_optimizer(self):
                opt=super()._select_optimizer();self.actual_updates=0;before={}
                params=[p for p in self.model.parameters() if p.requires_grad]
                def pre(_opt,_args,_kwargs):
                    if self.actual_updates>=1880:raise RuntimeError('Predeclared optimizer-update ceiling')
                    grads=[p.grad for p in params if p.grad is not None]
                    if not grads or not all(torch.isfinite(g).all() for g in grads):
                        raise FloatingPointError('Missing/nonfinite actual training gradient')
                    before['parameters']=[p.detach().clone() for p in params]
                    before['gradient_l2']=float(torch.stack([g.detach().float().norm() for g in grads]).norm())
                    before['gradient_abs_max']=max(float(g.detach().abs().max()) for g in grads)
                def post(_opt,_args,_kwargs):
                    delta=max(float((p.detach()-b).abs().max()) for p,b in zip(params,before.pop('parameters')))
                    if not math.isfinite(delta):raise FloatingPointError('Nonfinite optimizer parameter delta')
                    self.actual_updates+=1
                    events.append({'kind':'optimizer_step','optimizer_step':self.actual_updates,
                        'model_step':self.model.step,'lr':[g['lr'] for g in opt.param_groups],
                        'gradient_l2':before['gradient_l2'],'gradient_abs_max':before['gradient_abs_max'],
                        'parameter_delta_abs_max':delta,'effective_nonzero_update':delta>0})
                    if torch.cuda.max_memory_allocated()>20*1024**3:raise MemoryError('20GiB cap exceeded')
                opt.register_step_pre_hook(pre);opt.register_step_post_hook(post)
                return opt

            def vali(self,loader):
                value=super().vali(loader)
                if not math.isfinite(float(value)):raise FloatingPointError('Nonfinite normal SELECT loss')
                events.append({'kind':'normal_validation','loss':float(value),'model_step':self.model.step,
                    'epoch':1+sum(e['kind']=='normal_validation' for e in events)})
                return value

        args=SimpleNamespace(data='PSM',data_origin='DADA',model_class='Left',gpu=0,use_gpu=True,
            gpu_type='cuda',use_multi_gpu=False,configs_path=str(root/'configs'),
            model_save_path=str(out/'model'),results_save_path=str(out/'unused_results'))
        exp=ObservedNormal(args,id=0)
        if uuid()!=cfg['expected_gpu_uuid']:raise RuntimeError('Native device acquisition changed actual GPU')
        report['GPU_mapping']={'physical_index':cfg['physical_gpu_index'],'expected_uuid':cfg['expected_gpu_uuid'],
            'actual_uuid_after_native_Exp':uuid(),'logical_device':'cuda:0',
            'CUDA_VISIBLE_DEVICES_before_initialization':before_environment,
            'CUDA_VISIBLE_DEVICES_after_native_Exp':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'native_acquire_device_modified':False}
        report['parameter_count']=sum(p.numel() for p in exp.model.parameters())
        report['initial_model_state_and_RNG']=state_summary(exp.model)
        def forward_observation(model,_inputs,result):
            loss,_auxiliary=result
            if not bool(torch.isfinite(loss).all()):raise FloatingPointError('Nonfinite native forward loss')
            events.append({'kind':'native_forward','step':model.step,'training':model.training,
                'loss':float(loss.detach()),'batch':int(_inputs[0].shape[0])})
        exp.model.register_forward_hook(forward_observation)
        curriculum=exp.model._curriculum_lambda
        def observed_curriculum():
            value=curriculum();events.append({'kind':'curriculum','step':exp.model.step,
                'training':exp.model.training,'lambda':float(value)});return value
        exp.model._curriculum_lambda=observed_curriculum
        for bank_name in ('proto_t','proto_f'):
            bank=getattr(exp.model,bank_name);native=bank.ema_update_hard
            def observed_ema(*a,_native=native,_bank=bank,_name=bank_name,**kw):
                before=_bank.prototypes.detach().clone()
                result=_native(*a,**kw)
                events.append({'kind':'native_ema_update_hard','bank':_name,'step':exp.model.step,
                    'selected_samples':int(a[0].shape[0]),
                    'prototype_delta_abs_max':float((_bank.prototypes.detach()-before).abs().max())})
                return result
            bank.ema_update_hard=observed_ema
        with eligible_checkpoints(official,events):
            exp.train()  # Original optimizer/loss/training/validation/LR/earlystop body.
        validations=[e for e in events if e['kind']=='normal_validation']
        eligible=[e for e in validations if e['model_step']>=369]
        if not eligible:raise RuntimeError('No curriculum-complete validation checkpoint')
        best=min(eligible,key=lambda e:(e['loss'],-e['epoch']))
        checkpoint=Path(exp.model_save_path)/'checkpoint.pth'
        (out/'CHECKPOINT_STATE_KEYS.json').write_text(json.dumps({k:{'shape':list(v.shape),'dtype':str(v.dtype)}
            for k,v in exp.model.state_dict().items()},indent=2))
        updates=[e for e in events if e['kind']=='optimizer_step']
        if not updates or not any(e['effective_nonzero_update'] for e in updates):
            raise RuntimeError('No actual nonzero optimizer update')
        report.update(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',optimizer_steps=exp.actual_updates,
            final_python_step=exp.model.step,validation_count=len(validations),best_epoch=best['epoch'],
            best_checkpoint_python_step_at_save=best['model_step'],
            best_eligible_SELECT_loss=best['loss'],first_eligible_validation=eligible[0],
            effective_nonzero_updates=sum(e['effective_nonzero_update'] for e in updates),
            checkpoint={'path':str(checkpoint),'sha256':baseline.file_sha(checkpoint),'bytes':checkpoint.stat().st_size},
            scaler_sha256=baseline.file_sha(out/'SCALER_TEP.json'),
            ema_call_count=sum(e['kind']=='native_ema_update_hard' for e in events),
            native_good_any_evidence=('OBSERVED_NONEMPTY_NATIVE_EMA_CALLS' if any(e['kind']=='native_ema_update_hard'
                and e['selected_samples']>0 for e in events) else 'INCOMPLETE_NO_NATIVE_EMA_CALL_OBSERVED'),
            observation='all optimizer calls: GPU parameter clones and numeric finite gradient/LR/max delta reductions; no full state/RNG copy per step; observer synchronization cost included in OS wall; EMA delegates unchanged',
            checkpoint_scope='native best state_dict only, not exact optimizer/Python-step resume',
            final_reload_does_not_restore_best_python_step=True,
            scientific_scope='normal prediction/reconstruction training only; no AD evidence')
    except BaseException:
        report.update(status='FAILED_NO_AUTOMATIC_RETRY',traceback=traceback.format_exc());raise
    finally:
        if torch is None:torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            props=torch.cuda.get_device_properties(0)
            report['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'GPU':props.name,
                'uuid':str(props.uuid),'allocated_peak_bytes':torch.cuda.max_memory_allocated(),
                'reserved_peak_bytes':torch.cuda.max_memory_reserved(),'peak_reset':'once before model construction'}
        report['whole_seconds']=time.monotonic()-started
        (out/'normal_events.json').write_text(json.dumps(events,indent=2,allow_nan=False))
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def profile(config_path,release_path):
    cfg,root,parsed=preview(config_path)
    approved=baseline.read_json(release_path)
    if (approved.get('status')!='PRIMARY_RELEASED_LEFT_TEP_C3_H100_PROFILE' or approved.get('approved_by')!='root'
        or approved.get('config_sha256')!=baseline.file_sha(config_path)
        or approved.get('adapter_hashes')!=cfg['adapter_hashes'] or approved.get('allow_only_synthetic') is not True
        or approved.get('allow_real_normal') is not False or approved.get('allow_query_labels') is not False):
        raise PermissionError('Exact separate synthetic L0 release required before Torch/model/optimizer')
    # This branch is prepared only; no release has been issued during CPU preparation.
    import time
    import numpy as np
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    report={'status':'Running','real_arrays_read':False,'checkpoint_saved':False,'steps':[],
            'forced_probes':[],'infer_calls':[]}
    torch=None
    try:
        torch,Left,loaded=baseline.model_runtime(cfg,root)
        props=torch.cuda.get_device_properties(0)
        if str(props.uuid).lower().removeprefix('gpu-')!=cfg['expected_gpu_uuid'].lower().removeprefix('gpu-'):
            raise RuntimeError('Actual GPU UUID differs from approved mapping')
        report.update(loaded_modules=loaded,torch=torch.__version__,cuda=torch.version.cuda,
                      GPU_UUID=str(props.uuid),effective_configs=parsed)
        model=Left(SimpleNamespace(**parsed['model'])).cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-4)
        model.train()
        for i in range(3):
            x=torch.randn(128,100,3,device='cuda:0');opt.zero_grad();torch.cuda.synchronize();t=time.monotonic()
            before=model.step;loss,_=model(x,None,None,None)
            assert loss.ndim==0 and torch.isfinite(loss).item() and model.step==before+1
            loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all().item() for g in grads)
            assert any(torch.count_nonzero(g).item()>0 for g in grads)
            tracked=next((name,p) for name,p in model.named_parameters() if p.grad is not None and torch.count_nonzero(p.grad).item()>0)
            before_update=tracked[1].detach().clone()
            opt.step();torch.cuda.synchronize()
            delta=float((tracked[1].detach()-before_update).abs().max());assert delta>0 and math.isfinite(delta)
            report['steps'].append(dict(update=i+1,loss=float(loss.detach()),seconds=time.monotonic()-t,
                model_step_before=before,model_step=model.step,train_lambda=float(model._curriculum_lambda()),
                gradient_l2=float(torch.stack([g.detach().float().norm() for g in grads]).norm()),
                tracked_parameter=tracked[0],tracked_parameter_delta_abs_max=delta))
        retained_step=model.step;bounds=[]
        for step in (122,123,124,368,369):
            model.step=step;bounds.append({'probe_step':step,'lambda':float(model._curriculum_lambda())})
        model.step=retained_step
        import hashlib
        def tensor_digest(named):
            h=hashlib.sha256()
            for name,value in named:
                h.update(name.encode())
                h.update(b'None' if value is None else value.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        def rng_identity():
            ns=np.random.get_state()
            return {'python':hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                'numpy':hashlib.sha256(ns[1].tobytes()+repr((ns[0],ns[2:])).encode()).hexdigest(),
                'torch_cpu':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                'cuda':hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
        def fingerprint(m):
            return {'step':m.step,'state_sha256':tensor_digest(m.state_dict().items()),
                'gradient_sha256':tensor_digest((n,p.grad) for n,p in m.named_parameters()),
                'modes':{n:q.training for n,q in m.named_modules()},'rng':rng_identity(),
                'private_ms_cache_identity':id(getattr(m,'_ms_cache',None))}
        main_snapshot={k:v.detach().clone() for k,v in model.state_dict().items()}
        main_before_probes=fingerprint(model)
        for forced_counter in (123,368):
            python_rng=random.getstate();numpy_rng=np.random.get_state();whole_probe_start=time.monotonic()
            try:
                with torch.random.fork_rng(devices=[0]):
                    probe=Left(SimpleNamespace(**parsed['model'])).cuda()
                    probe.load_state_dict(main_snapshot,strict=True);probe.step=forced_counter;probe.train()
                    assert getattr(probe,'_ms_cache',None) is not getattr(model,'_ms_cache',None)
                    assert all(torch.equal(v,main_snapshot[k]) for k,v in probe.state_dict().items())
                    assert all(v.numel()==0 or v.data_ptr()!=model.state_dict()[k].data_ptr() for k,v in probe.state_dict().items())
                    calls=[];original_lambda=probe._curriculum_lambda
                    def observed_lambda():
                        value=original_lambda();calls.append({'step':probe.step,'lambda':float(value)});return value
                    probe._curriculum_lambda=observed_lambda
                    probe.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.monotonic()
                    probe_loss,_=probe(x.detach().clone(),None,None,None)
                    assert probe.step==forced_counter+1 and probe_loss.ndim==0 and torch.isfinite(probe_loss).item()
                    probe_loss.backward();torch.cuda.synchronize();forward_backward_seconds=time.monotonic()-t
                    pg=[p.grad for p in probe.parameters() if p.grad is not None]
                    assert pg and all(torch.isfinite(g).all().item() for g in pg)
                    changed={}
                    for k,v in probe.state_dict().items():
                        if not torch.equal(v,main_snapshot[k]):
                            diff=torch.logical_xor(v,main_snapshot[k]).float() if v.dtype==torch.bool else (v-main_snapshot[k]).abs()
                            changed[k]=float(diff.max())
                    report['forced_probes'].append({'counter_forced_before':forced_counter,'actual_step_after_forward':probe.step,
                        'actual_lambda_calls':calls,'loss':float(probe_loss.detach()),
                        'gradient_l2':float(torch.stack([g.detach().float().norm() for g in pg]).norm()),
                        'gradient_abs_max':max(float(g.abs().max()) for g in pg),
                        'state_changes_without_Adam':changed,'forward_backward_seconds':forward_backward_seconds,
                        'whole_probe_seconds':time.monotonic()-whole_probe_start,'optimizer_steps':0,
                        'scope':'counter-forced isolated clone of step3 weights; not 124/369 steps of training or EMA completion'})
                    del v
                    if changed:del diff
                    del probe._curriculum_lambda
                    del probe,probe_loss,pg,original_lambda,observed_lambda
            finally:
                random.setstate(python_rng);np.random.set_state(numpy_rng)
            assert fingerprint(model)==main_before_probes
            report['forced_probes'][-1]['main_state_gradient_modes_step_and_RNG_restored']=True
        del main_snapshot
        model.eval();eval_lambda=float(model._curriculum_lambda())
        inference_input=x[:1];scores=[]
        for call in range(10):
            whole_iteration=time.monotonic();before_infer=fingerprint(model)
            torch.cuda.synchronize();t=time.monotonic()
            with torch.no_grad():score=native_infer_score(model,inference_input)
            assert score.shape==(1,100,3) and torch.isfinite(score).all().item()
            cpu_score=score.detach().cpu().numpy().copy();torch.cuda.synchronize()
            compute_transfer_seconds=time.monotonic()-t;scores.append(cpu_score)
            after_infer=fingerprint(model)
            assert {k:v for k,v in before_infer.items() if k!='rng'}=={k:v for k,v in after_infer.items() if k!='rng'}
            report['infer_calls'].append({'call':call+1,'phase':'first_B1_after_training_and_probes' if call==0 else 'steady_B1',
                'shape':list(score.shape),'compute_and_CPU_transfer_seconds':compute_transfer_seconds,
                'whole_observed_iteration_seconds':time.monotonic()-whole_iteration,
                'score_sha256':hashlib.sha256(cpu_score.tobytes()).hexdigest(),
                'before':before_infer,'after':after_infer,'auxiliary_is_None':True})
        costs=np.asarray([r['compute_and_CPU_transfer_seconds'] for r in report['infer_calls'][1:]])
        observed=np.asarray([r['whole_observed_iteration_seconds'] for r in report['infer_calls'][1:]])
        stats={'steady_n':9,'steady_compute_transfer_mean_seconds':float(costs.mean()),
            'steady_compute_transfer_P50_seconds':float(np.quantile(costs,.5)),
            'steady_compute_transfer_P95_seconds':float(np.quantile(costs,.95)),
            'steady_observed_iteration_mean_seconds':float(observed.mean()),
            'CAL_calls_per_seed':8020,'query_calls_per_seed':344400,'total_calls_per_seed':352420,
            'projection_compute_transfer_seconds_per_seed':float(costs.mean()*352420),
            'projection_observed_iteration_seconds_per_seed':float(observed.mean()*352420),
            'not_a_bound':'synthetic nine-call estimate; excludes true data loading/archive/heterogeneity, not cold-process latency'}
        np.savez(out/'synthetic_infer.npz',training_last_input_B128=x.detach().cpu().numpy(),
                 inference_input_B1=inference_input.detach().cpu().numpy(),scores_B1=np.stack(scores),
                 inference_available_index=np.full(10,99,dtype=np.int64))
        report.update(status='Completed_pending_review',boundary_function_only=bounds,eval_lambda=eval_lambda,
                      inference_timing=stats,infer_shape=list(score.shape),model_step_frozen_in_eval=True,
                      prototype_training_good_any='INCOMPLETE',parameter_count=sum(p.numel() for p in model.parameters()))
    except BaseException as exc:
        report.update(status='Failed_no_automatic_retry',error_type=type(exc).__name__,error=str(exc),
                      traceback=__import__('traceback').format_exc())
        raise
    finally:
        report['whole_seconds']=time.monotonic()-start
        if torch is not None and torch.cuda.is_initialized():report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--release');p.add_argument('--execute',action='store_true');a=p.parse_args()
    train=baseline.read_json(a.config).get('operation')=='LEFT_TEP_C3_H100_NORMAL_TRAIN'
    if a.execute:
        if not a.release:p.error('Separate root profile release required')
        print(json.dumps(train_normal(a.config,a.release) if train else profile(a.config,a.release)),flush=True)
    else:
        _,_,parsed=preview_train(a.config) if train else preview(a.config)
        print(json.dumps({'status':'METADATA_AND_SYNTHETIC_PLAN_ONLY_NO_MODEL_OR_ARRAY_BODY',
            'effective_configs':parsed,'normal_layout':{'FIT_runs':30,'FIT_windows':12030,'FIT_batches':94,'FIT_last_batch':126,
            'SELECT_runs':20,'SELECT_windows':100,'SELECT_batches':1},'max_updates':1880,
            'inference_batch_size':1,'inference_no_batch_future_backfill':True,
            'profile_calls':{'optimizer_updates':3,'isolated_forced_backward_probes':2,'B1_infer':10},
            'eligibility_step':369,'earliest_eligible_epoch_if_step_equals_updates':4},indent=2))


if __name__=='__main__':main()
