"""LEFT C2 normal-run adaptation; external official model/Exp remain unchanged.

No real array/model is loaded by preview. Synthetic profiling and one normal-run
training job each require their own exact root release. No automatic DEV test.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from . import left_baseline as baseline


def load_normal_assets(references,scenario,role):
    """Future authorized FIT/SELECT loading only; never search data directories."""
    import numpy as np
    from .weak_lag import NormalRun, check_normal_runs
    if scenario not in ('LIN','WEAK') or role not in ('FIT','SELECT'):
        raise PermissionError('No DEV/CONFIRM/CAL/IRR loading through C2 normal training')
    expected_count=4 if role=='FIT' else 2
    if len(references)!=expected_count:
        raise ValueError('Exact pre-existing normal role count required')
    result=[]
    for i,ref in enumerate(references):
        path=Path(ref['path'])
        if ((ref['scenario'],ref['role'],ref['run_id'])!=(scenario,role,i) or
            tuple(path.parts[-3:-1])!=(scenario,role) or path.name!=f'run{i:02d}_N.npy'):
            raise PermissionError('Reference is not an exact normal role/run array')
        if baseline.file_sha(path)!=ref['sha256']:
            raise ValueError('Frozen normal source bytes differ; do not regenerate')
        values=np.load(path,allow_pickle=False)
        if values.shape!=(24576,2) or values.dtype!=np.float64:
            raise ValueError('Fixed normal run shape/dtype required; no H100 fallback')
        result.append(NormalRun(scenario,role,i,values))
    check_normal_runs(result,role)
    return result


class IndependentNormalWindows:
    """No concatenation of time axes, no DEV/cal/labels, FIT-only static scale."""
    def __init__(self, runs, scaler, flag, *, window=192):
        import numpy as np
        from . import weak_lag as wl
        if flag not in ('train', 'val'):
            raise PermissionError('Only independent FIT/train and SELECT/val permitted')
        if window != 192:
            raise ValueError('H192 is frozen; no implicit short-context fallback')
        role = 'FIT' if flag == 'train' else 'SELECT'
        wl.check_normal_runs(runs, role)
        self.runs, self.scaler, self.flag, self.window = runs, scaler, flag, window
        self.step = 1 if flag == 'train' else window
        if any(len(r.values) < window for r in runs):
            raise ValueError('Every admitted normal run needs its own complete H192 window')
        if not scaler.get('fit_ids') or any(row[1] != 'FIT' for row in scaler['fit_ids']):
            raise ValueError('Scaler must explicitly record FIT-only provenance')
        if set(row[0] for row in scaler['fit_ids']) != {runs[0].scenario}:
            raise ValueError('FIT scaler observation scene differs')
        if role == 'FIT' and scaler['fit_ids'] != [(r.scenario,r.role,r.run_id) for r in runs]:
            raise ValueError('Training pool and scaler FIT pool must match exactly')
        mean, scale = np.asarray(scaler['mean']), np.asarray(scaler['scale'])
        if mean.shape != (2,) or scale.shape != (2,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Finite two-channel FIT scaler required')
        self.normalized = [wl.standardize(r.values,scaler) for r in runs]
        counts = [1+(len(r.values)-window)//self.step for r in runs]
        self.cumulative = np.r_[0,np.cumsum(counts)]

    def __len__(self):
        return int(self.cumulative[-1])

    def footprint(self, index):
        import numpy as np
        if not 0 <= index < len(self):
            raise IndexError(index)
        run = int(np.searchsorted(self.cumulative,index,side='right')-1)
        start = (index-int(self.cumulative[run]))*self.step
        record = self.runs[run]
        return {'scenario':record.scenario,'role':record.role,'run_id':record.run_id,
                'start':start,'end_exclusive':start+self.window,'input_end':start+self.window-1}

    def __getitem__(self,index):
        import numpy as np
        fp = self.footprint(index)
        run = int(np.searchsorted(self.cumulative,index,side='right')-1)
        values = self.normalized[run][fp['start']:fp['end_exclusive']]
        return np.asarray(values,dtype=np.float32), np.zeros((self.window,1),np.float32)


def c2_model_config(author_config):
    """New C2 instance from original PSM hyperparameters, never C25 weights."""
    if author_config.get('seq_len') != 192 or author_config.get('enc_in') != 25:
        raise ValueError('Expected frozen official PSM configuration as source')
    copied = dict(author_config); copied['enc_in'] = 2
    return copied, {'enc_in':{'author':25,'C2':2},
                    'unchanged':sorted(k for k in author_config if k!='enc_in')}


def normal_exp_class(official,fit_runs,select_runs,scaler):
    """Actual inherited train/vali/optimizer/lr/early-stop/_basic; no model now.

    Factory creation is CPU metadata. Instantiating the returned class constructs
    the official model and is forbidden until a future formal train release.
    """
    train_data = IndependentNormalWindows(fit_runs,scaler,'train')
    select_data = IndependentNormalWindows(select_runs,scaler,'val')

    class C2NormalOnly(official.Exp_Anomaly_Detection):
        def _build_model(self):
            # Official __init__ reads original configs; only runtime C2 shapes
            # change. _basic uses model_configs, Left uses hyper_args.
            if self.model_configs.seq_len != 192 or self.hyper_args.seq_len != 192:
                raise ValueError('H192 required for this separate synthetic task')
            if self.model_configs.enc_in != 25 or self.hyper_args.enc_in != 25:
                raise ValueError('Original PSM source config expected before C2 override')
            self.model_configs.enc_in = 2
            self.hyper_args.enc_in = 2
            return super()._build_model()

        def _get_data(self,flag,step=None):
            if flag not in ('train','val'):
                raise PermissionError('No init/test/DEV/CAL/CONFIRM or label loader')
            expected_step = 1 if flag=='train' else 192
            if step is not None and step != expected_step:
                raise ValueError('Fixed train stride1 / SELECT stride192 only')
            from torch.utils.data import DataLoader
            ds = train_data if flag=='train' else select_data
            return ds,DataLoader(ds,batch_size=self.train_configs.batch_size,
                                 shuffle=flag=='train',drop_last=False,num_workers=0)

        def test(self,**kwargs):
            raise PermissionError('Normal train adapter never auto-evaluates DEV/query')

        def evaluate_spot(self,**kwargs):
            raise PermissionError('No automatic calibration or label evaluation')

    return C2NormalOnly


def validate_profile(path):
    """Identity-only preview: small source files, no array bodies or torch."""
    cfg = baseline.read_json(path)
    if (cfg.get('operation'),cfg.get('track'),cfg.get('seed')) != ('profile','WL_C2',2025):
        raise ValueError('Only separate C2 synthetic profile candidate is supported')
    if cfg.get('data') is not None or cfg.get('checkpoint') is not None or cfg.get('initialization') != 'random':
        raise ValueError('No normal/DEV values, PSM C25 checkpoint, or pretrained initialization for profile')
    if cfg.get('upstream_commit') != baseline.COMMIT:
        raise ValueError('Unexpected official commit')
    if (cfg.get('batch_size'),cfg.get('steps'),cfg.get('max_seconds'),cfg.get('max_allocated_GiB')) != (128,3,180,20):
        raise ValueError('Fixed B128/3-step/180s/20GiB synthetic profile')
    root = Path(cfg['source_root']).resolve()
    for name in baseline.MODEL_FILES+['configs/PSM/model_configs_Left.json']:
        if baseline.file_sha(root/name) != cfg['source_hashes'].get(name):
            raise ValueError('Official bytes differ: '+name)
    config,difference = c2_model_config(baseline.read_json(root/'configs/PSM/model_configs_Left.json'))
    return cfg,root,config,difference


def profile_c2(config_path):
    """Candidate only: requires root exact config+module release, then three steps.

    Uses existing model_runtime for fixed seed, original five imported files,
    preconstruction CUDA memory cap and reset. No formal array loader involved.
    """
    cfg,root,hyper,diff = validate_profile(config_path)
    approval = baseline.read_json(cfg['approval_path'])
    if (approval.get('status') != 'PRIMARY_RELEASED_LEFT_WL_C2_PROFILE' or
        approval.get('config_sha256') != baseline.file_sha(config_path) or
        approval.get('adapter_sha256') != baseline.file_sha(__file__) or
        approval.get('baseline_adapter_sha256') != baseline.file_sha(baseline.__file__) or
        approval.get('allow_real_data') is not False or approval.get('allow_labels') is not False):
        raise PermissionError('C2 GPU profile has not been separately released')
    output = Path(cfg['output']); output.mkdir(parents=True,exist_ok=False)
    report = {'status':'RUNNING','C2_difference':diff,'checkpoint_saved':False}
    # Same official cost loop as prior B128 profile; only C2 instance/input dims.
    import time
    torch,Left,loaded = baseline.model_runtime(cfg,root)
    report['loaded_modules'] = loaded
    props=torch.cuda.get_device_properties(0)
    report['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,
                       'GPU':props.name,'uuid':str(getattr(props,'uuid','UNAVAILABLE'))}
    model = Left(SimpleNamespace(**hyper)).cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    model.train(); rows=[]
    report['parameter_count'] = sum(p.numel() for p in model.parameters())
    for index in range(3):
        values=torch.randn(128,192,2,device='cuda:0')
        torch.cuda.synchronize();start=time.monotonic();optimizer.zero_grad()
        loss,_=model(values,None,None,None)
        assert loss.ndim==0 and torch.isfinite(loss)
        loss.backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(bool((g!=0).any()) for g in gradients)
        optimizer.step();torch.cuda.synchronize()
        rows.append({'step':index+1,'batch':128,'loss':float(loss.detach()),
                     'update_seconds':time.monotonic()-start,'model_step':model.step})
        if torch.cuda.max_memory_allocated()>20*1024**3:
            raise MemoryError('20GiB exceeded; no automatic retry')
    report.update(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',steps=rows,
        allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        profile_not_formal_training=True,real_data_loaded=False,
        coverage={'warmup1000':'INCOMPLETE','prototype_good_any':'INCOMPLETE'})
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report


def validate_train(path):
    """Only small metadata/source reads; never open normal array bodies here."""
    cfg=baseline.read_json(path)
    if ((cfg.get('operation'),cfg.get('track'),cfg.get('seed'))!=('train','WL_C2',2025) or
        cfg.get('scenario') not in ('LIN','WEAK')):
        raise ValueError('Only independent LIN/WEAK C2 normal training seed2025')
    if cfg.get('physical_gpu_index') not in (0,1) or not isinstance(cfg.get('expected_gpu_uuid'),str):
        raise ValueError('Explicit candidate physical GPU0/1 UUID identity required')
    if cfg.get('checkpoint') is not None or cfg.get('initialization')!='random':
        raise ValueError('No profile, PSM25 or pretrained weights; start from scratch')
    if cfg.get('upstream_commit')!=baseline.COMMIT or cfg.get('max_allocated_GiB')!=20:
        raise ValueError('Official commit / memory bound differs')
    if cfg.get('max_seconds')!=3600:
        raise ValueError('Candidate whole training bound is 3600 seconds')
    root=Path(cfg['source_root']).resolve()
    required=baseline.MODEL_FILES+['exp/exp_anomaly_detection.py',
        'configs/PSM/model_configs_Left.json','configs/PSM/model_configs_0.json','configs/PSM/train_configs.json']
    required+=['models/CrossAD/'+name+'.py' for name in ('Basic_CrossAD','Attention_Blocks','EncDec','Context_Blocks')]
    for name in required:
        if baseline.file_sha(root/name)!=cfg['source_hashes'].get(name):
            raise ValueError('Official training source identity differs: '+name)
    original_train=baseline.read_json(root/'configs/PSM/train_configs.json')
    if original_train!={'train_epochs':20,'batch_size':128,'optim':'adam','learning_rate':1e-4,'lradj':'type2','patience':3}:
        raise ValueError('Original optimizer/lr/earlystop budget changed')
    c2_model_config(baseline.read_json(root/'configs/PSM/model_configs_Left.json'))
    ref=cfg['normal_inputs']
    if baseline.file_sha(ref['path'])!=ref['sha256']:
        raise ValueError('Normal source manifest differs')
    inputs=baseline.read_json(ref['path'])
    accepted=cfg['cpu_acceptance']
    if baseline.file_sha(accepted['path'])!=accepted['sha256']:
        raise ValueError('Source CPU acceptance bytes differ')
    acceptance=baseline.read_json(accepted['path'])
    if (acceptance.get('status')!='PRIMARY_ACCEPTED_LIMITED_WL_CPU_RESULTS' or
        acceptance.get('original_manifest_sha256')!=inputs.get('source_manifest_sha256')):
        raise ValueError('Frozen existing source normal assets were not accepted')
    if set(inputs['roles'])!={'FIT','SELECT'} or inputs['scenario']!=cfg['scenario']:
        raise PermissionError('Only original matching FIT/SELECT references, no CAL/DEV/CONFIRM')
    for role,count in (('FIT',4),('SELECT',2)):
        if len(inputs['roles'][role])!=count:
            raise ValueError('Normal role counts changed')
        for i,row in enumerate(inputs['roles'][role]):
            p=Path(row['path'])
            if ((row['scenario'],row['role'],row['run_id'])!=(cfg['scenario'],role,i) or
                tuple(p.parts[-3:-1])!=(cfg['scenario'],role) or p.name!=f'run{i:02d}_N.npy'):
                raise PermissionError('Invalid independent normal role footprint')
    if Path(cfg['output']).exists():
        raise FileExistsError(cfg['output'])
    return cfg,root,inputs


def _require_train_approval(path,cfg):
    approval=baseline.read_json(cfg['approval_path'])
    checks={'status':'PRIMARY_RELEASED_LEFT_WL_C2_NORMAL_TRAIN',
        'config_sha256':baseline.file_sha(path),'adapter_sha256':baseline.file_sha(__file__),
        'baseline_adapter_sha256':baseline.file_sha(baseline.__file__),
        'weak_lag_sha256':baseline.file_sha(Path(__file__).with_name('weak_lag.py')),
        'scenario':cfg['scenario'],'seed':2025,'allow_train':True,'allow_CAL':False,
        'allow_DEV':False,'allow_CONFIRM':False,'allow_labels':False,'allow_checkpoint_initialization':False}
    if any(approval.get(k)!=value for k,value in checks.items()):
        raise PermissionError('Exact separate scene normal-train release required')
    return approval


def train_c2(config_path):
    """One separately released normal job; native Exp training, no auto-test."""
    cfg,root,inputs=validate_train(config_path)
    _require_train_approval(config_path,cfg)  # before numpy array load / torch import
    import signal
    import time
    import traceback
    import sys
    import os
    from . import weak_lag as wl
    output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();events=[]
    scene=cfg['scenario']
    report={'status':'RUNNING','operation':'train','track':'WL_C2','scenario':scene,'seed':2025,
            'config_sha256':baseline.file_sha(config_path),'adapter_sha256':baseline.file_sha(__file__),
            'normal_inputs_sha256':cfg['normal_inputs']['sha256'],
            'DEV_or_CONFIRM_values_read':False,'CAL_values_read':False,'labels_read':False,
            'checkpoint_initialization':False,'official_train_body_unchanged':True}
    def timed_out(*_):
        raise TimeoutError('C2 normal training time limit')
    old_handler=signal.signal(signal.SIGALRM,timed_out)
    signal.alarm(cfg['max_seconds'])
    torch=None
    try:
        fit=load_normal_assets(inputs['roles']['FIT'],scene,'FIT')
        select=load_normal_assets(inputs['roles']['SELECT'],scene,'SELECT')
        scaler=wl.fit_scaler(fit)
        scaler_record={'track':'WL_C2','scenario':scene,'fit_ids':scaler['fit_ids'],
            'mean':scaler['mean'].tolist(),'scale':scaler['scale'].tolist(),'ddof':0,
            'fit_points':sum(len(r.values) for r in fit),'SELECT_used_for_fit':False,
            'parent_normal_manifest_sha256':cfg['normal_inputs']['sha256']}
        (output/'SCALER_C2.json').write_text(json.dumps(scaler_record,indent=2))
        before_environment=os.environ.get('CUDA_VISIBLE_DEVICES')
        if before_environment!=str(cfg['physical_gpu_index']):
            raise RuntimeError('Physical CUDA launch mapping differs; no automatic switch')
        torch,_,loaded=baseline.model_runtime(cfg,root)
        normalized_uuid=lambda x:str(x).lower().removeprefix('gpu-')
        initialized_uuid=str(torch.cuda.get_device_properties(0).uuid)
        if normalized_uuid(initialized_uuid)!=normalized_uuid(cfg['expected_gpu_uuid']):
            raise RuntimeError('Initialized logical cuda0 UUID is not the approved physical card')
        official=baseline.import_official(root,'exp.exp_anomaly_detection')
        Base=normal_exp_class(official,fit,select,scaler)
        report['loaded_modules']=loaded

        class ObservedNormal(Base):
            def _basic(self):
                def snapshot():
                    return {'step':self.model.step,
                        'torch_rng':baseline.hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                        'cuda_rng':baseline.hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                        'modes':{k:v.training for k,v in self.model.named_modules()},
                        'state':{k:baseline.hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                                 for k,v in self.model.state_dict().items()}}
                before=snapshot()
                try:return super()._basic()
                finally:events.append({'kind':'official_basic','before':before,'after':snapshot()})

            def _get_data(self,flag,step=None):
                ds,loader=super()._get_data(flag,step)
                events.append({'kind':'loader','flag':flag,'windows':len(ds),'batches':len(loader),
                    'first':ds.footprint(0),'last':ds.footprint(len(ds)-1),'stride':ds.step,
                    'shuffle':flag=='train','drop_last':False,'workers':0})
                return ds,loader

            def _select_optimizer(self):
                optimizer=super()._select_optimizer();steps=[0]
                def observe(optimizer,_args,_kwargs):
                    steps[0]+=1
                    if steps[0]==1 or steps[0]%100==0:
                        gradients=[p.grad for p in self.model.parameters() if p.grad is not None]
                        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                            raise FloatingPointError('Nonfinite observed gradient; stop without retry')
                        events.append({'kind':'optimizer_step','optimizer_step':steps[0],
                            'model_step':self.model.step,'lr':[g['lr'] for g in optimizer.param_groups],
                            'gradient_l2':float(torch.stack([g.detach().float().norm() for g in gradients]).norm())})
                optimizer.register_step_post_hook(observe)
                self._observed_optimizer_steps=steps
                return optimizer

            def vali(self,loader):
                value=super().vali(loader)
                events.append({'kind':'normal_validation','loss':float(value),'model_step':self.model.step,
                               'epoch_by_validation':1+sum(row['kind']=='normal_validation' for row in events)})
                return value

        args=SimpleNamespace(data='PSM',data_origin='DADA',model_class='Left',gpu=0,use_gpu=True,
            gpu_type='cuda',use_multi_gpu=False,configs_path=str(root/'configs'),
            model_save_path=str(output/'model'),results_save_path=str(output/'unused_results'))
        exp=ObservedNormal(args,id=0)
        actual_uuid=str(torch.cuda.get_device_properties(0).uuid)
        if normalized_uuid(actual_uuid)!=normalized_uuid(cfg['expected_gpu_uuid']):
            raise RuntimeError('Native _acquire_device changed actual GPU identity; stop without retry')
        report['GPU_mapping']={'physical_index':cfg['physical_gpu_index'],'expected_uuid':cfg['expected_gpu_uuid'],
            'initialized_uuid_before_Exp':initialized_uuid,'actual_uuid_after_Exp':actual_uuid,
            'CUDA_VISIBLE_DEVICES_before_initialization':before_environment,
            'CUDA_VISIBLE_DEVICES_after_native_acquire':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'native_acquire_device_modified':False,'logical_device':'cuda:0'}
        report['C2_model_config_difference']={'model_configs.enc_in':[25,2],'hyper_args.enc_in':[25,2]}
        report['parameter_count']=sum(p.numel() for p in exp.model.parameters())
        exp.train()  # exact inherited _basic/optimizer/train/vali/lr/EarlyStopping; no run.py
        validations=[row for row in events if row['kind']=='normal_validation']
        # Native EarlyStopping saves ties as well; report last minimum only as a
        # derived identity, and retain all original val losses/stdout/checkpoint.
        best=min(range(len(validations)),key=lambda i:(validations[i]['loss'],-i))
        checkpoint=Path(exp.model_save_path)/'checkpoint.pth'
        report.update(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',
            optimizer_steps=exp._observed_optimizer_steps[0],final_python_step=exp.model.step,
            validation_count=len(validations),derived_best_epoch=best+1,
            best_normal_validation=validations[best]['loss'],
            checkpoint={'path':str(checkpoint),'sha256':baseline.file_sha(checkpoint),'bytes':checkpoint.stat().st_size},
            scaler_sha256=baseline.file_sha(output/'SCALER_C2.json'),
            gradient_observation='step1 and every100, not all steps',
            checkpoint_policy='native best state_dict only; no optimizer/int-step resume claim',
            diagnostic_scope='normal fit/select health only, not anomaly detection or new-method evidence')
        return report
    except BaseException:
        report.update(status='FAILED_NO_AUTOMATIC_RETRY',traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        if torch is None:torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            props=torch.cuda.get_device_properties(0)
            report['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'GPU':props.name,
                'uuid':str(getattr(props,'uuid','UNAVAILABLE')),'allocated_peak_bytes':torch.cuda.max_memory_allocated(),
                'reserved_peak_bytes':torch.cuda.max_memory_reserved(),'peak_reset':'once before model construction'}
        report['whole_operation_seconds']=time.monotonic()-start
        (output/'normal_events.json').write_text(json.dumps(events,indent=2))
        (output/'report.json').write_text(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    operation=baseline.read_json(args.config).get('operation')
    if args.execute:
        result=profile_c2(args.config) if operation=='profile' else train_c2(args.config)
    elif operation=='train':
        cfg,root,inputs=validate_train(args.config)
        result={'status':'PREVIEW_ONLY_NO_MODEL_OR_REAL_ARRAYS','operation':'train','config':cfg,
            'FIT_runs':len(inputs['roles']['FIT']),'SELECT_runs':len(inputs['roles']['SELECT']),
            'adapter_sha256':baseline.file_sha(__file__),
            'baseline_adapter_sha256':baseline.file_sha(baseline.__file__)}
    else:
        cfg,root,model,diff=validate_profile(args.config)
        result={'status':'PREVIEW_ONLY_NO_MODEL_OR_REAL_ARRAYS','config':cfg,
                'model_config_difference':diff,'channels':model['enc_in'],'H':model['seq_len'],
                'adapter_sha256':baseline.file_sha(__file__),
                'baseline_adapter_sha256':baseline.file_sha(baseline.__file__)}
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
