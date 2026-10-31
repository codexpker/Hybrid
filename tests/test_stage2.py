"""Synthetic CPU Stage2 tests. No real target data, GPU, query or labels."""
from contextlib import ExitStack,contextmanager
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hybrid_tsad.config import sha256
from hybrid_tsad.engine.readout_manifest import record
from hybrid_tsad.engine.stage2_identity import load_stage2,approve_stage2

HAS_TORCH=importlib.util.find_spec('torch') is not None
HAS_NUMPY=importlib.util.find_spec('numpy') is not None
ROOT=Path(__file__).resolve().parents[1]
OFFICIAL=os.environ.get('HYBRID_OFFICIAL_FILE')


def write(path,value):
    path.write_text(json.dumps(value),encoding='utf-8');return path


class Stage2SchemaTests(unittest.TestCase):
    def test_preview_is_separate_stage_and_no_fixture_steps_or_query_route(self):
        from hybrid_tsad.engine.stage2 import run_stage2,learning_rate
        for op in ('stage2-train','stage2-calibrate','stage2-l0'):
            result=run_stage2(ROOT/'configs'/(op+'.example.json'))
            self.assertFalse(result['executed']);self.assertEqual(result['operation_manifest']['stage'],'U_NORMAL_HEAD')
        self.assertAlmostEqual(learning_rate(1),.000003,places=18)
        self.assertEqual(learning_rate(100),.0003);self.assertEqual(learning_rate(1000),0.)
        self.assertGreater(learning_rate(101),learning_rate(999))
        with self.assertRaises(ValueError):learning_rate(0)
        with tempfile.TemporaryDirectory() as t:
            cfg=json.loads((ROOT/'configs/stage2-train.example.json').read_text())
            cfg['parameters']['steps']=5;p=write(Path(t)/'bad.json',cfg)
            with self.assertRaisesRegex(ValueError,'step override'):load_stage2(p)
            cfg['parameters'].pop('steps');cfg['operation']='stage2-query';write(p,cfg)
            with self.assertRaisesRegex(ValueError,'schema5'):load_stage2(p)


def toy_factory(official,*,mode,history,copies=2,temporal_depth=4,checkpoint=None,checkpoint_metadata=None,device='cpu'):
    import torch
    from hybrid_tsad.api import verify_checkpoint_contract
    from hybrid_tsad.model import HybridPredictor
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.history=history;self.mode=mode
            self.backbone=torch.nn.Linear(100,128);self.dropout=torch.nn.Dropout(.5);self.head=torch.nn.Linear(128,5)
            self.register_buffer('fixed_buffer',torch.tensor([3.]))
        def forward(self,x):
            z,valid,mean,scale=HybridPredictor.preprocess(self,x)
            b,h,c=z.shape;q=self.dropout(self.backbone(z.permute(0,2,1).reshape(b*c,h)))
            pred=self.head(q).reshape(b,c,5).permute(0,2,1)
            rec=None if mode=='temporal' else (z+torch.rand_like(z)*.01).unsqueeze(0).expand(2,-1,-1,-1)
            return dict(prediction_normalized=pred,prediction=pred*scale+mean,mean=mean,scale=scale,
                        tf_reconstruction=rec,tf_history_normalized=z,tf_valid=valid)
    model=Toy().to(device)
    if checkpoint is not None:
        verify_checkpoint_contract(checkpoint,checkpoint_metadata,history=history,mode=mode,copies=copies,temporal_depth=temporal_depth)
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True),strict=True)
    return model.eval()


class Fixture:
    def __init__(self,root):
        import numpy as np
        import torch
        from hybrid_tsad.engine.stage2_data import ROLES,traces
        from test_multi_readout import source_fixture_catalog
        self.root=root;self.contract=root/'contract.md';self.contract.write_text('SYNTHETIC ONLY NOT A SCIENTIFIC RELEASE')
        official=root/'official.py';official.write_text('synthetic placeholder; never imported')
        models=[]
        for seed in range(3):
            for mode in ('temporal','tf','hybrid'):
                torch.manual_seed(1000+seed)
                model=toy_factory(official,mode=mode,history=100)
                cp=root/f'parent_{mode}_{seed}.pt';torch.save(model.state_dict(),cp)
                meta=write(root/f'parent_{mode}_{seed}.json',dict(checkpoint_sha256=sha256(cp),history=100,horizon=5,
                    mode=mode,copies=2,temporal_depth=4,seed=seed,step=10000,excluded_target='TEP',source_domains=['PSM','SMD_SUBSET'],operation_manifest_sha256='a'*64))
                models.append(dict(seed=seed,mode=mode,checkpoint=record(cp),checkpoint_metadata=record(meta)))
        profile=dict(name='SYNTHETIC',faults=[1],runs=[31],points=120,channels=52,event_start=110,sample_interval_minutes=3)
        source_fixture_catalog(root/'source',profile,record(official),models)
        definition=root/'source/definition.json';refs=json.loads(definition.read_text());refs['parent_catalog_definition']=record(definition)
        self.refs=write(root/'source_refs.json',refs)
        raw=root/'TEP_FaultFree_Training.RData';raw.write_text('synthetic raw identity only; no RData decoding')
        data=np.random.default_rng(9123).normal(size=(100,500,52)).astype('float32')
        data[20:50]=np.nan  # Unauthorized rows must never be sampled/read for model calls.
        cache=root/'normal.npy';np.save(cache,data)
        lineage=write(root/'lineage.json',dict(status='PASS',raw=record(raw),cache=record(cache)))
        columns=[f'xmeas_{i}' for i in range(1,42)]+[f'xmv_{i}' for i in range(1,12)]
        parent_rows=[dict(dataset='TEP',entity=f'normal_training_run_{r}',role='fit' if r<=60 else 'select' if r<=80 else 'cal',
            selector=dict(faultNumber=0,simulationRun=r),cache_row=r-1,source=str(raw),columns=columns,
            history=100,horizon=5,local_start=0,local_end_exclusive=500,forecast_origins=list(range(100,496,5))) for r in range(1,101)]
        parent_manifest=write(root/'parent_manifest.json',dict(records=parent_rows))
        records=[]
        for role,runs in ROLES.items():
            for r in runs:
                row=dict(parent_rows[r-1]);row.update(parent_role=row['role'],role=role,stage='U_NORMAL_HEAD',array_path=str(cache),source_identity=sha256(raw));records.append(row)
        self.target=write(root/'target.json',dict(schema_version='U_NORMAL_HEAD_PREPARATION_V1',stage='U_NORMAL_HEAD',
            data_prior='KNOWN_NORMAL_TRAIN',roles=ROLES,excluded_training_run_numbers=list(range(21,51)),
            faulty_testing_access_allowed=False,original_manifests_modified=False,parent_manifest=record(parent_manifest),
            normal_lineage=record(lineage),cache=record(cache),records=records))
        plans={}
        for seed in range(3):
            p=root/f'plan{seed}.npz';np.savez_compressed(p,**traces(seed))
            source=refs['source_assets'][str(seed)]
            with np.load(source['plan']['path']) as z:trace={k:z[k] for k in ('cal_record','cal_origin','cal_channel')}
            s=root/f'source_trace{seed}.npz';np.savez_compressed(s,**trace)
            plans[str(seed)]=dict(target_plan=record(p),source_cal_trace=record(s),original_source_assets=source)
        jobs=[]
        for m in refs['models']:
            for init in ('source_pretrained','random_init'):
                seed=m['seed'];mode=m['mode']
                jobs.append(dict(job_id=f'u_normal_head_{mode}_s{seed}_{init}',stage='U_NORMAL_HEAD',mode=mode,seed=seed,initialization=init,
                    initial_checkpoint=m['checkpoint'] if init=='source_pretrained' else None,paired_reference_parent=m['checkpoint'],
                    head_reset_after_load=False,random_init_seed=seed if init=='random_init' else None,
                    target_plan=plans[str(seed)]['target_plan'],source_cal_trace=plans[str(seed)]['source_cal_trace']))
        self.prep=write(root/'preparation.json',dict(schema_version='RESEARCH_PREPARATION_V1_NOT_ENGINE_OPERATION',stage='U_NORMAL_HEAD',
            target_manifest=record(self.target),plans=plans,jobs=jobs))
        from hybrid_tsad.engine.stage2_identity import runtime_identity
        self.runtime=write(root/'runtime.json',runtime_identity())
        accepted={j['job_id']:{'recipe':sha256(ROOT/'configs/stage2_u_normal_head.recipe.json'),'target_manifest':sha256(self.target),
            'target_plan':j['target_plan']['sha256'],'source_cal_trace':j['source_cal_trace']['sha256'],
            'source_references':sha256(self.refs),'runtime':sha256(self.runtime)} for j in jobs}
        self.gate=write(root/'gate.json',dict(stage='U_NORMAL_HEAD',status='CPU_ENGINEERING_ONLY',allowed_jobs=[j['job_id'] for j in jobs],
            accepted_inputs_by_job=accepted,weights_discarded=True,query_or_labels_read=False,GPU_executed=False))
        self.requirements=write(root/'requirements.json',dict(stage='U_NORMAL_HEAD',contract=record(self.contract),
            allowed_jobs_if_separately_released=[dict(job_id=j['job_id']) for j in jobs if j['mode']=='hybrid' and j['seed']==0],
            shared_inputs=dict(target_normal_manifest=record(self.target),target_plan=plans['0']['target_plan'],source_cal_trace=plans['0']['source_cal_trace'])))
        self.pins=dict(CONTRACT_SHA=sha256(self.contract),NORMAL_RAW_SHA=sha256(raw),NORMAL_CACHE_SHA=sha256(cache),
            LINEAGE_SHA=sha256(lineage),PARENT_MANIFEST_SHA=sha256(parent_manifest),PREPARATION_SHA=sha256(self.prep),
            SOURCE_REFERENCES_SHA=sha256(self.refs),OFFICIAL_SHA=sha256(official),L0_REQUIREMENTS_SHA=sha256(self.requirements))
        self.plans=plans

    @contextmanager
    def context(self):
        with ExitStack() as s:
            for k,v in self.pins.items():s.enter_context(patch('hybrid_tsad.engine.stage2_identity.'+k,v))
            s.enter_context(patch('hybrid_tsad.api.create_predictor',side_effect=toy_factory))
            yield

    def operation(self,op,init='source_pretrained',*,trained=None,name=None):
        name=name or op+'_'+init
        approval=write(self.root/(name+'_approval.json'),{})
        inputs=dict(recipe=record(ROOT/'configs/stage2_u_normal_head.recipe.json'),preparation=record(self.prep),
            target_manifest=record(self.target),target_plan=self.plans['0']['target_plan'],source_references=record(self.refs),source_cal_trace=self.plans['0']['source_cal_trace'],runtime=record(self.runtime))
        if op=='stage2-train':inputs['entry_gate']=record(self.gate)
        elif op=='stage2-l0':inputs['requirements']=record(self.requirements)
        else:
            inputs.update(checkpoint=record(trained/'step1000.pt'),checkpoint_metadata=record(trained/'checkpoint_metadata.json'),training_summary=record(trained/'summary.json'))
        cfg=dict(engine_schema_version=5,operation=op,stage='U_NORMAL_HEAD',contract=record(self.contract),approval=record(approval),
            inputs=inputs,parameters=dict(job_id=f'u_normal_head_hybrid_s0_{init}',device='cpu'),output=str(self.root/(name+'_output')),
            score_space='history_normalized_abs_channelmean_div_source_median_v1')
        p=write(self.root/(name+'.json'),cfg);*_,digest=load_stage2(p)
        write(approval,dict(status='APPROVED',stage='U_NORMAL_HEAD',scope='SYNTHETIC_ENGINEERING_ONLY',contract_sha256=sha256(self.contract),
            operation_manifest_sha256=digest,allowed_jobs=[cfg['parameters']['job_id']],permissions=dict(train=op=='stage2-train',calibrate=op=='stage2-calibrate',l0=op=='stage2-l0',query=False,labels=False,F=False)))
        cfg['approval']=record(approval);write(p,cfg);return p


@unittest.skipUnless(HAS_TORCH and HAS_NUMPY,'existing torch/numpy required; synthetic CPU only')
class Stage2CPUTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2);self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.f=Fixture(Path(self.tmp.name))

    def test_full1000_train_both_initializations_then_source_target_cal_one_object(self):
        import numpy as np
        import torch
        from hybrid_tsad.engine.stage2 import run_stage2,calibration_tracks
        from hybrid_tsad.engine.identity import source_calibration
        with self.f.context():
            for init in ('source_pretrained','random_init'):
                path=self.f.operation('stage2-train',init);report=run_stage2(path,execute=True)
                self.assertEqual(report['steps'],1000);self.assertEqual(report['trainable_count'],645)
                self.assertTrue(all(v>0 for v in report['head_changes'].values()))
                self.assertTrue(report['head_mode_after_select']['head']);self.assertFalse(report['head_mode_after_select']['dropout'])
                out=Path(json.loads(path.read_text())['output'])
                curve=json.loads((out/'curve.json').read_text());self.assertEqual(len(curve),1000)
                self.assertEqual(curve[99]['lr'],.0003);self.assertEqual(curve[-1]['lr'],0.)
                self.assertTrue(all(r['TF_head_gradient_path']=='NO_GRAD_CONNECTION' for r in curve))
                self.assertEqual(report['identity']['frozen_before'],report['identity']['frozen_after']);self.assertNotIn('excluded_target',report['identity'])
                calpath=self.f.operation('stage2-calibrate',init,trained=out);cal=run_stage2(calpath,execute=True)
                calout=Path(json.loads(calpath.read_text())['output']);self.assertFalse(cal['query_access'])
                self.assertEqual(cal['threshold_tracks']['source']['score_object'],cal['threshold_tracks']['target']['score_object'])
                self.assertIs(cal['threshold_tracks']['source']['score_object'],cal['threshold_tracks']['target']['score_object'])
                self.assertTrue(cal['module_modes_after_cal']['head']);self.assertFalse(cal['module_modes_after_cal']['dropout'])
                with np.load(calout/'target_canonical_scores.npz') as z:
                    self.assertEqual(z['canonical_score'].dtype,np.float32);self.assertEqual(z['canonical_score'].shape,(256,5))
                    self.assertTrue(np.array_equal(z['origin'][:2],[465,100]));self.assertTrue(np.array_equal(z['run'][:2],[95,95]))
                with self.assertRaises(ValueError):source_calibration(calout/'target_threshold.json')
                from hybrid_tsad.engine.readout_controls import prepare_catalog
                definition=json.loads((self.f.root/'source/definition.json').read_text())
                selected=next(r for r in definition['models'] if r['mode']=='hybrid' and r['seed']==0)
                selected.update(checkpoint=record(out/'step1000.pt'),checkpoint_metadata=record(out/'checkpoint_metadata.json'))
                bad_z=write(self.f.root/('bad_Z_'+init+'.json'),definition)
                with self.assertRaisesRegex(ValueError,'not TEP-excluded100'):
                    prepare_catalog(bad_z,self.f.root/('bad_Z_output_'+init),dict(name='SYNTHETIC'))
                target=json.loads((calout/'target_threshold.json').read_text());target['source_error_scale']*=2;write(calout/'wrong_target.json',target)
                with self.assertRaisesRegex(ValueError,'replace the fitted source scale'):
                    calibration_tracks(calout/'source_calibration.json',calout/'wrong_target.json',report['identity']['checkpoint_sha256'])
                target=json.loads((calout/'target_threshold.json').read_text());target['threshold99']+=1;write(calout/'wrong_threshold.json',target)
                with self.assertRaisesRegex(ValueError,'single canonical'):
                    calibration_tracks(calout/'source_calibration.json',calout/'wrong_threshold.json',report['identity']['checkpoint_sha256'])
                target['normal_error_scale']=2;write(calout/'extra_target_scale.json',target)
                with self.assertRaisesRegex(ValueError,'target-rescale'):
                    calibration_tracks(calout/'source_calibration.json',calout/'extra_target_scale.json',report['identity']['checkpoint_sha256'])
                # Original audit counterexample: leave physical bank/source units intact,
                # multiply only canonical scores and update NPZ SHA plus matching q99.
                target=json.loads((calout/'target_threshold.json').read_text())
                self.assertEqual(target['schema_version'],2)
                self.assertEqual(target['target_cal_bank'],record(calout/'target_cal_bank.npz'))
                with np.load(calout/'target_canonical_scores.npz') as z:tampered={k:z[k].copy() for k in z.files}
                tampered['canonical_score']*=np.float32(2)
                doubled=calout/'doubled_scores.npz';np.savez_compressed(doubled,**tampered)
                target['canonical_scores']=record(doubled);target['threshold99']=float(np.quantile(tampered['canonical_score'],.99,method='linear'))
                write(calout/'rescaled_false_flag.json',target)
                with self.assertRaisesRegex(ValueError,'bound physical calibration bank'):
                    calibration_tracks(calout/'source_calibration.json',calout/'rescaled_false_flag.json',report['identity']['checkpoint_sha256'])
                # Even consistent reordering of physical bank and score object is not
                # the approved nonmonotone target-cal plan.
                with np.load(calout/'target_cal_bank.npz') as z:reordered={k:z[k][::-1].copy() for k in z.files}
                changed_bank=calout/'reordered_bank.npz';np.savez_compressed(changed_bank,**reordered)
                target=json.loads((calout/'target_threshold.json').read_text());target['target_cal_bank']=record(changed_bank)
                write(calout/'wrong_bank_order.json',target)
                with self.assertRaisesRegex(ValueError,'frozen trace order'):
                    calibration_tracks(calout/'source_calibration.json',calout/'wrong_bank_order.json',report['identity']['checkpoint_sha256'])
                target=json.loads((calout/'target_threshold.json').read_text());target['target_plan_sha256']='0'*64
                write(calout/'wrong_trace_identity.json',target)
                with self.assertRaisesRegex(ValueError,'manifest/trace identity'):
                    calibration_tracks(calout/'source_calibration.json',calout/'wrong_trace_identity.json',report['identity']['checkpoint_sha256'])
                target=json.loads((calout/'target_threshold.json').read_text())
                wrong_fitted=json.loads((out/'checkpoint_metadata.json').read_text());wrong_fitted['seed']=1
                wrong_fitted_path=write(calout/'wrong_fitted_metadata.json',wrong_fitted)
                target['fitted_checkpoint_metadata']=record(wrong_fitted_path);write(calout/'wrong_fitted_trace.json',target)
                with self.assertRaisesRegex(ValueError,'fitted checkpoint/trace'):
                    calibration_tracks(calout/'source_calibration.json',calout/'wrong_fitted_trace.json',report['identity']['checkpoint_sha256'])
                correct=calibration_tracks(calout/'source_calibration.json',calout/'target_threshold.json',report['identity']['checkpoint_sha256'])
                self.assertEqual(correct['target']['score_object'],record(calout/'target_canonical_scores.npz'))
                with self.assertRaisesRegex(ValueError,'fitted checkpoint'):
                    calibration_tracks(calout/'source_calibration.json',calout/'target_threshold.json','0'*64)
                wrong=self.f.operation('stage2-calibrate',init,trained=out,name='wrong_checkpoint_'+init)
                v=json.loads(wrong.read_text());v['inputs']['checkpoint']=report['identity']['paired_reference_parent'];write(wrong,v)
                _,_,_,_,digest=load_stage2(wrong);ap=Path(v['approval']['path']);aa=json.loads(ap.read_text());aa['operation_manifest_sha256']=digest;write(ap,aa);v['approval']=record(ap);write(wrong,v)
                with self.assertRaisesRegex(ValueError,'training identity mismatch'):run_stage2(wrong,execute=True)

    def test_l0_fixed_pair_five_updates_discard_routes_no_checkpoints_and_no_authority(self):
        import torch
        from hybrid_tsad.engine.stage2 import run_stage2
        with self.f.context():
            for init in ('source_pretrained','random_init'):
                p=self.f.operation('stage2-l0',init);cfg,paths,*_=load_stage2(p)
                with patch.object(torch,'save',side_effect=AssertionError('L0 wrote reusable model/optimizer')):
                    r=run_stage2(p,execute=True)
                out=Path(cfg['output']);self.assertEqual(r['steps'],5);self.assertTrue(r['weights_discarded'])
                self.assertFalse(list(out.rglob('*.pt')));self.assertFalse((out/'source_calibration.json').exists())
                self.assertEqual(r['target_cal_order'],[dict(run=95,origin=465),dict(run=95,origin=100)])
                self.assertEqual(r['source_first64_domains'],['PSM']);self.assertFalse(r['source_prefix_is_full_domain_coverage'])
                self.assertEqual(r['rng_after_updates'],r['rng_after_evaluations']);self.assertEqual(r['frozen_before'],r['frozen_after'])
                self.assertTrue(r['diagnostics']['total_head_grad_equals_future_only']);self.assertFalse(r['GPU_executed'])
                self.assertFalse(r['calibration_reusable']);self.assertTrue(r['module_flags_preserved'])
                # A raw diagnostic report cannot promote itself to formal training.
                train=self.f.operation('stage2-train',init,name='bad_gate_'+init);v=json.loads(train.read_text())
                v['inputs']['entry_gate']=record(out/'summary.json');write(train,v)
                _,_,_,_,digest=load_stage2(train);approval=Path(v['approval']['path']);a=json.loads(approval.read_text())
                a['operation_manifest_sha256']=digest;write(approval,a);v['approval']=record(approval);write(train,v)
                with self.assertRaisesRegex(ValueError,'separately accepted'):
                    run_stage2(train,execute=True)

    def test_runtime_l0_scope_and_cpu_fixture_cannot_be_approved_as_gpu(self):
        from hybrid_tsad.engine.stage2 import run_stage2
        from hybrid_tsad.engine.stage2_identity import validate_assets
        with self.f.context():
            p=self.f.operation('stage2-l0');cfg,paths,out,_,_=load_stage2(p)
            cfg['parameters']['device']='cuda:0';write(p,cfg);_,_,_,_,digest=load_stage2(p)
            approval=Path(cfg['approval']['path']);a=json.loads(approval.read_text());a['operation_manifest_sha256']=digest;write(approval,a)
            cfg['approval']=record(approval);write(p,cfg)
            with self.assertRaisesRegex(ValueError,'CPU-only'):run_stage2(p,execute=True)
            p=self.f.operation('stage2-train',name='runtime_bad');cfg,paths,*_=load_stage2(p)
            runtime=json.loads(self.f.runtime.read_text());runtime['files']['src/hybrid_tsad/model.py']='0'*64
            bad=write(self.f.root/'wrong_runtime.json',runtime);cfg['inputs']['runtime']=record(bad);paths['runtime']=bad
            with self.assertRaisesRegex(ValueError,'runtime bytes'):validate_assets(cfg,paths)

    def test_wrong_parent_stage_is_rejected_even_after_updating_asset_records(self):
        from hybrid_tsad.engine.stage2_identity import validate_assets
        with self.f.context():
            p=self.f.operation('stage2-train');cfg,paths,*_=load_stage2(p)
            refs=json.loads(self.f.refs.read_text());parent=next(r for r in refs['models'] if r['seed']==0 and r['mode']=='hybrid')
            meta_path=Path(parent['checkpoint_metadata']['path']);m=json.loads(meta_path.read_text());m['stage']='U_NORMAL_HEAD';write(meta_path,m)
            parent['checkpoint_metadata']=record(meta_path)
            definition=Path(refs['parent_catalog_definition']['path']);d=json.loads(definition.read_text());d['models']=refs['models'];write(definition,d)
            refs['parent_catalog_definition']=record(definition);write(self.f.refs,refs);cfg['inputs']['source_references']=record(self.f.refs)
            with patch('hybrid_tsad.engine.stage2_identity.SOURCE_REFERENCES_SHA',sha256(self.f.refs)),self.assertRaisesRegex(ValueError,'Wrong parent'):
                validate_assets(cfg,paths)

    def test_formal_loop_detects_buffer_drift_without_publishing_primary(self):
        from hybrid_tsad.engine.stage2 import run_stage2
        with self.f.context():
            def corrupt_factory(*args,**kwargs):
                model=toy_factory(*args,**kwargs);forward=model.forward
                def changed(x):
                    model.fixed_buffer.add_(1)
                    return forward(x)
                model.forward=changed;return model
            p=self.f.operation('stage2-train',name='buffer_corruption')
            with patch('hybrid_tsad.api.create_predictor',side_effect=corrupt_factory),self.assertRaisesRegex(ValueError,'Frozen backbone/buffer'):
                run_stage2(p,execute=True)
            out=Path(json.loads(p.read_text())['output'])
            self.assertTrue((out/'failure.json').exists());self.assertFalse((out/'summary.json').exists());self.assertFalse((out/'step1000.pt').exists())

    def test_warmstart_head_retained_random_never_loads_parent_and_eval_rng_modes_restore(self):
        import numpy as np
        import torch
        from hybrid_tsad.engine.stage2_identity import validate_assets
        from hybrid_tsad.engine.stage2 import create_stage2_model,configure_head_only,head_only_step,evaluate_c1,evaluate_target_cal,module_modes,frozen_hashes
        from hybrid_tsad.engine.stage2_data import TargetNormal
        with self.f.context():
            p=self.f.operation('stage2-train');cfg,paths,*_=load_stage2(p);a=validate_assets(cfg,paths)
            model=create_stage2_model(a,'cpu');state=torch.load(a['parent']['checkpoint']['path'],weights_only=True)
            self.assertTrue(torch.equal(model.head.weight,state['head.weight']))
            target=TargetNormal(a['target'],a['plan'],a['target']['cache']['path'])
            old=frozen_hashes(model);flags=module_modes(model);rng=torch.get_rng_state().clone()
            _,first=evaluate_c1(model,lambda i:target.batch('select',i,device='cpu'),device='cpu',count=64)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(module_modes(model),flags)
            _,second=evaluate_c1(model,lambda i:target.batch('select',i,device='cpu'),device='cpu',count=64)
            self.assertTrue(np.array_equal(first['prediction'],second['prediction']))
            evaluate_target_cal(model,target,1.,device='cpu',count=2)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(module_modes(model),flags)
            x,y=target.batch('fit',0,device='cpu');optimizer=torch.optim.AdamW(configure_head_only(model),lr=.0003,weight_decay=.01)
            result=head_only_step(model,optimizer,x,y,1);self.assertGreater(result['head_gradient_max']['head.weight'],0)
            self.assertEqual(frozen_hashes(model),old)
            p=self.f.operation('stage2-train','random_init');cfg,paths,*_=load_stage2(p);a=validate_assets(cfg,paths)
            with patch.object(torch,'load',side_effect=AssertionError('random path loaded weights')):
                random=create_stage2_model(a,'cpu')
            self.assertFalse(torch.equal(random.head.weight,state['head.weight']))

    def test_roles_trace_parent_permission_and_future_only_target_rejections(self):
        import numpy as np
        import torch
        from hybrid_tsad.engine.stage2_identity import validate_assets
        from hybrid_tsad.engine.stage2 import run_stage2,create_stage2_model,evaluation_scope
        from hybrid_tsad.engine.stage2_data import TargetNormal
        from hybrid_tsad.model import objective
        with self.f.context():
            p=self.f.operation('stage2-train');cfg,paths,out,_,digest=load_stage2(p);a=validate_assets(cfg,paths)
            changed=json.loads(p.read_text());changed['parameters']['device']='cuda:0';write(p,changed)
            parsed,pp,oo,_,dd=load_stage2(p)
            with self.assertRaisesRegex(ValueError,'exact operation'):approve_stage2(parsed,pp,oo,dd)
            t=TargetNormal(a['target'],a['plan'],a['target']['cache']['path']);original_run=int(t.plan['fit_run'][0]);t.plan['fit_run'][0]=21
            with self.assertRaisesRegex(ValueError,'role/run'):t.batch('fit',0,device='cpu')
            t.plan['fit_run'][0]=original_run
            with np.load(paths['target_plan']) as z:bad={k:z[k].copy() for k in z.files}
            bad['select_run'][0]=1;badpath=self.f.root/'bad_trace.npz';np.savez(badpath,**bad)
            from hybrid_tsad.engine.stage2_data import validate_target
            with self.assertRaisesRegex(ValueError,'fixed PCG64'):validate_target(paths['target_manifest'],badpath,0,a['assets'])
            m=json.loads(self.f.target.read_text());m['records'][0]['role']='cal';badmanifest=write(self.f.root/'bad_target.json',m)
            with self.assertRaisesRegex(ValueError,'records differ'):validate_target(badmanifest,paths['target_plan'],0,a['assets'])
            model=create_stage2_model(a,'cpu');x,y=t.batch('select',0,device='cpu')
            with evaluation_scope(model,'cpu'):o1=model(x)
            with evaluation_scope(model,'cpu'):o2=model(x)
            self.assertTrue(torch.equal(o1['prediction_normalized'],o2['prediction_normalized']))
            l1=objective(o1,y)[0];l2=objective(o2,y+10)[0];self.assertNotEqual(float(l1),float(l2))
            parent=Path(a['parent']['checkpoint_metadata']['path']);meta=json.loads(parent.read_text());meta['stage']='U_NORMAL_HEAD';write(parent,meta)
            # Even updating a ref must not turn an adapted child into an original parent.
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):validate_assets(cfg,paths)


@unittest.skipUnless(HAS_TORCH and OFFICIAL,'existing torch plus external pinned official source; no CUDA')
class ActualCoreStage2Tests(unittest.TestCase):
    def test_three_real_modes_existing_head_gradient_buffers_and_no_future_input(self):
        import torch
        from hybrid_tsad.model import HybridPredictor
        from hybrid_tsad.engine.stage2 import (configure_head_only,head_only_step,frozen_hashes,evaluation_scope,module_modes,
            diagnostic_replay,evaluate_c1,evaluate_target_cal,source_scale_only)
        torch.set_num_threads(2)
        for mode in ('temporal','tf','hybrid'):
            torch.manual_seed(71);model=HybridPredictor(OFFICIAL,history=100,mode=mode);head=configure_head_only(model)
            self.assertEqual(sum(p.numel() for p in head),645);before=frozen_hashes(model);flags=module_modes(model)
            x=torch.randn(64,100,1);y=torch.randn(64,5,1)
            with evaluation_scope(model,'cpu'):a=model(x)
            with evaluation_scope(model,'cpu'):b=model(x)
            self.assertTrue(torch.equal(a['prediction_normalized'],b['prediction_normalized']))
            opt=torch.optim.AdamW(head,lr=.0003,weight_decay=.01)
            for step in range(1,6):result=head_only_step(model,opt,x,y,step)
            self.assertIn(result['TF_head_gradient_max'],(None,0.));self.assertGreater(result['head_gradient_max']['head.weight'],0)
            self.assertTrue(diagnostic_replay(model,x,y,'cpu')['rng_restored'])
            _,source=evaluate_c1(model,lambda _: (x,y),device='cpu',count=64)
            xx=torch.randn(1,100,52);yy=torch.randn(5,52).numpy()
            class Pool:
                def cal_block(self,index,*,device):return xx,yy,95,[465,100][index]
            target=evaluate_target_cal(model,Pool(),source_scale_only(source),device='cpu',count=2)
            self.assertEqual(target['origin'].tolist(),[465,100])
            self.assertEqual(before,frozen_hashes(model));self.assertEqual(flags,module_modes(model))
