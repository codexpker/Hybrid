"""Schema6 synthetic CPU fixtures, no real target/query/labels/production release."""
from contextlib import ExitStack,contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hybrid_tsad.config import sha256
from hybrid_tsad.engine.readout_manifest import record
from hybrid_tsad.engine.stage2_readout_identity import JOBS,STAGE,canonical_sha,u_key,runtime_identity

HAS_TORCH=importlib.util.find_spec('torch') is not None

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8');return path

def restage(root,name,shard):
    folder=root/name;put(folder/'summary.json',shard);return put(folder/'shard.json',shard)

class Fixture:
    def __init__(self,root):
        import numpy as np
        import torch
        from test_stage2 import Fixture as SourceFixture,toy_factory
        from hybrid_tsad.engine.stage2 import frozen_hashes
        from hybrid_tsad.engine.identity import source_calibration
        from hybrid_tsad.engine.bank import collect_forecasts,write_bank,inference_metadata
        self.root=root;self.f=SourceFixture(root);self.profile=dict(name='SYNTHETIC',faults=[1,2],runs=[31],points=120,channels=52,event_start=110,sample_interval_minutes=3)
        refs=json.loads(self.f.refs.read_text());rows=[]
        for fault in (1,2):
            p=root/f'query_synthetic_{fault}.npy';np.save(p,np.random.default_rng(fault).normal(size=(120,52)).astype('float32'))
            rows.append(dict(entity=f'fault{fault:02}_run031',fault_id=fault,run_id=31,role='query',history=100,horizon=5,local_start=0,local_end_exclusive=120,
                guard=0,channels=52,origins=[100,105,110,115],array_path=str(p),array_sha256_from_frozen_parent=sha256(p)))
        self.query=put(root/'query.json',dict(dataset='SYNTHETIC',records=rows))
        zc=json.loads((root/'source/prepared/catalog.json').read_text());zc['profile']=self.profile
        self.zcatalog=put(root/'z_catalog.json',zc)
        original={(x['seed'],x['mode']):x for x in refs['models']};native={};scores={}
        for (seed,mode),spec in original.items():
            model=toy_factory(root/'official.py',mode=mode,history=100,checkpoint=spec['checkpoint']['path'],checkpoint_metadata=spec['checkpoint_metadata']['path'])
            scale=source_calibration(spec['calibration']['path'])['normal_error_scale']
            for q in rows:
                arrays,rng,_=collect_forecasts(model,np.load(q['array_path']),q['origins'],scale,mask_seed=84551)
                key=f's{seed}/{mode}/'+q['entity'];folder=root/'old_Z'/key
                meta=inference_metadata(key,dict(history=100,seed=seed,mode=mode,entity=q['entity']),points=20,blocks=4,mask_seed=84551,
                    checkpoint_sha256=spec['checkpoint']['sha256'],query_array_sha256=q['array_sha256_from_frozen_parent'])
                write_bank(folder,arrays,rng,meta);native[key]={k:record(folder/name) for k,name in [('bank','predictions_and_scores.npz'),('rng','rng_states.npz'),('metadata','inference.json')]}
                scores[key]=dict(file=record(folder/'predictions_and_scores.npz'))
        for seed in range(3):
            for mode in ('late','history_mean','persistence'):
                for q in rows:scores[f's{seed}/{mode}/'+q['entity']]=dict(file=native['s0/temporal/'+q['entity']]['bank'])
        self.zseal=put(root/'old_Z_seal.json',dict(status='ALL_FIXED_SCORES_SEALED_LABELS_LOCKED',labels_opened=False,complete_group=True,
            profile=self.profile,catalog=record(self.zcatalog),query_manifest=record(self.query),native_banks=native,scores=scores))
        self.bridge=put(root/'bridge.json',dict(status='PRIMARY_ACCEPTED_Z_PHYSICAL_REFERENCE_BRIDGE',profile=self.profile,z_seal=record(self.zseal),
            query_manifest=record(self.query),entities={q['entity']:native['s0/temporal/'+q['entity']] for q in rows}))
        trainroot=root/'old_training_runtime';p=trainroot/'src/hybrid_tsad/engine/stage2.py';p.parent.mkdir(parents=True);p.write_text('# synthetic metadata runtime only')
        truntime=put(root/'old_training_runtime.json',dict(stage='U_NORMAL_HEAD',files={'src/hybrid_tsad/engine/stage2.py':sha256(p)}))
        models=[]
        for job,(seed,mode,init) in JOBS.items():
            parent=original[seed,mode];folder=root/'U'/job;train=folder/'train';cal=folder/'cal';train.mkdir(parents=True);cal.mkdir()
            torch.manual_seed(seed if init=='random_init' else 42)
            model=toy_factory(root/'official.py',mode=mode,history=100,checkpoint=parent['checkpoint']['path'] if init=='source_pretrained' else None,
                checkpoint_metadata=parent['checkpoint_metadata']['path'] if init=='source_pretrained' else None)
            cp=train/'step1000.pt';torch.save(model.state_dict(),cp)
            header=dict(stage='U_NORMAL_HEAD',seed=seed,mode=mode,initialization=init,step=1000)
            m=dict(**header,checkpoint_sha256=sha256(cp),history=100,horizon=5,copies=2,temporal_depth=4,
                parent_checkpoint_sha256=parent['checkpoint']['sha256'],parent_weights_loaded=init=='source_pretrained',
                initial_checkpoint_sha256=parent['checkpoint']['sha256'] if init=='source_pretrained' else None,
                pretraining_domains=['PSM','SMD_SUBSET'] if init=='source_pretrained' else [],pretraining_excluded_target='TEP' if init=='source_pretrained' else None,
                adaptation_target='TEP',target_labels_used=False,query_access=False,trainable_count=645,trainable_parameters=['head.weight','head.bias'],
                target_manifest_sha256=sha256(self.f.target),target_plan_sha256=self.f.plans[str(seed)]['target_plan']['sha256'],
                source_cal_trace_sha256=self.f.plans[str(seed)]['source_cal_trace']['sha256'],source_references_sha256=sha256(self.f.refs),runtime_sha256=sha256(truntime),
                recipe_sha256=sha256(Path(__file__).resolve().parents[1]/'configs/stage2_u_normal_head.recipe.json'),
                frozen_before=frozen_hashes(model),frozen_after=frozen_hashes(model))
            mp=put(train/'checkpoint_metadata.json',m);summary=put(train/'summary.json',dict(status='U_NORMAL_HEAD_STEP1000_COMPLETE',job_id=job,primary=record(cp),steps=1000,identity=m))
            with np.load(parent['cal_bank']['path']) as z:bank={k:z[k].copy() for k in z.files}
            from hybrid_tsad.engine.source import calibrate_source
            arr,params=calibrate_source(bank);np.savez_compressed(cal/'source_cal_bank.npz',**bank);np.savez_compressed(cal/'source_canonical.npz',**arr)
            source=put(cal/'source_calibration.json',dict(stage='U_NORMAL_HEAD',role='source_cal_after_adaptation',fitted_checkpoint_sha256=sha256(cp),runtime_sha256=sha256(truntime),operation_manifest_sha256='b'*64,
                source_cal_trace_sha256=m['source_cal_trace_sha256'],source_references_sha256=sha256(self.f.refs),source_bank=record(cal/'source_cal_bank.npz'),canonical_scores=record(cal/'source_canonical.npz'),**params))
            with np.load(self.f.plans[str(seed)]['target_plan']['path']) as z:rr=z['cal_run'].copy();oo=z['cal_origin'].copy()
            target=np.ones((256,5,52),np.float32);scales=np.ones((256,1,52),np.float32);prediction=target+np.float32(.125)
            from hybrid_tsad.protocol import score_block
            canonical=np.stack([score_block(p,y,s,params['normal_error_scale']) for p,y,s in zip(prediction,target,scales)])
            np.savez_compressed(cal/'target_cal_bank.npz',prediction=prediction,target=target,history_scale=scales,run=rr,origin=oo)
            np.savez_compressed(cal/'target_canonical.npz',canonical_score=canonical,run=rr,origin=oo)
            threshold=put(cal/'target_threshold.json',dict(schema_version=2,stage='U_NORMAL_HEAD',role='target_threshold_only',fitted_checkpoint_sha256=sha256(cp),runtime_sha256=sha256(truntime),operation_manifest_sha256='b'*64,
                source_calibration=record(source),source_error_scale=params['normal_error_scale'],target_manifest_sha256=sha256(self.f.target),target_plan_sha256=m['target_plan_sha256'],
                canonical_scores=record(cal/'target_canonical.npz'),threshold99=float(np.quantile(canonical,.99,method='linear')),target_global_rescale=False,pool_is_contiguous_event=False,
                target_manifest=record(self.f.target),target_plan=self.f.plans[str(seed)]['target_plan'],seed=seed,target_cal_bank=record(cal/'target_cal_bank.npz'),fitted_checkpoint_metadata=record(mp)))
            for directory,op in [(train,'stage2-train'),(cal,'stage2-calibrate')]:put(directory/'archive_manifest.json',dict(stage='U_NORMAL_HEAD',status='STAGE2_ARCHIVE_COMPLETE',operation=op,files={x.name:sha256(x) for x in directory.iterdir()}))
            accepted_train=put(folder/'primary_train.json',dict(status='PRIMARY_ACCEPTED_FORMAL_TRAIN_PAIR',stage='U_NORMAL_HEAD',allowed_jobs=[job],summaries={job:record(summary)}))
            cal_review=put(folder/'cal_review.json',dict(status='PASS_LIMITED_WAVE2_CAL',fixture_only=True))
            accepted=put(folder/'primary_wave.json',dict(status='PRIMARY_ACCEPTED_NAMED_WAVE',stage='U_NORMAL_HEAD',wave=2,accepted_train=record(accepted_train),independent_cal_review=record(cal_review),fixture_only=True))
            models.append(dict(job_id=job,**header,checkpoint=record(cp),checkpoint_metadata=record(mp),training_summary=record(summary),train_archive=record(train/'archive_manifest.json'),
                cal_archive=record(cal/'archive_manifest.json'),source_calibration=record(source),target_threshold=record(threshold),primary_acceptance=record(accepted)))
        acceptance=put(root/'all18.json',dict(status='PRIMARY_ACCEPTED_ALL18_U_TRAIN_CAL',stage='U_NORMAL_HEAD',models={x['job_id']:canonical_sha(x) for x in models},fixture_only=True))
        d=dict(schema_version=1,stage=STAGE,profile=self.profile,official=record(root/'official.py'),query_manifest=record(self.query),source_references=record(self.f.refs),z_catalog=record(self.zcatalog),z_seal=record(self.zseal),
            normal_manifest=record(self.f.target),normal_plans={s:v['target_plan'] for s,v in self.f.plans.items()},source_traces={s:v['source_cal_trace'] for s,v in self.f.plans.items()},
            training_runtime=record(truntime),training_runtime_root=str(trainroot),u_models=models,all18_acceptance=record(acceptance),physical_bridge=record(self.bridge))
        self.definition=put(root/'definition6.json',d);self.models=models
        self.runtime=put(root/'readout_runtime.json',runtime_identity())

    @contextmanager
    def context(self):
        with ExitStack() as s:
            s.enter_context(self.f.context());s.enter_context(patch('hybrid_tsad.engine.stage2_readout_identity.CONTRACT_SHA',sha256(self.f.contract)))
            s.enter_context(patch('hybrid_tsad.engine.stage2_readout_identity.OFFICIAL_SHA',sha256(self.root/'official.py')))
            yield

    def operation(self,op,inputs,*,scope='full',name=None):
        from hybrid_tsad.engine.stage2_readout_identity import load_operation
        name=name or op;ap=put(self.root/(name+'_approval.json'),{})
        params=dict(profile=self.profile,scope=scope,runtime=record(self.runtime))
        if op in ('infer','z-normal-cal'):params['device']='cpu'
        c=dict(engine_schema_version=6,stage=STAGE,operation=op,contract=record(self.f.contract),approval=record(ap),inputs={k:record(v) for k,v in inputs.items()},parameters=params,
            output=str(self.root/(name+'_out')),score_space='history_normalized_abs_channelmean_div_source_median_v1')
        p=put(self.root/(name+'.json'),c);*_,digest=load_operation(p)
        put(ap,dict(status='APPROVED',stage=STAGE,scope='SYNTHETIC_CPU_ENGINEERING_ONLY',contract_sha256=sha256(self.f.contract),operation_manifest_sha256=digest,
            permissions=dict(operation=op,training=False,source_fit=False,query_infer=op=='infer',target_normal_cal=op=='z-normal-cal',labels=False,F=False)))
        c['approval']=record(ap);put(p,c);return p


@unittest.skipUnless(HAS_TORCH,'existing torch required; synthetic CPU only')
class ReadoutTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2);self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.f=Fixture(Path(self.tmp.name))

    def test_complete_config_catalog_zcal_infer_seal_gate_one_object_and_original_rng(self):
        import numpy as np
        from hybrid_tsad.engine.stage2_readout import run_readout,load_catalog,infer
        from hybrid_tsad.engine.bank import collect_forecasts
        from test_stage2 import toy_factory
        with self.f.context():
            p=self.f.operation('catalog',dict(definition=self.f.definition));run_readout(p,execute=True);catalog=self.f.root/'catalog_out/catalog.json'
            z=self.f.operation('z-normal-cal',dict(catalog=catalog));zr=run_readout(z,execute=True);zpath=self.f.root/'z-normal-cal_out/z_thresholds.json'
            self.assertEqual(len(zr['thresholds']),9);self.assertEqual(zr['query_forwards'],0)
            first=next(iter(zr['thresholds'].values()))
            identity=json.loads(Path(first['path']).read_text());self.assertEqual(identity['role'],'Z_TARGET_NORMAL_THRESHOLD_ONLY')
            with np.load(identity['bank']['path']) as bank:self.assertEqual(bank['origin'][:2].tolist(),[465,100])
            a=load_catalog(catalog,self.f.profile)
            cells=[dict(job_id=j,entity=e,origins=q['origins']) for j in JOBS for e,q in a['entities'].items()]
            wl=put(self.f.root/'whitelist.json',cells);p=self.f.operation('infer',dict(catalog=catalog,whitelist=wl),scope='shard');r=run_readout(p,execute=True)
            self.assertEqual(len(r['banks']),36);self.assertEqual(len(set(r['banks'])),36)
            # Differential against direct original collector under same state and device RNG.
            job=cells[0]['job_id'];entity=cells[0]['entity'];spec=a['models'][job];q=a['entities'][entity]
            model=toy_factory(a['official'],mode=JOBS[job][1],history=100,checkpoint=spec['paths']['checkpoint'],checkpoint_metadata=spec['paths']['checkpoint_metadata'])
            expected,rng,_=collect_forecasts(model,np.load(q['array_path']),q['origins'],spec['source']['normal_error_scale'],mask_seed=84551)
            actual=r['banks'][u_key(job,entity)]
            with np.load(actual['bank']['path']) as bank:
                for k,v in expected.items():self.assertTrue(np.array_equal(bank[k],v),k)
            with np.load(actual['rng']['path']) as saved:self.assertTrue(np.array_equal(saved['before_each_block'],rng))
            shards=put(self.f.root/'shards.json',[record(self.f.root/'infer_out/shard.json')]);ex=put(self.f.root/'executions.json',dict(scope='SYNTHETIC_CPU_ENGINEERING_ONLY'))
            p=self.f.operation('seal',dict(catalog=catalog,shards=shards,z_thresholds=zpath,executions=ex));sealed=run_readout(p,execute=True)
            self.assertEqual(sealed['counts'],dict(u_banks=36,z_native_banks=18,z_fixed_scores=36));self.assertFalse(sealed['labels_opened'])
            for value in sealed['u_banks'].values():self.assertEqual(value['tracks']['source']['score_object'],value['tracks']['target']['score_object'])
            p=self.f.operation('gate',dict(catalog=catalog,seal=self.f.root/'seal_out/seal.json'));self.assertFalse(run_readout(p,execute=True)['labels_permission'])
            # Missing full bank refuses seal even when shard hash is updated.
            shard=json.loads((self.f.root/'infer_out/shard.json').read_text());shard['banks'].pop(next(iter(shard['banks'])));bad=restage(self.f.root,'missing_bank_archive',shard)
            badlist=put(self.f.root/'missinglist.json',[record(bad)]);p=self.f.operation('seal',dict(catalog=catalog,shards=badlist,z_thresholds=zpath,executions=ex),name='missing')
            with self.assertRaisesRegex(ValueError,'Whitelist/output'):run_readout(p,execute=True)
            good=json.loads((self.f.root/'infer_out/shard.json').read_text());removed=good['whitelist'].pop(0)
            good['banks'].pop(u_key(removed['job_id'],removed['entity']))
            sp=restage(self.f.root,'partial_bank_archive',good);lst=put(self.f.root/'partial_shards.json',[record(sp)])
            p=self.f.operation('seal',dict(catalog=catalog,shards=lst,z_thresholds=zpath,executions=ex),name='partial_group')
            with self.assertRaisesRegex(ValueError,'INCOMPLETE'):run_readout(p,execute=True)
            duplicate=put(self.f.root/'duplicate_shards.json',[record(self.f.root/'infer_out/shard.json')]*2)
            p=self.f.operation('seal',dict(catalog=catalog,shards=duplicate,z_thresholds=zpath,executions=ex),name='duplicate_group')
            with self.assertRaisesRegex(ValueError,'Duplicate'):run_readout(p,execute=True)
            # Same physical target/index, altered canonical only, with fresh bank/meta/shard SHA.
            shard=json.loads((self.f.root/'infer_out/shard.json').read_text());key=next(iter(shard['banks']));v=shard['banks'][key]
            with np.load(v['bank']['path']) as z:values={k:z[k].copy() for k in z.files}
            values['standardized_score']*=np.float32(2);bp=self.f.root/'bad_unit_bank.npz';np.savez_compressed(bp,**values)
            meta=json.loads(Path(v['metadata']['path']).read_text());meta['bank_sha256']=sha256(bp);mp=put(self.f.root/'bad_unit_meta.json',meta)
            v.update(bank=record(bp),metadata=record(mp));sp=restage(self.f.root,'bad_unit_archive',shard);badlist=put(self.f.root/'bad_unit_shards.json',[record(sp)])
            p=self.f.operation('seal',dict(catalog=catalog,shards=badlist,z_thresholds=zpath,executions=ex),name='bad_unit')
            with self.assertRaisesRegex(ValueError,'closed'):run_readout(p,execute=True)
            missing_bridge=json.loads(self.f.bridge.read_text());missing_bridge['entities'].pop(next(iter(missing_bridge['entities'])))
            from hybrid_tsad.engine.stage2_readout import physical_bridge
            a['definition']['physical_bridge']=record(put(self.f.root/'missing_bridge.json',missing_bridge))
            with self.assertRaisesRegex(ValueError,'lacks actual'):physical_bridge(a,self.f.profile)

    def test_wrong_parent_role_plan_primary_and_incomplete_model_set_refused(self):
        from hybrid_tsad.engine.stage2_readout_identity import validate_definition
        with self.f.context():
            original=json.loads(self.f.definition.read_text())
            for case in ('missing','parent','role','plan','best','approval','train_only'):
                d=copy.deepcopy(original)
                if case=='missing':d['u_models'].pop()
                elif case=='parent':d['u_models'][0]['initialization']='random_init'
                elif case=='role':d['u_models'][0]['stage']='SOURCE_PRETRAIN'
                elif case=='plan':d['normal_plans']['0']=d['normal_plans']['1']
                elif case=='best':
                    cp=Path(d['u_models'][0]['checkpoint']['path']);best=cp.parent/'best.pt';best.write_bytes(cp.read_bytes());d['u_models'][0]['checkpoint']=record(best)
                elif case=='train_only':d['u_models'][0]['primary_acceptance']=record(Path(d['u_models'][0]['primary_acceptance']['path']).parent/'primary_train.json')
                else:d['all18_acceptance']=record(put(self.f.root/'PENDING.json',dict(status='PENDING')))
                if case not in ('missing','approval'):
                    acceptance=put(self.f.root/(case+'_accepted.json'),dict(status='PRIMARY_ACCEPTED_ALL18_U_TRAIN_CAL',stage='U_NORMAL_HEAD',models={x['job_id']:canonical_sha(x) for x in d['u_models']}));d['all18_acceptance']=record(acceptance)
                with self.subTest(case=case),self.assertRaises(ValueError):validate_definition(d,self.f.definition,self.f.profile)

    def test_cpu_fixture_permission_digest_and_no_labels_route(self):
        from hybrid_tsad.engine.stage2_readout_identity import load_operation,approve
        from hybrid_tsad.engine.stage2_readout import run_readout
        with self.f.context():
            p=self.f.operation('catalog',dict(definition=self.f.definition));self.assertFalse(run_readout(p)['executed'])
            c=json.loads(p.read_text());c['parameters']['profile']['name']='TEP100';put(p,c);cfg,paths,out,_,digest=load_operation(p)
            with self.assertRaisesRegex(ValueError,'bind'):approve(cfg,paths,out,digest,p)
            c=json.loads(p.read_text());c['inputs']['labels']=record(self.f.definition);put(p,c)
            with self.assertRaisesRegex(ValueError,'schema6'):load_operation(p)
            p=self.f.operation('infer',dict(catalog=self.f.definition,whitelist=self.f.definition),scope='shard',name='fake_gpu')
            c=json.loads(p.read_text());c['parameters']['device']='cuda:0';put(p,c);cfg,paths,out,_,digest=load_operation(p)
            ap=Path(c['approval']['path']);v=json.loads(ap.read_text());v['operation_manifest_sha256']=digest;put(ap,v);c['approval']=record(ap);put(p,c)
            cfg,paths,out,_,digest=load_operation(p)
            with self.assertRaisesRegex(ValueError,'CPU-only'):approve(cfg,paths,out,digest,p)
            from hybrid_tsad.engine.stage2_readout_identity import whitelist
            entities=json.loads(self.f.query.read_text())['records'];entities={q['entity']:q for q in entities};first=min(entities)
            l0=[dict(job_id=j,entity=first,origins=entities[first]['origins'][:2]) for j in JOBS if JOBS[j][0]==0]
            self.assertEqual(len(whitelist(l0,entities,'l0')),6)
            with self.assertRaisesRegex(ValueError,'exact six'):whitelist(l0[:-1],entities,'l0')
            from hybrid_tsad.engine.stage2_readout import production_backend
            with patch('torch.cuda.is_available',return_value=False),self.assertRaisesRegex(ValueError,'unavailable'):
                production_backend(dict(name='TEP100',channels=52),'cuda:0')
            self.assertNotEqual(u_key('u_normal_head_hybrid_s0_source_pretrained','fault01_run031'),u_key('u_normal_head_hybrid_s0_random_init','fault01_run031'))

    def test_z_normal_cal_x2_resigned_q99_and_trace_change_refused(self):
        import numpy as np
        from hybrid_tsad.engine.stage2_readout import run_readout,load_catalog,validate_z_threshold
        with self.f.context():
            run_readout(self.f.operation('catalog',dict(definition=self.f.definition)),execute=True);catalog=self.f.root/'catalog_out/catalog.json'
            r=run_readout(self.f.operation('z-normal-cal',dict(catalog=catalog)),execute=True)
            path=Path(r['thresholds']['s0/hybrid']['path']);t=json.loads(path.read_text());a=load_catalog(catalog,self.f.profile)
            with np.load(t['canonical']['path']) as z:values={k:z[k].copy() for k in z.files}
            values['canonical_score']*=np.float32(2);bad=path.parent/'x2.npz';np.savez_compressed(bad,**values)
            t['canonical']=record(bad);t['threshold99']=float(np.quantile(values['canonical_score'],.99,method='linear'));bp=put(path.parent/'x2.json',t)
            with self.assertRaisesRegex(ValueError,'closed'):validate_z_threshold(bp,a,0,'hybrid')
            t=json.loads(path.read_text());t['normal_plan']=self.f.f.plans['1']['target_plan'];bp=put(path.parent/'wrong_plan.json',t)
            with self.assertRaisesRegex(ValueError,'plan'):validate_z_threshold(bp,a,0,'hybrid')

    def test_replay_future_isolation_and_nonmonotone_cal_are_original_helpers(self):
        import numpy as np
        import torch
        from test_stage2 import toy_factory
        from hybrid_tsad.engine.bank import collect_forecasts
        model=toy_factory(self.f.root/'official.py',mode='hybrid',history=100)
        data=np.random.default_rng(5).normal(size=(110,52)).astype('float32');future=data.copy();future[100:105]+=10
        a,ar,_=collect_forecasts(model,data,[100],1.,mask_seed=84551);b,br,_=collect_forecasts(model,future,[100],1.,mask_seed=84551)
        self.assertTrue(np.array_equal(a['prediction'],b['prediction']));self.assertTrue(np.array_equal(ar,br));self.assertFalse(np.array_equal(a['standardized_score'],b['standardized_score']))

if __name__=='__main__':unittest.main()


@unittest.skipUnless(HAS_TORCH and os.environ.get('HYBRID_OFFICIAL_FILE'),'existing torch/external official source required')
class ActualCoreTransportTests(unittest.TestCase):
    def test_real_three_modes_wrapper_matches_old_collector_exact_arrays_rng(self):
        import numpy as np
        import torch
        from hybrid_tsad.model import HybridPredictor
        from hybrid_tsad.engine.stage2_readout import infer
        from hybrid_tsad.engine.bank import collect_forecasts
        from hybrid_tsad.api import create_predictor
        from hybrid_tsad.engine.stage2_identity import Assets
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);official=Path(os.environ['HYBRID_OFFICIAL_FILE']);data=np.random.default_rng(2).normal(size=(110,52)).astype('float32')
            array=root/'synthetic.npy';np.save(array,data);entity='fault01_run031';catalog=put(root/'transport_only.json',dict(scope='synthetic transport only; no identity authority'))
            for mode in ('temporal','tf','hybrid'):
                torch.manual_seed(7);model=HybridPredictor(official,history=100,mode=mode);cp=root/(mode+'.pt');torch.save(model.state_dict(),cp)
                mp=put(root/(mode+'_meta.json'),dict(history=100,horizon=5,mode=mode,copies=2,temporal_depth=4,checkpoint_sha256=sha256(cp)))
                job=f'u_normal_head_{mode}_s0_source_pretrained';spec=dict(paths=dict(checkpoint=cp,checkpoint_metadata=mp),source=dict(normal_error_scale=1.),
                    row=dict(checkpoint=record(cp),source_calibration={'sha256':'a'*64},target_threshold={'sha256':'b'*64}))
                q=dict(array_path=str(array),array_sha256_from_frozen_parent=sha256(array),fault_id=1,run_id=31,origins=[100,105])
                a=dict(assets=Assets(),base=catalog,official=official,catalog_path=catalog,entities={entity:q},models={job:spec})
                output=root/('infer_'+mode);output.mkdir()
                report=infer(a,[dict(job_id=job,entity=entity,origins=[100,105])],output,profile=dict(name='SYNTHETIC',points=110,channels=52),scope='shard',device='cpu')
                old=create_predictor(official,mode=mode,history=100,checkpoint=cp,checkpoint_metadata=mp)
                expected,rng,_=collect_forecasts(old,data,[100,105],1.,mask_seed=84551)
                desc=next(iter(report['banks'].values()))
                with np.load(desc['bank']['path']) as actual:
                    for k,v in expected.items():self.assertTrue(np.array_equal(actual[k],v),(mode,k))
                with np.load(desc['rng']['path']) as actual:self.assertTrue(np.array_equal(actual['before_each_block'],rng),mode)
