"""Only synthetic labels/scores/metadata; never imports Torch or runs a gate."""
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from hybrid_tsad.config import sha256
from hybrid_tsad.engine.readout_manifest import record
from hybrid_tsad.engine.identity import operation_manifest
from hybrid_tsad.engine.stage2_identity import canonical_sha
from hybrid_tsad.engine.stage2_evaluation import (STAGE,READOUT_STAGE,load_operation,runtime_identity,run_evaluation,threshold_only)
from hybrid_tsad.engine.stage2_run_statistics import CONFIGS,NATIVE,LEGACY,tracks,run_statistics,normal_prefix_summary,delay_summary

AFF=os.environ.get('HYBRID_OFFICIAL_AFF_DIR')
FROZEN=Path(os.environ.get('HYBRID_FROZEN_READOUT_SNAPSHOT',str(Path(__file__).resolve().parents[2]/'preparations/runtime_stage2_readout_a712825a_v1')))

def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v),encoding='utf-8');return p


class Fixture:
    def __init__(self,root):
        self.root=root;self.profile=dict(name='SYNTHETIC',faults=[1,2],runs=[31,32],points=120,channels=52,event_start=110,sample_interval_minutes=3)
        self.contract=Path(__file__).resolve().parents[2]/'experiments/contracts/HYBRID_TEP100_U_EVALUATION_CPU_20261101.md'
        if not self.contract.exists():self.contract=Path(os.environ['HYBRID_EVALUATION_CONTRACT'])
        self.frozen=root/'frozen_gate_copy';shutil.copytree(FROZEN,self.frozen)
        self.runtime=put(root/'eval_runtime.json',runtime_identity());self.oldruntime=self.frozen/'READOUT_RUNTIME_IDENTITY.json';self.copymanifest=self.frozen/'COPY_MANIFEST.json'
        dummy=put(root/'synthetic_metadata.json',dict(fixture_only=True));self.q=[]
        for run in (31,32):
            common=np.random.default_rng(run).normal(size=(120,52)).astype('float32')
            for fault in (1,2):
                data=common.copy();data[110:]+=fault
                array=root/f'features_{fault}_{run}.npy';np.save(array,data);label=root/f'labels_{fault}_{run}.npy';np.save(label,(np.arange(120)>=110).astype('uint8'))
                self.q.append(dict(entity=f'fault{fault:02}_run{run:03}',fault_id=fault,run_id=run,role='query',history=100,horizon=5,local_start=0,local_end_exclusive=120,
                    guard=0,channels=52,origins=[100,105,110,115],array_path=str(array),array_sha256_from_frozen_parent=sha256(array),label_path=str(label),label_sha256_from_frozen_parent=sha256(label),event_start_0based=110,event_end_exclusive=120))
        self.query=put(root/'query_metadata.json',dict(dataset='SYNTHETIC',records=self.q));urows=[];ubanks={};zscores={};znative={};zmodels=[];controls=[];zt={}
        for seed in range(3):
            for mode in ('temporal','tf','hybrid'):
                cp=root/f'Z_{seed}_{mode}.pt';cp.write_bytes(b'synthetic identity only; not a model')
                cal=put(root/f'Zcal_{seed}_{mode}.json',dict(calibration_schema_version=2,history=100,role='source_cal',score_space='history_normalized_abs_channelmean_div_source_median_v1',source_bank_sha256='a'*64,
                    normal_error_scale=1.,threshold99=.9,operation_manifest_sha256='b'*64))
                zmodels.append(dict(seed=seed,mode=mode,checkpoint=record(cp),calibration=record(cal)))
                target=put(root/f'Ztarget_{seed}_{mode}.json',dict(role='Z_TARGET_NORMAL_THRESHOLD_ONLY',checkpoint=record(cp),source_calibration=record(cal),source_error_scale=1.,threshold99=1.2,target_global_rescale=False))
                zt[f's{seed}/{mode}']=record(target)
            for mode in ('temporal','tf','hybrid','late','history_mean','persistence'):controls.append(dict(seed=seed,method=mode,threshold=.9,scale=1.))
        for stage,init,method in CONFIGS:
            for seed in range(3):
                checkpoint=root/f'{stage}_{init}_{method}_{seed}.pt';checkpoint.write_bytes(b'synthetic checkpoint hash only, no Torch')
                if stage=='U_NORMAL_HEAD':
                    job=f'u_normal_head_{method}_s{seed}_{init}'
                    source=put(root/(job+'_source.json'),dict(role='source_cal_after_adaptation',fitted_checkpoint_sha256=sha256(checkpoint),threshold99=.9))
                    target=put(root/(job+'_target.json'),dict(role='target_threshold_only',fitted_checkpoint_sha256=sha256(checkpoint),threshold99=1.2))
                    row=dict(job_id=job,checkpoint=record(checkpoint),source_calibration=record(source),target_threshold=record(target))
                    row.update({k:record(dummy) for k in ('checkpoint_metadata','training_summary','train_archive','cal_archive','primary_acceptance')});urows.append(row)
                for q in self.q:
                    index=np.arange(100,120);rng=np.random.default_rng(seed+q['run_id']*7+CONFIGS.index((stage,init,method))*100)
                    score=rng.uniform(0,2,size=20).astype('float32');score[10:]+=q['fault_id']*.01
                    folder=root/'scores'/stage/init/method/f's{seed}'/q['entity'];folder.mkdir(parents=True)
                    bank=folder/'score.npz';field='standardized_score' if stage=='U_NORMAL_HEAD' else 'score';np.savez_compressed(bank,time_index=index,**{field:score})
                    if stage=='U_NORMAL_HEAD':
                        obj=dict(file=record(bank),score_field=field,index_field='time_index');key=READOUT_STAGE+'/'+job+'/'+q['entity']
                        ubanks[key]=dict(job_id=job,entity=q['entity'],bank=record(bank),rng=record(dummy),metadata=record(dummy),canonical=obj,
                            tracks={t:dict(score_object=obj,threshold=.9 if t=='source' else 1.2) for t in ('source','target')})
                    else:
                        key=f's{seed}/{method}/'+q['entity'];zscores[key]=dict(seed=seed,method=method,entity=q['entity'],file=record(bank),threshold=.9)
                        if method in ('temporal','tf','hybrid'):znative[key]=dict(bank=record(bank))
        self.zseal=put(root/'old_Z_seal.json',dict(query_manifest=record(self.query),scores=zscores,native_banks=znative))
        zcatalog=put(root/'old_Z_catalog.json',dict(models=zmodels,controls=controls))
        definition=put(root/'readout_definition.json',dict(query_manifest=record(self.query),u_models=urows,z_catalog=record(zcatalog)))
        catalog=put(root/'accepted_catalog.json',dict(definition=record(definition)))
        zthresholds=put(root/'zthresholds.json',dict(thresholds=zt));self.seal=put(root/'seal.json',dict(status='ALL_U_CANONICAL_SCORES_SEALED_LABELS_LOCKED',stage=READOUT_STAGE,
            complete_group=True,labels_opened=False,catalog=record(catalog),u_banks=ubanks,z_seal=record(self.zseal),z_thresholds=record(zthresholds),counts=dict(u_banks=72,z_native_banks=36,z_fixed_scores=72)))
        gate_dir=root/'synthetic_gate';gate_dir.mkdir();gp=root/'gate_approval.json';put(gp,{})
        gc=dict(engine_schema_version=6,stage=READOUT_STAGE,operation='gate',contract=record(dummy),approval=record(gp),inputs=dict(catalog=record(catalog),seal=record(self.seal)),
            parameters=dict(profile=self.profile,scope='full',runtime=record(self.oldruntime)),output=str(gate_dir),score_space='history_normalized_abs_channelmean_div_source_median_v1')
        gate_config=put(root/'gate_config.json',gc)
        paths={k:Path(v['path']) for k,v in {'contract':gc['contract'],'approval':gc['approval'],**gc['inputs']}.items()};manifest,_=operation_manifest(gc,paths,gate_dir)
        manifest.update(stage=READOUT_STAGE,permissions='no training/source-fit/labels/F; explicit operation only')
        put(gp,dict(status='APPROVED',stage=READOUT_STAGE,contract_sha256=gc['contract']['sha256'],operation_manifest_sha256=canonical_sha(manifest),
            permissions=dict(operation='gate',training=False,source_fit=False,query_infer=False,target_normal_cal=False,labels=False,F=False)))
        gc['approval']=record(gp);put(gate_config,gc)
        gate_summary=put(gate_dir/'summary.json',dict(status='COMPLETE_SEAL_ELIGIBLE_FOR_SEPARATE_PRIMARY_LABEL_RELEASE',labels_permission=False,seal=record(self.seal),counts=json.loads(self.seal.read_text())['counts']))
        oswait=put(root/'synthetic_OS_WAIT.json',dict(actual_OS_wait_exit_code=0,process_seconds=1.,synthetic_fixture=True))
        frozen_identity=json.loads(self.oldruntime.read_text());resource=put(root/'synthetic_RESOURCE.json',dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',summary_sha256=sha256(gate_summary),synthetic_fixture=True,
            current_runtime=dict(shared_sha256=frozen_identity['files']['src/hybrid_tsad/engine/stage2_readout.py'],runtime_identity_sha256=sha256(self.oldruntime))))
        self.gate_review=put(root/'synthetic_gate_review.json',dict(status='SYNTHETIC_REVIEW_ONLY',gate_executed=False))
        self.inputs=dict(seal=record(self.seal),gate_summary=record(gate_summary),gate_config=record(gate_config),gate_os_wait=record(oswait),gate_resource=record(resource),
            readout_runtime=record(self.oldruntime),readout_copy_manifest=record(self.copymanifest),query_manifest=record(self.query),runtime=record(self.runtime))
        self.accept=put(root/'synthetic_gate_acceptance.json',dict(status='SYNTHETIC_GATE_ACCEPTANCE_ONLY',stage=READOUT_STAGE,independent_review=record(self.gate_review),
            **{k:self.inputs[k] for k in ('seal','gate_summary','gate_config','gate_os_wait','gate_resource','readout_runtime','readout_copy_manifest','query_manifest')}))
        self.inputs['gate_acceptance']=record(self.accept)
        from hybrid_tsad.engine.official_affiliation import OFFICIAL_FILES
        self.aff=put(root/'aff_identity.json',dict(schema_version=1,directory=AFF or '/MISSING_EXTERNAL_OFFICIAL_AFF',files=OFFICIAL_FILES,
            license=dict(status='UNVERIFIED_EXISTING_LOCAL_DEPENDENCY',source='existing frozen six-file audit; no redistribution',redistribute=False)))
        self.inputs['affiliation']=record(self.aff)

    def config(self,name='evaluation'):
        approval=put(self.root/(name+'_approval.json'),{})
        c=dict(engine_schema_version=7,stage=STAGE,operation='evaluate',contract=record(self.contract),approval=record(approval),inputs=self.inputs,
            parameters=dict(profile=self.profile,readout_runtime_root=str(self.frozen)),output=str(self.root/(name+'_out')),score_space='history_normalized_abs_channelmean_div_source_median_v1')
        p=put(self.root/(name+'.json'),c);self.resign(p);return p

    def resign(self,path):
        c=json.loads(path.read_text());_,_,_,_,digest=load_operation(path);ap=Path(c['approval']['path'])
        put(ap,dict(status='APPROVED',stage=STAGE,scope='SYNTHETIC_CPU_ENGINEERING_ONLY',contract_sha256=sha256(self.contract),operation_manifest_sha256=digest,
            permissions=dict(labels=True,normal_prefix_values=True,evaluate=True,inference=False,training=False,calibration=False,F=False)))
        c['approval']=record(ap);put(path,c)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.f=Fixture(Path(self.tmp.name))

    @unittest.skipUnless(AFF,'BLOCKED: existing external official Affiliation required, never download/fallback')
    def test_complete_synthetic_config_rank_once_and_independent_runtime(self):
        import sklearn.metrics as metrics
        with patch('sklearn.metrics.average_precision_score',wraps=metrics.average_precision_score) as ap,patch('sklearn.metrics.roc_auc_score',wraps=metrics.roc_auc_score) as roc:
            result=run_evaluation(self.f.config(),execute=True)
        self.assertEqual((ap.call_count,roc.call_count),(144,144));self.assertEqual(result['rank_call_counts'],dict(native=108,legacy=36));self.assertEqual(result['metric_rows'],252)
        self.assertTrue(result['synthetic_fixture']);self.assertFalse(result['GPU_executed']);self.assertNotIn('torch',sys.modules)
        out=self.f.root/'evaluation_out';self.assertTrue((out/'evaluation_manifest.json').is_file())
        with np.load(out/'paired_run_bootstrap.npz') as z:
            self.assertEqual(z['draws'].shape,(2000,2));self.assertEqual(z['AP_difference'].shape,(2000,6))
        self.assertEqual(len(list((self.f.frozen/'src/hybrid_tsad').rglob('*.py'))),32)

    def test_invalid_approval_gate_runtime_scope_and_missing_bank_fail_before_arrays(self):
        for case in ('PENDING','gate_unaccepted','wrong_gate_OS','production_fixture','old_extra_py','missing_bank','different_ST_object'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as t:
                f=Fixture(Path(t));p=f.config();c=json.loads(p.read_text())
                if case=='PENDING':
                    ap=Path(c['approval']['path']);v=json.loads(ap.read_text());v['status']='PENDING';put(ap,v);c['approval']=record(ap);put(p,c)
                elif case=='gate_unaccepted':
                    v=json.loads(f.accept.read_text());v['status']='PENDING';put(f.accept,v);c['inputs']['gate_acceptance']=record(f.accept);put(p,c);f.resign(p)
                elif case=='wrong_gate_OS':
                    op=Path(c['inputs']['gate_os_wait']['path']);put(op,dict(actual_OS_wait_exit_code=124,process_seconds=1.));c['inputs']['gate_os_wait']=record(op);put(p,c);f.resign(p)
                elif case=='production_fixture':
                    ap=Path(c['approval']['path']);v=json.loads(ap.read_text());v['scope']='U_NORMAL_HEAD_EVALUATION_EXPLICIT_LABEL_RELEASE';put(ap,v);c['approval']=record(ap);put(p,c)
                elif case=='old_extra_py':(f.frozen/'src/hybrid_tsad/forbidden_evaluation.py').write_text('# never insert new module into frozen gate')
                else:
                    v=json.loads(f.seal.read_text())
                    if case=='missing_bank':v['u_banks'].pop(next(iter(v['u_banks'])))
                    else:next(iter(v['u_banks'].values()))['tracks']['target']['score_object']['score_field']='raw_score'
                    put(f.seal,v);c['inputs']['seal']=record(f.seal);put(p,c);f.resign(p)
                with patch('numpy.load',side_effect=AssertionError('array read before gate release')),self.assertRaises(ValueError):run_evaluation(p,execute=True)

    def test_preview_and_missing_official_dependency_never_fallback(self):
        self.assertFalse(run_evaluation(self.f.config())['executed'])
        from hybrid_tsad.engine.official_affiliation import OfficialAffiliationBlocked
        aff=json.loads(self.f.aff.read_text());aff['directory']='/MISSING_EXTERNAL_OFFICIAL_AFF';put(self.f.aff,aff);self.f.inputs['affiliation']=record(self.f.aff)
        with patch('numpy.load',side_effect=AssertionError('labels before official dependency')),self.assertRaises(OfficialAffiliationBlocked):run_evaluation(self.f.config('missing_aff'),execute=True)

    def test_registry_semantic_missing_init_and_nonshared_tracks_before_arrays(self):
        from hybrid_tsad.engine.stage2_evaluation import registry
        from hybrid_tsad.engine.stage2_identity import Assets
        baseline=json.loads(self.f.seal.read_text());entities={q['entity']:q for q in self.f.q}
        for case in ('missing','wrong_init_key','different_ST'):
            seal=copy.deepcopy(baseline);key=next(iter(seal['u_banks']))
            if case=='missing':seal['u_banks'].pop(key)
            elif case=='wrong_init_key':seal['u_banks'][key]['job_id']=seal['u_banks'][key]['job_id'].replace('source_pretrained','random_init')
            else:seal['u_banks'][key]['tracks']['target']['score_object']['score_field']='raw_score'
            with self.subTest(case=case),patch('numpy.load',side_effect=AssertionError('array read before registry semantic closure')),self.assertRaises(ValueError):
                registry(Assets(),seal,entities,self.f.seal,record(self.f.query))


class ThresholdAndStatisticsTests(unittest.TestCase):
    def test_prelabel_profile_resource_caps_and_overrun(self):
        from hybrid_tsad.engine.stage2_evaluation import evaluation_limit_seconds
        self.assertEqual(evaluation_limit_seconds({'name':'SYNTHETIC'}),300)
        self.assertEqual(evaluation_limit_seconds({'name':'TEP100'}),900)
        for profile in ({},{'name':'UNKNOWN'},{'name':'tep100'}):
            with self.assertRaisesRegex(ValueError,'Unknown evaluation resource profile'):evaluation_limit_seconds(profile)
        with tempfile.TemporaryDirectory() as t:
            fixture=Fixture(Path(t));config=fixture.config('overrun')
            with patch('hybrid_tsad.engine.stage2_evaluation.time.monotonic',side_effect=[0.,301.]):
                with self.assertRaisesRegex(ValueError,'profile time budget exceeded'):run_evaluation(config,execute=True)
            out=Path(json.loads(config.read_text())['output'])
            self.assertEqual(json.loads((out/'evaluation_failure.json').read_text())['status'],'FAILED_NO_COMPLETE_REPORT')
            self.assertFalse((out/'evaluation_summary.json').exists())

    def test_threshold_only_exact_old_kernel_oracle_edge_cases(self):
        from hybrid_tsad.engine.evaluation import evaluate_fixed_scores
        from hybrid_tsad.engine.official_affiliation import load_official_affiliation,OFFICIAL_FILES
        if not AFF:self.skipTest('BLOCKED: external official Aff required for threshold/Aff differential')
        with tempfile.TemporaryDirectory() as t:
            identity=put(Path(t)/'aff.json',dict(schema_version=1,directory=AFF,files=OFFICIAL_FILES,
                license=dict(status='UNVERIFIED_EXISTING_LOCAL_DEPENDENCY',source='existing six-file source audit',redistribute=False)))
            callback,_=load_official_affiliation(identity)
        cases=[(np.arange(12),np.array([0]*4+[1]*8),np.zeros(12),0.),
               (np.arange(12),np.array([0]*4+[1]*8),np.ones(12),0.),
               (np.arange(12),np.array([0]*4+[1]*8),np.array([0,1,1,1,1,0,0,0,0,0,0,0]),.5),
               (np.array([0,1,4,5,6,7]),np.array([0,0,0,1,1,1]),np.array([0,1,0,1,0,1]),.5)]
        for index,y,s,threshold in cases:
            start=4 if len(y)==12 else 5;end=12 if len(y)==12 else 8
            coverage=[dict(event_id='event',start=start,end_exclusive=end,eligible_complete=True,scored_points=int((y==1).sum()),total_points=end-start)]
            source=evaluate_fixed_scores(index,y,s,threshold,coverage,affiliation=callback)
            actual=threshold_only(index,y,s,threshold,coverage,rank=source['metrics'],affiliation=callback)
            def clean(v):
                from hybrid_tsad.engine.pipeline import _json_safe
                return json.dumps(_json_safe({k:x.tolist() if isinstance(x,np.ndarray) else x for k,x in v.items()}),sort_keys=True)
            self.assertEqual(clean(source),clean(actual))
        index=np.arange(8);y=np.array([0,0,1,1,1,1,1,1]);coverage=[dict(event_id='e',start=2,end_exclusive=10,eligible_complete=False,scored_points=6,total_points=8)]
        source=evaluate_fixed_scores(index,y,np.arange(8),3,coverage,affiliation=callback);actual=threshold_only(index,y,np.arange(8),3,coverage,rank=source['metrics'],affiliation=callback)
        self.assertEqual(source['events'],actual['events']);self.assertFalse(actual['events'][0]['complete'])

    def test_shared_draws_macro_grid_and_nonfinite_rank_not_redrawn(self):
        profile=dict(name='SYNTHETIC',faults=[1,2],runs=[31,32]);rows=[]
        for config in CONFIGS:
            for track in tracks(config):
                for s in range(3):
                    for f in profile['faults']:
                        for r in profile['runs']:
                            value=.3+.01*s+.02*f+.001*r+(.1+.05*(r-31) if config[0]=='U_NORMAL_HEAD' and config[1]=='source_pretrained' else 0)
                            rows.append(dict(stage=config[0],initialization=config[1],method=config[2],threshold_track=track,seed=s,fault_id=f,run_id=r,AP=value,ROC=.5,positive_fraction=.5,rank_object_id=str((config,s,f,r))))
        a=run_statistics(rows,profile);b=a['bootstrap_arrays'];self.assertTrue(np.array_equal(b['draws'],np.random.default_rng(99174).integers(0,2,(2000,2))))
        self.assertTrue(np.allclose(b['paired_run_difference'],np.array([[.1],[.15]])))
        manual=.1+.05*np.sum(b['draws']==1,axis=1)/2
        self.assertTrue(np.allclose(b['AP_difference'],manual[:,None]))
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'):run_statistics(rows[:-1],profile)
        with self.assertRaisesRegex(ValueError,'duplicate'):run_statistics(rows+[rows[0]],profile)
        broken=copy.deepcopy(rows)
        for row in broken:
            if row['run_id']==31 and row['stage']=='U_NORMAL_HEAD' and row['initialization']=='random_init':row['AP']=float('nan')
        failed=run_statistics(broken,profile);self.assertEqual(failed['status'],'INCOMPLETE');self.assertTrue(np.isnan(failed['bootstrap_arrays']['AP_difference'][np.any(b['draws']==0,axis=1)]).all())

    def test_prefix_count_denominator_and_miss_delay_retained(self):
        profile=dict(faults=[1,2],sample_interval_minutes=3);obs=[]
        for fault in (1,2):obs.append(dict(stage='Z',initialization='source_pretrained',method='hybrid',threshold_track='target',seed=0,run_id=31,fault_id=fault,entity=str(fault),feature_prefix_sha256='f',score_prefix_sha256='s',binary_prefix_sha256='b',normal_points=60,normal_false_points=3,normal_alarm_fragments=1))
        row=normal_prefix_summary(obs,profile)[0];self.assertEqual((row['duplicated_hours'],row['unique_normal_hours'],row['unique_normal_false_points']),(6.,3.,3))
        obs[1]['binary_prefix_sha256']='different';self.assertIsNone(normal_prefix_summary(obs,profile)[0]['unique_normal_hours'])
        events=[dict(stage='Z',initialization='source_pretrained',method='hybrid',threshold_track='target',seed=0,complete=True,any_hit=hit,delay_minutes=delay) for hit,delay in [(True,0),(False,None)]]
        row=delay_summary(events)[0];self.assertEqual((row['eligible'],row['detected'],row['missed']),(2,1,1));self.assertEqual(row['median_detected_only_delay_minutes'],0);self.assertIsNone(row['all_eligible_mean_delay'])

if __name__=='__main__':
    assert 'torch' not in sys.modules
    unittest.main()
