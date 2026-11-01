"""Synthetic CPU evaluation fixtures only. No official data/model/score files."""
import csv
import importlib.util
import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad import left_normal_scores as s
from hybrid_tsad.engine import left_normal_evaluation as e


def truth_refs(root):
    return [dict(scenario=scene,role='DEV',run_id=i,
                 path=str(Path(root)/scene/'DEV'/f'run{i:02d}_truth.json'),sha256='a'*64)
            for scene in ('LIN','WEAK') for i in range(8)]


def config(root):
    return dict(status='PENDING',operation='LEFT_WL_C2_LABEL_EVALUATION',
        source_manifest_sha256=s.MANIFEST_SHA,seed=2025,bootstrap_repeats=10000,
        max_seconds=300,max_rss_GiB=2,gpu_budget=0,truths=truth_refs(root),
        score_seal={'path':str(Path(root)/'score_seal.json'),'sha256':'PENDING'},
        score_reports=[dict(scenario=scene,path=str(Path(root)/'scores'/scene/'report.json'),sha256='PENDING',
            config={'path':str(Path(root)/'scores'/scene/'config.json'),'sha256':'PENDING'},
            checkpoint_sha256='b'*64,scaler_sha256='c'*64) for scene in ('LIN','WEAK')],
        cpu_all_layers={'path':str(Path(root)/'all_layers.csv'),'sha256':'PENDING'},
        output=str(Path(root)/'evaluation'))


def synthetic_truth(scene='LIN',rid=0):
    start=24*s.H;end=start+12*s.H
    labels={k:np.zeros(s.N,dtype=np.int8) for k in ('A','N','S0','SB')}
    labels['A'][start:end]=1;labels['SB'][start:start+s.H]=1;labels['SB'][end-s.H:end]=1
    return dict(scenario=scene,role='DEV',run_id=rid,normal_lag=np.full(s.N,24),
                event_start=start,event_end=end,labels=labels)


def cpu_rows():
    return [dict(scenario=scene,model=m,readout=r,run_id=str(i),variant='A',layer='CORE',AP='.4')
            for scene in ('LIN','WEAK') for m,r in [('MULTI191','last'),('PEAK191','last'),
                ('SYNC191','last'),('MULTI64','mean128'),('SYNC64','mean128')] for i in range(8)]


def synthetic_score_artifacts(root):
    """Tiny explicit score/report fixtures, never claimed as actual LEFT output."""
    from hybrid_tsad.left_baseline import MODEL_FILES
    p=Path(root);cfg=config(root);cfg['status']='FROZEN';bank_rows=[]
    for scene,item in zip(('LIN','WEAK'),cfg['score_reports']):
        job=Path(item['path']).parent;job.mkdir(parents=True)
        snapshot=job/'source';snapshot.mkdir()
        shutil.copyfile(s.__file__,snapshot/'left_normal_scores.py')
        hashes={}
        for rel in MODEL_FILES:
            source=snapshot/rel;source.parent.mkdir(parents=True,exist_ok=True)
            source.write_text('# synthetic provenance fixture only; never imported\n')
            hashes[rel]=s.sha(source)
        scfg=dict(status='FROZEN',operation='LEFT_WL_C2_FIXED_SCORE',scenario=scene,seed=2025,
            source_manifest_sha256=s.MANIFEST_SHA,source_root=str(job/'not_loaded_original_source'),
            source_hashes=hashes,checkpoint={'path':'not_loaded_synthetic_checkpoint','sha256':'b'*64},
            scaler={'path':'not_loaded_synthetic_scaler','sha256':'c'*64},banks={})
        report_rows=[];cal=[]
        for role in ('CAL','DEV'):
            scfg['banks'][role]=[]
            for i in range(8):
                for variant in (('N',) if role=='CAL' else ('A','N','S0','SB')):
                    raw=np.zeros((128,192,2),np.float32)
                    if role=='CAL':raw[:]=np.arange(128,dtype=np.float32)[:,None,None]/1000
                    if role=='DEV' and variant=='A':raw[24:36]=3.
                    if role=='DEV' and variant=='SB':raw[[24,35]]=3.
                    file=job/f'{role}_run{i:02d}_{variant}.npz'
                    arrays=s.reduce_raw(raw);np.savez_compressed(file,**arrays,**s.coordinates())
                    feature_ref=dict(scenario=scene,role=role,run_id=i,variant=variant,
                        path=str(p/'not_opened_features'/scene/role/f'run{i:02d}_{variant}.npy'),sha256='a'*64)
                    scfg['banks'][role].append(feature_ref)
                    bank_rows.append(dict(reference=feature_ref,path=str(file),sha256=s.sha(file)))
                    report_rows.append(dict(reference=feature_ref,file=file.name,sha256=s.sha(file),raw_shape=[128,192,2],dtype='float32'))
                    if role=='CAL':cal.append(dict(reference=feature_ref,**{r:arrays[r] for r in s.READOUTS}))
        e.dump(item['config']['path'],scfg);item['config']['sha256']=s.sha(item['config']['path'])
        state=dict(step=0,modes={'':False},state_sha256={'synthetic_science':'a'*64})
        report=dict(status='Completed',label_files_read=False,unscored_points=0,banks=report_rows,
            config_sha256=item['config']['sha256'],module_sha256=s.sha(s.__file__),
            checkpoint=scfg['checkpoint'],scaler=scfg['scaler'],model_config={'enc_in':2,'seq_len':192},
            state_before=state,state_after=state,calibration=s.calibrate_frozen(cal,scene),
            loaded_modules={rel:{'path':str(Path(scfg['source_root'])/rel),'sha256':h} for rel,h in hashes.items()})
        e.dump(item['path'],report);item['sha256']=s.sha(item['path'])
    seal=dict(status='APPROVED',approved_by='root',operation='LEFT_WL_C2_LABEL_EVALUATION',
        allow_truth=True,module_sha256=s.sha(s.__file__),source_manifest_sha256=s.MANIFEST_SHA,banks=bank_rows)
    return cfg,seal


class EvaluationTests(unittest.TestCase):
    def test_preview_and_exact_truth_role_permissions(self):
        cfg=config('/synthetic')
        self.assertEqual(e.validate_config(cfg)['status'],'PENDING')
        for field,value in [('seed',2026),('gpu_budget',1),('bootstrap_repeats',100),('max_rss_GiB',3)]:
            bad=dict(cfg);bad[field]=value
            with self.assertRaises(ValueError):e.validate_config(bad)
        refs=truth_refs('/synthetic');refs[0]['role']='CAL'
        with self.assertRaises(PermissionError):e.validate_truth_references(refs)
        refs=truth_refs('/synthetic');refs[-1]=refs[0]
        with self.assertRaises(PermissionError):e.validate_truth_references(refs)
        refs=truth_refs('/synthetic');refs[0]['path']='/synthetic/IRR/DEV/run00_truth.json'
        with self.assertRaises(PermissionError):e.validate_truth_references(refs)

    def test_pending_fails_before_seal_or_truth_io(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);e.dump(p/'config.json',config(root));e.dump(p/'release.json',{'status':'PENDING'})
            with patch.object(e.score,'require_evaluation_seal',side_effect=AssertionError('must not open seal')):
                with self.assertRaises(PermissionError):e.require_release(p/'config.json',p/'release.json')

    def test_score_banks_validate_raw_coordinates_and_no_pickle(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'synthetic.npz';raw=np.zeros((128,192,2),np.float32)
            arrays=dict(s.reduce_raw(raw),**s.coordinates());np.savez_compressed(p,**arrays)
            got=e.load_score_bank({'path':str(p),'sha256':s.sha(p)})
            self.assertTrue(np.array_equal(got['last'],np.zeros(128)))
            arrays['endpoint'][0]+=1;np.savez_compressed(p,**arrays)
            with self.assertRaises(ValueError):e.load_score_bank({'path':str(p),'sha256':s.sha(p)})

    def test_truth_semantics_enforce_original_event_and_shams(self):
        truth=synthetic_truth();e.validate_truth(truth,'LIN',0)
        with self.assertRaises(PermissionError):e.validate_truth(truth,'WEAK',0)
        truth['labels']['S0'][0]=1
        with self.assertRaises(ValueError):e.validate_truth(truth,'LIN',0)

    def test_cpu_pairs_complete_and_na_not_zero(self):
        rows=cpu_rows();rows[0]['AP']=''
        got=e.cpu_core_ap(rows);self.assertEqual(len(got),80)
        self.assertIsNone(got[('LIN','MULTI191','last',0)])
        with self.assertRaises(ValueError):e.cpu_core_ap(rows[:-1])
        with self.assertRaises(ValueError):e.cpu_core_ap(rows+[rows[0]])
        rows[0]['AP']='nan'
        with self.assertRaises(ValueError):e.cpu_core_ap(rows)

    def test_counts_macro_na_event_censoring_and_matched_denominators(self):
        alarm=np.zeros(128,bool);alarm[:2]=True;y=np.zeros(128,int);y[1:3]=1
        counts=e.binary_counts(alarm,y,np.ones(128,bool))
        self.assertEqual([counts[k] for k in ('TP','FP','FN','TN')],[1,1,1,125])
        layers=[dict(scenario='LIN',readout='last',variant='A',layer='CORE',run_id=i,
                     AP=None if i==7 else .5,ROC_AUC=.6,normal_FPR=1/126,
                     points=128,positive_points=2,normal_points=126,**counts) for i in range(8)]
        events=[dict(scenario='LIN',readout='last',variant='A',run_id=i,hit=i<3,
                     delay_points=191 if i<3 else None,right_censored=i>=3) for i in range(8)]
        matched=[dict(scenario='LIN',readout='last',variant='N',run_id=i,
                      matched_normal_interval_hit=i<2) for i in range(8)]
        out=e.macro_summary(layers,events,matched)
        self.assertEqual(out['layers'][0]['AP_valid_runs'],7)
        self.assertEqual(out['layers'][0]['normal_FPR_micro_denominator'],1008)
        self.assertEqual(out['events'][0]['right_censored'],5)
        self.assertEqual(out['events'][0]['delay_hit_only_mean'],191.)
        self.assertEqual(out['matched_normal'][0]['false_hit_numerator'],2)
        with self.assertRaises(ValueError):e.macro_summary(layers[:-1],events,matched)

    def test_complete_gate_rejects_alias_text_missing_report_and_mapping(self):
        with tempfile.TemporaryDirectory() as root:
            cfg,seal=synthetic_score_artifacts(root)
            banks,_=e.validate_evaluation_inputs(cfg,seal)
            self.assertEqual(len(banks),80)
            # Identical N/S0 bytes are legal; sharing one physical file is not.
            first,second=seal['banks'][0],seal['banks'][1]
            original_path=second['path'];second['path']=first['path']
            with self.assertRaises(PermissionError):e.validate_evaluation_inputs(cfg,seal)
            second['path']=original_path
            hardlink=Path(root)/'same_physical_bank.npz';os.link(first['path'],hardlink)
            second['path']=str(hardlink)
            with self.assertRaises(PermissionError):e.validate_evaluation_inputs(cfg,seal)
            second['path']=original_path
            # A unique .npz path holding text is still rejected before truth IO.
            file=Path(first['path']);original_bytes=file.read_bytes();file.write_text('not a numpy archive')
            first['sha256']=s.sha(file)
            report_path=Path(cfg['score_reports'][0]['path']);report=json.loads(report_path.read_text())
            original_report=json.loads(report_path.read_text())
            report['banks'][0]['sha256']=first['sha256'];e.dump(report_path,report)
            cfg['score_reports'][0]['sha256']=s.sha(report_path)
            with self.assertRaises(ValueError):e.validate_evaluation_inputs(cfg,seal)
            file.write_bytes(original_bytes);first['sha256']=s.sha(file)
            e.dump(report_path,original_report);cfg['score_reports'][0]['sha256']=s.sha(report_path)
            # Hash-pinned report alone does not excuse missing completion evidence.
            report=dict(original_report);report.pop('state_after');e.dump(report_path,report)
            cfg['score_reports'][0]['sha256']=s.sha(report_path)
            with self.assertRaises(ValueError):e.validate_evaluation_inputs(cfg,seal)
            report=dict(original_report);report['banks'][0]['file']='DEV_run00_N.npz'
            e.dump(report_path,report);cfg['score_reports'][0]['sha256']=s.sha(report_path)
            with self.assertRaises(ValueError):e.validate_evaluation_inputs(cfg,seal)
            # Source/scaler identity mismatch is rejected even with new report SHA.
            report=dict(original_report);report['scaler']={'path':'wrong','sha256':'d'*64}
            e.dump(report_path,report);cfg['score_reports'][0]['sha256']=s.sha(report_path)
            with self.assertRaises(ValueError):e.validate_evaluation_inputs(cfg,seal)

    @unittest.skipUnless(importlib.util.find_spec('sklearn') is not None,
                         'SKIP integration: local existing Python has no scikit-learn; no dependency install')
    def test_full_cli_pipeline_on_explicit_synthetic80_bank_fixture(self):
        # These approvals are confined to temporary, explicitly synthetic test files.
        # No generate_run, model, checkpoint, public CAL/DEV, or real truth is read.
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);cfg,seal=synthetic_score_artifacts(root)
            e.dump(p/'score_seal.json',seal);cfg['score_seal']['sha256']=s.sha(p/'score_seal.json')
            for ref in cfg['truths']:
                file=Path(ref['path']);file.parent.mkdir(parents=True,exist_ok=True)
                e.dump(file,synthetic_truth(ref['scenario'],ref['run_id']));ref['sha256']=s.sha(file)
            e.write_csv(p/'all_layers.csv',cpu_rows());cfg['cpu_all_layers']['sha256']=s.sha(p/'all_layers.csv')
            e.dump(p/'config.json',cfg)
            release=dict(status='APPROVED',approved_by='root',allow_truth=True,operation=cfg['operation'],
                config_sha256=s.sha(p/'config.json'),module_sha256=s.sha(e.__file__),
                score_module_sha256=s.sha(s.__file__),weak_lag_module_sha256=s.sha(e.wl.__file__))
            e.dump(p/'release.json',release)
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):report=e.evaluate(p/'config.json',p/'release.json')
            self.assertEqual(report['status'],'Completed');self.assertEqual(report['forward_calls'],0)
            summary=json.loads((p/'evaluation/macro_summary.json').read_text())
            self.assertTrue(all(r['event_recall']==1 for r in summary['events']))
            self.assertTrue(all(r['false_hit_rate']==0 for r in summary['matched_normal']))
            bootstrap=json.loads((p/'evaluation/paired_bootstrap.json').read_text())
            self.assertEqual(len(bootstrap),12)
            self.assertEqual(bootstrap['LIN_LEFT.last_minus_MULTI191.last']['mean_difference'],.6)


if __name__=='__main__':unittest.main()
