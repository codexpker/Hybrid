"""Prepared synthetic CPU tests; no real data, labels or model dependencies."""
import copy,hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import smd_baselines as b,baselines,normal_readout as q,smd_recalibration as store
from hybrid_tsad.hybrid_v1.data import Run
from hybrid_tsad.hybrid_v1.chunked_bank import IDENTITY_KEYS,read_records,read_entries,_fingerprint
from hybrid_tsad.hybrid_v1.cli import score_records

class Baselines(unittest.TestCase):
    def test_same_fixed512_CAL_origins_and_original_fit(self):
        x=np.random.default_rng(99173).normal(size=(3010,3)).astype(np.float32);run=Run('SMD','machine-2-1','machine-2-1_test','NORMAL_CAL',x,np.ones_like(x,bool),13575)
        points=[dict(domain=run.domain,entity=run.entity,run=run.run,role=run.role,index=t,footprint=[t-512,t+32]) for t in range(14087,14599)]
        for method in b.METHODS:
            rows,cal=b.fit_target([run],points,method)
            self.assertEqual([r['index'] for r in rows],list(range(14087,14599)));self.assertEqual(len(rows),512)
            self.assertEqual(rows[0]['footprint'],[13575,14088]);self.assertFalse(cal['source_domain_equal_weight']);self.assertEqual(cal['active_components'],['r_pred'])
            expected=b.fit_calibration(rows,source=False,active_components=['r_pred']);self.assertEqual(cal,expected)
    def test_one_raw_iterator_per_method_and_offline_target_view(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);x=np.repeat(np.arange(6565,dtype=np.float32)[:,None],3,axis=1);x[511]=-0.;obs=np.ones_like(x,bool);x[575]=np.nan;obs[575]=False
            run=Run('SMD','machine-2-1','machine-2-1_test','QUERY',x,obs,17129)
            cal=dict(active_components=['r_pred'],weights={'r_pred':1.},scales={'r_pred':1.},threshold=-1.,source_domain_equal_weight=True,domains=['SMD','Sunspot'])
            for method in b.METHODS:
                calls=[]
                def iterator(*args,**kwargs):calls.append(kwargs['method']);return baselines.iter_baselines(*args,**kwargs)
                ident={k:'a'*64 for k in IDENTITY_KEYS};ident['checkpoint_sha256']=b.algorithm(method);ident['input_runs']=[q.input_binding(run,'e'*64)]
                directory=root/method/'source';b.write_source([run],method,cal,directory,identity=ident,iterator=iterator)
                self.assertEqual(calls,[method]);m=q.read(directory/'manifest.json');self.assertEqual((m['committed_records'],len(m['chunks']),m['chunks'][-1]['records']),(6053,95,37))
                rows=list(read_records(directory));self.assertEqual(rows[-1]['index'],23693)
                for local in (512,575,576,6564):
                    expected=score_records([baselines.baseline_record(run,local,method=method)],cal)[0];got=rows[local-512]
                    self.assertEqual(_fingerprint(expected),_fingerprint({k:got[k] for k in expected}))
                parent=dict(directory=str(directory),manifest_sha256=q.sha(directory/'manifest.json'),identity=ident,ranges=store.RANGES)
                tc=dict(cal,scales={'r_pred':2.},threshold=.01,source_domain_equal_weight=False,domains=['SMD']);destination=dict(directory=str(root/method/'target'),identity=dict(ident),ranges=store.RANGES,active_components=['r_pred'])
                before={p.name:q.sha(p) for p in directory.iterdir() if p.is_file()};store.build_overlay(parent,destination,tc)
                self.assertEqual(calls,[method]);self.assertEqual(before,{p.name:q.sha(p) for p in directory.iterdir() if p.is_file()})
                derived=list(read_entries(directory));self.assertFalse(derived[63]['raw']['valid']);self.assertFalse(derived[64]['derived']['confirmed3_current'])
                self.assertTrue(derived[67 if method=='persistence' else 66]['derived']['confirmed3_current'])
    def test_future_value_changes_do_not_change_issued_prediction(self):
        x=np.arange(1800,dtype=np.float32).reshape(600,3);run=Run('SMD','fixture','q','QUERY',x,np.ones_like(x,bool),17129)
        for method in b.METHODS:
            a=baselines.baseline_record(run,520,method=method);run.values[521:]+=100000
            c=baselines.baseline_record(run,520,method=method);self.assertEqual(_fingerprint(a),_fingerprint(c))
    def test_phase_requires_CPU_permissions_and_exact_method_set(self):
        for methods in (['persistence'],['persistence','other']):
            cfg=dict(operation=b.OPERATION,history=512,methods=methods,permissions=dict(phase='normal_calibrate',labels=False,GPU=False,model=False))
            with self.assertRaises(PermissionError):b.preflight(cfg,'normal_calibrate')
    def test_unknown_method_and_borrowed_model_CAL_rejected(self):
        with self.assertRaises(ValueError):b.algorithm('model')
        with self.assertRaises(PermissionError):b.source_cal(dict(path='/never/model.json',sha256='a'*64),{},'persistence')
    def test_all_missing_CAL_is_not_repaired_or_resampled(self):
        x=np.full((520,1),np.nan,np.float32);run=Run('SMD','fixture','cal','NORMAL_CAL',x,np.zeros_like(x,bool),100)
        p=[dict(domain='SMD',entity='fixture',run='cal',role='NORMAL_CAL',index=612,footprint=[100,644])]
        with self.assertRaises(ValueError):b.fit_target([run],p,'persistence')
    def test_execute_query_source_completes_both_methods(self):
        with tempfile.TemporaryDirectory() as temp:
            x=np.random.default_rng(99173).normal(size=(6565,38)).astype(np.float32)
            run=Run('SMD','machine-2-1','machine-2-1_test','QUERY',x,np.ones_like(x,bool),17129)
            cal=dict(active_components=['r_pred'],weights={'r_pred':1.},scales={'r_pred':1.},threshold=.5,source_domain_equal_weight=True,domains=['SMD','Sunspot'])
            cfg=dict(operation=b.OPERATION,output=str(Path(temp)/'output'),history=512,methods=list(b.METHODS),execution_config_sha256='a'*64,source_sha256={'synthetic':'fixture'},
                assets=dict(query_manifest=dict(path='/synthetic/query_manifest.json',sha256='b'*64),source_acceptance=dict(path='/synthetic/source_acceptance.json',sha256='c'*64),
                    **{method+'_source_calibration':dict(path='/synthetic/'+method+'.json',sha256=b.SOURCE_CAL[method]) for method in b.METHODS}))
            # Only admission/file loading is injected; execute, both original
            # NumPy baseline loops, bank serialization and receipts are real.
            with patch.object(b,'preflight'),patch('hybrid_tsad.hybrid_v1.data.load_manifest',return_value=[run]),\
                 patch.object(b.storage,'metadata',return_value={'runs':[{'sha256':'e'*64}]}),patch.object(b,'source_cal',return_value=cal):
                receipt=b.execute(cfg,'query_source')
            self.assertEqual(set(receipt['methods']),set(b.METHODS))
            for method in b.METHODS:
                bank=Path(cfg['output'])/method/'bank';m=q.read(bank/'manifest.json')
                self.assertTrue(m['complete']);self.assertEqual((m['committed_records'],len(m['chunks']),m['chunks'][-1]['records']),(6053,95,37))
                self.assertEqual(m['contract']['identity']['input_runs'][0]['sha256'],'e'*64)
                self.assertEqual(m['contract']['identity']['checkpoint_sha256'],b.algorithm(method))
                self.assertEqual(receipt['methods'][method]['manifest_sha256'],q.sha(bank/'manifest.json'))
                rows=list(read_records(bank));self.assertEqual((rows[0]['index'],rows[-1]['index']),(17641,23693));self.assertEqual(rows[-1]['method'],method)

if __name__=='__main__':unittest.main()
