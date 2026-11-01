"""Only synthetic NumPy CPU fixtures. Never import torch or official model."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from hybrid_tsad import left_normal_scores as s


def ref(role='CAL', i=0, variant='N', scene='LIN'):
    return dict(path=f'/synthetic/data/{scene}/{role}/run{i:02d}_{variant}.npy',
                scenario=scene, role=role, run_id=i, variant=variant, sha256='a'*64)


class FixedScoreTests(unittest.TestCase):
    def test_indices_cover_whole_run_and_availability(self):
        c = s.coordinates()
        self.assertTrue(np.array_equal(c['absolute_run_index'].ravel(), np.arange(24576)))
        self.assertTrue(np.array_equal(c['endpoint'], np.arange(191, 24576, 192)))
        self.assertTrue(np.array_equal(c['window_end_exclusive'][:-1], c['window_start'][1:]))
        self.assertTrue((c['batch_input_end'] == 24575).all())

    def test_exact_numpy_reduction_and_y_diagnostic(self):
        raw = np.random.default_rng(2025).normal(size=(128, 192, 2)).astype('float32')
        got = s.reduce_raw(raw)
        oracle = np.mean(np.array(np.concatenate([raw], axis=0).reshape(-1, 2)), axis=-1)
        self.assertTrue(np.array_equal(got['last'], oracle.reshape(128,192)[:, -1]))
        self.assertTrue(np.array_equal(got['mean128'], raw[:,64:,:].mean(axis=(1,2))))
        self.assertTrue(np.array_equal(got['y_last'], raw[:,-1,1]))
        self.assertTrue(np.array_equal(got['y_mean128'], raw[:,64:,1].mean(axis=1)))
        for bad in (raw.astype('float64'), raw[:127], np.full_like(raw,np.nan)):
            with self.assertRaises(ValueError): s.reduce_raw(bad)
        core=np.arange(128)%2==0
        sham={k:v.copy() for k,v in got.items()}
        sham['channel_scores'][~core]+=1
        self.assertEqual(s.control_equivalence(got,got,sham,core)['core_windows'],64)
        sham['last'][0]+=1
        with self.assertRaises(ValueError): s.control_equivalence(got,got,sham,core)

    def test_feature_permissions_rejected_before_io(self):
        for role, variant, scene in [('CAL','A','LIN'), ('CONFIRM','N','LIN'),
                                     ('FIT','N','LIN'), ('DEV','A','IRR')]:
            with self.assertRaises(PermissionError): s.load_feature(ref(role,variant=variant,scene=scene),scene,role)
        r=ref();r['path']='/synthetic/data/LIN/CAL/run00_truth.json'
        with self.assertRaises(PermissionError): s.load_feature(r,'LIN','CAL')
        r=ref();r['role']='DEV'
        with self.assertRaises(PermissionError): s.load_feature(r,'LIN','CAL')

    def test_complete_banks_and_cal_dev_isolation(self):
        records=[dict(reference=ref(i=i), last=np.arange(128)+i*128,
                      mean128=np.arange(128)+i*128+20) for i in range(8)]
        got=s.calibrate_frozen(records,'LIN')
        self.assertEqual(got['last']['threshold'],1014.)
        self.assertEqual(got['last']['order_1based'],1015)
        self.assertEqual(got['mean128']['threshold'],1034.)
        self.assertEqual((np.array([1014.,1015.]) > got['last']['threshold']).tolist(),[False,True])
        with self.assertRaises(ValueError): s.calibrate_frozen(records[:-1],'LIN')
        records[0]['reference']=ref('DEV',variant='N')
        with self.assertRaises(PermissionError): s.calibrate_frozen(records,'LIN')

    def test_fit_scaler_and_no_refit(self):
        scaler=dict(fit_ids=[['LIN','FIT',i] for i in range(4)], scenario='LIN',
                    ddof=0, fit_points=98304, SELECT_used_for_fit=False,
                    mean=[3.,4.],scale=[2.,5.])
        values=np.arange(24576*2,dtype='float64').reshape(24576,2)
        got=s.fixed_windows(values,scaler,'LIN')
        expected=((values-[3.,4.])/[2.,5.]).astype('float32').reshape(128,192,2)
        self.assertTrue(np.array_equal(got,expected))
        with self.assertRaises(PermissionError): s.fixed_windows(values,scaler,'WEAK')
        scaler['fit_ids'][0][1]='CAL'
        with self.assertRaises(PermissionError): s.fixed_windows(values,scaler,'LIN')

    def test_pending_release_and_truth_gate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'release.json';p.write_text(json.dumps({'status':'PENDING'}))
            with self.assertRaises(PermissionError): s.require_evaluation_seal(p)
            cfg=Path(d)/'cfg.json';cfg.write_text('{}')
            with self.assertRaises(PermissionError): s.require_score_release(cfg,p)

    def test_checkpoint_missing_scientific_keys_rejected_without_model(self):
        # A shape-free state interface stub; no torch, official class or weights.
        class StateStub:
            def state_dict(self): return {'science':None}
            def load_state_dict(self,state,strict=False):
                from types import SimpleNamespace
                return SimpleNamespace(missing_keys=[] if 'science' in state else ['science'],
                                       unexpected_keys=list(set(state)-{'science'}))
        m=StateStub()
        self.assertEqual(s.load_c2_checkpoint(m,{'science':1,'layer.total_ops':0},['layer.total_ops'])['missing_keys'],[])
        self.assertEqual(s.load_c2_checkpoint(m,{'science':1,'total_ops':0},['total_ops'])['missing_keys'],[])
        with self.assertRaises(ValueError): s.load_c2_checkpoint(m,{'layer.total_ops':0},['layer.total_ops'])
        with self.assertRaises(ValueError): s.load_c2_checkpoint(m,{'science':1,'evil':0},['layer.total_ops'])

    def test_evaluation_requires_seal_before_touching_truth(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'release.json';p.write_text(json.dumps({'status':'PENDING'}))
            # Invalid truth is intentionally never accessed before permission fails.
            with self.assertRaises(PermissionError):
                s.evaluate_fixed({},None,'A',{},seal_path=p)

    def test_common_eight_run_bootstrap_preserves_na_and_fixed_comparisons(self):
        ap={(scene,m,r,i): .5+i/100 for scene in ('LIN','WEAK')
            for m in ('LEFT','MULTI191','PEAK191','SYNC191','MULTI64','SYNC64')
            for r in ('last','mean128') for i in range(8)}
        ap[('WEAK','LEFT','mean128',7)]=None
        result,draws=s.paired_core_ap_comparisons(ap)
        self.assertEqual(len(result),12)
        self.assertEqual(draws.shape,(10000,8))
        self.assertTrue(((draws>=0)&(draws<8)).all())
        self.assertEqual(result['WEAK_LEFT.mean128_minus_LEFT.last']['valid_pairs'],7)
        self.assertEqual(result['LIN_LEFT.last_minus_PEAK191.last']['mean_difference'],0.)


if __name__ == '__main__': unittest.main()
