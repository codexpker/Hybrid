"""Pure tiny synthetic CSV/missingness/index fixtures, no real data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import target_data as t


class TargetPreparation(unittest.TestCase):
    def test_preview_and_pending_do_not_open_raw(self):
        with patch('pathlib.Path.open',side_effect=AssertionError('No files')),contextlib.redirect_stdout(io.StringIO()):t.main(['--config','missing'])
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'cfg';f.write_text('{}')
            with self.assertRaises(PermissionError):t.admit(f,None,None)

    def test_roles_unique_draw_order_and_global_rng_unchanged(self):
        observed=np.ones((60,2),bool);roles=dict(FIT=[0,20],SELECT=[24,40],CAL=[44,60]);before=np.random.get_state()
        plan,counts=t.fixed_positions(observed,roles,history=4,horizon=2,counts=dict(SELECT=2,CAL=3),seed=2025,identity=dict(domain='S',entity='E',run='R'))
        after=np.random.get_state();np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(counts,dict(SELECT=11,CAL=12))
        rng=np.random.default_rng(2025)
        for role,candidates,n,p in [('SELECT',np.arange(28,39),2,2),('CAL',np.arange(48,60),3,1)]:
            self.assertEqual([e['index'] for e in plan[role]],rng.choice(candidates,size=n,replace=False).tolist())
            self.assertEqual(len({e['index'] for e in plan[role]}),n)
            self.assertTrue(all(roles[role][0]<=e['footprint'][0] and e['footprint'][1]<=roles[role][1] for e in plan[role]))

    def test_missing_teacher_and_overlap_reject_no_fallback(self):
        observed=np.ones((60,2),bool);observed[44:]=False;roles=dict(FIT=[0,20],SELECT=[24,40],CAL=[44,60])
        with self.assertRaises(ValueError):t.fixed_positions(observed,roles,history=4,horizon=2,counts=dict(SELECT=2,CAL=3),seed=2025,identity={})
        with self.assertRaises(ValueError):t.fixed_positions(np.ones((60,2),bool),dict(roles,SELECT=[19,40]),history=4,horizon=2,counts=dict(SELECT=2,CAL=3),seed=2025,identity={})

    def test_partial_current_channel_missing_is_not_a_legal_position(self):
        observed=np.ones((60,2),bool);observed[48,1]=False
        roles=dict(FIT=[0,20],SELECT=[24,40],CAL=[44,60])
        plan,counts=t.fixed_positions(observed,roles,history=4,horizon=2,
            counts=dict(SELECT=2,CAL=11),seed=2025,identity={})
        # Channel0 is sufficient by itself, but channel1 has no current GT.
        self.assertEqual(counts['CAL'],11)
        self.assertNotIn(48,[e['index'] for e in plan['CAL']])
        self.assertTrue(all(observed[e['index']].all() for e in plan['CAL']))

    def test_one_channel_lacks_two_conservative_past_observations(self):
        observed=np.ones((60,2),bool);observed[45:47,1]=False
        roles=dict(FIT=[0,20],SELECT=[24,40],CAL=[44,60])
        self.assertTrue(observed[48].all())
        self.assertEqual(observed[45:48,1].sum(),1)  # [t-H+1,t), not widened
        plan,counts=t.fixed_positions(observed,roles,history=4,horizon=2,
            counts=dict(SELECT=2,CAL=11),seed=2025,identity={})
        self.assertEqual(counts['CAL'],11)
        self.assertNotIn(48,[e['index'] for e in plan['CAL']])
        self.assertTrue(all((observed[e['index']-3:e['index']].sum(0)>=2).all() for e in plan['CAL']))

    def test_csv_loader_parses_once_keeps_nan_and_never_guesses_zero_mask(self):
        from hybrid_tsad.hybrid_v1.data import load_manifest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);csv=root/'synthetic.csv';values=np.arange(120,dtype=np.float64).reshape(60,2);values[3,0]=np.nan;values[4:7,1]=0.
            np.savetxt(csv,np.column_stack([np.arange(60),values]),delimiter=',',header='timestamp_(min),feature_0,feature_1',comments='')
            original=t.sha(csv);cfg=dict(raw_csv=dict(path=str(csv),sha256=original,format='csv',has_header=True,columns=['feature_0','feature_1'],expected_shape=[60,2]),output=str(root/'derived'),timestamp_column='timestamp_(min)',domain='S',entity='E',run='R',roles=dict(FIT=[0,20],SELECT=[24,40],CAL=[44,60]),history=4,horizon=2,selection_counts=dict(SELECT=2,CAL=3),sampling_seed=2025)
            with patch('numpy.genfromtxt',wraps=np.genfromtxt) as parsed:report=t.derive(cfg)
            self.assertEqual(parsed.call_count,1);self.assertEqual(t.sha(csv),original)
            with np.load(root/'derived/values_observed_original_axis.npz',allow_pickle=False) as a:
                self.assertEqual(a['values'].dtype,np.float32);self.assertTrue(np.isnan(a['values'][3,0]));self.assertFalse(a['observed'][3,0])
                self.assertTrue(a['observed'][4:7,1].all());self.assertEqual(a['original_index'].tolist(),list(range(60)))
            runs=load_manifest(root/'derived/target_role_manifest.json')
            self.assertEqual([(r.role,r.start,len(r.values)) for r in runs],[('FIT',0,20),('SELECT',24,16),('CAL',44,16)])
            self.assertFalse(report['zero_repeat_or_LOCF_mask_guess'])


if __name__=='__main__':unittest.main()
