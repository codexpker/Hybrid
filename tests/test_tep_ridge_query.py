import copy,unittest
from unittest.mock import patch
import numpy as np
import test_tep_ridge_readout as fixture_module
from hybrid_tsad.tep_ridge_readout import score_run
from hybrid_tsad.tep_ridge_query import score_values,require_release,METHODS

class RidgeQuery(unittest.TestCase):
    def test500_original_all_arrays_bitexact(self):
        selection,weights,scaler=fixture_module.RidgeReadout().fixture()
        x=np.arange(1500,dtype=np.float64).reshape(500,3)/1500
        a=score_values(x,scaler,selection,weights);b=score_run(x,scaler,selection,weights,role='CAL',run_id=81)
        self.assertEqual(set(a),set(b))
        for k in b:np.testing.assert_array_equal(a[k],b[k])
    def test960_complete_coordinates_and_first_mean(self):
        s,w,c=fixture_module.RidgeReadout().fixture();x=np.arange(2880,dtype=np.float64).reshape(960,3)
        a=score_values(x,c,s,w)
        self.assertEqual(a['SELF_prediction_z'].shape,(896,3))
        np.testing.assert_array_equal(a['prediction_t'],np.arange(64,960))
        for m in METHODS:
            self.assertEqual(a[m+'_last'].shape,(861,));np.testing.assert_array_equal(a[m+'_end_index_0based'],np.arange(99,960))
        self.assertEqual(a['ZMAX_mean32'][0],a['ZMAX_point_score'][68:100].mean())
        self.assertEqual(a['SELF_mean32'][0],a['SELF_point_score'][4:36].mean())
    def test_past_prediction_does_not_change_with_future_input(self):
        s,w,c=fixture_module.RidgeReadout().fixture();x=np.arange(2880,dtype=np.float64).reshape(960,3);y=x.copy();y[700:]+=500
        a=score_values(x,c,s,w);b=score_values(y,c,s,w)
        for m in METHODS:np.testing.assert_array_equal(a[m+'_point_score'][:(700-64 if m!='ZMAX' else 700)],b[m+'_point_score'][:(700-64 if m!='ZMAX' else 700)])
        bad=copy.deepcopy(s);bad['features']['SELF_target0'][0]['lag']=0
        with self.assertRaises((PermissionError,ValueError)):score_values(x,c,bad,w)
    def test_pending_both_acceptances_refuses_before_release_file(self):
        with patch('hybrid_tsad.tep_ridge_query.read_json',side_effect=AssertionError('No refs')) as r:
            with self.assertRaises(PermissionError):require_release('no','no',dict(binding_state='PENDING_INPUT_AND_CAL_ACCEPTANCE',input_acceptance=None,calibration_acceptance=None))
            r.assert_not_called()

if __name__=='__main__':unittest.main()
