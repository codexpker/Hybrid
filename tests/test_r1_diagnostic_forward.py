"""NumPy/schema-only tests: no Torch, real arrays/checkpoints or forward."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import unittest
import numpy as np
from hybrid_tsad.hybrid_v1 import r1_diagnostic_forward as r


def fake():
    stats=dict(mean=np.zeros((1,1,2)),scale=np.full((1,1,2),2.),
               channel_valid=np.ones((1,1,2),bool),visible_count=np.full((1,1,2),4))
    f=dict(prediction=np.zeros((1,2,2)),valid=np.ones((1,2,2),bool),stats=stats,
           t_features=np.zeros((1,2,3)),f_features=np.zeros((1,2,3)),copy_masks=np.zeros((1,1,2),bool))
    recon=dict(stats=copy.deepcopy(stats),hidden_mask=np.zeros((1,2,4),bool),visible_mask=np.ones((1,2,4),bool),
               transform_hidden_mask=np.zeros((1,2,2),bool),observed_mask=np.ones((1,4,2),bool))
    loss=dict(parts={k:np.array(4.) for k in ('tf','ms','pred','consistency')},
              valid_counts={k:np.array([2]) for k in ('tf','ms','pred','consistency')},total=np.array(12.4))
    return dict(forecast=f,reconstruction=recon,loss=loss,
                batch=dict(future_observed=np.ones((1,2,2),bool)),raw_residuals=dict(r_pred=np.array([1.])))


class R1ForwardTests(unittest.TestCase):
    def test_distillation_is_not_loss_difference(self):
        s=fake();u=copy.deepcopy(s);u['forecast']['prediction'][:]=4.
        u['loss']['parts']['pred']=np.array(1.);u['loss']['total']=np.array(9.4)
        self.assertEqual(r.discrepancy(s,u)['D_pred_teacher_scale_MSE'],4.)
        self.assertEqual(r.loss_delta(s,u)['parts']['pred']['student_minus_source'],-3.)
        self.assertEqual(r.discrepancy(s,s)['D_pred_teacher_scale_MSE'],0.)

    def test_mask_or_stats_difference_fails_before_comparison(self):
        s=fake();u=copy.deepcopy(s);u['reconstruction']['hidden_mask'][0,0,0]=True
        with self.assertRaises(ValueError):r.discrepancy(s,u)
        u=copy.deepcopy(s);u['forecast']['stats']['scale'][0,0,0]=3.
        with self.assertRaises(ValueError):r.discrepancy(s,u)

    def test_zero_support_is_NA_not_zero_error(self):
        s=fake();u=copy.deepcopy(s);s['forecast']['valid'][:]=False;u['forecast']['valid'][:]=False
        self.assertIsNone(r.discrepancy(s,u)['D_pred_teacher_scale_MSE'])
        s['loss']['valid_counts']['tf'][:]=0
        self.assertIsNone(r.loss_delta(s,u)['parts']['tf']['student_minus_source'])

    def test_full544_geometry_and_no_labels(self):
        u=dict(history=512,horizon=32,runtime_role='R1_DIAGNOSTIC_HOLDOUT',target=1280,
               footprint=[768,1312],mask_seed=5,encoder_asset=dict(role='DIAGNOSTIC_HOLDOUT',expected_shape=[3072,3]))
        r.validate_unit(u)
        u['footprint']=[769,1312]
        with self.assertRaises(PermissionError):r.validate_unit(u)
        u['footprint']=[768,1312];u['label']=1
        with self.assertRaises(PermissionError):r.validate_unit(u)

    def test_numeric_scalar_array_serialization_and_noTorch(self):
        value=r.loss_delta(fake(),fake())
        restored=json.loads(json.dumps(r.jsonable(value),allow_nan=False))
        self.assertEqual(restored['source_total'],12.4)
        self.assertEqual(restored['source_raw_residuals']['r_pred'],[1.])
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main(verbosity=2)
