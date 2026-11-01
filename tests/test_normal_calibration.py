"""Synthetic-only tests: no real normal arrays, model, checkpoint or Torch."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.normal_b1_readout import (calibrate_normal_pool,score_identity,
    require_runtime_release,validate_remote_paths,SEED_DEVICES)


class NormalCalibration(unittest.TestCase):
    def banks(self,seed=2025):
        result=[]
        for i,run in enumerate(range(81,101)):
            meta=dict(domain='TEP_REACTOR',role='CAL',purpose='NORMAL_CALIBRATION',run_id=run,
                fault_number=0,seed=seed,columns=['xmeas_9','xmeas_21','xmv_10'],window=100,batch_size=1,points=500,
                array_sha256='a'*64,scaler_sha256='b'*64,checkpoint_sha256='c'*64,feature_source_sha256='d'*64,
                best_checkpoint_step=1786 if seed==2026 else 1880)
            v=np.arange(401,dtype=np.float32)+401*i
            result.append(dict(metadata=meta,last=v,mean32=v*.5,end_index_0based=np.arange(99,500)))
        return result

    def test8020_rank7941_and_endpoint_reset(self):
        dist,report=calibrate_normal_pool(self.banks(),seed=2025,checkpoint_sha256='c'*64,scaler_sha256='b'*64)
        self.assertEqual((report['n'],report['rank_1based']),(8020,7941))
        self.assertEqual(report['thresholds'],{'last':7940.,'mean32':3970.})
        self.assertEqual(report['observed_CAL_point_exceedance']['last'],79)
        np.testing.assert_array_equal(dist['end_index_0based'].reshape(20,401),np.tile(np.arange(99,500),(20,1)))
        self.assertEqual((dist['run_id'][400],dist['run_id'][401]),(81,82))

    def test_threshold_ties_use_strict_greater(self):
        banks=self.banks()
        for b in banks:b['last'][:]=5;b['mean32'][:]=2
        _,r=calibrate_normal_pool(banks,seed=2025,checkpoint_sha256='c'*64,scaler_sha256='b'*64)
        self.assertEqual(r['thresholds'],{'last':5.,'mean32':2.})
        self.assertEqual(r['observed_CAL_point_exceedance'],{'last':0,'mean32':0})

    def test_mixed_seed_weights_scaler_or_extractor_rejected(self):
        for k,v in [('seed',2026),('checkpoint_sha256','e'*64),('scaler_sha256','e'*64),('feature_source_sha256','e'*64)]:
            banks=self.banks();banks[5]['metadata'][k]=v
            with self.assertRaises(PermissionError):calibrate_normal_pool(banks,seed=2025,checkpoint_sha256='c'*64,scaler_sha256='b'*64)

    def test_role_and_complete_run_order_are_fixed(self):
        banks=self.banks();banks[0]['metadata']['role']='SELECT'
        with self.assertRaises(PermissionError):calibrate_normal_pool(banks,seed=2025,checkpoint_sha256='c'*64,scaler_sha256='b'*64)
        banks=self.banks();banks[0]['metadata']['run_id']=82
        with self.assertRaises(ValueError):calibrate_normal_pool(banks,seed=2025,checkpoint_sha256='c'*64,scaler_sha256='b'*64)
        banks=self.banks();banks[0]['end_index_0based']=np.arange(98,499)
        with self.assertRaises(ValueError):calibrate_normal_pool(banks,seed=2025,checkpoint_sha256='c'*64,scaler_sha256='b'*64)

    def test_each_predeclared_seed_retains_its_own_pool(self):
        for seed in SEED_DEVICES:
            _,r=calibrate_normal_pool(self.banks(seed),seed=seed,checkpoint_sha256='c'*64,scaler_sha256='b'*64)
            self.assertEqual(r['seed'],seed)

    def test_wrong_seed_device_or_run_release_fails_before_file(self):
        cfg=dict(operation='LEFT_TEP_NORMAL_CALIBRATION',seed=2026,run_ids=list(range(81,101)),
                 max_seconds=300,infer_calls=8020,physical_gpu_index=0,expected_gpu_uuid=SEED_DEVICES[2026][1],
                 quantile_rank_1based=7941,threshold_comparison='strict_greater')
        with patch('hybrid_tsad.normal_b1_readout.read_json',side_effect=AssertionError('No release IO')) as read:
            with self.assertRaises(PermissionError):require_runtime_release(cfg,'not-opened','not-opened')
            read.assert_not_called()


if __name__=='__main__':unittest.main()
