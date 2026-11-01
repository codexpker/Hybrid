"""Tiny NumPy fixtures only; no torch import, official model or real role arrays."""
import unittest
from types import SimpleNamespace
import numpy as np
from hybrid_tsad import weak_lag as wl
from hybrid_tsad.left_normal_runs import IndependentNormalWindows, c2_model_config, normal_exp_class, load_normal_assets


class NormalRunsTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(322)
        fit=[wl.NormalRun('LIN','FIT',i,rng.normal(size=(384,2))+i*20) for i in range(2)]
        select=[wl.NormalRun('LIN','SELECT',i,rng.normal(size=(384,2))+200) for i in range(2)]
        return fit,select,wl.fit_scaler(fit)

    def test_no_cross_run_windows_and_exact_float32_oracle(self):
        fit,select,scaler=self.fixture()
        train=IndependentNormalWindows(fit,scaler,'train')
        val=IndependentNormalWindows(select,scaler,'val')
        self.assertEqual(len(train),2*(384-192+1))
        self.assertEqual(len(val),4)
        for i in range(len(train)):
            fp=train.footprint(i);x,y=train[i]
            self.assertLessEqual(fp['end_exclusive'],384)
            self.assertEqual(fp['end_exclusive']-fp['start'],192)
            ref=((fit[fp['run_id']].values[fp['start']:fp['end_exclusive']]-scaler['mean'])/scaler['scale']).astype(np.float32)
            self.assertTrue(np.array_equal(ref,x))
            self.assertEqual((y.shape,y.dtype),((192,1),np.dtype('float32')))
            self.assertFalse(y.any())
        self.assertEqual(train.footprint(192)['run_id'],0)
        self.assertEqual(train.footprint(193)['run_id'],1)
        self.assertEqual(train.footprint(193)['start'],0)
        self.assertEqual([val.footprint(i)['start'] for i in range(4)],[0,192,0,192])
        # Formal counts are arithmetic assertions, no full pool allocation.
        self.assertEqual(4*(24576-192+1),97540)
        self.assertEqual(2*(24576//192),256)
        self.assertEqual((97540+127)//128,763)
        self.assertEqual(97540%128,4)

    def test_FIT_only_scaler_and_future_outside_window(self):
        fit,select,scaler=self.fixture()
        before=scaler['mean'].copy(),scaler['scale'].copy()
        val=IndependentNormalWindows(select,scaler,'val');x,_=val[0]
        # Transforming SELECT must not fit/refit; strong holdout shift persists.
        self.assertGreater(float(x.mean()),10.)
        self.assertTrue(np.array_equal(before[0],scaler['mean']))
        self.assertTrue(np.array_equal(before[1],scaler['scale']))
        changed=[wl.NormalRun(r.scenario,r.role,r.run_id,r.values.copy()) for r in select]
        changed[0].values[192:]+=1e8
        xx,_=IndependentNormalWindows(changed,scaler,'val')[0]
        self.assertTrue(np.array_equal(x,xx))
        for flag in ('test','init','DEV','CAL','CONFIRM'):
            with self.assertRaises(PermissionError):
                IndependentNormalWindows(select,scaler,flag)
        with self.assertRaises(ValueError):
            IndependentNormalWindows(select,scaler,'val',window=100)
        short=[wl.NormalRun('LIN','SELECT',0,np.ones((160,2)))]
        with self.assertRaises(ValueError):
            IndependentNormalWindows(short,scaler,'val')

    def test_C2_only_config_change_and_inherited_original_control_flow(self):
        config={'seq_len':192,'enc_in':25,'d_model':128,'lambda_max':.3}
        derived,difference=c2_model_config(config)
        self.assertEqual(config['enc_in'],25)
        self.assertEqual(derived,dict(config,enc_in=2))
        self.assertEqual(set(k for k in config if config[k]!=derived[k]),{'enc_in'})
        class OfficialFixture:
            def train(self): pass
            def vali(self,loader): pass
            def _basic(self): pass
            def _adjust_learning_rate(self,*args): pass
            def _select_optimizer(self): pass
        fit,select,scaler=self.fixture()
        cls=normal_exp_class(SimpleNamespace(Exp_Anomaly_Detection=OfficialFixture),fit,select,scaler)
        for name in ('train','vali','_basic','_adjust_learning_rate','_select_optimizer'):
            self.assertIs(getattr(cls,name),getattr(OfficialFixture,name))
        # Creation is not official model instantiation. This dummy only tests
        # permission rejection before torch import, not model/train equivalence.
        instance=cls()
        with self.assertRaises(PermissionError):instance._get_data('test')
        with self.assertRaises(PermissionError):instance.test()

    def test_actual_role_asset_loader_refuses_nontraining_roles_before_read(self):
        for role in ('DEV','CONFIRM','CAL'):
            with self.assertRaises(PermissionError):load_normal_assets([], 'LIN', role)
        with self.assertRaises(ValueError):load_normal_assets([], 'LIN','FIT')

    def test_real_CPU_DataLoader_has_native_batch_shape_and_SELECT_order(self):
        # Torch's CPU DataLoader only; no official module/model/optimizer/CUDA.
        fit,select,scaler=self.fixture()
        class OfficialFixture:pass
        cls=normal_exp_class(SimpleNamespace(Exp_Anomaly_Detection=OfficialFixture),fit,select,scaler)
        instance=cls();instance.train_configs=SimpleNamespace(batch_size=128)
        ds,loader=instance._get_data('val')
        batches=list(loader)
        self.assertEqual(len(batches),1)
        x,y=batches[0]
        np.testing.assert_array_equal(x.numpy(),np.stack([ds[i][0] for i in range(4)]))
        self.assertEqual(tuple(x.shape),(4,192,2));self.assertEqual(tuple(y.shape),(4,192,1))
        ds,loader=instance._get_data('train',1)
        self.assertEqual(len(loader),4)
        self.assertEqual(sum(len(x) for x,y in loader),len(ds))
        self.assertFalse(loader.drop_last)
        with self.assertRaises(ValueError):instance._get_data('val',1)

    def test_formal_train_pending_rejected_before_model_or_arrays(self):
        import tempfile,json
        from pathlib import Path
        from hybrid_tsad.left_normal_runs import _require_train_approval
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'config.json').write_text('{}')
            (p/'approval.json').write_text(json.dumps({'status':'PENDING'}))
            with self.assertRaises(PermissionError):
                _require_train_approval(p/'config.json',{'approval_path':str(p/'approval.json'),'scenario':'LIN'})


if __name__=='__main__':unittest.main()
