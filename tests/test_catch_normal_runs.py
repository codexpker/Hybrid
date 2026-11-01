"""CPU tiny fixtures only; original fit flow is stopped before optimizer creation."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hybrid_tsad import catch_baseline as cb
from hybrid_tsad.catch_normal_runs import independent_data,production_layout,PARAMS,observed_native_step
from hybrid_tsad.weak_lag import NormalRun


class CatchNormalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'CATCH_SOURCE_ROOT' not in os.environ:raise RuntimeError('Fixed external source required; no skip')
        cls.api=cb.external_api(Path(os.environ['CATCH_SOURCE_ROOT']))

    def fixture(self,order='C'):
        rng=np.random.default_rng(911)
        fit=[NormalRun('LIN','FIT',i,np.array(rng.normal(size=(384,2))+10*i,order=order)) for i in range(4)]
        select=[NormalRun('LIN','SELECT',i,np.array(rng.normal(size=(384,2))+100,order=order)) for i in range(2)]
        return fit,select

    def data_calls(self,detector,fit_frame,select,trace):
        a,b=self.api.train_val_split(fit_frame,.8,None)
        detector.scaler.fit(a.values)
        for mode,df in [('train',a),('val',b)]:
            transformed=pd.DataFrame(detector.scaler.transform(df.values),columns=df.columns,index=df.index)
            yield mode,self.api.anomaly_detection_data_provider(transformed,128,192,1,mode)

    def test_exact_shared_windows_scaler_and_no_run_join(self):
        for order in ['C','F']:
            fit,select=self.fixture(order);det=SimpleNamespace(scaler=StandardScaler());trace=[]
            with independent_data(self.api,det,fit,select,trace) as (frame,datasets):
                loaders=dict(self.data_calls(det,frame,select,trace))
                self.assertEqual(len(datasets['train']),4*193)
                self.assertEqual(len(datasets['val']),4)
                self.assertEqual(len(loaders['val']),1)
                for mode,runs in [('train',fit),('val',select)]:
                    ds=datasets[mode]
                    for i in range(len(ds)):
                        fp=ds.footprint(i);x,_=ds[i]
                        self.assertLessEqual(fp['end_exclusive'],384)
                        self.assertEqual(fp['end_exclusive']-fp['start'],192)
                        run=runs[fp['run_id']]
                        expected=det.scaler.transform(run.values)[fp['start']:fp['end_exclusive']].astype(np.float32)
                        np.testing.assert_array_equal(x,expected)
                    self.assertFalse(loaders[mode].drop_last)
                expected=StandardScaler().fit(np.concatenate([r.values for r in fit]))
                np.testing.assert_array_equal(det.scaler.mean_,expected.mean_)
                np.testing.assert_array_equal(det.scaler.scale_,expected.scale_)

    def test_select_cannot_change_fit_or_scaler(self):
        fit,select=self.fixture();means=[];inputs=[]
        for scale in [1.,-200.]:
            other=[NormalRun(r.scenario,r.role,r.run_id,r.values*scale) for r in select]
            det=SimpleNamespace(scaler=StandardScaler())
            with independent_data(self.api,det,fit,other,[]) as (frame,datasets):
                list(self.data_calls(det,frame,other,[]));means.append(det.scaler.mean_.copy());inputs.append(datasets['train'][0][0])
        np.testing.assert_array_equal(means[0],means[1]);np.testing.assert_array_equal(inputs[0],inputs[1])

    def test_permissions_and_finally_restoration(self):
        fit,select=self.fixture();a,b=self.api.train_val_split,self.api.anomaly_detection_data_provider
        with self.assertRaisesRegex(RuntimeError,'fixture abort'):
            with independent_data(self.api,SimpleNamespace(scaler=StandardScaler()),fit,select,[]) as (frame,_):
                with self.assertRaises(PermissionError):self.api.train_val_split(frame,.9,None)
                with self.assertRaises(PermissionError):self.api.anomaly_detection_data_provider(frame,128,192,1,'thre')
                with self.assertRaises(RuntimeError):
                    with independent_data(self.api,SimpleNamespace(),fit,select,[]):pass
                raise RuntimeError('fixture abort')
        self.assertIs(self.api.train_val_split,a);self.assertIs(self.api.anomaly_detection_data_provider,b)
        with self.assertRaises(ValueError):
            bad=[NormalRun('LIN','DEV',0,select[0].values)]
            with independent_data(self.api,SimpleNamespace(),fit,bad,[]):pass

    def test_real_native_detect_fit_reaches_hooks_before_optimizer(self):
        class StopBeforeOptimizer(Exception):pass
        class FakeModel:
            def __init__(self,*a):pass
            def to(self,*a):return self
            def parameters(self):return iter(())
            def named_parameters(self):raise StopBeforeOptimizer()
        original=self.api.CATCHModel
        split_original=self.api.train_val_split
        fit,select=self.fixture();trace=[]
        try:
            self.api.CATCHModel=FakeModel
            det=self.api.CATCH(**PARAMS)
            with self.assertRaises(StopBeforeOptimizer):
                with independent_data(self.api,det,fit,select,trace) as (frame,datasets):
                    det.detect_fit(frame,None)
            self.assertEqual(det.scaler.n_samples_seen_,1536)
            self.assertEqual([x['event'] for x in trace],['split_override','provider_override','provider_override'])
            self.assertEqual([x.get('mode') for x in trace[1:]],['val','train'])
            self.assertFalse(hasattr(det,'optimizer'))
        finally:self.api.CATCHModel=original
        self.assertIs(self.api.train_val_split,split_original)

    def test_production_layout_is_predeclared_not_performance_selected(self):
        x=production_layout()
        self.assertEqual(4*(24576-192+1),x['fit_windows'])
        self.assertEqual((x['fit_batches'],x['fit_last_batch'],x['select_windows']), (763,4,256))
        self.assertEqual((x['mask_interval'],x['mask_calls_if_complete'],x['main_steps_if_complete']), (76,30,2289))
        self.assertEqual(PARAMS['num_epochs'],3);self.assertFalse(x['equal_optimization_budget'])

    def test_observer_calls_native_once_and_preserves_math_rng(self):
        import torch
        p=torch.nn.Parameter(torch.tensor([.5,-.25]));q=torch.nn.Parameter(p.detach().clone())
        a=torch.optim.Adam([p],lr=.005);b=torch.optim.Adam([q],lr=.005)
        p.grad=torch.tensor([.3,-.4]);q.grad=p.grad.clone();state=torch.get_rng_state().clone()
        calls=[];native=torch.optim.Adam.step
        def original(opt,*args,**kwargs):calls.append(1);return native(opt,*args,**kwargs)
        _,stats=observed_native_step(a,original);native(b)
        self.assertEqual(calls,[1]);self.assertTrue(torch.equal(p,q))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertGreater(stats['parameter_delta_abs_max'],0)
        self.assertAlmostEqual(stats['gradient_l2_norm'],.5,places=6)
        self.assertAlmostEqual(stats['gradient_abs_max'],.4,places=6)
        self.assertEqual(stats['gradient_tensors'],1)
        p.grad=None;_,empty=observed_native_step(a,native)
        self.assertEqual(empty['parameter_delta_abs_max'],0.)
        self.assertEqual(empty['gradient_l2_norm'],0.)
        p.grad=torch.tensor([float('nan'),0.])
        with self.assertRaises(AssertionError):observed_native_step(a,native)


if __name__=='__main__':unittest.main()
