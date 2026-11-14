"""New pure metadata/NumPy fixtures only; real Torch gates run on owned GPU."""
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from types import SimpleNamespace
import unittest
import numpy as np
from hybrid_tsad.hybrid_v1 import r2_learnability as r2
from hybrid_tsad.hybrid_v1.config import HybridV1Config


class Parameter:
    def __init__(self):self.requires_grad=True
    def requires_grad_(self,value):self.requires_grad=value


class FakeModel:
    def __init__(self):
        self.config=HybridV1Config();self.lag=SimpleNamespace(config=self.config)
        self.other=SimpleNamespace(config=self.config)
        self.params={k:Parameter() for k in ('lag.query.weight','fusion.weight','temporal.weight','tf_backbone.weight','ms_filter.weight')}
    def modules(self):return [self,self.lag,self.other]
    def named_parameters(self):return self.params.items()


class Tensor:
    def __init__(self,x):self.x=np.asarray(x)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.x


class Tests(unittest.TestCase):
    def test_common_plan_1000_32_512_full544_role_isolation(self):
        runs=[dict(domain='d',entity='e',run='run'+str(i),role=role) for i,role in enumerate(('FIT','SELECT','CAL'))]
        p=r2.fixed_plan(runs)
        self.assertEqual([len(p[k]) for k in ('FIT','SELECT','CAL')],[1000,32,512])
        self.assertEqual(p,r2.fixed_plan(runs))
        for role in ('FIT','SELECT','CAL'):
            for row in p[role]:
                self.assertEqual(row['footprint'],[row['index']-512,row['index']+(1 if role=='CAL' else 32)])
                self.assertGreaterEqual(row['footprint'][0],0);self.assertLessEqual(row['footprint'][1],2560)
                self.assertLessEqual(row['index']+32,2560)
            if role!='FIT':self.assertEqual(len({r['index'] for r in p[role]}),len(p[role]))
        self.assertTrue(set(r['run'] for r in p['FIT']).isdisjoint(r['run'] for r in p['SELECT']))

    def test_cross_off_both_config_references_and_self_retained(self):
        source=FakeModel();m,names=r2.clone_for_arm(source,'CROSS_OFF_LAG_U')
        self.assertIs(m.config,m.lag.config);self.assertIs(m.config,m.other.config)
        self.assertFalse(m.config.cross_lag_enabled);self.assertTrue(m.config.self_lag_enabled)
        self.assertTrue(source.config.cross_lag_enabled)
        self.assertTrue(m.params['lag.query.weight'].requires_grad)
        self.assertFalse(m.params['temporal.weight'].requires_grad)
        self.assertFalse(m.params['tf_backbone.weight'].requires_grad)
        self.assertEqual(set(names),{'lag.query.weight','fusion.weight'})
        u,_=r2.clone_for_arm(source,'U');self.assertFalse(u.params['lag.query.weight'].requires_grad)

    def test_arm_state_allowlist_not_old_U_only(self):
        a={'lag.query.weight':1,'fusion.weight':1,'temporal.weight':1}
        b=dict(a,**{'lag.query.weight':2})
        self.assertTrue(r2.state_change(a,b,'LAG_U')['passed'])
        self.assertFalse(r2.state_change(a,b,'U')['passed'])
        self.assertFalse(r2.state_change(a,dict(b,**{'temporal.weight':2}),'LAG_U')['passed'])

    def test_QK_signed_xy_coefficient_observation(self):
        q=SimpleNamespace(weight=Tensor([[1.],[2.]]),bias=Tensor([.1,.2]))
        k=SimpleNamespace(weight=Tensor([[-2.],[1.]]),bias=Tensor([.3,.4]))
        s=r2.qk_signature(SimpleNamespace(lag=SimpleNamespace(query=q,key=k)))
        self.assertEqual(s['raw_xy_coefficient'],0.)
        self.assertTrue(np.array_equal(s['query_bias'],[.1,.2]))

    def test_scaled_generator_fixture_independent_null_and_ramp(self):
        x,_,_=r2.build_query_inputs(n=160,burn=64,query_run=901,independent_run=902,role='UNIT_R2_MATCHED',event=(64,96))
        self.assertEqual(set(x),set(r2.CASES))
        n=x['CROSS_N']['values'];null=x['NO_DRIVER0_COUPLING']['values']
        self.assertTrue(np.array_equal(n[:,0],null[:,0]))
        self.assertFalse(np.array_equal(n[:,1:],null[:,1:]))
        self.assertTrue(np.array_equal(x['NORMAL_DRIFT']['values'],(n.astype(np.float64)+np.arange(160)[:,None]/159).astype(np.float32)))
        self.assertTrue(np.array_equal(x['CROSS_LAG12_20']['values'][:64],n[:64]))

    def test_original_AR_self_ARX12_fit_predict_no_future(self):
        original=Path(__file__).resolve().parents[2]/'notes/paper_convergence_20261105/R2_bounded_mechanism_v1/run_cpu_controls.py'
        helper=r2.load_original_controls(original)
        x=np.random.default_rng(91888).normal(size=(2560,3))
        t=np.arange(512,1512);models=r2.fit_controls(helper,x,t)
        a=r2.predict_controls(helper,models,x,np.array([1512]))
        altered=x.copy();altered[1512:]+=99
        b=r2.predict_controls(helper,models,altered,np.array([1512]))
        self.assertEqual(set(a),{'AR_SELF','ARX12'})
        for key in a:self.assertTrue(np.array_equal(a[key],b[key]))
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main(verbosity=2)
