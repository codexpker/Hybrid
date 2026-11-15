import unittest
import numpy as np
import torch
from hybrid_tsad.g36_persistent_model import PersistentResponse
from hybrid_tsad.normality_density_runner import validation_mode


def dense(model,h,prior=None):
    A,B,C,D,Q,R,m,P=model.parameters_gaussian(prior)
    n=h.shape[0];means=[];state_cov=[];cross={}
    for t in range(n):
        m=A@m+B@h[t];P=A@P@A.T+Q
        means.append(C@m+D@h[t]);state_cov.append(P)
        for j in range(t):cross[t,j]=A@(state_cov[j]if t-1==j else cross[t-1,j])
    cov=R.new_zeros(n*16,n*16)
    for t in range(n):
        for j in range(n):
            value=state_cov[t]if t==j else cross[t,j]if t>j else cross[j,t].T
            cov[t*16:(t+1)*16,j*16:(j+1)*16]=C@value@C.T+(R if t==j else 0)
    return torch.cat(means),cov


class PersistentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def model(self,v='full'):
        torch.manual_seed(2025);return PersistentResponse(v)
    def test_inverse_jacobian_and_gradients(self):
        m=self.model();x=torch.randn(2,52,9,requires_grad=True)
        y=x[:,4:,:4].reshape(2,12,4,4);z,ld=m.transform(y);r,ild=m.transform(z,True)
        torch.testing.assert_close(y,r,atol=1e-5,rtol=1e-5);torch.testing.assert_close(ld,-ild)
        loss=m.loss(x);loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for name in ['context.weight_ih_l0','raw_A','B','C','D','raw_Q','raw_R','steps.0.conditioner.output.weight']:
            p=dict(m.named_parameters())[name];self.assertIsNotNone(p.grad,name);self.assertGreater(float(p.grad.abs().sum()),0,name)
        self.assertEqual(float(x.grad[:,48:,4:].abs().sum()),0)
    def test_nonidentity_inverse_and_autograd_jacobian(self):
        m=self.model()
        with torch.no_grad():
            for step in m.steps:
                step.conditioner.output.weight.normal_(0,.015)
                step.conditioner.output.bias.normal_(0,.015)
        y=torch.randn(1,4,4,requires_grad=True)
        z,ld=m.transform(y);back,ild=m.transform(z,inverse=True)
        torch.testing.assert_close(back,y,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(ld,-ild,atol=1e-5,rtol=1e-5)
        jac=torch.autograd.functional.jacobian(lambda flat:m.transform(flat.reshape(1,4,4))[0].flatten(),y.flatten())
        torch.testing.assert_close(torch.linalg.slogdet(jac)[1],ld[0],atol=2e-5,rtol=2e-5)

    def test_dense_gaussian_kalman_and_zero_prior(self):
        m=self.model('linear');h=torch.randn(2,3,32,dtype=torch.double);z=torch.randn(2,3,16,dtype=torch.double)
        for prior in [None,torch.linspace(.2,1.,16,dtype=torch.double)]:
            log,_,_,_=m.kalman(z,h,prior)
            for i in range(2):
                mu,cov=dense(m,h[i],prior);expected=torch.distributions.MultivariateNormal(mu,covariance_matrix=cov).log_prob(z[i].flatten())
                torch.testing.assert_close(log[i],expected,rtol=1e-10,atol=1e-10)
        a=m.kalman(z,h)[0];b=m.kalman(z,h,torch.zeros(16))[0]
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    def test_hypotheses_distinct_persistent_offset_and_normal_condition_drift(self):
        m=self.model('linear');h=torch.randn(12,32,dtype=torch.double)
        mu,cov0=dense(m,h);mu1,cov1=dense(m,h,torch.ones(16,dtype=torch.double))
        torch.testing.assert_close(mu,mu1)
        delta=cov1-cov0;torch.testing.assert_close(delta,torch.ones(12,12,dtype=torch.double).kron(torch.eye(16,dtype=torch.double)),atol=1e-10,rtol=1e-10)
        def lr(z):return m.kalman(z[None],h[None],torch.ones(16))[0]-m.kalman(z[None],h[None])[0]
        normal=mu.reshape(12,16);self.assertLess(float(lr(normal).detach()),0)
        self.assertGreater(float(lr(normal+3).detach()),float(lr(normal).detach()))
        self.assertGreater(float(lr(normal+3).detach()),0)
    def test_same_shared_initial_and_terminal_prediction_causality(self):
        a,b=self.model(),self.model('linear')
        for k,v in b.state_dict().items():self.assertTrue(torch.equal(v,a.state_dict()[k]),k)
        x=torch.randn(2,52,9);y=x.clone();y[:,-4:,:4]+=20;y[:,-4:,4:]+=100
        with torch.no_grad():
            p=a.components(x,torch.ones(16));q=a.components(y,torch.ones(16))
        torch.testing.assert_close(p['terminal_prediction'],q['terminal_prediction'],rtol=0,atol=0)
        self.assertFalse(torch.equal(p['score'],q['score']))
    def test_validation_rng_heterogeneous_modes_and_exception(self):
        m=self.model();m.train();m.context.eval();before=[p.training for p in m.modules()];rng=torch.get_rng_state()
        with self.assertRaises(RuntimeError):
            with validation_mode(m,torch.device('cpu')):torch.randn(4);raise RuntimeError('intentional')
        self.assertEqual(before,[p.training for p in m.modules()]);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_prior_only_nonoverlap_fit(self):
        m=self.model();fit=torch.randn(29,9)
        with torch.no_grad():
            prior=m.fit_prior(fit);z,_=m.transform(fit[:28,:4].reshape(7,4,4));expected=z.flatten(1).double().var(0,unbiased=False)+1e-4
        torch.testing.assert_close(prior,expected);self.assertTrue((prior>0).all())


if __name__=='__main__':unittest.main()
