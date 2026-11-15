"""Scientific numerical gates for the new joint-density calculation."""
import unittest
import torch
from hybrid_tsad.normality_density import NormalityDensity, haar


class DensityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2025)
        torch.set_num_threads(1)

    def test_haar_inverse_energy_and_dc(self):
        for h in (100,192):
            x=torch.randn(2,h,3,dtype=torch.float64)
            y=haar(x)
            torch.testing.assert_close(haar(y,True),x,atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(y.square().sum(),x.square().sum(),atol=1e-10,rtol=1e-12)
            dc=haar(torch.ones_like(x))
            self.assertGreater(float(dc.square().sum()),0.)

    def test_nontrivial_flow_inverse_and_exact_jacobian(self):
        for variant in ('full','time_only','local'):
            m=NormalityDensity(2,4,variant,hidden=4).double()
            for step in m.steps:
                torch.nn.init.normal_(step.conditioner.output.weight,std=.03)
            x=torch.randn(1,4,2,dtype=torch.float64)
            z,ld=m.transform(x);recovered,ild=m.transform(z,True)
            torch.testing.assert_close(recovered,x,atol=1e-10,rtol=1e-10)
            torch.testing.assert_close(ld.sum()+ild.sum(),torch.tensor(0.,dtype=torch.float64),atol=1e-10,rtol=0)
            jac=torch.autograd.functional.jacobian(lambda v:m.transform(v.reshape(1,4,2))[0].flatten(),x.flatten())
            sign,actual=torch.linalg.slogdet(jac)
            self.assertEqual(float(sign.abs()),1.)
            torch.testing.assert_close(actual,ld.sum(),atol=1e-9,rtol=1e-9)

    def test_objective_score_exact_same_joint_density(self):
        m=NormalityDensity(3,100)
        x=torch.randn(3,101,3)
        torch.testing.assert_close(m.loss(x),m.score(x).mean(),rtol=0,atol=1e-7)
        z,ld=m.transform(x)
        p=torch.distributions.StudentT(torch.tensor(5.))
        torch.testing.assert_close(m.evidence(x),-p.log_prob(z)-ld,rtol=1e-6,atol=1e-6)

    def test_samples_independent_and_explicit_footprint(self):
        m=NormalityDensity(3,100).eval()
        x=torch.randn(2,101,3)
        y=x.clone();y[:,0]=999
        torch.testing.assert_close(m.score(x),m.score(y),rtol=0,atol=0)
        for i in range(2):
            torch.testing.assert_close(m.score(x)[i],m.score(x[i:i+1])[0],rtol=1e-5,atol=1e-6)
        y=x.clone();y[:,-1]+=100
        self.assertFalse(torch.equal(m.score(x),m.score(y)))

    def test_no_per_window_location_erasure(self):
        m=NormalityDensity(3,100)
        x=torch.zeros(2,100,3)
        self.assertGreater(float((m.score(x+5)-m.score(x)).mean().detach()),.5)

    def test_gradients_train_and_variants_equal_capacity(self):
        sizes=[]
        for variant in ('full','time_only','local'):
            m=NormalityDensity(3,100,variant,hidden=8)
            sizes.append(sum(p.numel()for p in m.parameters()))
            opt=torch.optim.Adam(m.parameters(),lr=1e-3)
            x=torch.randn(8,100,3)*.2
            first=float(m.loss(x).detach())
            for _ in range(3):
                opt.zero_grad();loss=m.loss(x);loss.backward();opt.step()
            self.assertTrue(all(torch.isfinite(p).all()for p in m.parameters()))
            self.assertLess(float(m.loss(x).detach()),first)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()for p in m.parameters()))
        self.assertEqual(len(set(sizes)),1)


if __name__=='__main__':unittest.main()
