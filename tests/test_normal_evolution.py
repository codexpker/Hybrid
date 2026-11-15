import itertools
import math
import unittest

import torch

from hybrid_tsad.normal_evolution import NormalEvolution


class NormalEvolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_forward_matches_all_state_paths(self):
        torch.manual_seed(5)
        emission=torch.randn(2,4,2,dtype=torch.float64,requires_grad=True)
        pi=torch.tensor([.2,-.3],dtype=torch.float64,requires_grad=True)
        transition=torch.tensor([[1.,-.4],[.6,.2]],dtype=torch.float64,requires_grad=True)
        actual=NormalEvolution.forward_log_likelihood(emission,pi,transition)
        paths=[]
        for path in itertools.product(range(2),repeat=4):
            term=pi.log_softmax(-1)[path[0]]+emission[:,0,path[0]]
            for t in range(1,4):term=term+transition.log_softmax(-1)[path[t-1],path[t]]+emission[:,t,path[t]]
            paths.append(term)
        expected=torch.stack(paths,-1).logsumexp(-1)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        actual.sum().backward();self.assertTrue((transition.grad.abs().sum()>0).item())

    def test_mixture_rosenblatt_uses_exclusive_coordinate_prefix(self):
        z=torch.tensor([[.4,-.7]],dtype=torch.float64)
        prior=torch.tensor([[.3,.7]],dtype=torch.float64)
        mu=torch.tensor([[[0.,1.],[1.,-1.]]],dtype=torch.float64);sigma=torch.ones(2,2,dtype=torch.float64)
        result=NormalEvolution.rosenblatt(z,prior.log(),mu,sigma)
        cdf=torch.special.ndtr(z[:,None,:]-mu)
        first=(prior*cdf[:,:,0]).sum(1)
        weight=prior*torch.exp(-.5*(z[:,None,0]-mu[:,:,0]).square())
        second=((weight/weight.sum(1,keepdim=True))*cdf[:,:,1]).sum(1)
        torch.testing.assert_close(result,torch.stack((first,second),-1),rtol=1e-12,atol=1e-12)
        single=NormalEvolution.rosenblatt(z,torch.zeros(1,1,dtype=z.dtype),torch.zeros(1,1,2,dtype=z.dtype),torch.ones(1,2,dtype=z.dtype))
        torch.testing.assert_close(single,torch.special.ndtr(z),rtol=1e-12,atol=1e-12)

    def test_nontrivial_local_map_jacobian_and_inverse(self):
        torch.manual_seed(8);model=NormalEvolution(1,4,hidden=4).double()
        with torch.no_grad():
            for step in model.steps:
                step.conditioner.output.weight.normal_(0,.02);step.conditioner.output.bias.normal_(0,.03)
        x=torch.randn(1,4,1,dtype=torch.float64)
        z,ld=model.block_transform(x)
        jac=torch.autograd.functional.jacobian(lambda v:model.block_transform(v.reshape(1,4,1))[0].flatten(),x.flatten())
        torch.testing.assert_close(torch.linalg.slogdet(jac)[1],ld[0],rtol=1e-9,atol=1e-9)
        recovered,ild=model.block_transform(z,inverse=True)
        torch.testing.assert_close(recovered,x,rtol=1e-9,atol=1e-9)
        torch.testing.assert_close(ld+ild,torch.zeros_like(ld),rtol=1e-9,atol=1e-9)

    def test_initial_active_tensors_shared_and_dynamics_stable(self):
        fit=torch.randn(2,32,2);models=[]
        for variant in ('full','time_only','no_evolution'):
            torch.manual_seed(2025);m=NormalEvolution(2,12,variant=variant,hidden=4);m.initialise_from_fit(fit);models.append(m)
        for a in models[1:]:
            for k,v in models[0].state_dict().items():self.assertTrue(torch.equal(v,a.state_dict()[k]),k)
        x=torch.randn(2,4,2);z,_=models[0].block_transform(x)
        torch.testing.assert_close(z,x,rtol=1e-5,atol=1e-6)
        with torch.no_grad():models[0].diagonal.normal_(0,20);models[0].low_u.normal_(0,20);models[0].low_v.normal_(0,20)
        self.assertLessEqual(float(torch.linalg.matrix_norm(models[0].dynamics(),ord=2).max().detach()),.900001)
        self.assertTrue(torch.equal(models[2].dynamics(),torch.zeros_like(models[2].dynamics())))

    def test_training_gradients_and_lowrank_means(self):
        torch.manual_seed(11);m=NormalEvolution(2,12,hidden=4);m.initialise_from_fit(torch.randn(3,24,2))
        arrived=torch.randn(3,13,2);z,ld=m.encode(arrived);means=m.means(z)
        explicit=m.mu[None,None]+torch.einsum('kde,blke->blkd',m.dynamics(),z[:,:-1,None]-m.mu[None,None])
        torch.testing.assert_close(means[:,1:],explicit,rtol=1e-5,atol=1e-6)
        loss=m.loss(arrived);loss.backward()
        for name in ['mu','raw_noise','transition_logits','diagonal','low_u','low_v']:
            g=getattr(m,name).grad;self.assertIsNotNone(g);self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().sum()),0,name)
        self.assertGreater(sum(float(p.grad.abs().sum())for p in m.steps.parameters()if p.grad is not None),0)

    def test_current_block_cannot_change_predictive_prior(self):
        torch.manual_seed(22);m=NormalEvolution(2,12,hidden=4);m.initialise_from_fit(torch.randn(2,24,2));m.eval()
        x=torch.randn(3,13,2);y=x.clone();y[:,-4:]+=10
        z,_=m.encode(x);w,_=m.encode(y);self.assertTrue(torch.equal(z[:,:-1],w[:,:-1]))
        p,a=m.final_predictive(z);q,b=m.final_predictive(w)
        self.assertTrue(torch.equal(p,q));self.assertTrue(torch.equal(a,b))
        torch.testing.assert_close(p.exp().sum(-1),torch.ones(3),rtol=1e-6,atol=1e-6)
        s=m.scalar_score(x);permutation=torch.tensor([2,0,1]);other=m.scalar_score(x[permutation])
        torch.testing.assert_close(other,s[permutation],rtol=1e-5,atol=1e-6)
        repeated=m.score(x);self.assertTrue(torch.equal(repeated,repeated[:,:1].expand_as(repeated)))
        with self.assertRaises(ValueError):m.loss(torch.randn(3,14,2))

    def test_no_evolution_removes_filter_and_past_response(self):
        torch.manual_seed(3);m=NormalEvolution(2,12,variant='no_evolution',hidden=4);m.initialise_from_fit(torch.randn(2,24,2))
        x=torch.randn(3,13,2);y=x.clone();y[:,:-4]+=100
        torch.testing.assert_close(m.scalar_score(x),m.scalar_score(y),rtol=0,atol=0)
        z,_=m.encode(x);means=m.means(z);emission=m.gaussian_coordinates(z[:,:,None],means,m.noise()[None,None]).sum(-1)
        expected=torch.logsumexp(emission+m.initial_logits.log_softmax(-1),-1).sum(-1)
        torch.testing.assert_close(m.log_likelihood(z),expected)


if __name__=='__main__':unittest.main()
