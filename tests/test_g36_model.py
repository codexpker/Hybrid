import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock
import torch
from hybrid_tsad.g36_model import ConditionalResponseFlow


class G36ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def random_map(self,variant='full',dtype=torch.float32):
        torch.manual_seed(13);model=ConditionalResponseFlow(variant,hidden=8).to(dtype=dtype)
        with torch.no_grad():
            for step in model.steps:
                step.net[-1].weight.normal_(0,.02);step.net[-1].bias.normal_(0,.01)
        return model

    def test_nontrivial_inverse_and_conditional_jacobian(self):
        model=self.random_map(dtype=torch.float64);target=torch.randn(1,4,4,dtype=torch.float64);context=torch.randn(1,8,dtype=torch.float64)
        z,ld=model.transform(target,context);back,invld=model.transform(z,context,inverse=True)
        torch.testing.assert_close(back,target,rtol=1e-9,atol=1e-9)
        torch.testing.assert_close(ld+invld,torch.zeros_like(ld),rtol=1e-9,atol=1e-9)
        jac=torch.autograd.functional.jacobian(lambda flat:model.transform(flat.reshape(1,4,4),context)[0].flatten(),target.flatten())
        torch.testing.assert_close(torch.linalg.slogdet(jac)[1],ld[0],rtol=1e-9,atol=1e-9)

    def test_target_and_future_conditions_cannot_enter_context(self):
        model=self.random_map();x=torch.randn(2,52,9);y=x.clone();y[:,48:]+=100
        torch.testing.assert_close(model.context(model.split(x)[0]),model.context(model.split(y)[0]),rtol=0,atol=0)
        y=x.clone();y[:,48:,4:]+=100
        torch.testing.assert_close(model.score(x),model.score(y),rtol=0,atol=0)
        with self.assertRaises(ValueError):model.score(torch.randn(2,53,9))

    def test_history_only_independent_of_five_past_conditions(self):
        model=self.random_map('history_only');x=torch.randn(3,52,9);y=x.clone();y[:,:48,4:]=100*torch.randn_like(y[:,:48,4:])
        torch.testing.assert_close(model.score(x),model.score(y),rtol=0,atol=0)
        x.requires_grad_();model.loss(x).backward();self.assertTrue(torch.equal(x.grad[:,:48,4:],torch.zeros_like(x.grad[:,:48,4:])))

    def test_full_condition_gradient_and_pointwise_loss(self):
        model=self.random_map();x=torch.randn(3,52,9,requires_grad=True);score=model.score(x)
        self.assertEqual(score.shape,(3,));self.assertTrue(torch.equal(score.mean(),model.loss(x)))
        model.loss(x).backward();self.assertGreater(float(x.grad[:,:48,4:].abs().sum()),0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_shared_initial_tensors_and_effective_context(self):
        models=[]
        for variant in ['full','history_only']:
            torch.manual_seed(2025);models.append(ConditionalResponseFlow(variant,hidden=8))
        for k,v in models[0].state_dict().items():self.assertTrue(torch.equal(v,models[1].state_dict()[k]),k)
        x=torch.randn(2,48,9);y=x.clone();y[:,:17]+=100
        torch.testing.assert_close(models[0].context(x),models[0].context(y),rtol=0,atol=0)

    def test_role_permission_is_checked_before_opening_query(self):
        from hybrid_tsad.g36_runner import Data
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);(root/'manifest.json').write_text(json.dumps(dict(history=48,block=4,response_columns=list(range(4)),context_columns=list(range(5)))))
            data=Data(root,{'data':'.'},['FIT'])
            with mock.patch('hybrid_tsad.g36_runner.sha')as hasher:
                with self.assertRaises(PermissionError):data.load(dict(role='QUERY',path='missing.npz'))
                hasher.assert_not_called()


if __name__=='__main__':unittest.main()
