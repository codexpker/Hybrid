import unittest
import numpy as np
import torch
from torch import nn
from hybrid_tsad.g36_response_correction import frozen_features,fusion_capture,feature_stats,design,fit_head,ridge_predict


class FakeFusion(nn.Module):
    def forward(self,t,f,m):return t,f,m


class FakeLEFT(nn.Module):
    def __init__(self):
        super().__init__();self.fusion=FakeFusion();self.weight=nn.Parameter(torch.ones(1),requires_grad=False)
        self.register_buffer('prototypes',torch.randn(2,4));self.step=2100;self.fail=False
    def forward(self,h,*args):
        if self.training:self.step+=1
        self.last_shape=h.shape
        out=self.fusion(h[:,:,:4],h[:,:,4:8],h[:,:,:4])
        if self.fail:raise RuntimeError('intentional')
        return out[0].sum()


class CorrectionTests(unittest.TestCase):
    def test_target_is_not_encoder_input_and_frozen_state(self):
        model=FakeLEFT().eval();x=torch.randn(3,52,9);y=x.clone();y[:,48:]+=100
        before={k:v.clone()for k,v in model.state_dict().items()}
        a=frozen_features(model,x);b=frozen_features(model,y)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(model.last_shape,(3,48,9));self.assertEqual(model.step,2100)
        self.assertEqual(a.shape,(3,8));self.assertFalse(a.requires_grad)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        self.assertEqual(len(model.fusion._forward_hooks),0)
    def test_all_modules_eval_and_cleanup_exception(self):
        model=FakeLEFT().eval();model.fusion.train()
        with self.assertRaises(ValueError):frozen_features(model,torch.randn(1,52,9))
        model.eval();model.fail=True
        with self.assertRaises(RuntimeError):frozen_features(model,torch.randn(1,52,9))
        self.assertEqual(len(model.fusion._forward_hooks),0)
    def test_fit_only_statistics_and_matched_feature_slots(self):
        fit=np.array([[1,2,3,0],[3,2,5,0]],float);mean,scale=feature_stats(fit)
        np.testing.assert_array_equal(mean,[2,2,4,0]);self.assertEqual(scale[1],1e-6)
        select=fit+100;before=mean.copy();a=design(select,mean,scale,'T_only');b=design(select,mean,scale,'TF')
        np.testing.assert_array_equal(mean,before);np.testing.assert_array_equal(a[:,:2],b[:,:2]);self.assertFalse(a[:,2:].any())
        self.assertEqual(a.shape,b.shape)
    def test_closed_form_matches_mean_scalar_loss(self):
        rng=np.random.default_rng(2025);phi=rng.normal(size=(23,8));target=rng.normal(size=(23,4,4));mean,scale=feature_stats(phi)
        for method in ['TF','T_only']:
            W,stats=fit_head(phi,target,mean,scale,method)
            x=design(phi,mean,scale,method);y=target.reshape(23,16)
            grad=2*x.T@(x@W-y)/(23*16)+2*W
            np.testing.assert_allclose(grad,0,atol=1e-14)
            self.assertEqual(stats['alpha_sumloss'],23*16)
            self.assertAlmostEqual(stats['objective'],np.mean((x@W-y)**2)+np.sum(W**2))
            if method=='T_only':np.testing.assert_array_equal(W[4:],0)
    def test_ridge_target_order(self):
        h=np.arange(2*48*9).reshape(2,48,9);coef=np.zeros((16,432));coef[:,0]=np.arange(16);bias=np.arange(16)*.1
        prediction=ridge_predict(h,coef,bias)
        np.testing.assert_allclose(prediction.reshape(2,16),h[:,0,0,None]*np.arange(16)[None]+bias)


if __name__=='__main__':unittest.main()
