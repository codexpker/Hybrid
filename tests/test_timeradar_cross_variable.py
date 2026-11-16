"""Synthetic adapter tests. Actual official forward parity is a separate artifact."""
import copy
import unittest
import torch
from torch import nn
from hybrid_tsad.timeradar_cross_variable import ComplexChannelAdapter,TimeRadarCrossVariable
from hybrid_tsad.normality_density_runner import validation_mode


class FakeDecoder(nn.Module):
    def __init__(self,d):super().__init__();self.layer=nn.Linear(d,3);self.fail=False
    def forward(self,z):
        if self.fail:raise RuntimeError('intentional')
        return torch.complex(self.layer(z.real),self.layer(z.imag)).flatten(-2)


class FakeNative(nn.Module):
    def __init__(self):
        super().__init__();self.repr_dim=3;self.mask_mode='c';self.encoder=nn.Sequential(nn.Linear(3,3),nn.Dropout(.1));self.decoder=FakeDecoder(3)
    def forward(self,x,norm=0,mask_mode=None,copies=10):
        b,t,c=x.shape;z=x.transpose(1,2).reshape(b*c,2,3);a=self.encoder(z);base=torch.complex(a,a*.2)
        k=1 if mask_mode=='nomask'else copies
        if k>1:
            mask=torch.rand(k//2,b*c,2,1)<.5;mask=torch.cat((mask,~mask),0)
            rep=base[None].expand(k,-1,-1,-1).masked_fill(mask,0).reshape(k*b*c,2,3)
        else:rep=base
        y=self.decoder(rep).abs().reshape(k,b,c,t).permute(0,1,3,2)
        return rep,y
    def cal_anomaly_score(self,x,y,anomaly_criterion=None,L=1):
        score=y.var(0).mean(-1)
        if anomaly_criterion is None:return score
        return L*score+(1-L)*anomaly_criterion(y.mean(0),x).mean(-1)


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def adapter(self,mode='cross'):
        torch.manual_seed(2024);return ComplexChannelAdapter(4,4,mode)
    def nonzero(self,m):
        with torch.no_grad():
            m.down.weight.zero_();m.down.weight[:,:4]=torch.eye(4);m.down.bias.zero_();m.up.weight.fill_(.1);m.up.bias.zero_()
    def test_zero_identity_matched_parameters_and_axes(self):
        a,b=self.adapter(),self.adapter('self');z=torch.randn(12,2,4,dtype=torch.complex64)
        for key,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[key]))
        self.assertTrue(torch.equal(a(z,2,2,3),z));self.assertTrue(torch.equal(b(z,2,2,3),z))
        with self.assertRaises(ValueError):a(z,2,1,3)
    def test_only_other_channels_same_copy_sample_patch(self):
        a=self.adapter();self.nonzero(a)
        z=torch.randn(2,2,3,2,4,dtype=torch.complex64);changed=z.clone();changed[0,0,0,0]+=10
        delta=a(z.reshape(12,2,4),2,2,3).reshape_as(z)-z
        other=a(changed.reshape(12,2,4),2,2,3).reshape_as(z)-changed
        torch.testing.assert_close(delta[1],other[1],rtol=0,atol=0)
        torch.testing.assert_close(delta[0,1],other[0,1],rtol=0,atol=0)
        torch.testing.assert_close(delta[0,0,:,1],other[0,0,:,1],rtol=0,atol=0)
        torch.testing.assert_close(delta[0,0,0,0],other[0,0,0,0],atol=2e-6,rtol=0)
        self.assertFalse(torch.equal(delta[0,0,1,0],other[0,0,1,0]))
    def test_one_channel_falls_back_to_self(self):
        a,b=self.adapter(),self.adapter('self');self.nonzero(a);b.load_state_dict(a.state_dict());z=torch.randn(4,2,4,dtype=torch.complex64)
        torch.testing.assert_close(a(z,2,2,1),b(z,2,2,1),atol=0,rtol=0)
    def test_disabled_and_zero_wrapper_match_native_mask_rng(self):
        torch.manual_seed(2024);base=FakeNative().eval();wrapper=TimeRadarCrossVariable(copy.deepcopy(base),'cross',2).eval();x=torch.randn(2,6,3)
        state=torch.get_rng_state();expected=base(x,copies=4)
        torch.set_rng_state(state);actual=wrapper(x,copies=4)
        for a,b in zip(expected,actual):self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(base.cal_anomaly_score(x,expected[1],L=1),wrapper.cal_anomaly_score(x,actual[1],L=1)))
        with torch.no_grad():wrapper.adapter.up.weight.fill_(1)
        wrapper.adapter_enabled=False;torch.set_rng_state(state);disabled=wrapper(x,copies=4)
        for a,b in zip(expected,disabled):self.assertTrue(torch.equal(a,b))
        self.assertIsNone(wrapper._layout);wrapper.close();self.assertEqual(len(wrapper.native.decoder._forward_pre_hooks),0)
    def test_frozen_encoder_eval_decoder_adapter_gradient(self):
        wrapper=TimeRadarCrossVariable(FakeNative(),'cross',2).train();x=torch.randn(2,6,3)
        self.assertFalse(wrapper.native.training);self.assertFalse(wrapper.native.encoder.training);self.assertTrue(wrapper.native.decoder.training)
        loss=(wrapper(x,copies=4)[1]-x[None]).square().mean();loss.backward()
        self.assertGreater(float(wrapper.adapter.up.weight.grad.abs().sum()),0)
        self.assertGreater(float(wrapper.native.decoder.layer.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in wrapper.native.encoder.parameters()))
    def test_exception_layout_cleanup_and_validation_restore(self):
        w=TimeRadarCrossVariable(FakeNative(),'self',2).train();w.native.decoder.fail=True
        with self.assertRaises(RuntimeError):w(torch.randn(2,6,3),copies=4)
        self.assertIsNone(w._layout)
        modes=[m.training for m in w.modules()];rng=torch.get_rng_state()
        with self.assertRaises(RuntimeError):
            with validation_mode(w,torch.device('cpu')):
                torch.randn(2);raise RuntimeError('intentional')
        self.assertEqual(modes,[m.training for m in w.modules()]);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        w.close()


if __name__=='__main__':unittest.main()
