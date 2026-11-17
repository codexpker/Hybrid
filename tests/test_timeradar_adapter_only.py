"""Only synthetic fixtures; real fixed FIT diagnostics have separate receipts."""
import unittest
import copy
import torch
from hybrid_tsad.timeradar_cross_variable import TimeRadarCrossVariable
from hybrid_tsad.timeradar_adapter_only import AdapterOnlyTimeRadar
from hybrid_tsad.timeradar_cross_variable_runner import state_hashes, validation_mode
from test_timeradar_cross_variable import FakeNative


class AdapterOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def wrapper(self, arm='cross'):
        torch.manual_seed(2024)
        native=FakeNative();native.register_buffer('tau_fixture',torch.tensor(.25))
        old=TimeRadarCrossVariable(native,arm,2)
        before=copy.deepcopy(old.state_dict());rng=torch.get_rng_state().clone()
        result=AdapterOnlyTimeRadar(old)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in result.state_dict().items():self.assertTrue(torch.equal(value,before[key]))
        return result

    def test_modes_and_only_adapter_trainable(self):
        for arm in ('self','cross'):
            m=self.wrapper(arm).train()
            self.assertTrue(m.training and m.adapter.training)
            self.assertTrue(all(not q.training for q in m.native.modules()))
            self.assertTrue(all(not p.requires_grad for p in m.native.parameters()))
            self.assertTrue(all(n.startswith('adapter.')for n in m.trainable_names()))

    def test_frozen_decoder_transmits_gradient_without_state_updates(self):
        for arm in ('self','cross'):
            m=self.wrapper(arm).train();initial=state_hashes(m.native);x=torch.randn(2,6,3)
            optimizer=torch.optim.AdamW(m.adapter.parameters(),lr=.01);captured=[]
            def hook(module,args,output):output.retain_grad();captured.append(output)
            handle=m.adapter.register_forward_hook(hook)
            for _ in range(2):
                optimizer.zero_grad();captured.clear()
                output=m(x,copies=4)[1];self.assertTrue(output.requires_grad)
                (output-x[None]).square().mean().backward()
                self.assertTrue(torch.isfinite(captured[-1].grad).all())
                self.assertGreater(float(captured[-1].grad.abs().max()),0.)
                optimizer.step();self.assertEqual(initial,state_hashes(m.native))
                self.assertTrue(all(p.grad is None for p in m.native.parameters()))
            self.assertGreater(float(m.adapter.down.weight.grad.abs().sum()),0.)
            handle.remove();m.close()

    def test_zero_disabled_native_identity_and_rng(self):
        for arm in ('self','cross'):
            m=self.wrapper(arm).eval();x=torch.randn(2,6,3)
            m._hook.remove();m._hook=None;rng=torch.get_rng_state()
            expected=m.native(x,copies=4);after=torch.get_rng_state()
            m._hook=m.native.decoder.register_forward_pre_hook(m._adapt)
            torch.set_rng_state(rng);actual=m(x,copies=4)
            self.assertTrue(torch.equal(after,torch.get_rng_state()))
            for a,b in zip(expected,actual):self.assertTrue(torch.equal(a,b))
            with torch.no_grad():m.adapter.up.weight.fill_(.1)
            m.adapter_enabled=False;torch.set_rng_state(rng);disabled=m(x,copies=4)
            for a,b in zip(expected,disabled):self.assertTrue(torch.equal(a,b))
            m.close()

    def test_selection_and_score_keep_entire_native_and_restore_rng(self):
        m=self.wrapper().train();x=torch.randn(2,6,3);initial=state_hashes(m.native)
        modes=[q.training for q in m.modules()];rng=torch.get_rng_state()
        with validation_mode(m,torch.device('cpu')):
            result=m(x,copies=4);m.cal_anomaly_score(x,result[1],None,L=1)
        self.assertEqual(modes,[q.training for q in m.modules()]);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(initial,state_hashes(m.native));m.close()


if __name__=='__main__':unittest.main()
