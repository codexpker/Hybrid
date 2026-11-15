"""Optional synthetic CPU path check using the actual externally stored LEFT."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from hybrid_tsad.left_predictive_cycle import predictive_cycle_class
from hybrid_tsad.terminal_aligned_cycle import terminal_aligned_cycle_class


@unittest.skipUnless(os.environ.get('LEFT_SOURCE_ROOT'), 'LEFT_SOURCE_ROOT not bound')
class OfficialTerminalPaths(unittest.TestCase):
    def test_both_domains_real_loss_gradients_modes_and_time_permissions(self):
        sys.path.insert(0, os.environ['LEFT_SOURCE_ROOT'])
        from models.Left.LEFT import Left
        torch.set_num_threads(1)
        configs = Path(__file__).resolve().parents[1] / 'review_artifacts/code/preparations/hybrid_native_predictive_cycle_20261113_v1/configs'
        for domain in ('TEP', 'PSM'):
            cfg = SimpleNamespace(**json.loads((configs / (domain + '.json')).read_text()))
            x = torch.randn(2, cfg.seq_len+1, cfg.enc_in, generator=torch.Generator().manual_seed(2025))
            for mode in ('native', 'full'):
                with self.subTest(domain=domain, mode=mode):
                    torch.manual_seed(2025)
                    old = predictive_cycle_class(Left, mode)(cfg)
                    new = terminal_aligned_cycle_class(Left, mode)(cfg)
                    new.load_state_dict(old.state_dict(), strict=True)
                    for m in (old, new):
                        m.train(); m.step = cfg.warmup_steps + cfg.ramp_steps + 1
                    torch.manual_seed(84551)
                    a = old(x)[0]; a.backward(); rng = torch.get_rng_state().clone()
                    torch.manual_seed(84551)
                    b = new(x)[0]; b.backward()
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    for (n, p), (n1, p1) in zip(old.named_parameters(), new.named_parameters()):
                        self.assertEqual(n, n1)
                        if p.grad is None:
                            self.assertIsNone(p1.grad)
                        else:
                            torch.testing.assert_close(p.grad, p1.grad, rtol=0, atol=0)
                    old.eval(); new.eval()
                    before = copy.deepcopy(new.state_dict())
                    with torch.no_grad():
                        if mode == 'native':
                            torch.testing.assert_close(old.infer(x)[0], new.infer(x)[0], rtol=0, atol=0)
                            expected = Left.forward(new, x[:,1:], None, None, None)[0]
                            torch.testing.assert_close(new.selection_loss(x), expected, rtol=0, atol=0)
                        else:
                            pred = new.prediction_forward(x[:,:-1])
                            evidence, _ = new.endpoint_evidence(pred, x[:,1:])
                            torch.testing.assert_close(evidence.mean(), old._terminal(pred, x[:,1:])[0])
                            changed = x.clone(); changed[:,-1] += 100
                            other = new.prediction_forward(changed[:,:-1])
                            for i in (3,4,9):
                                torch.testing.assert_close(pred[i], other[i], rtol=0, atol=0)
                            self.assertFalse(torch.equal(new.infer(x)[0], new.infer(changed)[0]))
                            torch.testing.assert_close(new.selection_loss(x), old.selection_loss(x), rtol=0, atol=0)
                    for name, value in new.state_dict().items():
                        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
