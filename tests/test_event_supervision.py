import unittest
import torch
from torch import nn
from hybrid_tsad.event_supervision import EventSupervisedLEFT,event_balanced_loss,TRAIN_PATHS


class EventSupervisionTest(unittest.TestCase):
    def test_tail_is_inside_each_group_and_uses_eight_points(self):
        logits=torch.linspace(-4,4,32).repeat(4,1).requires_grad_()
        labels=torch.cat((torch.ones(2,32),torch.zeros(2,32)))
        loss,group,raw=event_balanced_loss(logits,labels,.25)
        expected=raw.topk(8,dim=1).values.mean(1)
        self.assertTrue(torch.equal(group,expected))
        self.assertTrue(torch.equal(loss,.5*expected[:2].mean()+.5*expected[2:].mean()))
        loss.backward();self.assertEqual((logits.grad!=0).sum(1).tolist(),[8,8,8,8])

    def test_mean_arm_is_same_balanced_bce_at_q_one(self):
        x=torch.randn(4,32,requires_grad=True);y=torch.cat((torch.ones(2,32),torch.zeros(2,32)))
        loss,_,raw=event_balanced_loss(x,y,1.)
        self.assertTrue(torch.allclose(loss,raw.mean()));loss.backward();self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_group_labels_or_q_rejected(self):
        x=torch.zeros(4,32);y=torch.cat((torch.ones(2,32),torch.zeros(2,32)))
        with self.assertRaises(ValueError):event_balanced_loss(x,y,.5)
        with self.assertRaises(ValueError):event_balanced_loss(x,y.flip(0),.25)

    def test_frozen_children_keep_eval_and_future_rows_rejected(self):
        native=nn.Module();native.seq_len=192;native.enc_in=31;native.d_model=4
        for name in TRAIN_PATHS:setattr(native,name,nn.Linear(4,4))
        native.proto_t=nn.Sequential(nn.Linear(4,4),nn.Dropout(.5))
        native.Dt=nn.Linear(4,4)
        m=EventSupervisedLEFT(native);m.train()
        self.assertTrue(m.native.Et.training);self.assertFalse(m.native.proto_t.training)
        self.assertFalse(m.native.Dt.training)
        self.assertTrue(all(not p.requires_grad for p in m.native.proto_t.parameters()))
        self.assertTrue(all(p.requires_grad for p in m.native.fusion.parameters()))
        with self.assertRaises(ValueError):m(torch.zeros(1,193,31))


if __name__=='__main__':unittest.main()
