"""Event-balanced terminal supervision on native LEFT's pre-memory fused tokens.

The official components are external assets. This module implements only the
explicit token readout and group-wise loss reduction; it never calls native
reconstruction, prototype association, memory mixing, or decoder functions.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


TRAIN_PATHS=('tf','ms_utils','pos_embedding','patch_embedding','encoder','Et','Ef','fusion')


class EventSupervisedLEFT(nn.Module):
    def __init__(self,native):
        super().__init__();self.native=native
        native.requires_grad_(False)
        for name in TRAIN_PATHS:getattr(native,name).requires_grad_(True)
        self.head=nn.Linear(3*native.d_model,1)
        self.train(False)

    def train(self,mode=True):
        super().train(mode)
        # Frozen official submodules keep evaluation mode even during F fitting.
        for name,module in self.native.named_children():
            if name not in TRAIN_PATHS:module.eval()
        return self

    def features(self,x):
        m=self.native
        if x.ndim!=3 or x.shape[1:]!=(m.seq_len,m.enc_in):
            raise ValueError('Expected one arrived W-by-C window, without future rows')
        B,T,C=x.shape;meta=m._get_ms_meta(T,x.device)
        views,_=m.ms_utils(x,return_residual=True)
        for i in range(m.n_scales):views[i],_=m.patch_embedding(views[i].permute(0,2,1))
        multi=m.ms_utils.concat_sampling_list(views)+meta['ms_pos']
        multi,_=m.encoder(multi,meta['scale_ind_mask'])
        time=m.Et(x);freq=m.Ef(m.tf(x))
        multi=multi.reshape(B,C,-1,m.d_model).reshape(B,-1,m.d_model)
        t,f,s=m.fusion(time,freq,multi)
        return torch.cat((t.mean(1),f.mean(1),s.mean(1)),dim=-1)

    def forward(self,x):return self.head(self.features(x)).squeeze(-1)


def event_balanced_loss(logits,labels,q):
    """Two positive event groups, then two normal blocks; 32 points each.

    This is a 32-point minibatch tail approximation, not an unbiased estimator
    of the exact full-event CVaR. Tail reduction occurs inside each group.
    Positive and negative classes each
    receive half of the objective, independently of any event's duration.
    """
    if logits.shape!=(4,32)or labels.shape!=logits.shape:
        raise ValueError('Four separate groups of 32 points are required')
    if q not in (.25,1.):raise ValueError('Only the fixed paired tail and mean arms exist')
    if not bool((labels[:2]==1).all()and(labels[2:]==0).all()):
        raise ValueError('Group labels must be two positive events and two normal blocks')
    loss=F.binary_cross_entropy_with_logits(logits,labels,reduction='none')
    k=math.ceil(32*q)
    grouped=loss.topk(k,dim=1).values.mean(1)
    return .5*grouped[:2].mean()+.5*grouped[2:].mean(),grouped,loss
