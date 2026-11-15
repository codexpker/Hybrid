"""Conditional normal dynamics versus an integrated persistent response offset.

All computations use arrived data only. Gaussian state integrations are exact;
learned time/Haar coordinates do not establish physical causal identification.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .normality_density import Coupling, haar


class PersistentResponse(nn.Module):
    def __init__(self, variant='full', hidden=32, state_dim=8):
        super().__init__()
        if variant not in ('full', 'linear'):
            raise ValueError(variant)
        self.variant, self.state_dim = variant, state_dim
        # Shared dynamics are initialized BEFORE optional flow parameters.
        self.context = nn.GRU(20, hidden, batch_first=True)
        self.raw_A = nn.Parameter(torch.eye(state_dim)*.3)
        self.B = nn.Parameter(torch.randn(state_dim, hidden)*.02)
        self.C = nn.Parameter(torch.randn(16, state_dim)*.1)
        self.D = nn.Parameter(torch.randn(16, hidden)*.02)
        self.raw_Q = nn.Parameter(torch.full((state_dim,), -2.))
        self.raw_R = nn.Parameter(torch.zeros(16))
        self.initial_mean = nn.Parameter(torch.zeros(state_dim))
        self.raw_P = nn.Parameter(torch.zeros(state_dim))
        self.steps = nn.ModuleList(Coupling(4,4,i,hidden,True) for i in range(4)) if variant=='full' else nn.ModuleList()

    def dynamics(self):
        return .98*self.raw_A/(1+self.raw_A.norm())

    def transform(self, y, inverse=False):
        shape=y.shape
        z=y.reshape(-1,4,4)
        ld=z.new_zeros(z.shape[0])
        order=reversed(range(len(self.steps))) if inverse else range(len(self.steps))
        for i in order:
            if i%2: z=haar(z)
            z,change=self.steps[i](z,inverse=inverse)
            ld=ld+change.sum((1,2))
            if i%2: z=haar(z,inverse=True)
        return z.reshape(shape),ld.reshape(shape[:-2])

    def encode(self, arrived):
        if arrived.ndim!=3 or arrived.shape[1:]!=(52,9) or not torch.isfinite(arrived).all():
            raise ValueError('Finite 52x9 role-local arrived window required')
        y=arrived[:,4:,:4].reshape(-1,12,4,4)
        conditions=arrived[:,:48,4:].reshape(-1,12,20)
        h,_=self.context(conditions)
        z,ld=self.transform(y)
        return z.flatten(-2),ld,h

    def parameters_gaussian(self, prior=None):
        A=self.dynamics().double()
        B,C,D=self.B.double(),self.C.double(),self.D.double()
        Q=torch.diag(.001+F.softplus(self.raw_Q.double()))
        R=torch.diag(.001+F.softplus(self.raw_R.double()))
        m=self.initial_mean.double()
        P=torch.diag(.001+F.softplus(self.raw_P.double()))
        if prior is not None:
            dim=self.state_dim
            A=torch.block_diag(A,torch.eye(16,device=A.device,dtype=A.dtype))
            B=torch.cat((B,B.new_zeros(16,B.shape[1])),0)
            C=torch.cat((C,torch.eye(16,device=C.device,dtype=C.dtype)),1)
            Q=torch.block_diag(Q,Q.new_zeros(16,16))
            m=torch.cat((m,m.new_zeros(16)))
            P=torch.block_diag(P,torch.diag(prior.to(dtype=P.dtype,device=P.device)))
        return A,B,C,D,Q,R,m,P

    def kalman(self,z,h,prior=None):
        A,B,C,D,Q,R,m,P=self.parameters_gaussian(prior)
        z,h=z.double(),h.double()
        m=m[None].expand(len(z),-1)
        eye=torch.eye(A.shape[0],dtype=A.dtype,device=A.device)
        logp=z.new_zeros(len(z)); means=[]; innov=[]; covs=[]
        for j in range(z.shape[1]):
            m=m@A.T+h[:,j]@B.T
            P=A@P@A.T+Q
            pred=m@C.T+h[:,j]@D.T
            S=C@P@C.T+R
            S=(S+S.T)/2
            L=torch.linalg.cholesky(S)
            e=z[:,j]-pred
            whiten=torch.linalg.solve_triangular(L,e.T,upper=False).T
            logp=logp-.5*(whiten.square().sum(-1)+2*L.diagonal().log().sum()+16*math.log(2*math.pi))
            K=torch.cholesky_solve(C@P,L).T
            m=m+e@K.T
            left=eye-K@C
            P=left@P@left.T+K@R@K.T
            P=(P+P.T)/2
            means.append(pred);innov.append(e);covs.append(S)
        return logp,torch.stack(means,1),torch.stack(innov,1),torch.stack(covs)[None].expand(len(z),-1,-1,-1)

    def loss_per_sample(self,arrived):
        z,ld,h=self.encode(arrived)
        logp,_,_,_=self.kalman(z,h)
        return -(logp+ld.double().sum(1))/192

    def loss(self,arrived):
        return self.loss_per_sample(arrived).mean()

    def components(self,arrived,prior):
        z,ld,h=self.encode(arrived)
        log0,pred,e,cov=self.kalman(z,h)
        log1,_,_,_=self.kalman(z,h,prior)
        nll=-(log0+ld.double().sum(1))/192
        terminal,_=self.transform(pred[:,-1].float().reshape(-1,4,4),inverse=True)
        return dict(score=log1-log0,normal_nll=nll,logp_H0=log0,logp_H1=log1,
                    log_jacobian=ld.double().sum(1),terminal_prediction=terminal, # inverse of latent predictive mean, NOT E[y|past]
                    
                    terminal_target=arrived[:,-4:,:4],terminal_innovation=e[:,-1],
                    terminal_innovation_cov=cov[:,-1])

    @torch.no_grad()
    def fit_prior(self,fit):
        n=len(fit)//4
        y=fit[:n*4,:4].reshape(n,4,4)
        z,_=self.transform(y)
        return z.flatten(1).double().var(0,unbiased=False)+1e-4

    @torch.no_grad()
    def joint_covariance(self,prior=None,blocks=12):
        A,B,C,D,Q,R,m,P=self.parameters_gaussian(prior)
        states=[];cross={}
        for t in range(blocks):
            P=A@P@A.T+Q;states.append(P)
            for j in range(t):cross[t,j]=A@(states[j] if t-1==j else cross[t-1,j])
        rows=[]
        for t in range(blocks):
            row=[]
            for j in range(blocks):
                v=states[t] if t==j else cross[t,j] if t>j else cross[j,t].T
                row.append(C@v@C.T+(R if t==j else 0))
            rows.append(torch.cat(row,1))
        return torch.cat(rows,0)
