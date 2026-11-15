"""Locally invertible time-frequency normal evolution with predictive compliance.

Affine coupling and Haar are mature primitives. The switching Gaussian forward
recursion is exact; this is an independent implementation, not the official SDS
EM experiment. Statistical regimes do not authenticate physical commands.
"""
import math

import torch
from torch import nn

from .normality_density import Coupling,haar


class NormalEvolution(nn.Module):
    score_is_replicated_scalar=True

    def __init__(self,channels,history,variant='full',hidden=64,states=4,block=4,rank=8):
        super().__init__()
        if variant not in ('full','time_only','no_evolution')or block!=4 or history%block:
            raise ValueError('Fixed four-point blocks and three predeclared variants required')
        self.channels,self.history,self.variant=channels,history,variant
        self.block,self.states,self.dim=block,states,block*channels
        self.rank=min(rank,self.dim)
        self.steps=nn.ModuleList(Coupling(channels,block,i,hidden,local=True)for i in range(4))
        self.mu=nn.Parameter(torch.zeros(states,self.dim))
        self.raw_noise=nn.Parameter(torch.zeros(states,self.dim))
        self.initial_logits=nn.Parameter(torch.zeros(states))
        transition=torch.full((states,states),.1/(states-1));transition.fill_diagonal_(.9)
        self.transition_logits=nn.Parameter(transition.log())
        self.diagonal=nn.Parameter(torch.zeros(states,self.dim))
        self.low_u=nn.Parameter(torch.randn(states,self.dim,self.rank)*.02)
        self.low_v=nn.Parameter(torch.randn(states,self.dim,self.rank)*.002)
        if variant=='no_evolution':
            for name in ('transition_logits','diagonal','low_u','low_v'):getattr(self,name).requires_grad_(False)

    def _window(self,arrived):
        if arrived.ndim!=3 or arrived.shape[2]!=self.channels or arrived.shape[1]not in(self.history,self.history+1):
            raise ValueError('Expected B,H[, +1],C containing only arrived observations')
        if not bool(torch.isfinite(arrived).all()):raise ValueError('Finite FIT-standardized inputs required')
        return arrived[:,-self.history:]

    def block_transform(self,blocks,inverse=False):
        if blocks.shape[-2:]!=(4,self.channels):raise ValueError('Wrong local block shape')
        shape=blocks.shape;z=blocks.reshape(-1,4,self.channels);logdet=z.new_zeros(z.shape[0])
        order=reversed(range(4))if inverse else range(4)
        for i in order:
            frequency=self.variant!='time_only'and i%2==1
            # Conjugate odd couplings by Haar. Identity couplings imply g=I,
            # unlike a nonzero net coordinate rotation under an iid base prior.
            if frequency:z=haar(z)
            z,ld=self.steps[i](z,inverse=inverse);logdet=logdet+ld.sum((1,2))
            if frequency:z=haar(z,inverse=True)
        return z.reshape(shape),logdet.reshape(shape[:-2])

    def encode(self,arrived):
        x=self._window(arrived);B=x.shape[0];blocks=x.reshape(B,self.history//4,4,self.channels)
        z,ld=self.block_transform(blocks)
        return z.flatten(-2,-1),ld

    def noise(self):return .05+9.95*torch.sigmoid(self.raw_noise)

    def dynamics(self):
        """Explicit matrices for verification; training uses the low-rank form."""
        if self.variant=='no_evolution':return self.mu.new_zeros(self.states,self.dim,self.dim)
        denom=self.low_u.norm(dim=(1,2))*self.low_v.norm(dim=(1,2))+1e-8
        low=self.low_u@self.low_v.transpose(-1,-2)/denom[:,None,None]
        return .45*(torch.diag_embed(torch.tanh(self.diagonal))+low)

    def means(self,z):
        B,L,D=z.shape
        first=self.mu[None,None].expand(B,1,-1,-1)
        if self.variant=='no_evolution':return self.mu[None,None].expand(B,L,-1,-1)
        centered=z[:,:-1,None,:]-self.mu[None,None]
        diag=centered*torch.tanh(self.diagonal)[None,None]
        latent=torch.einsum('blkd,kdr->blkr',centered,self.low_v)
        low=torch.einsum('blkr,kdr->blkd',latent,self.low_u)
        denom=self.low_u.norm(dim=(1,2))*self.low_v.norm(dim=(1,2))+1e-8
        later=self.mu[None,None]+.45*(diag+low/denom[None,None,:,None])
        return torch.cat((first,later),dim=1)

    @staticmethod
    def gaussian_coordinates(z,means,sigma):
        return -.5*((z-means)/sigma).square()-sigma.log()-.5*math.log(2*math.pi)

    @staticmethod
    def forward_log_likelihood(log_emission,initial_logits,transition_logits,iid=False):
        """Exact normalized forward pass: emissions B,L,K, returns B."""
        logpi=initial_logits.log_softmax(-1);logT=transition_logits.log_softmax(-1)
        if iid:return torch.logsumexp(log_emission+logpi,dim=-1).sum(-1)
        alpha=log_emission[:,0]+logpi;scale=torch.logsumexp(alpha,-1);alpha=alpha-scale[:,None];total=scale
        for t in range(1,log_emission.shape[1]):
            prior=torch.logsumexp(alpha[:,:,None]+logT[None],dim=1)
            alpha=prior+log_emission[:,t];scale=torch.logsumexp(alpha,-1)
            alpha=alpha-scale[:,None];total=total+scale
        return total

    def log_likelihood(self,z):
        mean=self.means(z);coords=self.gaussian_coordinates(z[:,:,None,:],mean,self.noise()[None,None])
        return self.forward_log_likelihood(coords.sum(-1),self.initial_logits,self.transition_logits,self.variant=='no_evolution')

    def final_predictive(self,z):
        """Prior of the final block depends exclusively on earlier blocks."""
        mean=self.means(z);logpi=self.initial_logits.log_softmax(-1)
        if self.variant=='no_evolution'or z.shape[1]==1:
            return logpi[None].expand(z.shape[0],-1),mean[:,-1]
        emission=self.gaussian_coordinates(z[:,:-1,None,:],mean[:,:-1],self.noise()[None,None]).sum(-1)
        alpha=emission[:,0]+logpi;alpha=alpha-alpha.logsumexp(-1,keepdim=True)
        logT=self.transition_logits.log_softmax(-1)
        for t in range(1,emission.shape[1]):
            alpha=torch.logsumexp(alpha[:,:,None]+logT[None],dim=1)+emission[:,t]
            alpha=alpha-alpha.logsumexp(-1,keepdim=True)
        prior=torch.logsumexp(alpha[:,:,None]+logT[None],dim=1)
        return prior-prior.logsumexp(-1,keepdim=True),mean[:,-1]

    @staticmethod
    def rosenblatt(z,logprior,means,sigma):
        """Exact fixed-order Gaussian-mixture transform using exclusive prefixes."""
        logcoord=NormalEvolution.gaussian_coordinates(z[:,None,:],means,sigma[None])
        prefix=torch.cat((torch.zeros_like(logcoord[:,:,:1]),logcoord.cumsum(-1)[:,:,:-1]),-1)
        posterior=(logprior[:,:,None]+prefix).softmax(dim=1)
        phi=torch.special.ndtr((z[:,None,:]-means)/sigma[None])
        return(posterior*phi).sum(1)

    def scalar_score(self,arrived):
        z,_=self.encode(arrived);prior,mean=self.final_predictive(z)
        v=self.rosenblatt(z[:,-1],prior,mean,self.noise()).clamp(1e-7,1-1e-7)
        return-(2*torch.minimum(v,1-v)).log().mean(-1)

    def score(self,arrived):
        # Compatibility container only: coordinates are NOT channel densities.
        return self.scalar_score(arrived)[:,None].expand(-1,self.channels)

    def loss(self,arrived):
        z,ld=self.encode(arrived)
        return-((self.log_likelihood(z)+ld.sum(-1))/(self.history*self.channels)).mean()

    @torch.no_grad()
    def initialise_from_fit(self,fit):
        values=torch.as_tensor(fit,dtype=self.mu.dtype,device=self.mu.device)
        if values.ndim!=3 or values.shape[-1]!=self.channels:raise ValueError('FIT must be R,T,C')
        n=values.shape[1]//4
        blocks=values[:,:n*4].reshape(-1,self.dim)
        index=torch.linspace(0,len(blocks)-1,self.states,device=blocks.device).long()
        # g is identity at initialization. Copy identical raw block coordinates
        # into all variants to avoid variant-dependent initialization data.
        self.mu.copy_(blocks[index])
        sigma=blocks.std(dim=0,unbiased=False)
        p=((sigma-.05)/9.95).clamp(1e-6,1-1e-6)
        self.raw_noise.copy_(torch.logit(p)[None].expand_as(self.raw_noise))
        return dict(FIT_shape=list(values.shape),block_grid_start=0,blocks=len(blocks),
                    selected_block_indices=index.cpu().tolist(),noise_min=float(self.noise().min()),
                    noise_max=float(self.noise().max()),initial_map='identity couplings; paired Haar conjugation')
