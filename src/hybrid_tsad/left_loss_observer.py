"""Detached native LEFT loss accounting; no loss, RNG or gradient modification."""
def attach_native_loss_observer(model, torch):
    import types
    F=torch.nn.functional
    original=model._forward
    state={'last':None}
    def observed(self,*args,**kwargs):
        result=original(*args,**kwargs)
        gt,dec,lengths,from_s,xhat,x,cycle,ms,cons,*_=result
        with torch.no_grad():
            err=F.smooth_l1_loss(dec.detach(),gt.detach(),reduction='none')
            segments=list(torch.split(err,lengths[1:],dim=1))
            weights=getattr(self,'ms_seg_weights',None)
            weights=[1.]*len(segments)if weights is None else weights
            w=torch.as_tensor(weights,device=err.device,dtype=err.dtype);w=w/(w.sum()+1e-12)
            each_segment=torch.stack([s.mean(dim=(1,2))for s in segments],dim=1)
            each_ms=(each_segment*w).sum(dim=1)
            band=float(getattr(self,'alpha_band_reg',0.))
            if band:each_ms=each_ms+band*self.ms_utils.band_regularizer().detach()
            each_ft=F.smooth_l1_loss(from_s.detach(),x.detach(),reduction='none').mean(dim=(1,2))
            # TF is deterministic; same-forward outputs, no second model pass.
            each_tf=F.smooth_l1_loss(self.tf(xhat.detach()),self.tf(x.detach()),reduction='none').flatten(1).mean(dim=1)
            each_cycle=each_ft+each_tf
            each_cons=F.smooth_l1_loss(result[9].detach(),from_s.detach(),reduction='none').mean(dim=(1,2))
            each_total=self.alpha_cycle_l*each_cycle+self.alpha_ms_l*each_ms+self.alpha_cons_l*each_cons
            expected=self.alpha_cycle_l*cycle.detach()+self.alpha_ms_l*ms.detach()+self.alpha_cons_l*cons.detach()
            assert torch.allclose(each_ms.mean(),ms.detach(),atol=1e-6,rtol=1e-6)
            assert torch.allclose(each_cycle.mean(),cycle.detach(),atol=1e-6,rtol=1e-6)
            assert torch.allclose(each_total.mean(),expected,atol=1e-6,rtol=1e-6)
            state['last']=dict(ms=each_ms.cpu().numpy(),cycle_ft=each_ft.cpu().numpy(),cycle_tf=each_tf.cpu().numpy(),cycle=each_cycle.cpu().numpy(),cons=each_cons.cpu().numpy(),total=each_total.cpu().numpy(),MS_segment=each_segment.cpu().numpy(),MS_weights=w.cpu().numpy(),MS_lengths=list(lengths[1:]),loss_scalar_mean_matches=True)
        return result
    model._forward=types.MethodType(observed,model)
    return state
