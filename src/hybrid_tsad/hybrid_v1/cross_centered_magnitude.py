"""One experimental Cross routing-score mapping; original V1 is untouched.

Not a new network or anomaly score. Checkpoints contain state_dict only, so a
consumer MUST reapply and verify this policy after loading each candidate.
"""
import hashlib
import math
from pathlib import Path
import torch
from .lag import FiniteLag

POLICY = 'CROSS_CENTERED_QK_MAGNITUDE_V1'


class CenteredMagnitudeFiniteLag(FiniteLag):
    def mapped_cross_score(self, student, visible, raw_score, count):
        """Identical joint overlaps for means and original Q/K numerator."""
        x = student[:, -self.config.lag_window:].transpose(1, 2)
        m = visible[:, -self.config.lag_window:].transpose(1, 2)
        q, k = self.query(x[..., None]), self.key(x[..., None])
        centers = []
        for lag in range(self.config.max_lag + 1):
            qa, ka = q[:, :, lag:], k[:, :, :x.shape[-1]-lag]
            qm, km = m[:, :, lag:], m[:, :, :x.shape[-1]-lag]
            sum_q = torch.einsum('bisd,bjs->bijd', qa * qm[..., None], km.float())
            sum_k = torch.einsum('bis,bjsd->bijd', qm.float(), ka * km[..., None])
            n = count[..., lag].clamp_min(1)
            centers.append((sum_q * sum_k).sum(-1) / (n.square() * math.sqrt(self.config.lag_width)))
        center = torch.stack(centers, -1)
        # For zero overlaps both original numerator and projected sums are zero.
        centered = torch.where(count > 0, raw_score-center, torch.zeros_like(raw_score))
        return centered.abs(), center

    def forward(self, student, visible):
        # Same original correlation/count/endpoint V construction and Self path.
        score, count, valid, values = self.correlations(student, visible)
        c = student.shape[-1]
        eye = torch.eye(c, dtype=torch.bool, device=student.device)[None, :, :, None]
        lags = torch.arange(self.config.max_lag+1, device=student.device)
        self_valid = valid & eye & (lags > 0)
        cross_valid = valid & ~eye
        if self.config.cross_synchronous_only:
            cross_valid = cross_valid & (lags == 0)
        own, own_meta = self._aggregate(score, self_valid, values)
        mapped, center = self.mapped_cross_score(student, visible, score, count)
        cross, cross_meta = self._aggregate(mapped, cross_valid, values)
        own = self.self_out(own) if self.config.self_lag_enabled else torch.zeros_like(own)
        cross = self.cross_out(cross) if self.config.cross_lag_enabled else torch.zeros_like(cross)
        return own, cross, dict(correlation=score, overlap_count=count,
            self_candidate_valid=self_valid, cross_candidate_valid=cross_valid,
            self=own_meta, cross=cross_meta,
            source_sample_offset=student.shape[1]-1-lags,
            target_axis='i', source_axis='j', lag_unit='original_sample', circular=False,
            cross_routing_policy=POLICY,
            cross_routing_score=mapped, cross_centering_term=center,
            raw_correlation_semantics='Original signed uncentered Q/K; Cross weights use cross_routing_score',
            mapped_score_semantics='abs(raw_QK - dot(joint_overlap_meanQ,joint_overlap_meanK)/sqrtD)',
            signed_values_unchanged=True)


def policy_identity(model):
    if type(model.lag) is not CenteredMagnitudeFiniteLag:
        raise PermissionError('Candidate checkpoint requires explicit Cross policy installation')
    if model.config is not model.lag.config:
        raise ValueError('Model and lag configuration references differ')
    return dict(policy=POLICY, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        legacy_lag_sha256=hashlib.sha256(Path(__file__).with_name('lag.py').read_bytes()).hexdigest(),
        new_parameters=0, Self='ORIGINAL_SIGNED', values='ORIGINAL_SIGNED_ENDPOINT_V',
        changes='Cross routing score only; original candidate masks, projections, losses and anomaly scoring')


def install(model):
    """Instance-only class dispatch: no initialization, RNG draw or state change.

    Call on a fresh independent candidate, never the retained baseline instance.
    deepcopy keeps this top-level class; state_dict loading alone does not.
    """
    if type(model.lag) not in (FiniteLag, CenteredMagnitudeFiniteLag):
        raise TypeError('Only original FiniteLag or this exact policy can be installed')
    model.lag.__class__ = CenteredMagnitudeFiniteLag
    return policy_identity(model)
