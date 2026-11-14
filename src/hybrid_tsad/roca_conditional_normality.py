"""One fixed command-conditioned reference control; no model optimization.

Anonymous published command encodings are contexts, not identified healthy
physical regimes. All references are same-entity nominal FIT64, not certified
clean. No context-specific threshold, center clipping, clustering, or head.
"""
import numpy as np

from hybrid_tsad.roca_fixed_reference_geometry import fixed_five_nn, normalized

MIN_SUPPORT = 5
EPS = 1e-12
METHODS = ('GLOBAL_FIT64', 'COMMAND_FIT64', 'JOINT_NN5', 'SHAM_COMMAND')


def shared_center(q, qr):
    q, qr = np.asarray(q, np.float64), np.asarray(qr, np.float64)
    if q.ndim != 2 or q.shape != qr.shape or len(q) == 0:
        raise ValueError('Same nonempty q/qr rows required')
    # Same builder for both global and conditional centers; no eps clipping.
    return np.concatenate((q, qr), axis=0).mean(axis=0)


def center_valid(center):
    return bool(np.isfinite(center).all() and np.linalg.norm(center) > EPS)


def cosine_sum(q, qr, centers):
    q, qr, centers = [np.asarray(x, np.float64) for x in (q, qr, centers)]
    if centers.ndim == 1:
        centers = np.broadcast_to(centers, q.shape)
    if q.shape != qr.shape or centers.shape != q.shape:
        raise ValueError('Both branches and shared centers must align')
    uq, vq = normalized(q)
    ur, vr = normalized(qr)
    uc, vc = normalized(centers)
    valid = vq & vr & vc
    # Existing audited two-step normalize/cosine formula, not a new weighting.
    den_c = np.maximum(np.linalg.norm(uc, axis=1), 1e-6)
    dq = 1 - (uq * uc).sum(axis=1) / (np.maximum(np.linalg.norm(uq, axis=1), 1e-6) * den_c)
    dr = 1 - (ur * uc).sum(axis=1) / (np.maximum(np.linalg.norm(ur, axis=1), 1e-6) * den_c)
    score = dq + dr
    score[~valid] = np.nan
    return score, valid


def fit_reference(q, qr, command_key, *, reference_role):
    if reference_role != 'FIT':
        raise PermissionError('No QUERY or CAL reference fitting')
    q, qr, key = np.asarray(q), np.asarray(qr), np.asarray(command_key)
    if len(q) != 64 or len(qr) != 64 or key.shape != (64,):
        raise PermissionError('Keep original fixed same-entity FIT64')
    if not (normalized(q)[1] & normalized(qr)[1]).all():
        raise ValueError('Do not silently replace invalid fixed references')
    global_center = shared_center(q, qr)
    contexts = {}
    for k in np.unique(key):
        use = key == k
        n = int(use.sum())
        center = shared_center(q[use], qr[use])
        contexts[int(k)] = dict(count=n, center=center, usable=n >= MIN_SUPPORT and center_valid(center))
    return dict(q=q.astype(np.float64), qr=qr.astype(np.float64), command_key=key,
                global_center=global_center, global_valid=center_valid(global_center), contexts=contexts)


def conditional_assignment(reference, keys, q_shape):
    centers = np.tile(reference['global_center'], (q_shape[0], 1))
    count = np.zeros(q_shape[0], dtype=np.int64)
    conditioned = np.zeros(q_shape[0], dtype=bool)
    invalid_context_center = np.zeros(q_shape[0], dtype=bool)
    for key in np.unique(keys):
        use = keys == key
        info = reference['contexts'].get(int(key))
        if info is None:
            continue
        count[use] = info['count']
        if info['usable']:
            centers[use] = info['center']
            conditioned[use] = True
        elif info['count'] >= MIN_SUPPORT:
            invalid_context_center[use] = True
    return centers, count, conditioned, invalid_context_center


def score_reference(reference, q, qr, command_key, *, sham_reference=None):
    q, qr, keys = np.asarray(q), np.asarray(qr), np.asarray(command_key)
    if keys.shape != (len(q),):
        raise ValueError('Same-time full context keys required')
    centers, count, conditioned, invalid_context_center = conditional_assignment(reference, keys, q.shape)
    global_score, global_valid = cosine_sum(q, qr, reference['global_center'])
    conditional_score, conditional_valid = cosine_sum(q, qr, centers)
    # Fixed fair simple control uses both branches and the exact same FIT64.
    ref_joint = np.concatenate((normalized(reference['q'])[0], normalized(reference['qr'])[0]), axis=1)
    query_joint = np.concatenate((normalized(q)[0], normalized(qr)[0]), axis=1)
    nn = fixed_five_nn(ref_joint, query_joint, reference_role='FIT', normalization='raw')
    branch_valid = normalized(q)[1] & normalized(qr)[1]
    nn_valid = nn['valid'] & branch_valid
    nn['score'][~nn_valid] = np.nan
    scores = {'GLOBAL_FIT64': (global_score, global_valid),
              'COMMAND_FIT64': (conditional_score, conditional_valid),
              'JOINT_NN5': (nn['score'], nn_valid)}
    if sham_reference is not None:
        sham_centers, sham_count, sham_conditioned, sham_invalid = conditional_assignment(sham_reference, keys, q.shape)
        if not np.array_equal(sham_count, count):
            raise ValueError('Sham must preserve exact command-key reference counts')
        scores['SHAM_COMMAND'] = cosine_sum(q, qr, sham_centers)
    return dict(scores=scores,
                context_reference_count=count, conditioned=conditioned,
                invalid_context_center=invalid_context_center)
