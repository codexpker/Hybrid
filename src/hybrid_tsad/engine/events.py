"""Extracted unchanged from project-owned frozen research helpers."""
import numpy as np


def real_blocks(index, valid):
    ix = np.flatnonzero(valid)
    if not len(ix):
        return []
    cut = np.r_[0, np.flatnonzero(np.diff(index[ix]) != 1) + 1, len(ix)]
    return [ix[a:b] for a, b in zip(cut[:-1], cut[1:])]


def binary_events(v):
    d = np.diff(np.r_[0, np.asarray(v, int), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(),
                    np.flatnonzero(d == -1).tolist()))


def greedy_assign(predicted, truth):
    """truth: sorted (event_id,start,end); call separately per real block."""
    assert all(a < b for a, b in predicted)
    assert all(a[1] <= b[0] for a, b in zip(predicted, predicted[1:]))
    assert all(a[2] <= b[1] for a, b in zip(truth, truth[1:]))
    matched, assignments = set(), []
    for lo, hi in predicted:
        candidates = [eid for eid, start, end in truth
                      if lo < end and hi > start and eid not in matched]
        hit = candidates[0] if candidates else None
        if hit is not None:
            matched.add(hit)
        assignments.append(hit)
    return assignments
