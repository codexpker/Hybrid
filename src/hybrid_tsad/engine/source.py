"""Frozen source sampling and calibration extracted from hybrid_10k_wave.py.

history=400, horizon=5, domain batch=64, rejection policy and RNG offsets are
unchanged. Small role_sizes is for synthetic engineering fixtures only.
"""
import numpy as np


def read_plan(path):
    with np.load(path) as stream:
        return {key: stream[key] for key in stream.files}


def draw_source_plan(sampler, rows, sources, sample_seed, *, role_sizes=None):
    role_sizes = role_sizes or {"fit": 640000, "select": 256, "cal": 256}
    arrays = {}
    quality = {}
    for role, offset in (("fit", 0), ("select", 50000), ("cal", 60000)):
        total = role_sizes[role]
        rng = np.random.default_rng(sample_seed + offset)
        rr, bb, cc = [], [], []
        drawn, rejected = 0, 0
        pools = {s: [j for j, r in enumerate(rows) if r["role"] == role and r["dataset"] == s]
                 for s in sources}
        assert all(pools.values())
        for n in range(total):
            domain = sources[(n // 64) % len(sources)]
            for _ in range(1000):
                j = int(rng.choice(pools[domain]))
                r = rows[j]
                b = int(r["forecast_origins"][rng.integers(len(r["forecast_origins"]))])
                c = int(rng.integers(len(r["columns"])))
                ar = sampler.array(r)
                x, y = ar[b - 400:b, c], ar[b:b + 5, c]
                drawn += 1
                ok = np.isfinite(x[0]) and not np.isinf(x).any() and not np.isinf(y).any() and np.isfinite(y).any()
                if ok:
                    break
                rejected += 1
            else:
                raise ValueError("No eligible source tuple; never relax quality or role rules")
            rr.append(j)
            bb.append(b)
            cc.append(c)
        arrays[role + "_record"] = np.array(rr, dtype="uint16")
        arrays[role + "_origin"] = np.array(bb, dtype="uint32")
        arrays[role + "_channel"] = np.array(cc, dtype="uint8")
        quality[role] = {"requested": total, "drawn": drawn, "rejected": rejected,
                         "rng_seed": sample_seed + offset}
    return arrays, quality


def calibrate_source(bank):
    """Exact legacy train() source-cal formula; never fits on query scores."""
    e = np.abs(bank["prediction"] - bank["target"]) / bank["history_scale"]
    valid = bank["valid_target"]
    pointvalid = valid.any(-1)
    point = np.divide(np.where(valid, e, 0).sum(-1), valid.sum(-1),
                      out=np.full(pointvalid.shape, np.nan, dtype="float32"), where=pointvalid)
    scale = max(float(np.median(point[pointvalid])), 1e-8)
    scores = point / scale
    return {"raw_residual": point, "standardized_score": scores, "valid": pointvalid}, {
        "normal_error_scale": scale, "threshold99": float(np.quantile(scores[pointvalid], .99))}
