"""Source-frozen retrospective scores on the labelled public tasks; labels are opened by metrics only."""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from . import left_multisource as multi
from . import left_smd_targets as smd
from . import left_smd_reliability_eval as smd_metrics
from . import left_context_unmodulated as previous
from . import left_multisource_eval as old_scores
from . import external_endpoint_metrics as external
from .left_decoder_context_metrics import AUTHOR, MANIFEST
from .left_decoder_context_run import PUBLIC
from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .left_swat_author import sha, read, dump
from .native_full_window_metrics import truth, causal
from .native_tep_readout import AXIS_SHA
from .trajectory_evaluation import labels, independent_check
from .trajectory_experiment import ROLES, target

OUT = Path("results/left_history_relation_20261126_v1")
W, H = 192, 256
RANK = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC")


def members(w, d):
    return sorted(smd.smd.bindings(w)) if d == "SMD" else [d]


def test_values(w, d, e):
    if d == "SMD":
        rec = smd.smd.bindings(w)[e]
        _, sc = smd.normal_scaler(w, rec, True)
        return sc.transform(smd.smd.checked_data(w, rec, "test")).astype(np.float32), 0
    _, _, sc, _ = multi.target_inputs(w, d)
    return sc.transform(target(w, d, "TEST")).astype(np.float32), ROLES[d]["TEST"][0]


def public_bank(w, d, arm):
    from .left_history_relation import load_model, score_batch

    model, scales = load_model(w, d, arm)
    model.eval()
    fingerprint = multi.fingerprint(model.state_dict())
    checkpoint = w / OUT / d / arm / "selected.pt"
    checkpoint_hash = sha(checkpoint)
    initial_step = model.left.step
    for e in members(w, d):
        folder = w / OUT / "public" / d / e
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (arm + ".npz")
        if path.with_suffix(".json").exists():
            existing = read(path.with_suffix(".json"))
            assert (
                sha(path) == existing["sha256"]
                and checkpoint_hash == existing["checkpoint"]
            )
            continue
        x, offset = test_values(w, d, e)
        starts = np.arange(2 * W, len(x) - W + 1, W)
        assert starts[0] == 384 and starts[-1] + W == len(x) // W * W
        arrays = {}
        torch.cuda.synchronize()
        tic = time.monotonic()
        with torch.no_grad():
            for i in range(0, len(starts), 16):
                first = starts[i : i + 16]
                batch = torch.from_numpy(x[first[:, None] + np.arange(-H, W)]).cuda()
                result = score_batch(model, batch, scales)
                for k, v in result.items():
                    assert v.shape == (len(first), W) and torch.isfinite(v).all()
                    arrays.setdefault(k, []).append(
                        v.detach().cpu().numpy().reshape(-1)
                    )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - tic
        idx = np.arange(starts[0] + offset, starts[-1] + W + offset)
        np.savez_compressed(
            path,
            index=idx,
            history_start=np.repeat(starts - H + offset, W),
            current_start=np.repeat(starts + offset, W),
            available=np.repeat(starts + W - 1 + offset, W),
            **{k: np.concatenate(v) for k, v in arrays.items()},
        )
        assert (
            multi.fingerprint(model.state_dict()) == fingerprint
            and model.left.step == initial_step
        )
        dump(
            path.with_suffix(".json"),
            dict(
                domain=d,
                entity=e,
                arm=arm,
                points=len(idx),
                grid_origin=offset,
                seconds=elapsed,
                sha256=sha(path),
                checkpoint=checkpoint_hash,
                frozen_model_fingerprint=fingerprint,
                history=H,
                window=W,
                labels_read=False,
            ),
        )
        print("sealed", d, e, arm, len(idx), round(elapsed, 2), flush=True)


def archived_scores(w, d, e, n):
    """Return (index, score, bank identity). Every comparator uses its own authorized preprocessing."""
    if d == "SMD":
        left = {
            k: smd_metrics.bank(w, e, arm, "TEST")
            for k, arm in (
                ("frozen", "source"),
                ("scratch", "scratch_N"),
                ("target_normal", "N"),
            )
        }
        out = {k: (i, s, h) for k, (i, s, _, h) in left.items()}
        p = w / PUBLIC / "level" / "SMD" / (e + ".npz")
        with np.load(p) as z:
            out["level"] = (z["index"], z["score"], sha(p))
    else:
        out = {}
        for k, arm in (("frozen", "source"), ("target_normal", "N")):
            i, s, _ = old_scores.read_score(w, d, arm, "TEST")
            out[k] = (
                i,
                s,
                read(w / multi.OUT / d / "banks" / (arm + "_TEST.json"))["sha256"],
            )
        p = w / multi.NATIVE / d / "native" / "raw_scores.npz"
        digest = read(p.parent / "SCORE_RECEIPT.json")["bank_sha256"]
        assert digest == sha(p)
        with np.load(p) as z:
            out["scratch"] = (
                z["absolute_time_index"] - (132481 if d == "PSM" else 0),
                z["canonical_score"],
                digest,
            )
        p = w / previous.ROOT / d / "level.npz"
        digest = read(w / previous.ROOT / "METRICS.json")["domains"][d]["level"][
            "bank_sha256"
        ]
        assert digest == sha(p)
        with np.load(p) as z:
            out["level"] = (
                z["absolute_time_index"] - (132481 if d == "PSM" else 0),
                z["score"],
                digest,
            )
        pnl = next(
            p
            for p in read(w / MANIFEST)["panels"]
            if p["dataset"] == d and p["track"] == "offline-readout"
        )
        axis = np.arange(*pnl["axis"])
        for row in pnl["rows"]:
            name = row["method"]
            if name not in ("CATCH W192", "PCA residual", "TimeRadar official block0"):
                continue
            key = {
                "CATCH W192": "catch_" + str(row.get("archived_seed")),
                "PCA residual": "pca",
                "TimeRadar official block0": "timeradar_block",
            }[name]
            assert sha(w / row["path"]) == row["sha256"]
            out[key] = axis, previous.read_archive_curve(w, row, axis), row["sha256"]
    if d in ("PSM", "SMD"):
        protocol = w / external.OUT / "AXES_AND_INPUTS.json"
        assert sha(protocol) == AXIS_SHA
        row = next(
            r
            for r in read(protocol)["records"]
            if r["domain"] == d and r["entity"] == e
        )
        assert row["test_points"] == n
        out["dada"], _ = external.new_bank(
            w, row, "DADA", external.DADA_SHA["pytorch_model.bin"], AXIS_SHA
        )
        arm = external.ARMS[1]
        data, digest = external.cached_rcd(w, e, n)[arm]
        assert digest == row["TimeRCD_QUERY_sha256"][arm]
        out["timercd"] = data["index"], data["score"], digest
        ix, raw, bindings = causal(w, d, e)
        out["timeradar_endpoint"] = ix, raw["official_V"], bindings[0]["sha256"]
    return out


def history_panel(w, d, e, y):
    """The 512-history comparator has a separate, shorter intersection."""
    if d == "PSM":
        hist = external.history_ref(w, d, e, y)
    elif d == "SMD":
        roots = (
            "timeradar_cross_variable_history512_20261116_v1",
            "timeradar_smd_full_history512_20261116_v1",
        )
        hits = [
            w / "results" / r / e / "history_mean512" / "score" / "QUERY.npz"
            for r in roots
        ]
        (p,) = [path for path in hits if path.exists()]
        receipt = read(p.parent / "SCORE_RECEIPT.json")
        digest = next(b["sha256"] for b in receipt["banks"] if b["role"] == "QUERY")
        assert sha(p) == digest and receipt["record"] == e
        with np.load(p) as z:
            ix, score = z["index"].ravel(), z["score"].ravel()
            np.testing.assert_array_equal(ix, np.arange(512, len(y)))
            np.testing.assert_array_equal(ix, z["available_index"].ravel())
        hist = ix, score, digest
    else:
        raise ValueError(d)
    parts = {"history512": hist}
    for arm, key in (("continue", "native"), ("joint", "joint")):
        p = w / OUT / "public" / d / e / (arm + ".npz")
        digest = read(p.with_suffix(".json"))["sha256"]
        assert sha(p) == digest
        with np.load(p) as z:
            parts[arm + "_" + key] = z["index"], z[key], digest
    return parts


def metric_job(task):
    w, d, e, subset = task
    root = w / OUT / "public" / d / e
    root.mkdir(parents=True, exist_ok=True)
    dest = root / ("metrics_" + subset + ".json")
    if dest.exists():
        return
    if d == "SMD":
        y = truth(w, d, e)
        offset = 0
        n = len(y)
    else:
        y = labels(w, d)
        offset, stop = ROLES[d]["TEST"]
        n = stop - offset
    start = offset + 384
    stop = offset + n // W * W
    common_stop = min(stop, offset + n // 100 * 100) if d in ("PSM", "SMD") else stop
    if subset == "history":
        data = history_panel(w, d, e, y)
    elif subset == "archived":
        data = archived_scores(w, d, e, n)
    else:
        path = root / (subset + ".npz")
        digest = read(path.with_suffix(".json"))["sha256"]
        assert digest == sha(path)
        with np.load(path) as z:
            idx = z["index"]
            starts = (idx - offset) // W * W + offset
            np.testing.assert_array_equal(idx, np.arange(start, stop))
            np.testing.assert_array_equal(starts - H, z["history_start"])
            np.testing.assert_array_equal(starts + W - 1, z["available"])
            data = {
                subset + "_" + key: (idx, z[key], digest)
                for key in z.files
                if key not in ("index", "history_start", "current_start", "available")
            }
    metric = load_author_metric(w / AUTHOR)
    rows = []
    t0 = time.monotonic()
    for name, (ix, s, digest) in data.items():
        assert ix.ndim == s.ndim == 1 and len(s) == len(ix) and np.all(np.diff(ix) > 0)
        cache = {}
        panels = (
            [("history512", common_stop)]
            if subset == "history"
            else [("complete", stop), ("external", common_stop)]
        )
        for panel, end in panels:
            if panel == "history512":
                start = 512
            if (
                name in ("dada", "timeradar_endpoint", "timercd")
                and panel != "external"
            ):
                continue
            axis = np.arange(start, end)
            loc = np.searchsorted(ix, axis)
            np.testing.assert_array_equal(ix[loc], axis)
            if end not in cache:
                val = s[loc]
                yy = y[axis]
                assert np.isfinite(val).all() and len(np.unique(yy)) == 2
                met, _ = continuous_metrics(yy, val, metric)
                err = float(
                    max(
                        abs(
                            independent_check(yy, val)
                            - np.array([met["AP"], met["ROC_AUC"]])
                        )
                    )
                )
                assert err < 1e-10
                cache[end] = {m: met[m] for m in RANK} | dict(check_ap_roc_maxerr=err)
            rows.append(
                dict(
                    domain=d,
                    entity=e,
                    method=name,
                    panel=panel,
                    start=start,
                    stop=end,
                    points=end - start,
                    positives=int(y[start:end].sum()),
                    sha256=digest,
                    **cache[end],
                )
            )
    tmp = dest.with_suffix(".part")
    dump(tmp, dict(rows=rows, seconds=time.monotonic() - t0))
    os.replace(tmp, dest)
    print(d, e, subset, len(rows), flush=True)


def metrics(w, subset, domains, workers):
    tasks = [(w, d, e, subset) for d in domains for e in members(w, d)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(metric_job, tasks))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("action", choices=("score", "metrics"))
    p.add_argument("arm", choices=("continue", "joint", "archived", "history"))
    p.add_argument("domains", nargs="+", choices=("PSM", "SWAT", "SMD"))
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.action == "metrics":
        metrics(a.workspace, a.arm, a.domains, a.workers)
    else:
        for d in a.domains:
            public_bank(a.workspace, d, a.arm)
