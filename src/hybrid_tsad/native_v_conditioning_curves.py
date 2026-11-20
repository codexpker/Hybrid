"""Cached fixed-axis curves; thresholds are read from accepted evaluation only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from .native_v_conditioning_evaluation import (
    AXES, METHODS, TARGET_MEAN_ARM, EXTERNAL_MANIFEST, EXTERNAL_MANIFEST_SHA, checked, load_bank, read, sha)
from .conditional_response_evaluation import select_bank


LOW_FPR_DEFINITION = {
    "version": "low_fpr_empirical_ties_causal_C3_v1",
    "integral_bounds": [.01, .05], "FPR_caps": [.01, .03, .05],
    "pAUC": "piecewise-linear empirical ROC trapezoids, linearly clipped at bound; no convex hull",
    "normalized_pAUC": "area / bound; not McClish or sklearn max_fpr standardization",
    "short_curve": "NA if final attainable FPR is below integration bound; no extrapolation",
    "cap_selection": "max attainable tie-group TPR, then minimum FPR, then highest cutoff",
    "C3": "min of last3 scores only with3 consecutive valid observations; block first2 always false",
    "denominators": "all legal QUERY positive/negative points, including C3-unavailable block first2",
    "event_policy": "selected actual C3 bits through existing fixed event evaluator, no second confirmation",
    "use": "retrospective QUERY description only; no CAL changes or deployment thresholds",
}


def c3_scores(index, score, valid=None):
    """Causal confirmation score and its immutable eligibility, without backfill."""
    index, score = np.asarray(index), np.asarray(score, dtype=np.float64)
    valid = np.ones(len(index), bool) if valid is None else np.asarray(valid)
    if (index.ndim != 1 or index.dtype.kind not in "iu" or not len(index)
            or score.shape != index.shape or not np.isfinite(score).all()
            or np.any(np.diff(index) <= 0) or valid.shape != index.shape or valid.dtype != np.bool_):
        raise ValueError("Finite scores, ordered original indices and boolean validity required")
    eligible = np.zeros(len(index), bool)
    z = np.zeros(len(index), np.float64)
    if len(index) >= 3:
        eligible[2:] = (valid[2:] & valid[1:-1] & valid[:-2]
            & (np.diff(index)[1:] == 1) & (np.diff(index)[:-1] == 1))
        z[2:] = np.minimum(np.minimum(score[2:], score[1:-1]), score[:-2])
    return z, eligible


def tie_group_roc(labels, score, eligible=None):
    """All achievable tied-score ROC points; denominator never drops history gaps."""
    y, s = np.asarray(labels), np.asarray(score, dtype=np.float64)
    ok = np.ones(len(y), bool) if eligible is None else np.asarray(eligible)
    if (y.ndim != 1 or not len(y) or s.shape != y.shape or not np.isfinite(s).all()
            or not np.isin(y, [0, 1]).all() or not (y == 0).any() or not (y == 1).any()
            or ok.shape != y.shape or ok.dtype != np.bool_):
        raise ValueError("Both binary classes, finite scores, boolean eligibility required")
    chosen = np.flatnonzero(ok)
    order = chosen[np.argsort(-s[chosen], kind="stable")]
    ordered = s[order]
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(order)-1] if len(order) else np.array([], int)
    tp = np.r_[0, np.cumsum(y[order] == 1)[ends]].astype(np.int64)
    fp = np.r_[0, np.cumsum(y[order] == 0)[ends]].astype(np.int64)
    return dict(FPR=fp / int((y == 0).sum()), TPR=tp / int((y == 1).sum()), TP=tp, FP=fp,
        cutoffs=np.r_[np.inf, ordered[ends]], positive_points=int((y == 1).sum()),
        normal_points=int((y == 0).sum()), eligible_points=int(ok.sum()))


def partial_roc_area(fpr, tpr, bound):
    """Empirical polyline integral; duplicate-FPR vertical segments retain order."""
    x, y = np.asarray(fpr, float), np.asarray(tpr, float)
    if (x.ndim != 1 or y.shape != x.shape or not len(x) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or np.any(np.diff(x) < 0) or np.any(np.diff(y) < 0)
            or x[0] != 0 or y[0] != 0 or not 0 < bound <= 1):
        raise ValueError("Ordered empirical ROC beginning at origin and positive bound required")
    if x[-1] < bound:
        return None
    stop = np.searchsorted(x, bound, side="right")
    xx, yy = x[:stop], y[:stop]
    if xx[-1] < bound:
        edge = yy[-1] + (y[stop]-yy[-1]) * (bound-xx[-1]) / (x[stop]-xx[-1])
        xx, yy = np.r_[xx, bound], np.r_[yy, edge]
    return float(np.sum(np.diff(xx) * (yy[1:]+yy[:-1]) / 2))


def attainable_cap_point(curve, cap):
    """Select a real discrete point, never interpolate recall or optimize events."""
    if not 0 <= cap <= 1:
        raise ValueError("FPR cap must be in [0,1]")
    legal = np.flatnonzero(curve["FPR"] <= cap)
    best_tpr = np.max(curve["TPR"][legal])
    # Curves descend in cutoff and ascend in FPR/TPR, so first wins both ties.
    return int(legal[np.flatnonzero(curve["TPR"][legal] == best_tpr)[0]])


def fixed_bit_events(index, bits, labels):
    """Feed the actual selected stream to the existing strict-bit event channel."""
    from .timercd_swat_evaluation import evaluate_score
    index, bits, labels = np.asarray(index), np.asarray(bits, bool), np.asarray(labels)
    rows, _, _, _ = evaluate_score(index, bits.astype(float), labels, .5, index,
        query_start=int(index[0]), query_stop=int(index[-1])+1, original_length=len(labels),
        expected_events=None, ranking={"AP": 0., "ROC_AUC": 0.})
    row = next(r for r in rows if r["stream"] == "strict")
    fields = ("M", "Ntrue", "auxiliary_any_hit_events", "event_point_coverage",
        "long_steady_macro_coverage", "long_steady_point_coverage", "max_event_miss_points")
    return {key: row[key] for key in fields}


def low_fpr_entry(index, score, full_labels, accepted_rows, identity, *, strict_reference=None,
                  event_summary=fixed_bit_events):
    """Reusable two-stream analysis for one already sealed bank and metric pair.

    Future seeds call this same function with their own bank/accepted rows; noU
    and fixed TT/SS references retain their actual seed metadata.
    """
    index, score, full_labels = np.asarray(index), np.asarray(score, float), np.asarray(full_labels)
    y = full_labels[index]
    z, eligible = c3_scores(index, score)
    rows, arrays = [], {}
    accepted = {r["stream"]: r for r in accepted_rows}
    if set(accepted) != {"strict", "confirmed3"}:
        raise ValueError("Both original accepted CAL operating rows required")
    if accepted["strict"]["threshold"] != accepted["confirmed3"]["threshold"]:
        raise ValueError("Original shared CAL threshold must remain unchanged")
    for stream, s, ok in (("strict", score, np.ones(len(score), bool)), ("confirmed3", z, eligible)):
        curve = tie_group_roc(y, s, ok)
        if stream == "strict":
            if strict_reference is not None:
                for key in ("FPR", "TPR", "cutoffs"):
                    np.testing.assert_array_equal(curve[key], strict_reference[key])
            if abs(partial_roc_area(curve["FPR"], curve["TPR"], 1.)-accepted[stream]["ROC_AUC"]) > 1e-12:
                raise ValueError("Strict same-score full ROC must reproduce accepted AUC")
        old = accepted[stream]
        original_alarm = ok & (s > old["threshold"])
        tp, fp = int(np.sum(original_alarm & (y == 1))), int(np.sum(original_alarm & (y == 0)))
        if tp != old["TP"] or fp != old["FP"]:
            raise ValueError("Original fixed CAL strict/C3 alarm counts changed")
        if (fp / curve["normal_points"] != old["normal_FPR"]
                or tp / curve["positive_points"] != old["Recall"]):
            raise ValueError("Original fixed CAL FPR/recall changed")
        if stream == "confirmed3" and event_summary is not None:
            prior_events = event_summary(index, original_alarm, full_labels)
            if any(prior_events[k] != old[k] for k in prior_events):
                raise ValueError("Original CAL C3 coverage/matching must reproduce accepted metrics")
        row = dict(identity, stream=stream, normal_points=curve["normal_points"],
            attack_points=curve["positive_points"], eligible_alarm_points=curve["eligible_points"],
            CAL_operating_QUERY_FPR=old["normal_FPR"], CAL_operating_QUERY_Recall=old["Recall"],
            CAL_operating_QUERY_FP=fp, CAL_operating_QUERY_TP=tp)
        for bound in LOW_FPR_DEFINITION["integral_bounds"]:
            suffix = f"{round(bound*100):02d}"
            area = partial_roc_area(curve["FPR"], curve["TPR"], bound)
            row["pAUC_"+suffix] = area
            row["pAUC_div_bound_"+suffix] = area/bound if area is not None else None
        for cap in LOW_FPR_DEFINITION["FPR_caps"]:
            suffix = f"cap{round(cap*100):02d}"
            pos = attainable_cap_point(curve, cap)
            bits = ok & (s >= curve["cutoffs"][pos])
            tp2, fp2 = int(np.sum(bits & (y == 1))), int(np.sum(bits & (y == 0)))
            if (tp2, fp2) != (int(curve["TP"][pos]), int(curve["FP"][pos])):
                raise ValueError("Chosen tied ROC point must reconstruct actual alarm bits")
            row.update({suffix+"_FPR": float(curve["FPR"][pos]), suffix+"_TPR": float(curve["TPR"][pos]),
                suffix+"_TP": tp2, suffix+"_FP": fp2, suffix+"_curve_point": pos})
            if stream == "confirmed3" and event_summary is not None:
                row.update({suffix+"_"+k: v for k,v in event_summary(index, bits, full_labels).items()})
            arrays[stream+"_"+suffix+"_alarm"] = bits
        arrays.update({stream+"_"+k: v for k,v in curve.items() if isinstance(v, np.ndarray) and k != "cutoffs"})
        arrays[stream+"_eligible"] = ok
        rows.append(row)
    return rows, arrays


def low_fpr_cached(workspace, root, metrics, strict_curves, output, *, expected_metrics_sha,
                   methods=("noU", "TT", "SS", "R", TARGET_MEAN_ARM), cases=None):
    """One CPU cache pass; optional exact(d,method,seed) cases avoid old repeats.

    ``cases`` is an allowlist, not a data-selected subset. A later repetition
    calls only the eight new state identities and reuses the frozen20 old rows.
    Public output contains no selected/deployment thresholds.
    """
    start = time.perf_counter()
    w, root, metrics, output = map(Path, (workspace, root, metrics, output))
    strict_curves = Path(strict_curves) if strict_curves is not None else None
    requested = set(map(tuple, cases)) if cases is not None else None
    seen = set()
    if output.exists():
        raise FileExistsError(output)
    if sha(metrics) != expected_metrics_sha:
        raise ValueError("Accepted metric identity changed")
    result = read(metrics)
    seal_key = "global_seal_sha256" if "global_seal_sha256" in result else "raw_seal_sha256"
    seal = root / "GLOBAL_RAW_BANK_SEAL.json"
    if result["status"] != "COMPLETED" or sha(seal) != result[seal_key]:
        raise ValueError("Only fully sealed accepted cache inputs permitted")
    manifest = read(checked(w, dict(relative=EXTERNAL_MANIFEST, sha256=EXTERNAL_MANIFEST_SHA)))
    inputs = {str(metrics.resolve()): expected_metrics_sha, str(seal.resolve()): sha(seal),
              str((w/EXTERNAL_MANIFEST).resolve()): EXTERNAL_MANIFEST_SHA}
    output.mkdir(parents=True)
    (output / "DEFINITION.json").write_text(json.dumps(LOW_FPR_DEFINITION, indent=2), encoding="utf-8")
    rows = []
    for domain, axis in AXES.items():
        descriptor = manifest["labels"][domain]
        path = checked(w, dict(relative=descriptor["path"], sha256=descriptor["sha256"]))
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                labels = z[descriptor["key"]].copy()
        else:
            labels = np.load(path, allow_pickle=False).reshape(-1)
        inputs[str(path)] = descriptor["sha256"]
        reference_path = strict_curves / (domain+"_FULL_CURVES.npz") if strict_curves is not None else None
        references = load_bank(reference_path) if reference_path is not None else {}
        if reference_path is not None:
            inputs[str(reference_path.resolve())] = sha(reference_path)
        for name in methods:
            accepted = [r for r in result["rows"] if r["domain"] == domain and r["method"] == name]
            seeds = list(dict.fromkeys(r.get("seed") for r in accepted))
            for seed in seeds:
                identity = (domain, name, seed)
                if requested is not None and identity not in requested:
                    continue
                seen.add(identity)
                pair = [r for r in accepted if r.get("seed") == seed]
                descriptors = [r for r in result["bank_manifest"] if r["domain"] == domain and r["method"] == name
                               and (len(seeds) == 1 or r.get("seed") == seed)]
                banks = {}
                for role in ("CAL", "QUERY"):
                    matches = [r for r in descriptors if r["role"] == role]
                    if len(matches) != 1:
                        raise ValueError("Exactly one actual seed/method/role bank required")
                    ref = matches[0]; p = checked(w, ref)
                    if any(r["bank_"+role+"_sha256"] != ref["sha256"] for r in pair):
                        raise ValueError("Bank must match accepted metric row identity")
                    banks[role] = load_bank(p); inputs[str(p)] = ref["sha256"]
                bank, cal = banks["QUERY"], banks["CAL"]
                index = bank["absolute_time_index"]
                np.testing.assert_array_equal(index, np.arange(*axis))
                np.testing.assert_array_equal(bank["available_index"], index)
                if "canonical_score" not in bank or "canonical_score" not in cal:
                    raise ValueError("Preserve native canonical score, no alternate score formula")
                tau = float(np.quantile(cal["canonical_score"].astype(float), .99, method="higher"))
                if len(cal["canonical_score"]) != 512 or any(r["threshold"] != tau for r in pair):
                    raise ValueError("Original512 CAL threshold changed")
                ref_curve = ({"FPR": references[name+"_FPR"], "TPR": references[name+"_TPR"],
                              "cutoffs": references[name+"_ROC_thresholds"]}
                             if name+"_FPR" in references and seed in (None, 2025) else None)
                rr, arrays = low_fpr_entry(index, bank["canonical_score"], labels, pair,
                    dict(domain=domain, method=name, seed=seed, reference_seed=2025 if name in ("TT", "SS") else None,
                         QUERY_bank_sha256=next(r["sha256"] for r in descriptors if r["role"] == "QUERY")),
                    strict_reference=ref_curve)
                rows.extend(rr)
                np.savez_compressed(output / f"{domain}_{name}_seed{seed}_LOW_FPR.npz", **arrays)
    if requested is not None and seen != requested:
        raise ValueError("Every predeclared domain/method/seed case must be present exactly")
    from .left_swat_replay_repeats import write_csv
    write_csv(output / "LOW_FPR_SUPPLEMENT.csv", rows)
    receipt = dict(status="COMPLETED", definition=LOW_FPR_DEFINITION, rows=len(rows), input_hashes=inputs,
        module_sha256=sha(__file__), cpu_wall_seconds=time.perf_counter()-start, GPU_hours=0,
        neural_forwards=0, training=False, CAL_thresholds_changed=False, QUERY_descriptive_only=True)
    receipt["output_hashes"] = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "MANIFEST.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return rows, receipt


def render(workspace, root, metrics, output):
    start = time.perf_counter()
    w, root, metrics, output = map(Path, (workspace, root, metrics, output))
    if output.exists():
        raise FileExistsError(output)
    result = read(metrics)
    target_mean = result.get("mode") == "native_target_mean"
    seal_key = "global_seal_sha256" if target_mean else "raw_seal_sha256"
    if result["status"] != "COMPLETED" or sha(root / "GLOBAL_RAW_BANK_SEAL.json") != result[seal_key]:
        raise ValueError("Only sealed, fully evaluated scores may be plotted")
    primary_names = ["noU", "TT", "SS", "R", TARGET_MEAN_ARM] if target_mean else list(METHODS.values())
    checked(w, dict(relative=EXTERNAL_MANIFEST, sha256=EXTERNAL_MANIFEST_SHA))
    manifest = read(w / EXTERNAL_MANIFEST)
    models, labels, inputs = {}, {}, {str(metrics): sha(metrics)}
    for d, axis in AXES.items():
        descriptor = manifest["labels"][d]
        p = checked(w, dict(relative=descriptor["path"], sha256=descriptor["sha256"]))
        with_label = np.load(p, allow_pickle=False)
        if p.suffix == ".npz":
            with with_label as z:
                values = z[descriptor["key"]].copy()
        else:
            values = with_label.reshape(-1)
        labels[d] = values[slice(*axis)]
        models[d] = {}
        for name in primary_names:
            row = next(r for r in result["rows"] if r["domain"] == d and r["method"] == name and r["stream"] == "strict")
            if target_mean:
                descriptor = next(b for b in result["bank_manifest"] if b["domain"] == d and b["method"] == name and b["role"] == "QUERY")
                p = checked(w, descriptor)
            else:
                m = next(m for m, n in METHODS.items() if n == name)
                p = root / "banks" / d / m / "QUERY" / "BANK.npz"
            if sha(p) != row["bank_QUERY_sha256"]:
                raise ValueError("New score identity changed")
            bank = load_bank(p)
            np.testing.assert_array_equal(bank["absolute_time_index"], np.arange(*axis))
            models[d][name] = (bank["canonical_score"] if "canonical_score" in bank else bank["V"].astype(np.float64), row)
            inputs[str(p)] = sha(p)
    for descriptor in manifest["compatible_models"] + manifest["ranking_only_models"]:
        d, name = descriptor["domain"], descriptor["method"]
        bank = descriptor["banks"]["QUERY"] if "banks" in descriptor else descriptor
        p = w / bank["path"]
        _, score, _ = select_bank(p, AXES[d], expected_sha=bank["sha256"],
            causal=descriptor["causal"], **descriptor.get("bank_keys", {}))
        rows = [r for r in result["rows"] if r["domain"] == d and r["method"] == name]
        row = next((r for r in rows if r.get("stream") == "strict"), rows[0])
        models[d][name] = (score, row)
        inputs[str(p)] = bank["sha256"]
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    all_points = []
    for d, group in models.items():
        arrays, points = {}, {}
        y = labels[d]
        for name, (score, row) in group.items():
            ap, auc = float(average_precision_score(y, score)), float(roc_auc_score(y, score))
            if abs(ap - row["AP"]) > 1e-12 or abs(auc - row["ROC_AUC"]) > 1e-12:
                raise ValueError("Curve ranking must match the same-axis table")
            precision, recall, pt = precision_recall_curve(y, score)
            fpr, tpr, rt = roc_curve(y, score, drop_intermediate=False)
            for k, v in dict(precision=precision, recall=recall, PR_thresholds=pt,
                            FPR=fpr, TPR=tpr, ROC_thresholds=rt).items():
                arrays[name + "_" + k] = v
            op = dict(domain=d, method=name, AP=ap, ROC_AUC=auc, threshold=row.get("threshold"),
                normal_points=int((y == 0).sum()), attack_points=int((y == 1).sum()),
                calibrated=row.get("threshold") is not None)
            if op["calibrated"]:
                alarm = score > op["threshold"]
                tp, fp = int((alarm & (y == 1)).sum()), int((alarm & (y == 0)).sum())
                if tp != row["TP"] or fp != row["FP"]:
                    raise ValueError("Saved CAL threshold must reproduce accepted TP/FP")
                op.update(TP=tp, FP=fp, FPR=fp / op["normal_points"], recall=tp / op["attack_points"],
                    precision=tp / (tp + fp) if tp + fp else None,
                    low_FPR_panel_outside=fp / op["normal_points"] > .05)
            points[name] = op
            all_points.append(op)
        np.savez_compressed(output / (d + "_FULL_CURVES.npz"), **arrays)
        for scope, names in (("TARGET_MEAN" if target_mean else "ABCD", primary_names), ("EXTERNAL", list(group))):
            fig, axes = plt.subplots(1, 3, figsize=(17, 7))
            outside, empty, rank_only = [], [], []
            for number, name in enumerate(names):
                primary_colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
                if target_mean:
                    primary_colors += ("#E69F00",)
                color = primary_colors[number] if number < len(primary_colors) else plt.get_cmap("tab20")((2 * number) % 20)
                style = "-" if name in primary_names else "--"
                op = points[name]
                kw = dict(color=color, linestyle=style, lw=1.5)
                axes[0].plot(arrays[name + "_recall"], arrays[name + "_precision"],
                    label=f"{name} ({op['AP']:.3f})", **kw)
                for ax in axes[1:]:
                    ax.plot(arrays[name + "_FPR"], arrays[name + "_TPR"],
                        label=f"{name} ({op['ROC_AUC']:.3f})", **kw)
                if not op["calibrated"]:
                    rank_only.append(name)
                    continue
                if op["precision"] is not None:
                    axes[0].scatter(op["recall"], op["precision"], edgecolors=[color], facecolors="white", s=35, zorder=5)
                else:
                    empty.append(name)
                for ax in axes[1:]:
                    ax.scatter(op["FPR"], op["recall"], edgecolors=[color], facecolors="white", s=35, zorder=5)
                if op["low_FPR_panel_outside"]:
                    outside.append(f"{name}: {op['FPR']:.2%} FPR, {op['recall']:.2%} recall")
            axes[0].set(title="Complete PR (legend: AP)", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
            for ax, limit, title in ((axes[1], 1, "Complete ROC (legend: AUC)"), (axes[2], .05, "ROC: normal FPR 0–5%")):
                ax.set(title=title, xlabel="Normal false-positive rate", ylabel="Recall", xlim=(0, limit), ylim=(0, 1.02))
            for ax in axes:
                ax.grid(alpha=.2)
                ax.xaxis.set_major_formatter(PercentFormatter(1))
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.legend(fontsize=6.5, loc="lower left", ncols=1)
            caption = "CAL markers outside low-FPR panel: " + ("; ".join(outside) or "none")
            if rank_only:
                caption += "\nRanking only, no invented CAL marker: " + ", ".join(rank_only)
            if empty:
                caption += "\nNo positive alarms; PR precision undefined: " + ", ".join(empty)
            import textwrap
            caption = "\n".join(textwrap.fill(line, width=170) for line in caption.splitlines())
            permission = "Native-V arms: current100 endpoint; external permissions differ" if target_mean else "A/B/C/D causal endpoint; external context and training permissions differ"
            fig.suptitle(f"{d} | seed2025 | fixed common axis {AXES[d]} | {scope}\n" + permission)
            fig.text(.02, .025, caption, fontsize=8, va="bottom")
            fig.tight_layout(rect=(0, .19, 1, .90))
            for extension in ("png", "pdf", "svg"):
                fig.savefig(output / f"{d}_{scope}.{extension}", dpi=180)
            plt.close(fig)
    from .left_swat_replay_repeats import write_csv
    write_csv(output / "OPERATING_POINTS.csv", all_points)
    receipt = dict(status="COMPLETED", cpu_wall_seconds=time.perf_counter() - start,
        input_hashes=inputs, thresholds_changed=False, full_numeric_arrays_saved=True,
        no_alarm_PR_precision="NA", evaluation_only=True)
    (output / "CURVE_RECEIPT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("workspace", "root", "metrics", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    render(a.workspace, a.root, a.metrics, a.out)
