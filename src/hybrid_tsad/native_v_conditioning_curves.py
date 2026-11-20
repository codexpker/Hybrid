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
