"""Fixed two-domain evaluation of aggregate and per-output source preservation."""

import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from .left_native_swat_metrics import continuous_metrics, load_author_metric, sha
from .left_swat_natural_u_evaluation import checked, read_json, write_json
from .left_swat_replay_repeats import KEYS, write_csv
from .timercd_swat_evaluation import evaluate_score


METRIC = Path("literature/novelty_review/left_source_20261031/ts_ad_evaluation/auc_vus/metrics.py")
MODELS = ("noU", "TT", "SS", "R", "P", "V")
BASELINES = {
    "SWAT": ("results/left_author_swat_source_replay_metrics_20261118_v1/METRICS.json",
             "353b96f7493bf7bb22ed4d44274116b54cfa7044238186bb3a82ebd173f0b885"),
    "PSM": ("results/left_author_psm_source_replay_metrics_20261118_v1/METRICS.json",
            "0ec57ff0474f5bcd88c60118a061bbe65cc220a54115e30c8957e4ee6ed10716"),
}


def score_bank(path, axis, channels, anchor):
    index = np.arange(*axis)
    starts = anchor + (index-anchor)//192*192
    with np.load(path, allow_pickle=False) as bank:
        score, per_channel = bank["canonical_score"], bank["per_channel"]
        if (not np.array_equal(bank["absolute_time_index"], index)
                or not np.array_equal(bank["available_index"], starts+191)
                or score.shape != index.shape or score.dtype != np.float32 or not np.isfinite(score).all()
                or per_channel.shape != (len(index), channels)
                or not np.array_equal(score, per_channel.mean(-1))):
            raise ValueError("Preservation bank differs from the fixed native scoring definition")
    return score


def baseline(workspace, domain):
    relative, expected = BASELINES[domain]
    old = read_json(checked(workspace / relative, expected))
    if domain == "SWAT":
        rows = [r for r in old["rows"] if r["method"] in MODELS[:4]]
        paths = {"noU": workspace / "results/left_author_swat_u_cache_20261118_v1/QUERY.npz"}
        for arm in ("TT", "SS", "R"):
            paths[arm] = workspace / f"results/left_author_swat_source_replay_20261118_v1/{arm}/QUERY.npz"
        cache_seal = read_json(paths["noU"].parent / "CACHE_SEAL.json")
        expected_scores = {"noU": cache_seal["banks"]["QUERY"]}
        expected_scores.update({arm:old["replay"]["arms"][arm]["score_sha256"]["QUERY"]
                                for arm in ("TT","SS","R")})
    else:
        rows = old["references"] + [r for r in old["rows"] if r["seed"] == 2025]
        root = workspace / "results/left_author_psm_source_replay_20261118_v1"
        paths = {"noU": root / "noU/QUERY.npz"}
        paths.update({arm: root / f"seed2025/{arm}/QUERY.npz" for arm in ("TT", "SS", "R")})
        expected_scores = {"noU":old["identities"]["None/noU"]["banks"]["QUERY"]}
        expected_scores.update({arm:old["identities"][f"2025/{arm}"]["banks"]["QUERY"]
                                for arm in ("TT","SS","R")})
    scores = {}
    for model, path in paths.items():
        with np.load(checked(path,expected_scores[model]), allow_pickle=False) as bank:
            scores[model] = bank["canonical_score"].copy()
    return [dict(row) for row in rows], scores


def validated_inputs(workspace, root):
    seal = read_json(root / "GLOBAL_RAW_BANK_SEAL.json")
    if (seal["status"] != "COMPLETED" or not seal["no_labels_decoded"]
            or len(seal["banks"]) != 8 or len(seal["jobs"]) != 4
            or any(job["OS_exit_code"] != 0 for job in seal["jobs"])):
        raise ValueError("Require all four fixed, completed P/V jobs before evaluation")
    expected = {f"{domain}/{mode}/{role}.npz" for domain in BASELINES for mode in ("P","V")
                for role in ("CAL","QUERY")}
    ids = {item["relative"].replace("\\","/"):item["sha256"] for item in seal["banks"]}
    if set(ids) != expected:
        raise ValueError("Preservation seal does not contain exactly the fixed eight banks")
    pending, identities = [], {"seal_sha256":sha(root / "GLOBAL_RAW_BANK_SEAL.json"),
                               "GPU_hours_including_diagnostic_upper_bound":seal["actual_GPU_hours"],
                               "measured_GPU_hours":seal["measured_GPU_hours"],
                               "diagnostic_cost":seal["diagnostic_cost"]}
    for domain in BASELINES:
        task_path = root / domain / "TASK.json"
        task = read_json(task_path)
        frozen = read_json(root / domain / "FROZEN.json")
        old_plan = workspace / ("results/left_author_swat_source_replay_20261118_v1/SAMPLING_PLAN.npz"
            if domain == "SWAT" else "results/left_author_psm_source_replay_20261118_v1/seed2025/SAMPLING_PLAN.npz")
        accepted_gate = read_json(root / domain / "ACCEPTED_GATE.json")
        gate_path = root / domain / accepted_gate["receipt_relative"]
        gate = read_json(checked(gate_path,accepted_gate["receipt_sha256"]))
        if (sha(root / domain / "SAMPLING_PLAN.npz") != sha(old_plan)
                or frozen["plan_sha256"] != sha(old_plan) or frozen["task_sha256"] != sha(task_path)
                or gate["status"] != "PASS"):
            raise ValueError("Preservation requires the original seed2025 replay draws and a passing incremental gate")
        channels = task["data"]["target_test"]["shape"][1]
        scores = {}
        for mode in ("P","V"):
            folder = root / domain / mode
            train = read_json(folder / "train/RECEIPT.json")
            score_receipt = read_json(folder / "RECEIPT.json")
            binding = read_json(folder / "PRESERVATION_BINDING.json")
            trace = read_json(folder / "train/PRESERVATION_TRACE.json")
            source = task["source"]
            if (train["status"] != "COMPLETED" or train["seed"] != 2025 or train["updates"] != 2000
                    or train["batch"] != 256 or train["native_forward_calls"] != 2000
                    or train["labels_read"] or train["CAL_QUERY_read"] or not train["endpoint_only"]
                    or train["initial_python_step"] != source["python_step"]
                    or train["final_python_step"] != source["python_step"]+2000
                    or train["source_checkpoint_sha256"] != source["checkpoint"]["sha256"]
                    or train["scaler_sha256"] != source["scaler"]["sha256"]
                    or train["plan_sha256"] != frozen["plan_sha256"]
                    or binding["mode"] != mode or binding["coefficient"] != 1
                    or not binding["teacher_eval"] or not binding["teacher_frozen"]
                    or binding["source_only_second_half"] != [128,256]
                    or binding["helper_sha256"] != frozen["helper_sha256"]
                    or not trace["teacher_state_unchanged"] or not trace["teacher_gradients_none"]
                    or not trace["teacher_only_source_FIT"]
                    or trace["native_training_forward_calls"] != 2000
                    or trace["student_source_eval_forward_calls"] != 2000):
                raise ValueError(f"{domain}/{mode} differs from the fixed source-preservation objective")
            if (score_receipt["labels_read"] or not score_receipt["readout_state_step_RNG_immutable"]
                    or not score_receipt["source_checkpoint_unchanged"]
                    or score_receipt["python_step"] != train["final_python_step"]
                    or score_receipt["checkpoint_sha256"] != train["checkpoint_sha256"]):
                raise ValueError("Preservation scoring must use the fixed endpoint and native state")
            checked(folder / "train/endpoint2000.pth",train["checkpoint_sha256"])
            scores[mode] = {}
            for role,axis in task["score_axes"].items():
                relative = f"{domain}/{mode}/{role}.npz"
                scores[mode][role] = score_bank(checked(root/relative,ids[relative]),axis,channels,
                                                task["scoring"]["anchor"])
            identities[f"{domain}/{mode}"] = dict(checkpoint_sha256=train["checkpoint_sha256"],
                plan_sha256=train["plan_sha256"], trace=trace, task_sha256=sha(task_path))
        pending.append((domain,task,scores))
    jobs = []
    for domain,task,scores in pending:
        descriptor = task["labels"]
        path = checked(workspace/descriptor["relative"],descriptor["sha256"])
        if path.suffix == ".npz":
            with np.load(path,allow_pickle=False) as z:
                labels = z["label"].copy()
        else:
            labels = np.load(path,allow_pickle=False).reshape(-1)
        if not np.isin(labels,(0,1)).all():
            raise ValueError("Binary native label asset required")
        jobs.append((domain,task,labels,scores))
    return jobs,identities


def curves(output, domain, labels, scores, rows):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    palette = ("#374151", "#D55E00", "#8C4E96", "#0072B2", "#C08A00", "#008661")
    data, handles, outside = {}, [], []
    for model, color in zip(MODELS, palette):
        precision, recall, _ = precision_recall_curve(labels, scores[model])
        fpr, tpr, _ = roc_curve(labels, scores[model])
        data.update({f"{model}_precision": precision, f"{model}_recall": recall,
                     f"{model}_fpr": fpr, f"{model}_tpr": tpr})
        line, = axes[0].plot(recall, precision, color=color, label=model)
        handles.append(line)
        for axis in axes[1:]:
            axis.plot(fpr, tpr, color=color)
        row = next(r for r in rows if r["method"] == model and r["stream"] == "strict")
        if row["Precision"] is not None:
            axes[0].scatter(row["Recall"], row["Precision"], edgecolor=color, facecolor="white", zorder=5)
        axes[1].scatter(row["normal_FPR"], row["Recall"], edgecolor=color, facecolor="white", zorder=5)
        if row["normal_FPR"] <= .05:
            axes[2].scatter(row["normal_FPR"], row["Recall"], edgecolor=color, facecolor="white", zorder=5)
        else:
            outside.append(f"{model} {row['normal_FPR']:.1%}")
    for axis, title, x, y, limit in zip(axes, ("Precision–recall", "ROC", "ROC at low false positive rates"),
            ("Point recall", "Normal false positive rate", "Normal false positive rate"),
            ("Point precision", "Point recall", "Point recall"), (1, 1, .05)):
        axis.set(title=title, xlabel=x, ylabel=y, xlim=(0,limit), ylim=(0,1.02))
        axis.grid(alpha=.15)
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False)
    caption = "Circles show each model’s fixed mixed-CAL operating point on QUERY."
    if outside:
        caption += " Outside low-FPR panel: " + ", ".join(outside)
    fig.text(.5,.02,caption,ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.92))
    fig.savefig(output / f"{domain}_curves.png", dpi=220)
    fig.savefig(output / f"{domain}_curves.pdf")
    plt.close(fig)
    np.savez_compressed(output / f"{domain}_curve_arrays.npz", **data)


def evaluate(workspace, root, output):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    jobs, identities = validated_inputs(workspace, root)
    output.mkdir(parents=True, exist_ok=False)
    all_rows, all_events, contrasts, surfaces = [], [], [], {}
    metric = load_author_metric(workspace / METRIC)
    for domain, task, labels, new_scores in jobs:
        rows, scores = baseline(workspace, domain)
        axis = task["score_axes"]["QUERY"]
        index = np.arange(*axis)
        anchor = task["scoring"]["anchor"]
        available = anchor + (index-anchor)//192*192 +191
        cal_axis = task["score_axes"]["CAL"]
        selected = np.linspace(cal_axis[0],cal_axis[1]-1,512,dtype=np.int64)
        for arm in ("P", "V"):
            values = new_scores[arm]
            tau = float(np.quantile(values["CAL"][selected-cal_axis[0]],.99,method="higher"))
            rank, surface = continuous_metrics(labels[index],values["QUERY"],metric)
            streams, events, _, _ = evaluate_score(index,values["QUERY"],labels,tau,available,
                query_start=axis[0],query_stop=axis[1],original_length=len(labels),expected_events=None,ranking=rank)
            for row in streams:
                row.update(rank, method=arm, seed=2025)
            rows.extend(streams)
            scores[arm] = values["QUERY"]
            all_events.extend(dict(domain=domain,method=arm,**r) for r in events)
            for key, array in zip(("TPR","FPR","precision","windows"),surface):
                surfaces[f"{domain}_{arm}_{key}"] = array
        lookup = {(row["method"],row["stream"]):row for row in rows}
        for left,right in (("V","R"),("V","P"),("P","R"),("V","noU"),("P","noU")):
            for stream in ("strict","confirmed3"):
                a,b = lookup[(left,stream)],lookup[(right,stream)]
                contrasts.append(dict(domain=domain,contrast=f"{left}−{right}",stream=stream,
                                      **{key:a[key]-b[key] for key in KEYS}))
        curves(output,domain,labels[index],scores,rows)
        all_rows.extend(dict(domain=domain,**row) for row in rows)
    write_csv(output / "comparison.csv", all_rows)
    write_csv(output / "contrasts.csv",contrasts)
    write_json(output / "EVENTS.json",all_events)
    np.savez_compressed(output / "VUS_SURFACES.npz",**surfaces)
    result = dict(rows=all_rows,contrasts=contrasts,identities=identities,cpu_wall_seconds=time.perf_counter()-started)
    write_json(output / "METRICS.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace","root","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    result=evaluate(args.workspace,args.root,args.out)
    print(f"Saved {len(result['rows'])} model/stream rows")


if __name__ == "__main__":
    main()
