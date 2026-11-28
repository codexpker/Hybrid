"""Summarize completed run receipts without evaluating or invoking a model."""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(root, dest):
    dest.mkdir(parents=True, exist_ok=True)
    training = []
    for domain in ("source", "PSM", "SWAT"):
        for arm in ("native", "history"):
            p = root / domain / arm / "receipt.json"
            r = read(p)
            training.append(dict(domain=domain, arm=arm, updates=r["steps"],
                selected_step=r["selected_step"], current_window_presentations=r["input_exposures"],
                process_wall_seconds=r["seconds"], peak_allocated_MiB=r["peak_bytes"] / 2**20,
                checkpoint_sha256=r["sha256"], source_checkpoint_sha256=(
                    read(root / "source" / arm / "receipt.json")["sha256"] if domain != "source" else ""),
                plan_sha256=r["plan_sha256"]))
    with (dest / "training_costs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(training[0]))
        writer.writeheader()
        writer.writerows(training)
    scoring = []
    independent_seconds = paired_seconds = 0.
    encoder_windows = 0
    for domain in ("PSM", "SWAT"):
        for stage in ("Z", "U"):
            for arm in ("native", "history"):
                chunks = [read(p) for p in (root / domain / "scores" / (stage + "_" + arm)).glob("*_chunk*.json")]
                scoring.append(dict(domain=domain, stage=stage, arm=arm,
                    scored_endpoint_count=sum(r["points"] for r in chunks),
                    allocated_chunk_wall_seconds=sum(r["seconds"] for r in chunks),
                    note="shared-pair chunks apportion joint wall time; not exclusive GPU time or separate latency"))
                old = [r for r in chunks if not r.get("frozen_encoder_reused", False)]
                independent_seconds += sum(r["seconds"] for r in old)
                encoder_windows += sum(r["points"] for r in old)
            for p in (root / domain / "scores" / ("paired_" + stage)).glob("*.json"):
                if p.name.startswith("parity_"): continue
                r = read(p)
                paired_seconds += r["seconds"]
                encoder_windows += r["frozen_encoder_forwards"]
    with (dest / "scoring_costs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scoring[0]))
        writer.writeheader()
        writer.writerows(scoring)
    result = dict(
        formal_source_fits=2, formal_U_fits=4, formal_F_fits=0,
        formal_optimizer_updates=sum(r["updates"] for r in training),
        formal_training_current_window_presentations=sum(r["current_window_presentations"] for r in training),
        source_SELECT_forward_windows=2*7*32,
        formal_scored_model_endpoints=sum(r["scored_endpoint_count"] for r in scoring),
        frozen_encoder_window_evaluations_from_completed_chunk_ledgers=encoder_windows,
        training_process_wall_seconds=sum(r["process_wall_seconds"] for r in training),
        completed_independent_score_chunk_wall_seconds=independent_seconds,
        completed_paired_score_chunk_wall_seconds=paired_seconds,
        summed_process_seconds_are_not_exclusive_A40_hours=True,
        execution="four A40 devices; independent processes shared devices to improve B1 throughput",
        exclusions="library startup, interrupted uncompleted scoring chunks and some CPU preparation lack complete timers; they are not silently counted as zero",
        optimization="frozen encoder reuse is bitwise checked; distinct decoder/history computations retained")
    (dest / "cost_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    run(args.run, args.destination)
