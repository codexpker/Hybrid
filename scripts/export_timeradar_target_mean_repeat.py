"""Public tables from frozen evaluation; never loads a model or score bank."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[2]
RAW = W / "results/timeradar_target_mean_repeat_20261120_v1"
EVAL = W / "results/timeradar_target_mean_repeat_metrics_20261120_v1"
OUT = W / "hybrid_ad/docs/experiments/timeradar_target_mean_repeat_20261120"
FIELDS = (
    "AP", "ROC_AUC", "VUS_PR", "VUS_ROC", "FP", "TP", "normal_FPR", "Recall", "F1",
    "M", "Ntrue", "auxiliary_any_hit_events", "one_to_one_eventPrecision",
    "one_to_one_eventRecall", "one_to_one_eventF1", "normal_alarm_fragments",
    "event_macro_coverage", "event_point_coverage", "long_steady_macro_coverage",
    "long_steady_point_coverage", "max_event_miss_points", "delay_hit_only_median",
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(p, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def mdtable(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def run():
    metrics = json.loads((EVAL / "METRICS.json").read_text())
    seal = json.loads((RAW / "GLOBAL_RAW_BANK_SEAL.json").read_text())
    assert metrics["status"] == "COMPLETED" and seal["status"] == "COMPLETED"
    assert sha(RAW / "GLOBAL_RAW_BANK_SEAL.json") == metrics["global_seal_sha256"]
    rows = metrics["rows"]
    assert len(rows) == 68
    native = [r for r in rows if r["method"] in ("noU", "TT", "SS", "R", "R_target_mean")]
    assert len(native) == 36
    assert all(r.get("seed") in (2025, 2026, 2027) for r in native if r["method"] in ("R", "R_target_mean"))
    assert all(r.get("seed") is None for r in native if r["method"] == "noU")
    write_csv(OUT / "methods.csv", rows)
    write_csv(OUT / "paired_seeds.csv", metrics["paired"])
    write_csv(OUT / "paired_summary.csv", metrics["paired_summary"])
    events = json.loads((EVAL / "EVENTS.json").read_text())
    write_csv(OUT / "events.csv", events)
    aggregates = []
    for d in ("SWAT", "PSM"):
        for method in ("noU", "TT", "SS", "R", "R_target_mean"):
            for stream in ("strict", "confirmed3"):
                group = [r for r in native if (r["domain"], r["method"], r["stream"]) == (d, method, stream)]
                expected = 3 if method in ("R", "R_target_mean") else 1
                assert len(group) == expected
                for key in FIELDS:
                    values = [r[key] for r in group if r.get(key) is not None]
                    aggregates.append(dict(domain=d, method=method, stream=stream, metric=key, n=len(values),
                        mean=float(np.mean(values)) if values else None,
                        sample_SD=float(np.std(values, ddof=1)) if len(values) > 1 else None, ddof=1,
                        scope="three adaptation seeds at one pretrained state" if expected == 3 else "one fixed reference; no fabricated SD"))
    write_csv(OUT / "method_summary.csv", aggregates)
    jobs = []
    for job in seal["jobs"]:
        public = {k: job.get(k) for k in ("identity", "stage", "status", "GPU", "pid", "UTC_end", "OS_exit_code", "actual_GPU_seconds", "guard_elapsed_seconds")}
        public["cleanup_status"] = (job.get("cleanup") or {}).get("status")
        candidates = list((RAW / "lifecycle").glob(f'*/{job["identity"]}/STARTED.json'))
        # Original rejected attempt has no child. Never assign recovery start to it.
        if job["status"] == "COMPLETED":
            matches = [p for p in candidates if json.loads(p.read_text()).get("pid") == job["pid"]]
            if len(matches) == 1:
                started = json.loads(matches[0].read_text())
                public["UTC_start"] = started.get("UTC_start", started.get("UTC"))
                public["start_receipt_sha256"] = sha(matches[0])
        jobs.append(public)
    assert len(jobs) == 53 and sum(r["status"] == "COMPLETED" for r in jobs) == 52
    assert abs(sum(r["actual_GPU_seconds"] for r in jobs) / 3600 - seal["actual_GPU_hours"]) < 1e-10
    write_csv(OUT / "COMPUTE.csv", jobs)
    identities = dict(official_pretrained_weights_published=False, native_checkpoints_embed_upstream_weights=True,
        checkpoints_and_optimizers="retained outside publicGit; hashes only", trainings=seal["trainings"],
        banks=seal["banks"], seed2025_references=seal["baseline_seed2025_references"])
    (OUT / "MODEL_AND_CACHE_IDENTITIES.json").write_text(json.dumps(identities, indent=2), encoding="utf-8")
    provenance = dict(status="COMPLETED", raw_seal_sha256=sha(RAW / "GLOBAL_RAW_BANK_SEAL.json"),
        metrics_sha256=sha(EVAL / "METRICS.json"), actual_GPU_hours=seal["actual_GPU_hours"],
        successful_GPU_jobs=52, prestart_guard_failures=1, failed_attempt_GPU_seconds=0,
        original_successful_readouts=30, recovered_previously_unstarted_readouts=10,
        recovery_plan_sha256=seal["resource_recovery"]["plan_sha256"],
        original_controller_exit="uncaught guard exception, retained traceback; exact OS status unavailable to non-parent",
        old_results_recomputed=False, training_source="public SWaT31/PSM25 only",
        fixed_pretrained_initial=True, adaptation_seeds=[2025, 2026, 2027],
        noU="one fixed reference, no sample SD", TT_SS="seed2025 reference only",
        source_teacher_or_new_modules=False, new_trainings=8, new_banks=16,
        calibration="same512 points, own q99/higher/strict >; SWaT34 attack / PSM512 attack; not a certified FPR budget",
        privacy="raw checkpoints, numeric banks/scalers and detailed resource diagnostics retained outside publicGit")
    runtime = W / "preparations/timeradar_target_mean_repeat_20261120_v1/RUNTIME_MANIFEST.json"
    provenance.update(runtime_manifest_sha256=sha(runtime), runtime_files=json.loads(runtime.read_text())["files"],
        gate_metadata_clarification_sha256=sha(RAW / "GATE_METADATA_CLARIFICATION.json"),
        annotations_sha256=sha(RAW / "ANNOTATIONS.json"),
        runtime_note="Future receipt wording and optional resource routing changed after launch; original executed bytes retained, no scientific worker changes.")
    (OUT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    rank = []
    for r in native:
        if r["stream"] == "strict":
            rank.append([r["domain"], r["method"].replace("R_target_mean", "R-MeanT"),
                         "fixed" if r.get("seed") is None else r["seed"],
                         *[f'{r[k]:.6f}' for k in ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC")]])
    ops = []
    for r in native:
        ops.append([r["domain"], r["method"].replace("R_target_mean", "R-MeanT"), r.get("seed") or "fixed", r["stream"],
            r["FP"], f'{100*r["normal_FPR"]:.3f}', f'{100*r["Recall"]:.3f}',
            f'{r["M"]}/{r["Ntrue"]}', r["auxiliary_any_hit_events"], f'{100*r["one_to_one_eventF1"]:.3f}',
            r["normal_alarm_fragments"], f'{100*r["event_point_coverage"]:.3f}',
            f'{100*r["long_steady_macro_coverage"]:.3f}/{100*r["long_steady_point_coverage"]:.3f}',
            r["max_event_miss_points"], r["delay_hit_only_median"]])
    pairheaders = ["Task", "Contrast", "Stream", "Metric", "n", "Mean difference", "Sample SD"]
    pairtable = []
    for r in metrics["paired_summary"]:
        rate = r["metric"] not in ("FP", "TP", "M", "auxiliary_any_hit_events", "max_event_miss_points", "normal_alarm_fragments", "delay_hit_only_median")
        scale = 100 if rate else 1
        pairtable.append([r["domain"], r["contrast"].replace("R_target_mean", "R-MeanT"), r["stream"],
            r["metric"] + (" (pp)" if rate else ""), r["n"],
            "NA" if r["mean"] is None else f'{scale*r["mean"]:.6f}',
            "NA" if r["sample_SD"] is None else f'{scale*r["sample_SD"]:.6f}'])
    report = """# Fixed R-MeanT adaptation repetition

Two exposed public tasks; seeds2025/2026/2027 share one official pretrained initial state. Only adaptation randomness changes. noU is one fixed reference; TT/SS retain only their accepted seed2025 results. No repeated pretraining, independent-domain or blind-test claim is made.

## Per-seed ranking

""" + mdtable(["Task", "Method", "Seed", "AP", "ROC-AUC", "VUS-PR", "VUS-ROC"], rank)
    report += "\n\nAP is Average Precision. VUS uses the previously pinned implementation/window100/threshold250. Original seed2025 metrics and external baselines are reused. Complete exact rows and permissions remain in methods.csv.\n\n## Original CAL operation\n\n"
    report += mdtable(["Task", "Method", "Seed", "Stream", "Normal FP", "FPR %", "Recall %", "Matched/events", "Any-hit", "Event F1 %", "Normal fragments", "Complete-event coverage %", "Long macro/pooled %", "Longest miss", "Hit-only median delay"], ops)
    report += "\n\nCAL is the original512 mixed points:34 SWaT attack points and512 PSM attack points. Each model uses its own q99/higher threshold with strict >; this is not a certified normal-FPR budget. Strict and past-three/no-backfill confirmation remain separate. Complete-event coverage and all-QUERY point recall use different denominators; any-hit is not one-to-one matching. Long events exceed512 points, with steady coverage from event start+512; macro and pooled are both retained. Nonhits are right-censored and appear in events.csv.\n\n## Paired changes\n\n"
    report += mdtable(pairheaders, pairtable)
    report += "\n\nDifferences are within-domain and within-seed against R; noU is the same frozen reference for all three contrasts. Sample SD uses ddof=1, n=3. Seeds are not datasets or independent event samples. Low-FPR areas/cap coverage are descriptive QUERY calculations in the single LOW_FPR_SUPPLEMENT table; their cutoffs never replace CAL thresholds.\n"
    report += """

## Technical decision

PSM ranking gains repeat in all three pairs: R-MeanT minus R AP is+0.864,+0.400,+1.408pp, mean+0.891+/-0.505pp; ROC-AUC mean+0.758+/-0.125pp. Against the one frozen noU reference AP is+1.029+/-0.122pp. This is a real but task-limited ranking effect, not broad superiority: VUS-PR changes-0.534+/-0.700pp and VUS-ROC-0.827+/-0.315pp versus R. At original CAL, C3 normal FPR increases+2.256+/-2.678pp with recall+5.393+/-2.816pp, long steady macro+5.230+/-5.329pp and pooled+13.491+/-4.417pp. Matched events12/10/11 versus R11/12/12 and any-hit14/11/12 versus11/12/12 do not establish a repeated extra-event advantage. Longest miss improves only in2025 (1228 to879); in2026/2027 both arms remain1228.

On SWaT, AP decreases in all seeds (-0.427+/-0.241pp), ROC-AUC improves+0.338+/-0.024pp, and VUS-PR/ROC decline relative to R. C3 FPR falls0.091+/-0.007pp alongside recall-1.276+/-0.867pp and long macro-3.007+/-5.043pp. This does not demonstrate the corresponding sustained-detection advantage. The apparent precision of pooled coverage does not replace the variable per-event macro coverage.

Low-FPR evidence qualifies the PSM positive: strict normalized partial areas improve over R, but the C3 effect is supported more clearly through5% than through1% FPR. Whole-tie attainable recall, continuous-area gain, event matching and long macro/pooled coverage can move differently; the supplement retains every case. These are retrospective comparisons, not threshold recommendations. Against noU, reduced PSM false alarms coexist with substantially lower coverage, so the learned candidate is not an unconditional replacement for noU.

Retain this fixed candidate as a reproducible, limited PSM learning effect with SWaT negatives and operating costs. It is not established cross-task evidence preservation or contamination robustness, and mature TimeRadar, replay and the error/disagreement decomposition keep their original attribution. This matrix is complete; no loss, threshold, seed, structure or data extension is queued.

## Computation and retained evidence

Eight new adaptations and forty new readout parts plus four normal engineering checks consumed10.828305763A40GPUh, including startup and cleanup. All52 actual GPU jobs completed and cleared. A separate pre-start guard rejection used zero GPU seconds and2.233744947CPU wall seconds. The original thirty successful readouts were retained; only ten never-started parts were recovered after guarded resources became available. The original controller exception, exact-status limitation, CPU pidfd fix, resource receipts and original runtime are preserved outside Git. No successful training/inference was repeated.

The final16-bank seal and metric hashes are in PROVENANCE.json; actual53 dispatch records are in COMPUTE.csv. Checkpoints/optimizers embed official pretrained weights and remain archived rather than republished. MODEL_AND_CACHE_IDENTITIES.json carries their hashes, plans, scalers and all bank identities. Full per-seed observations, n=3 method summaries, paired differences, event details and low-FPR tables are provided, with old results and source snapshots preserved. The key review is role-separated within the same model family, not external double-blind peer review.
"""
    (OUT / "RESULTS.md").write_text(report, encoding="utf-8")
    receipt = dict(status="EXPORTED_NOT_SCIENTIFIC_REEVALUATION", metrics_sha256=sha(EVAL / "METRICS.json"),
        seal_sha256=sha(RAW / "GLOBAL_RAW_BANK_SEAL.json"), exported_native_rows=len(native),
        complete_method_rows=len(rows), event_rows=len(events), actual_GPU_hours=seal["actual_GPU_hours"],
        source_script_sha256=sha(__file__), output_hashes={p.name:sha(p) for p in OUT.iterdir()
            if p.name in ("methods.csv", "paired_seeds.csv", "paired_summary.csv", "events.csv", "method_summary.csv", "RESULTS.md")})
    (EVAL / "PUBLIC_EXPORT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    run()
