"""Evaluate the two frozen-history endpoints against sealed same-QUERY references."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from hybrid_tsad.left_decoder_context_metrics import AUTHOR
from hybrid_tsad.left_multisource import target_inputs
from hybrid_tsad.native_v_conditioning import sha, state_hashes
from hybrid_tsad.timeradar_history_context import record_window
from hybrid_tsad.timeradar_history_evaluation import bank, bounds
from hybrid_tsad.timeradar_history_experiment import FROZEN_HISTORY, model, pools
from hybrid_tsad.trajectory_evaluation import continuous_metrics, independent_check, labels, load_author_metric
from hybrid_tsad.trajectory_experiment import ROLES, dump, target


RANKING = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC")


def read(path):
    return json.loads(path.read_text())


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_source_records(workspace, contract):
    records = pools(workspace)
    for role in ("fit", "select"):
        for domain, arrays in enumerate(records[role]):
            expected = contract["source_records"][role][domain]
            assert len(arrays) == len(expected)
            for values, sealed in zip(arrays, expected):
                assert list(values.shape) == sealed["shape"]
                assert hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest() == sealed["sha256"]


def verify_endpoint(workspace, out, archived, domain, data, receipt):
    """Check saved state and three fixed uncached endpoint forwards."""
    root = out / domain / FROZEN_HISTORY
    checkpoint = root / "selected.pt"
    assert sha(checkpoint) == receipt["sha256"]
    source = archived / "source/history/selected.pt"
    assert sha(source) == receipt["source_checkpoint_sha256"]
    initial = torch.load(source, map_location="cpu", weights_only=True)["model"]
    final = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    audit = read(root / "parameter_audit.json")
    detector = model(workspace, FROZEN_HISTORY)
    detector.load_state_dict(initial)
    initial_hashes = state_hashes(detector)
    detector.load_state_dict(final)
    final_hashes = state_hashes(detector)
    before = {k: v for k, v in initial_hashes.items() if not k.startswith("native.decoder.")}
    after = {k: final_hashes[k] for k in before}
    assert before == after == audit["frozen_state_before"] == audit["frozen_state_after"]
    decoder_keys = [k for k in initial if k.startswith("native.decoder.")]
    decoder_l2 = sum((final[k] - initial[k]).abs().square().sum().item() for k in decoder_keys) ** .5
    assert decoder_l2 > 0
    np.testing.assert_allclose(decoder_l2, audit["decoder_L2_displacement"], rtol=1e-7)
    _, _, scaler, binding = target_inputs(workspace, domain)
    values = torch.from_numpy(scaler.transform(target(workspace, domain, "QUERY")).astype(np.float32)).cuda()
    origin = ROLES[domain]["QUERY"][0]
    samples = []
    lo, hi = bounds(domain, "U")
    with torch.no_grad():
        for tick in (lo, (lo + hi)//2, hi-1):
            history, current, footprint = record_window(values, tick-origin-99, (0, len(values)), 512)
            _, copies = detector(history, current, **footprint)
            score = copies.var(0, correction=1).mean(-1)[0, -1].item()
            assert score == data["score"][tick-lo]
            samples.append(dict(index=tick, terminal_V=score, history_start=tick-611, current_start=tick-99))
    score_dir = out / domain / "scores/U_frozen_history"
    receipts = [read(score_dir / f"part{k}.json") for k in range(4)]
    assert all(r["checkpoint"] == receipt["sha256"] and r["normal_values_sha256"] == binding for r in receipts)
    assert all(r["scoring_B"] == 1 for r in receipts)
    return dict(decoder_L2=decoder_l2, history_L2=0.0, frozen_tensor_count=len(before),
                parameter_hashes_exact=True, samples=samples,
                inference_process_seconds=sum(r["seconds"] for r in receipts))


def evaluate(workspace, out, archived):
    torch.set_num_threads(1)
    started = time.monotonic()
    verified = {}
    scores_by_domain = {}
    # Check both frozen endpoints and seal the new bank identities before labels.
    contract = read(archived / "contract.json")
    assert sha(out / "plans.npz") == contract["plan_sha256"] == sha(archived / "plans.npz")
    verify_source_records(workspace, contract)
    for domain in ("PSM", "SWAT"):
        root = out / domain / FROZEN_HISTORY
        receipt = read(root / "receipt.json")
        assert receipt["steps"] == receipt["selected_step"] == 2000
        assert receipt["plan_sha256"] == contract["plan_sha256"]
        scope = read(root / "update_scope_check.json")
        assert scope["optimizer_decoder_only"] and scope["fresh_adamw"] and scope["frozen_gradients_absent"]
        assert scope["history_forward_calls"] == dict.fromkeys(("past_context", "query", "cross", "output"), 32)
        candidate = bank(out, domain, FROZEN_HISTORY, "U")
        idx = candidate["index"]
        np.testing.assert_array_equal(candidate["available"], idx)
        np.testing.assert_array_equal(candidate["history_start"], idx-611)
        np.testing.assert_array_equal(candidate["current_start"], idx-99)
        verified[domain] = verify_endpoint(workspace, out, archived, domain, candidate, receipt)
        verified[domain].update(checkpoint_sha256=receipt["sha256"], source_sha256=receipt["source_checkpoint_sha256"],
                                score_numeric_sha256=hashlib.sha256(candidate["score"].tobytes()).hexdigest(),
                                query_bounds=list(bounds(domain, "U")),
                                training_seconds=receipt["seconds"], peak_bytes=receipt["peak_bytes"])
        scores_by_domain[domain] = candidate
    dump(out / "sealed_bank_review.json", dict(plan_sha256=contract["plan_sha256"],
        source_records_verified=True, labels_read=False, by_domain=verified,
        review_process_seconds=time.monotonic()-started))

    metric_started = time.monotonic()
    with (archived / "metrics.csv").open(newline="") as stream:
        cached = list(csv.DictReader(stream))
    archived_diagnostic = read(archived / "history_diagnostics.json")
    rows, contrasts, diagnostics = [], [], []
    for domain, candidate in scores_by_domain.items():
        lo, hi = bounds(domain, "U")
        idx = candidate["index"]
        truth = labels(workspace, domain)[idx]
        metrics, _ = continuous_metrics(truth, candidate["score"], load_author_metric(workspace / AUTHOR))
        values = {"B": {k: float(metrics[k]) for k in RANKING}}
        for name, arm, stage in (("Z", "history", "Z"), ("A", "history", "U"), ("Native U", "native", "U")):
            reference = bank(archived, domain, arm, stage)
            take = (reference["index"] >= lo) & (reference["index"] < hi)
            np.testing.assert_array_equal(reference["index"][take], idx)
            cached_row = next(r for r in cached if (r["domain"], r["method"], r["stage"], r["panel"]) ==
                              (domain, arm, stage, "QUERY"))
            assert (int(cached_row["start"]), int(cached_row["stop"])) == (lo, hi)
            values[name] = {k: float(cached_row[k]) for k in RANKING}
            np.testing.assert_allclose(independent_check(truth, reference["score"][take]),
                                       [values[name]["AP"], values[name]["ROC_AUC"]], atol=1e-11, rtol=0)
        np.testing.assert_allclose(independent_check(truth, candidate["score"]),
                                   [values["B"]["AP"], values["B"]["ROC_AUC"]], atol=1e-11, rtol=0)
        for name in ("Z", "A", "B", "Native U"):
            rows.append(dict(dataset=domain, method=name, start=lo, stop=hi, points=len(idx), **values[name]))
        for left, right in (("A", "Z"), ("B", "Z"), ("B", "A"), ("B", "Native U")):
            contrasts.append(dict(dataset=domain, comparison=f"{left} - {right}",
                                  **{k: values[left][k]-values[right][k] for k in RANKING}))
        for name, arm, stage in (("Z", "history", "Z"), ("A", "history", "U"), ("Native U", "native", "U")):
            group = [r["score"]["correct"] for r in archived_diagnostic if (r["stage"], r["domain"], r["arm"])
                     == (stage, None if stage == "Z" else domain, arm)]
            assert len(group) == 32
            diagnostics.append(dict(dataset=domain, method=name,
                **{k: float(np.mean([r[k] for r in group])) for k in ("copy", "E_full", "V_full")}))
        b_select, = read(out / domain / FROZEN_HISTORY / "select.json")
        assert b_select["diagnostic_only"]
        diagnostics.append(dict(dataset=domain, method="B", **{k: b_select[k] for k in ("copy", "E_full", "V_full")}))
    write_rows(out / "metrics.csv", rows)
    write_rows(out / "contrasts.csv", contrasts)
    write_rows(out / "normal_select.csv", diagnostics)
    dump(out / "metric_review.json", dict(AP_ROC_rows_checked=len(rows), archived_VUS_reused=True,
        B_VUS_fixed_implementation=True, VUS_metric_sha256=sha(workspace / AUTHOR),
        metric_process_seconds=time.monotonic()-metric_started))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("workspace", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("reference", type=Path)
    args = cli.parse_args()
    evaluate(args.workspace, args.output, args.reference)
