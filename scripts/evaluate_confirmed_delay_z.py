"""Seal a four-arm public Z panel using the archived native LEFT task roles."""

import argparse
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, median

import numpy as np

from hybrid_tsad import confirmed_delay as method
from hybrid_tsad import left_timesfm_evaluation as prior
from hybrid_tsad import left_timesfm_training as training
from hybrid_tsad.left_decoder_context_run import fingerprint
from hybrid_tsad.left_history_public import members
from hybrid_tsad.left_timesfm_condition import load_model
from hybrid_tsad.trajectory_evaluation import save_rows

DOMAINS = ("PSM", "SWAT", "SMD")
CHANNELS = {"B": "residual", "C": "unconfirmed", "D": "delay"}
ARMS = ("A", *CHANNELS)
MIN_AP = 0.005
M = prior.METRICS


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def score_blocks(history, current):
    ref = method.fit_reference(history)
    return len(ref.links), method.score_window(ref, history, current)


def calibrate(w, output, domain):
    """Fill the absent point-level normal-source reference with frozen eval only."""
    import torch

    cache = w / prior.OUT / domain
    model, checkpoint = load_model(w, domain, "A")
    role_data, identities = training.source_data(w, domain)
    assert identities == prior.read(cache / "SOURCE.json")
    plan_path = cache / "PLANS.npz"
    assert hash_file(plan_path) == checkpoint["plan_sha256"]
    with np.load(plan_path) as plans:
        plan = plans["select"]
    assert len(plan) == 1024 and checkpoint["condition_parameters"] == 0
    step = model.left.step
    original = fingerprint(model.state_dict())
    native, rows = [], []
    inference = relation = 0.0
    elapsed = time.monotonic()
    for j in range(0, len(plan), 32):
        history, current, lengths = training.batch_arrays(
            role_data["select"], plan[j : j + 32]
        )
        assert (lengths >= method.HISTORY).all()
        torch.cuda.synchronize()
        tick = time.monotonic()
        with torch.no_grad():
            values = model.left.infer(
                torch.from_numpy(current).cuda(), None, None, None
            )[0].mean(-1)
            native.append(values.cpu().numpy())
        torch.cuda.synchronize()
        inference += time.monotonic() - tick
        tick = time.monotonic()
        rows.extend(
            score_blocks(h[-method.HISTORY :], x)[1]
            for h, x in zip(history, current, strict=True)
        )
        relation += time.monotonic() - tick
    assert step == model.left.step and original == fingerprint(model.state_dict())
    path = output / f"calibration_{domain}.npz"
    np.savez_compressed(
        path,
        A=np.concatenate(native).ravel(),
        **{
            a: np.array([getattr(b, statistic) for row in rows for b in row])
            for a, statistic in CHANNELS.items()
        },
    )
    receipt = dict(
        domain=domain,
        role="SOURCE SELECT",
        windows=len(plan),
        native_forward_calls=len(native),
        checkpoint=checkpoint["sha256"],
        plan_sha256=checkpoint["plan_sha256"],
        source_catalog_sha256=hash_file(cache / "SOURCE.json"),
        sha256=hash_file(path),
        labels_read=False,
        total_seconds=time.monotonic() - elapsed,
        native_inference_seconds=inference,
        relation_seconds=relation,
        training_updates=0,
    )
    prior.dump(path.with_suffix(".json"), receipt)
    print("Source sealed", domain, receipt["native_forward_calls"], flush=True)
    return path, receipt


def seal_entity(w, output, domain, entity, calibration, cal_receipt):
    values, offset, starts, axis = prior.role_input(w, domain, entity, "TEST")
    idx, native, available, old = prior.bank(w, domain, entity, "A", "TEST")
    for key in (
        "record_id",
        "starts",
        "input_sha256",
        "scaler_mean_sha256",
        "scaler_scale_sha256",
        "channel_mapping",
    ):
        assert old[key] == axis[key]
    assert (
        old["checkpoint"] == cal_receipt["checkpoint"] and old["labels_read"] is False
    )
    np.testing.assert_array_equal(
        idx, (starts[:, None] + np.arange(method.WINDOW)).ravel()
    )
    np.testing.assert_array_equal(
        available, np.repeat(starts + method.WINDOW - 1, method.WINDOW)
    )
    assert (
        starts[0] - method.HISTORY >= offset
        and starts[-1] + method.WINDOW <= axis["role_stop"]
    )

    tick = time.monotonic()
    counted = [
        score_blocks(
            values[s - offset - method.HISTORY : s - offset],
            values[s - offset : s - offset + method.WINDOW],
        )
        for s in starts
    ]
    relation_seconds = time.monotonic() - tick
    counts, rows = zip(*counted, strict=True)
    raw = {
        a: np.array([getattr(b, s) for row in rows for b in row])
        for a, s in CHANNELS.items()
    }
    assert np.all(raw["C"] >= raw["D"])
    support = {
        name: np.array([getattr(b, name) for row in rows for b in row])
        for name in (
            "available_links",
            "testable_links",
            "confirming_links",
        )
    }
    with np.load(calibration) as ref:
        tick = time.monotonic()
        native_tail = -np.log(method._tail(ref["A"], native))
        scores = {"A": native}
        rises = {}
        for a, statistic in CHANNELS.items():
            relation_tail = np.repeat(
                -np.log(method._tail(ref[a], raw[a])), method.BLOCK
            )
            scores[a] = np.maximum(native_tail, relation_tail)
            rises[a] = int(np.count_nonzero(relation_tail > native_tail))
            np.testing.assert_array_equal(
                scores[a][: method.WINDOW],
                method.combine(
                    native[: method.WINDOW],
                    rows[0],
                    ref["A"],
                    ref[a],
                    channel=statistic,
                ),
            )
        composition_seconds = time.monotonic() - tick
    path = output / domain / f"{entity}.npz"
    path.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        path,
        index=idx,
        available=available,
        current_start=starts,
        history_start=starts - method.HISTORY,
        historical_links=counts,
        **scores,
        **{f"raw_{a}": v for a, v in raw.items()},
        **support,
    )
    receipt = dict(
        domain=domain,
        entity=entity,
        sha256=hash_file(path),
        old_score_sha256=old["sha256"],
        source_reference_sha256=cal_receipt["sha256"],
        native_model_sha256=old["checkpoint"],
        labels_read=False,
        start=int(idx[0]),
        stop=int(idx[-1] + 1),
        points=len(idx),
        input_sha256=axis["input_sha256"],
        record=axis["record_id"],
        scaler_mean_sha256=axis["scaler_mean_sha256"],
        scaler_scale_sha256=axis["scaler_scale_sha256"],
        role_start=offset,
        role_stop=axis["role_stop"],
        excluded_prefix=int(idx[0] - offset),
        excluded_tail=int(axis["role_stop"] - 1 - idx[-1]),
        windows=len(starts),
        blocks=len(raw["D"]),
        history_no_reference_windows=int(np.count_nonzero(np.asarray(counts) == 0)),
        abstained_blocks=int(np.count_nonzero(support["available_links"] == 0)),
        no_legal_alternative_blocks=int(
            np.count_nonzero(support["testable_links"] == 0)
        ),
        eligible_blocks_without_confirmation=int(
            np.count_nonzero(
                (support["testable_links"] > 0) & (support["confirming_links"] == 0)
            )
        ),
        relation_nonzero={a: int(np.count_nonzero(v > 0)) for a, v in raw.items()},
        exceeds_native_tail_points=rises,
        relation_seconds=relation_seconds,
        composition_seconds=composition_seconds,
        added_test_forwards=0,
    )
    prior.dump(path.with_suffix(".json"), receipt)
    print("Test sealed without labels", domain, entity, len(idx), flush=True)


def seal(w, output):
    output.mkdir(parents=True, exist_ok=False)
    expected = []
    for domain in DOMAINS:
        calibration, receipt = calibrate(w, output, domain)
        for entity in members(w, domain):
            seal_entity(w, output, domain, entity, calibration, receipt)
            expected.append((domain, entity))
    assert len(expected) == 30
    prior.dump(
        output / "SCORE_SEAL.json",
        dict(
            labels_read=False,
            entities=expected,
            inputs=[
                dict(
                    domain=d,
                    entity=e,
                    receipt_sha256=hash_file(output / d / f"{e}.json"),
                )
                for d, e in expected
            ],
        ),
    )


def metrics_job(task):
    w, output, domain, entity, arm = task
    path = output / domain / f"{entity}.npz"
    receipt = prior.read(path.with_suffix(".json"))
    assert hash_file(path) == receipt["sha256"] and receipt["labels_read"] is False
    with np.load(path) as bank:
        idx, score = bank["index"], bank[arm]
    labels = (
        prior.truth(w, domain, entity) if domain == "SMD" else prior.labels(w, domain)
    )
    y = labels[idx]
    begin = time.monotonic()
    if arm == "A":
        path = w / prior.OUT / "metrics" / domain / entity / "A_TEST.json"
        row = next(r for r in prior.read(path)["rows"] if r["panel"] == "complete")
        assert row["score_sha256"] == receipt["old_score_sha256"]
        assert (row["start"], row["stop"], row["points"]) == (
            int(idx[0]),
            int(idx[-1]) + 1,
            len(idx),
        )
        computed = np.asarray(prior.independent_check(y, score))
        error = float(np.max(np.abs(computed - [row["AP"], row["ROC_AUC"]])))
        assert error < 1e-10
        result = {key: row[key] for key in M}
    else:
        result, error = prior.ranking(
            y, score, prior.load_author_metric(w / prior.AUTHOR)
        )
    return dict(
        domain=domain,
        entity=entity,
        arm=arm,
        points=len(idx),
        positives=int(y.sum()),
        **result,
        AP_ROC_check_error=error,
        metric_seconds=time.monotonic() - begin,
    )


def summarize(w, output, workers):
    seal = prior.read(output / "SCORE_SEAL.json")
    expected = [(d, e) for d in DOMAINS for e in members(w, d)]
    assert len(expected) == 30 and not seal["labels_read"]
    assert [tuple(v) for v in seal["entities"]] == expected
    for row in seal["inputs"]:
        assert (
            hash_file(output / row["domain"] / f"{row['entity']}.json")
            == row["receipt_sha256"]
        )
    tasks = [(w, output, d, e, a) for d, e in expected for a in ARMS]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(metrics_job, tasks))
    save_rows(output / "entity_metrics.csv", rows)
    table, paired, differences = [], [], []
    keyed = {(r["domain"], r["entity"], r["arm"]): r for r in rows}
    for d, e in expected:
        for a in ("A", "B", "C"):
            delta = {m: keyed[d, e, "D"][m] - keyed[d, e, a][m] for m in M}
            paired.append(dict(domain=d, entity=e, contrast="D-" + a, **delta))
    save_rows(output / "paired_entities.csv", paired)
    for domain in DOMAINS:
        for arm in ARMS:
            r = [v for v in rows if (v["domain"], v["arm"]) == (domain, arm)]
            table.append(
                dict(
                    domain=domain,
                    arm=arm,
                    entities=len(r),
                    points=sum(v["points"] for v in r),
                    **{m: mean(v[m] for v in r) for m in M},
                )
            )
        for arm in ("A", "B", "C"):
            rr = [
                v
                for v in paired
                if v["domain"] == domain and v["contrast"] == "D-" + arm
            ]
            fields = {}
            for m in M:
                delta = [v[m] for v in rr]
                fields.update(
                    {
                        m: mean(delta),
                        m + "_median": median(delta),
                        m + "_wins": sum(x > 0 for x in delta),
                        m + "_ties": sum(x == 0 for x in delta),
                        m + "_losses": sum(x < 0 for x in delta),
                    }
                )
            differences.append(
                dict(domain=domain, contrast="D-" + arm, entities=len(rr), **fields)
            )
    save_rows(output / "methods.csv", table)
    save_rows(output / "paired_summary.csv", differences)
    paired_ap = {(r["domain"], r["contrast"]): r["AP"] for r in differences}
    positive = [
        d for d in DOMAINS if min(paired_ap[d, "D-A"], paired_ap[d, "D-B"]) >= MIN_AP
    ]
    practical = (
        bool(positive)
        and min(paired_ap[d, c] for d in DOMAINS for c in ("D-A", "D-B")) >= -MIN_AP
    )
    separate = (
        practical
        and any(paired_ap[d, "D-C"] >= MIN_AP for d in positive)
        and min(paired_ap[d, "D-C"] for d in DOMAINS) >= -MIN_AP
    )
    prior.dump(
        output / "DECISION.json",
        dict(
            minimum_AP_increment=MIN_AP,
            practical=practical,
            independent_confirmation=separate,
            outcome="retain"
            if separate
            else ("benefit_without_needed_confirmation" if practical else "close"),
            practical_domains=positive,
            AP_contrasts=[
                {k: r[k] for k in ("domain", "contrast", "AP")} for r in differences
            ],
            metric_wall_seconds=time.monotonic() - started,
            AP_ROC_max_check_error=max(r["AP_ROC_check_error"] for r in rows),
            workers=workers,
        ),
    )
    diagnostics(output, expected)
    print("Public Z complete", len(rows), flush=True)


def diagnostics(output, expected):
    receipts = [prior.read(output / d / f"{e}.json") for d, e in expected]
    overview, details, costs = [], [], []
    for r in receipts:
        details.append(
            {
                k: r[k]
                for k in (
                    "domain",
                    "entity",
                    "record",
                    "role_start",
                    "role_stop",
                    "start",
                    "stop",
                    "points",
                    "windows",
                    "excluded_prefix",
                    "excluded_tail",
                    "blocks",
                    "history_no_reference_windows",
                    "abstained_blocks",
                    "no_legal_alternative_blocks",
                    "eligible_blocks_without_confirmation",
                    "relation_seconds",
                    "composition_seconds",
                )
            }
            | {f"{a}_nonzero_blocks": r["relation_nonzero"][a] for a in CHANNELS}
            | {
                f"{a}_raised_points": r["exceeds_native_tail_points"][a]
                for a in CHANNELS
            }
        )
    save_rows(output / "axis_and_coverage.csv", details)
    for domain in DOMAINS:
        selected = [r for r in details if r["domain"] == domain]
        total = {
            k: sum(r[k] for r in selected)
            for k in selected[0]
            if isinstance(selected[0][k], (int, float))
        }
        windows, blocks, points = total["windows"], total["blocks"], total["points"]
        overview.append(
            dict(
                domain=domain,
                entities=len(selected),
                windows=windows,
                blocks=blocks,
                points=points,
                no_history_reference_window_rate=total["history_no_reference_windows"]
                / windows,
                abstained_block_rate=total["abstained_blocks"] / blocks,
                no_legal_alternative_block_rate=total["no_legal_alternative_blocks"]
                / blocks,
                eligible_but_no_confirmation_block_rate=total[
                    "eligible_blocks_without_confirmation"
                ]
                / blocks,
                **{
                    f"{a}_nonzero_block_rate": total[f"{a}_nonzero_blocks"] / blocks
                    for a in CHANNELS
                },
                **{
                    f"{a}_raised_point_rate": total[f"{a}_raised_points"] / points
                    for a in CHANNELS
                },
            )
        )
        r = prior.read(output / f"calibration_{domain}.json")
        costs.append(
            dict(
                domain=domain,
                source_windows=r["windows"],
                source_LEFT_calls=r["native_forward_calls"],
                source_LEFT_seconds=r["native_inference_seconds"],
                source_cpu_relation_seconds=r["relation_seconds"],
                source_total_seconds=r["total_seconds"],
                test_LEFT_calls=0,
                test_windows=windows,
                test_cpu_relation_seconds=total["relation_seconds"],
                test_composition_seconds=total["composition_seconds"],
                training_updates=0,
            )
        )
    save_rows(output / "coverage.csv", overview)
    save_rows(output / "costs.csv", costs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("phase", choices=("seal", "metrics"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.phase == "seal":
        import torch

        torch.set_num_threads(1)
        seal(args.workspace, args.output)
    else:
        summarize(args.workspace, args.output, args.workers)


if __name__ == "__main__":
    main()
