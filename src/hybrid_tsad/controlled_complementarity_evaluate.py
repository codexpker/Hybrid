"""Evaluate fixed paired streams and rederive the key results from raw arrays."""

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from .controlled_complementarity_design import dump, read, sha, write_csv
from .controlled_complementarity_run import MODELS, csv_rows


def intervals(bits):
    edges = np.flatnonzero(np.diff(np.r_[False, bits, False].astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def regions(times, start, stop, core):
    def span(lo, hi):
        return (times >= lo) & (times < hi)

    return {
        "full_event": span(start, stop),
        "core": span(*core),
        "first32": span(start, min(stop, start + 32)),
        "entry_to_core": span(start, core[0]),
        "exit_after_core": span(core[1], stop),
        "recovery32": span(stop, stop + 32),
        "recovery33_to_512": span(stop + 32, stop + 512),
        "clean_pre": span(512, 640),
        "clean_post": span(2176, 2304),
        "all_outside": ~span(start, stop),
    }


def alarms(score, threshold):
    strict = score > threshold
    c3 = np.r_[False, False, strict[2:] & strict[1:-1] & strict[:-2]]
    return {"strict": strict, "confirmed3": c3}


def evaluate_alarms(bits, times, start, stop):
    event = (start <= times) & (times < stop)
    hits = times[bits & event]
    predicted = intervals(bits)
    any_hit = len(hits) > 0
    # Each synthetic stream has one fixed evaluation interval.
    matched = int(any_hit)
    return dict(any_hit=any_hit, one_to_one_matches=matched, predicted_alarm_fragments=len(predicted),
                one_to_one_event_precision=matched / len(predicted) if predicted else None,
                one_to_one_event_recall=matched, one_to_one_event_f1=2 * matched / (len(predicted) + 1),
                offset_of_first_hit=int(hits[0] - start) if any_hit else None,
                longest_missed_run=max((hi - lo for lo, hi in intervals(event & ~bits)), default=0),
                outside_alarm_points=int((bits & ~event).sum()),
                outside_alarm_fragments=len(intervals(bits & ~event)))


def independent_ranking(y, scores):
    """Independent AP tie groups and Mann–Whitney ROC, without sklearn metrics."""
    positives = y.sum()
    negatives = len(y) - positives
    ranks = rankdata(scores, method="average")
    auc = (ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    order = np.argsort(-scores, kind="stable")
    sorted_values = scores[order]
    groups = np.r_[np.flatnonzero(sorted_values[1:] != sorted_values[:-1]), len(y) - 1]
    tp = np.cumsum(y[order])[groups]
    ap = np.dot(np.diff(np.r_[0, tp]), tp / (groups + 1)) / positives
    return float(ap), float(auc)


def summary_rows(pair_rows):
    fields = ("delta_ap", "delta_auc", "delta_core_ap", "delta_core_auc", "target_time_win_fraction", "delta_strict_core_coverage",
              "delta_c3_core_coverage", "delta_c3_event_hit", "delta_strict_outside_alarm_points",
              "delta_c3_outside_alarm_points", "delta_strict_recovery_alarm_points")
    sort_keys = ("domain", "model", "condition", "control")
    result = []
    for keys, group in itertools.groupby(sorted(pair_rows, key=lambda x: tuple(x[k] for k in sort_keys)),
                                         key=lambda x: tuple(x[k] for k in sort_keys)):
        items = list(group)
        row = dict(zip(sort_keys, keys), blocks=len(items))
        for column in fields:
            vector = np.array([item[column] for item in items], dtype=float)
            row[column + "_block_mean"] = float(vector.mean())
            row[column + "_blocks_above_null"] = int(np.sum(vector > (0.5 if column == "target_time_win_fraction" else 0)))
            row[column + "_min"] = float(np.min(vector))
            row[column + "_max"] = float(np.max(vector))
        result.append(row)
    return result


def evaluate(root, execution, design, out):
    config = read(execution)
    plan = read(design)
    dispatch = read(root / "DISPATCH.json")
    assert dispatch["succeeded"] == dispatch["total_expected"] == 16
    case_rows = csv_rows(root / "CASES.csv")
    metrics, phase, pairs, invalid, audits = [], [], [], [], []
    for job in read(root / "JOBS.json"):
        domain, block, model = job["domain"], job["block"], job["model"]
        directory = root / f"{domain}_block{block}_{model}"
        bank = directory / "BANK.npz"
        receipt = read(directory / "RECEIPT.json")
        assert sha(bank) == receipt["bank_sha256"]
        assert sha(root / job["inputs"]) == job["input_sha256"]
        with np.load(bank, allow_pickle=False) as z, np.load(root / job["inputs"], allow_pickle=False) as inp:
            times, cases, scores, mapping, rr, cc = (z[k] for k in
                ("time", "cases", "scores", "reused_base_col", "forward_row", "forward_col"))
            values = inp["values"]
            times_expected = np.arange(*config["score_axis"])
            np.testing.assert_array_equal(times, times_expected)
            np.testing.assert_array_equal(times, z["available_index"])
            np.testing.assert_array_equal(times - MODELS[model], z["footprint_start"])
            np.testing.assert_array_equal(times + (model == "TimeRadar"), z["footprint_end_exclusive"])
            np.testing.assert_array_equal(cases, inp["cases"])
            np.testing.assert_array_equal((rr, cc), np.where(mapping == -1))
            # Independently rederive all native scores from terminal copies or forecast/target arrays.
            if model == "TimeRadar":
                copies = z["copies"].astype(np.float64)
                derived = ((copies - copies.mean(axis=1, keepdims=True)) ** 2).sum(axis=1).mean(axis=1) / 9
            else:
                actual = values[rr, times[cc]]
                np.testing.assert_array_equal(z["actual"], actual)
                derived = np.square(z["median"].astype(np.float64) - actual).mean(axis=-1)
            np.testing.assert_allclose(scores[rr, cc], derived, rtol=1e-5, atol=1e-7)
            for row, col in zip(*np.where(mapping >= 0)):
                t = times[col]
                base_t = times[mapping[row, col]]
                past = MODELS[model]
                assert scores[row, col] == scores[0, mapping[row, col]]
                assert np.array_equal(values[row, t-past:t+1].view(np.uint32),
                                      values[0, base_t-past:base_t+1].view(np.uint32))
            max_score_error = float(np.max(np.abs(scores[rr, cc] - derived)))
        arrays = dict(zip(cases.tolist(), scores))
        threshold = config["thresholds"][domain][model]
        checked_rankings = 0
        for case in case_rows:
            name = case["case"]
            kind = case["kind"]
            if case["domain"] != domain or int(case["block"]) != block or kind not in ("H1", "H2", "H3"):
                continue
            if case["eligible"] != "True":
                invalid.append(dict(domain=domain, block=block, model=model, condition=name, status="NA_PRESET_H3_GATE"))
                continue
            start = plan["intervention_start"]
            duration = int(name.rsplit("_d", 1)[1]) if kind == "H1" else plan[kind]["duration"]
            stop = start + duration
            core = (start, stop) if kind == "H1" else plan[kind]["primary_core"]
            slices = regions(times, start, stop, core)
            event = slices["full_event"]
            core_selection = slices["core"] | slices["clean_pre"] | slices["clean_post"]
            core_truth = slices["core"][core_selection]
            controls = {"target": name, "normal": "base", "pressure": name + "_amp_control"}
            if kind in ("H1", "H2"):
                controls["smooth"] = name + "_smooth_control"
            else:
                controls["sync_sham"] = name.replace("H3_lag", "H3_sync")
            if kind == "H3":
                # Verify shifted-sham scores exactly reproduce the known unchanged joint inputs.
                delay = int(name.split("lag")[1])
                offsets = np.where(slices["core"])[0]
                np.testing.assert_array_equal(arrays[controls["sync_sham"]][offsets], arrays["base"][offsets - delay])
            common = dict(domain=domain, block=block, model=model, hypothesis=kind, condition=name,
                          source_block_start=int(case["source_start"]), threshold=threshold)
            by_control = {}
            for control, case_name in controls.items():
                score = arrays[case_name]
                ap, auc = average_precision_score(event, score), roc_auc_score(event, score)
                np.testing.assert_allclose((ap, auc), independent_ranking(event, score), rtol=1e-13, atol=1e-14)
                core_score = score[core_selection]
                core_ap = average_precision_score(core_truth, core_score)
                core_auc = roc_auc_score(core_truth, core_score)
                np.testing.assert_allclose((core_ap, core_auc), independent_ranking(core_truth, core_score), rtol=1e-13, atol=1e-14)
                checked_rankings += 2
                predicted = alarms(score, threshold)
                strict = score > threshold
                independent_c3 = np.convolve(strict.astype(np.int16), np.ones(3, dtype=np.int16), mode="full")[:len(strict)] == 3
                np.testing.assert_array_equal(predicted["confirmed3"], independent_c3)
                outcomes = {}
                for mode, bits in predicted.items():
                    outcome = evaluate_alarms(bits, times, start, stop)
                    assert outcome["one_to_one_matches"] == int(np.any(bits[event]))
                    assert outcome["outside_alarm_points"] == len(np.intersect1d(np.flatnonzero(bits), np.flatnonzero(~event)))
                    metrics.append(dict(**common, control=control, case=case_name, ap=ap, roc_auc=auc, mode=mode,
                                        core_vs_clean_ap=core_ap, core_vs_clean_roc_auc=core_auc,
                                        event_coverage=float(bits[event].mean()), core_coverage=float(bits[slices["core"]].mean()),
                                        **outcome))
                    outcomes[mode] = outcome
                for region, mask in slices.items():
                    if not np.any(mask):
                        continue
                    phase.append(dict(**common, control=control, case=case_name, region=region, points=int(mask.sum()),
                                      score_mean=float(score[mask].mean()), score_median=float(np.median(score[mask])),
                                      strict_points=int(predicted["strict"][mask].sum()),
                                      c3_points=int(predicted["confirmed3"][mask].sum()),
                                      strict_fragments=len(intervals(predicted["strict"] & mask)),
                                      c3_fragments=len(intervals(predicted["confirmed3"] & mask))))
                by_control[control] = ap, auc, core_ap, core_auc, score, predicted, outcomes
            tap, tauc, target_core_ap, target_core_auc, target, target_bits, target_out = by_control["target"]
            for control, (ap, auc, core_ap, core_auc, score, bits, outcome) in by_control.items():
                if control == "target":
                    continue
                core_mask = slices["core"]
                recovery_mask = slices["recovery32"] | slices["recovery33_to_512"]
                delta = target[core_mask] - score[core_mask]
                pairs.append(dict(**common, control=control, delta_ap=float(tap-ap), delta_auc=float(tauc-auc),
                                  delta_core_ap=float(target_core_ap-core_ap), delta_core_auc=float(target_core_auc-core_auc),
                                  core_median_score_delta=float(np.median(delta)),
                                  target_time_win_fraction=float(np.mean(delta > 0) + .5*np.mean(delta == 0)),
                                  delta_strict_core_coverage=float(target_bits["strict"][core_mask].mean() - bits["strict"][core_mask].mean()),
                                  delta_c3_core_coverage=float(target_bits["confirmed3"][core_mask].mean() - bits["confirmed3"][core_mask].mean()),
                                  delta_c3_event_hit=target_out["confirmed3"]["one_to_one_matches"]-outcome["confirmed3"]["one_to_one_matches"],
                                  delta_strict_outside_alarm_points=target_out["strict"]["outside_alarm_points"]-outcome["strict"]["outside_alarm_points"],
                                  delta_c3_outside_alarm_points=target_out["confirmed3"]["outside_alarm_points"]-outcome["confirmed3"]["outside_alarm_points"],
                                  delta_strict_recovery_alarm_points=int(target_bits["strict"][recovery_mask].sum())-int(bits["strict"][recovery_mask].sum())))
        audits.append(dict(domain=domain, block=block, model=model, score_arrays_verified=int(scores.size),
                           forwarded_scores_rederived=len(rr), cached_footprints_verified=int(np.sum(mapping >= 0)),
                           metrics_rederived=checked_rankings, max_absolute_raw_score_difference=max_score_error))
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in [("EVENTS", metrics), ("PHASES", phase), ("PAIRED", pairs),
                       ("BLOCK_SUMMARY", summary_rows(pairs)), ("UNQUALIFIED", invalid), ("RECALCULATION", audits)]:
        write_csv(out / (name + ".csv"), rows)
    dump(out / "EVALUATION.json", dict(scored_streams_per_model=238, scored_points_per_model=426496,
                                      independently_checked_ap_roc=sum(row["metrics_rederived"] for row in audits),
                                      gpu_occupied_hours=dispatch["occupied_gpu_hours"],
                                      raw_banks_and_fixed_thresholds_verified=True, inference_repeated=False,
                                      no_automatic_successor=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("root", "execution", "design", "out"):
        parser.add_argument(path, type=Path)
    evaluate(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
