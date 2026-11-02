"""Export already evaluated candidate results; never load scores or raw labels.

Only evaluation JSONs, their complete receipt, and prepare/training receipts are read.
No model/threshold selection, numerical recomputation, or optional-metric fallback.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

MODES = ("tf", "lag", "concat", "gated")
STREAMS = ("strict_point", "confirmed3_current")
RANKING = ("AP", "trapezoid_PR_AUC", "ROC_AUC", "ranking_metric_NA_reason")
METRICS = ("point_Precision", "point_Recall", "point_F1", "normal_FPR", "FP",
           "normal_exposure_points", "normal_false_alarm_points", "normal_alarm_fragments",
           "complete_event_recall_any_hit_aux", "one_to_one_event_precision",
           "complete_event_hits", "complete_events", "original_events", "predicted_events",
           "raw_threshold", "delay_units", "Aff_F1", "Aff_F1_reason", "affiliation_status")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _read(path, root, inputs):
    data = path.read_bytes()
    inputs[path.relative_to(root).as_posix()] = digest(data)
    def invalid(value):
        raise ValueError("Nonstandard nonfinite JSON token: " + value)
    value = json.loads(data.decode("utf-8"), parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("JSON record must be an object: " + str(path))
    return value


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _cell(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite metric must be represented as null with its original reason")
        return format(value, ".6g")
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)] + [
        "| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]


def collect(run_dir):
    """Read a completed evaluation and bind every consumed input byte identity."""
    root = Path(run_dir).resolve()
    inputs = {}
    receipt = _read(root / "evaluate/receipt.json", root, inputs)
    expected = {"evaluate/" + mode + ".json" for mode in MODES}
    if (receipt.get("stage") != "evaluate" or receipt.get("labels_opened") is not True
            or set(receipt.get("files", {})) != expected
            or not receipt.get("protocol_sha256")):
        raise ValueError("Complete four-mode evaluation receipt required")
    preparation = _read(root / "prepare/receipt.json", root, inputs)
    if (preparation.get("stage") != "prepare"
            or preparation.get("protocol_sha256") != receipt["protocol_sha256"]):
        raise ValueError("Matching preparation record required")
    records, training = {}, {}
    rows = []
    for mode in MODES:
        name = "evaluate/" + mode + ".json"
        result = _read(root / name, root, inputs)
        if inputs[name] != receipt["files"][name]:
            raise ValueError("Evaluation artifact SHA256 mismatch: " + name)
        train = _read(root / ("train_" + mode + "/receipt.json"), root, inputs)
        if (train.get("stage") != "train" or train.get("mode") != mode
                or train.get("protocol_sha256") != receipt["protocol_sha256"]
                or type(train.get("updates")) is not int or train["updates"] < 1
                or type(train.get("selected_step")) is not int
                or not 1 <= train["selected_step"] <= train["updates"]):
            raise ValueError("Matching completed training record required: " + mode)
        if (not isinstance(result.get("ranking"), dict)
                or set(result.get("streams", {})) != set(STREAMS)
                or not isinstance(result.get("Range_VUS"), dict)):
            raise ValueError("Incomplete evaluation schema: " + mode)
        if not all(key in result["ranking"] for key in RANKING[:3]):
            raise ValueError("Raw ranking metrics missing: " + mode)
        records[mode], training[mode] = result, train
        for stream in STREAMS:
            source = result["streams"][stream]
            status = source.get("status", "EVALUATED")
            metrics = source.get("metrics", {})
            if status == "EVALUATED" and (not isinstance(metrics, dict) or
                    not all(key in metrics for key in METRICS) or
                    not isinstance(source.get("events"), list) or not isinstance(source.get("affinity"), list)):
                raise ValueError("Incomplete fixed-stream evaluation: " + mode + "/" + stream)
            if status != "EVALUATED" and not str(status).startswith("NA_"):
                raise ValueError("Unknown stream completion status")
            row = {"mode": mode, "stream": stream, "stream_status": status,
                   "updates": train["updates"], "selected_step": train["selected_step"],
                   "normal_SELECT_MSE": train.get("normal_SELECT_MSE")}
            row.update({"raw_" + key: result["ranking"].get(key) for key in RANKING})
            row.update({key: metrics.get(key) for key in METRICS})
            # Preserve exact event records: hit delays, censored misses, incomplete
            # coverage, and sample-point units. No invented mean/median delay.
            row["event_delays_and_censoring_json"] = _json(source.get("events", []))
            row["affinity_blocks_json"] = _json(source.get("affinity", []))
            row["Range_VUS_json"] = _json(result["Range_VUS"])
            row["affiliation_provenance_json"] = _json(result.get("affiliation_provenance"))
            rows.append(row)
    return root, receipt, preparation, records, training, rows, inputs


def summary(records, training, rows, receipt, preparation):
    from_scratch = preparation.get("from_scratch")
    initialization = str(from_scratch) if type(from_scratch) is bool else "未核 / not verified"
    updates = "; ".join(mode + "=" + str(training[mode]["updates"]) for mode in MODES)
    lines = ["# 完整候选结果 / Candidate results", "",
             "范围 / Scope：单run导出，不据此宣称多seed证据；该导出本身不构成官方原方法复现、最终多源/Z/U/F、新颖性或泛化证据。",
             "Single-run export; no multi-seed evidence, official reproduction, final Z/U/F, novelty or generalization is asserted by this report.",
             "", "from_scratch（原 prepare 记录 / as recorded）: " + initialization,
             "实际更新数 / Actual updates: " + updates,
             "", "原评价范围 / Original evaluation scope:", "```text", str(receipt.get("scope", "未核 / not verified")), "```",
             "", "Protocol SHA256: `" + receipt["protocol_sha256"] + "`", "",
             "## 原连续分数 / Raw continuous-score ranking", ""]
    lines += _table(["mode", "updates", "selected step", "normal SELECT MSE", "AP", "trapezoid PR", "ROC", "NA reason"], [
        [mode, training[mode]["updates"], training[mode]["selected_step"], training[mode].get("normal_SELECT_MSE")] +
        [records[mode]["ranking"].get(k) for k in RANKING] for mode in MODES])
    lines += ["", "## 固定阈值与报警流 / Fixed-threshold streams", "",
              "strict_point 是主点级流；confirmed3_current 从连续第三点确认，不回填。下表只转录已存指标，不重新选阈值或计算指标。NA 保留缺失/未定义含义，不替换成0。",
              "strict_point is the primary point stream; confirmed3_current starts at the third consecutive exceedance without backfilling. Values are copied, never recomputed.", ""]
    lines += _table(["mode", "stream", "P", "R", "F1", "normal FPR", "FP", "normal exposure", "false fragments", "event recall", "1:1 event P"], [
        [r["mode"], r["stream"]] + [r[k] for k in ("point_Precision", "point_Recall", "point_F1", "normal_FPR", "FP",
             "normal_exposure_points", "normal_alarm_fragments", "complete_event_recall_any_hit_aux", "one_to_one_event_precision")] for r in rows])
    lines += ["", "## 延迟、删失与可选指标 / Delay, censoring and optional metrics", "",
              "延迟为原样本点，不换算秒。未命中事件的删失和未完整覆盖事件原样保留；不以命中事件延迟掩盖漏检。",
              "Delays remain in original sample points. Censored misses and incomplete coverage remain explicit; no hit-only delay aggregate is added."]
    for row in rows:
        delays = []
        for event in json.loads(row["event_delays_and_censoring_json"]):
            if event.get("delay_censored"):
                value = "censored@" + _cell(event.get("censor_after_points"))
            elif not event.get("complete"):
                value = "incomplete; delay=" + _cell(event.get("delay_sample_points"))
            else:
                value = "delay=" + _cell(event.get("delay_sample_points"))
            delays.append(_cell(event.get("event_id")) + ": " + value)
        lines += ["", "### " + row["mode"] + " / " + row["stream"], "",
                  "Stream status: " + row["stream_status"],
                  "Delay units: " + _cell(row["delay_units"]),
                  "Aff F1: " + _cell(row["Aff_F1"]) + "; status: " + _cell(row["affiliation_status"]) + "; reason: " + _cell(row["Aff_F1_reason"]),
                  "", "Event delays (sample points; censored@ = 原记录删失截止): " + ("; ".join(delays) or "NA"),
                  "完整事件记录与 Aff block 原值/NA 原因见 metrics.csv 的 event_delays_and_censoring_json / affinity_blocks_json。",
                  "Range/VUS (unchanged):", "```json", row["Range_VUS_json"], "```"]
    return "\n".join(lines) + "\n"


def export_report(run_dir, output_dir):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Report output already exists; no overwrite")
    root, receipt, preparation, records, training, rows, inputs = collect(run_dir)
    text = summary(records, training, rows, receipt, preparation)
    source = Path(__file__).read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "metrics.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        # Explicit NA rather than zero. Original causes are separate columns/JSON.
        writer.writerows({k: "NA" if v is None else v for k, v in row.items()} for row in rows)
    (output / "SUMMARY.md").write_text(text, encoding="utf-8")
    (output / "candidate_report.py").write_bytes(source)
    manifest = {"status": "COMPLETE_FOUR_MODE_EXPORT", "run_directory": str(root),
                "protocol_sha256": receipt["protocol_sha256"], "exporter_source_sha256": digest(source),
                "input_sha256": inputs, "input_scope": "evaluation JSON/receipt and prepare/training receipts only; no raw arrays, labels or weights",
                "output_sha256": {p.name: digest(p.read_bytes()) for p in output.iterdir() if p.is_file()},
                "metrics_recomputed": False, "model_or_threshold_selected": False}
    (output / "REPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = export_report(args.run_dir, args.output_dir)
    print(json.dumps({"status": result["status"], "output_directory": str(Path(args.output_dir).resolve())}))


if __name__ == "__main__":
    main()
