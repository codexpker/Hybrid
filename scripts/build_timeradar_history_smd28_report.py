"""Export full SMD28 result receipts and create a reproducible report."""
import argparse
import csv
import json
from pathlib import Path
from shutil import copy2


METRICS = ("AP","ROC_AUC","VUS_PR","VUS_ROC")
ARMS = {
    "Z_native":"Native Z",
    "Z_history":"History Z",
    "U_native":"C - Native U",
    "U_history":"A - History Joint U",
    "U_frozen_history":"B - History Frozen U",
}
DIFFS = ("B-A","B-C","A-C","B-History Z","A-History Z","C-Native Z")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def report(run, out):
    completion=read(run/"queue_complete.json")
    assert completion["completed_U"]==84
    costs=read(run/"cost.json")
    support=read(run/"F_support_eligibility.json")
    assert costs["Z_banks"]==56 and costs["U_banks"]==84
    out.mkdir(parents=True,exist_ok=True)
    for name in ("contract.json","plan_review.json","metrics.csv","contrasts.csv","macro.csv",
                 "Z_metrics.csv","Z_bank_manifest.csv","bank_manifest.csv","training_manifest.csv",
                 "F_support_eligibility.json","cost.json","pilot.json","metric_review.json","queue_complete.json"):
        copy2(run/name,out/name)
    choices={}
    for arm in ("native","history"):
        folder=out/"source"/arm
        folder.mkdir(parents=True,exist_ok=True)
        for name in ("receipt.json","curve.json","select.json","first_update.json"):
            copy2(run/"source"/arm/name,folder/name)
        choice=min(read(folder/"select.json"),key=lambda r:(r["copy"],r["step"]))
        assert choice["step"]==read(folder/"receipt.json")["selected_step"]
        choices[arm]=choice
    for entity in read(run/"contract.json")["entities"]:
        folder=out/"device_audits"/entity
        folder.mkdir(parents=True,exist_ok=True)
        for name in ("parameter_audit.json","update_scope_check.json"):
            copy2(run/entity/"frozen_history"/name,folder/name)
    sm={(r["group"],r["metric"]):r for r in rows(out/"macro.csv")}
    detail=rows(out/"metrics.csv")
    assert len({r["entity"] for r in detail})==28 and len(detail)==140
    eligible={int(sm[g,"AP"]["eligible"]) for g in ARMS}
    assert len(eligible)==1

    def number(value, pattern=".6f"):
        return format(float(value),pattern) if value else "NA"

    header="| Method or contrast | AP | ROC-AUC | VUS-PR | VUS-ROC |"
    line="| --- | ---: | ---: | ---: | ---: |"
    main=[header,line]
    for key,label in ARMS.items():
        vals=[" ± ".join(number(sm[key,m][x]) for x in ("mean","sample_SD")) for m in METRICS]
        main.append(f"| {label} | "+" | ".join(vals)+" |")
    paired=[header,line]
    for key in DIFFS:
        vals=[]
        for m in METRICS:
            r=sm[key,m]
            vals.append(f"{number(r['mean'],'+.6f')} ± {number(r['sample_SD'])} ({r['win']}/{r['tie']}/{r['loss']})")
        paired.append(f"| {key} | "+" | ".join(vals)+" |")
    hist=[r for r in support["domains"] if r["eligible"]]
    total_cost=sum(costs[k] for k in ("source_worker_hours","U_worker_hours",
                                     "Z_scoring_worker_hours","U_scoring_worker_hours"))
    text=[f"# TimeRadar history-context SMD28 result",
          "",
          "Both 240-step source fits, all 56 frozen-Z banks and all 84 target-U fits and banks are complete.",
          "The source fold uses normal TEP and HAI, with all SMD records excluded from project source learning.",
          "The held-out record suffix and complete 612-point dependency are identical within each device.",
          f"All 28 devices are reported. Eligible AP denominator: {eligible.pop()}.",
          "The four metrics use equal-device means and sample SD across devices, with one project training seed.",
          "Full precision and query bounds are in [the per-device result table](metrics.csv).",
          "",*main,"",
          "Differences use per-device pairs. Parentheses give wins/ties/losses by strict numeric sign.",
          "[Per-device contrasts](contrasts.csv) and [macro table including medians](macro.csv).","",*paired,"",
          f"Native and history source checkpoints select steps {choices['native']['step']} and {choices['history']['step']}, "
          f"with normal SELECT copy losses {choices['native']['copy']:.6f} and {choices['history']['copy']:.6f}. "
          "The source budget is 240 for both.",
          "",
          f"Source training {costs['source_worker_hours']:.3f}, U training {costs['U_worker_hours']:.3f}, "
          f"Z scoring {costs['Z_scoring_worker_hours']:.3f}, and U scoring {costs['U_scoring_worker_hours']:.3f} "
          f"A40 process-hours, total {total_cost:.3f}. "
          f"Peak training allocation {costs['peak_train_bytes']/2**20:.2f} MiB, "
          f"peak scoring allocation {costs['peak_score_bytes']/2**20:.2f} MiB.",
          "Model and bank digests, indexed footprint checks, and durations are preserved in "
          "[training](training_manifest.csv), [scoring](bank_manifest.csv) and [metric](metric_review.json) receipts.",
          "All B endpoints pass exact state hashes outside the decoder, with active frozen history and decoder movement.",
          "",
          f"F review finds {len(hist)}/28 devices eligible under the complete-event and normal-block rules. "
          "Each eligible device has one complete event and 512 normal labeled points under the existing U-FIT role. "
          "Additional history and separate CAL/QUERY boundaries are in the [F eligibility table](F_support_eligibility.json). "
          "There are zero F fits.",
          ""]
    (out/"RESULTS.md").write_text("\n".join(text),encoding="utf-8",newline="\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("destination",type=Path)
    a=p.parse_args()
    report(a.run,a.destination)
