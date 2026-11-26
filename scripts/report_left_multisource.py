"""Assemble paper tables from native score receipts and compatible public caches."""
import csv
from pathlib import Path
from statistics import mean,stdev
import argparse



ROOT=Path("docs/experiments/left_multisource_20261126")
METRICS=("AP","ROC_AUC","VUS_PR","VUS_ROC")
DOMAINS=("PSM","SWAT")
NAMES={"scratch_N":"Target-normal LEFT from scratch","source":"Frozen multi-source LEFT (Z)",
       "N":"Multi-source LEFT + target-normal training (N)","ordinary":"Multi-source LEFT + ordinary U",
       "replay":"Multi-source LEFT + source replay U","random_u":"Random LEFT + ordinary U"}


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def grid(rows, names):
    parts=["| Method | "+" | ".join(d+" "+m.replace("_","-") for d in DOMAINS for m in METRICS)+" |",
           "| --- |"+" ---: |"*8]
    for arm in names:
        values=[]
        for d in DOMAINS:
            group=[r for r in rows if r["arm"]==arm and r["domain"]==d]
            for m in METRICS:
                sample=[float(r[m]) for r in group]
                values.append("NA" if not sample else f"{mean(sample):.4f}"+
                    (f" ± {stdev(sample):.4f}" if len(sample)>1 else ""))
        parts.append("| "+" | ".join([NAMES.get(arm,arm),*values])+" |")
    return "\n".join(parts)


def report(repo):
    out=repo/ROOT
    public=[r for d in DOMAINS for r in read(out/d/"public.csv")]
    mixed=[r for d in DOMAINS for r in read(out/d/"mixed.csv")]
    old=read(repo/"docs/experiments/left_decoder_context_20261124/public_extension/methods.csv")
    for r in old:
        if r["panel"]!="archived_full":
            continue
        raw=r["method"]
        arm=None
        if raw.startswith("CATCH"):
            arm="CATCH (3 seeds)"
        elif raw.startswith("TimeRadar official block0"):
            arm="Pretrained TimeRadar (native block grid)"
        elif raw in ("PCA residual","Pointwise level deviation"):
            arm=raw
        if arm:
            public.append(dict(r,arm=arm))
    differences=[]
    for data,pairs in [(public,[("N","scratch_N"),("N","source"),("source","scratch_N")]),
                       (mixed,[("ordinary","source"),("replay","source"),("ordinary","random_u"),
                               ("replay","ordinary"),("replay","random_u")])]:
        lookup={(r["domain"],r["arm"]):r for r in data}
        for a,b in pairs:
            for d in DOMAINS:
                x,y=lookup[d,a],lookup[d,b]
                assert (x["start"],x["stop"])==(y["start"],y["stop"])
                differences.append(dict(domain=d,role=x["role"],arm=a+" − "+b,
                    **{m:float(x[m])-float(y[m]) for m in METRICS}))
    with (out/"contrasts.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(differences[0]))
        writer.writeheader()
        writer.writerows(differences)
    print("## Complete public test\n")
    print(grid(public,dict.fromkeys(["scratch_N","source","N","Pretrained TimeRadar (native block grid)",
                                     "CATCH (3 seeds)","PCA residual","Pointwise level deviation"])))
    print("\n## Pretraining and target-normal learning\n")
    print(grid(differences,dict.fromkeys(r["arm"] for r in differences if r["role"]=="TEST")))
    print("\n## Natural mixed-target learning\n")
    print(grid(mixed,["source","ordinary","replay","random_u"]))
    print("\n"+grid(differences,dict.fromkeys(r["arm"] for r in differences if r["role"]=="QUERY")))
    print("\n## Operating costs on the isolated mixed suffix\n")
    print("| Domain | Method | Stream | Normal FPR | Recall | Normal fragments | Matched/eligible | Any hit | Event F1 | Long macro | Long pooled | Longest miss |")
    print("| --- | --- | --- |"+" ---: |"*9)
    for row in (r for d in DOMAINS for r in read(out/d/"operations.csv")):
        cols=[row["domain"],NAMES[row["arm"]],row["stream"]]
        cols += [f"{float(row[q]):.4f}" for q in ("normal_FPR","Recall")]
        cols += [row["normal_alarm_fragments"],row["M"]+"/"+row["Ntrue"],row["auxiliary_any_hit_events"]]
        cols += [f"{float(row[q]):.4f}" for q in ("one_to_one_eventF1","long_steady_macro_coverage","long_steady_point_coverage")]
        print("| "+" | ".join(cols+[row["max_event_miss_points"]])+" |")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("repo",type=Path)
    report(p.parse_args().repo)
