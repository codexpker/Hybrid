"""Render the three LEFT study figures directly from full-precision metric tables."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize_left_exposure_repeats import FREEZE, REPEAT, read_rows

LABELS = {"PSM": "PSM", "SWAT": "SWaT", "SMD28": "SMD28"}
COLORS = {"R": "#225b79", "W": "#a84c28"}
SEEDS = (2025, 2026, 2027)
MARKERS = ("o", "s", "^")


def save(fig, output, name):
    path = output/f"{name}.svg"
    fig.savefig(path, metadata={"Date": None})
    markup = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in markup.splitlines())+"\n",
                    encoding="utf-8", newline="\n")
    plt.close(fig)


def natural(repo, paired, output):
    archive = read_rows(repo/FREEZE/"main_metrics.csv")
    mixed = {(r["domain"],r["method_id"]): r for r in archive if r["panel"]=="mixed_suffix"}
    fig, ax = plt.subplots(figsize=(7.3, 2.8), layout="constrained")
    for i, d in enumerate(LABELS):
        dx = lambda left, right: 100*(float(mixed[d,left]["AP"])-float(mixed[d,right]["AP"]))
        ax.scatter(dx("D","C"),i+0.15,marker="D",s=40,color="#225b79",
                   label="Replay − target-only, one archived pair" if i==0 else None)
        if d=="SMD28":
            ax.scatter(dx("E","D"),i-0.16,s=44,color=COLORS["W"], marker="D",
                       label="Weighted − replay, one archived macro")
        else:
            for j, (seed, marker) in enumerate(zip(SEEDS,MARKERS)):
                x=100*float(paired[d,seed]["W_minus_R_A"])
                ax.scatter(x,i-0.10-0.12*j,marker=marker,s=42,color=COLORS["W"],
                           label=f"Weighted − replay, {seed}" if i==0 else None)
    ax.axvline(0,color="#374151",linewidth=0.8)
    ax.set_yticks(range(3),LABELS.values())
    ax.set_ylim(2.45,-0.43)
    ax.set_xlabel("Paired Δ AP (percentage points)")
    ax.legend(frameon=False,ncol=2,fontsize=8,loc="upper center",bbox_to_anchor=(0.5,-0.25))
    save(fig, output, "natural_adaptation")


def exposure(paired, output):
    fig, axes = plt.subplots(1,3,figsize=(8.1,2.9),width_ratios=(1,1,1.05),layout="constrained")
    for d, ax in zip(("PSM","SWAT"),axes[:2]):
        for j, (s, marker) in enumerate(zip(SEEDS,MARKERS)):
            row = paired[d,s]
            for m, offset in (("R",-0.13),("W",0.13)):
                x = [i+offset+(j-1)*0.09 for i in range(3)]
                y = [100*float(row[m+c]) for c in "ABC"]
                ax.plot(x,y,color=COLORS[m],alpha=0.68,linewidth=0.9)
                ax.scatter(x,y,marker=marker,color=COLORS[m],s=25,zorder=3)
        ax.set_xticks(range(3), list("ABC"))
        ax.set_title(LABELS[d])
        ax.set_ylabel("Absolute AP (%)")
        ax.set_xlabel("Exposure condition")
    ax=axes[2]
    for i,d in enumerate(("PSM","SWAT")):
        for j,(s, marker) in enumerate(zip(SEEDS,MARKERS)):
            ax.scatter(100*float(paired[d,s]["interaction"]),i-0.2+0.2*j,
                       marker=marker,s=30,color="#332a52")
    ax.set_yticks([0,1],("PSM","SWaT"))
    ax.set_ylim(1.45,-0.45)
    ax.axvline(0,color="#374151",linewidth=0.8)
    ax.set_title("B−C interaction")
    ax.set_xlabel("AP percentage points")
    handles = [plt.Line2D([],[],color=color,label=label)
               for color,label in ((COLORS["R"],"Replay"),(COLORS["W"],"Weighted"))]
    handles.extend(plt.Line2D([],[],marker=marker,color="#332a52",linestyle="none",label=str(seed))
                   for seed,marker in zip(SEEDS,MARKERS))
    fig.legend(handles=handles,ncol=5,loc="outside lower center",frameon=False,fontsize=8)
    save(fig, output, "controlled_exposure")


def smd(repo, output):
    archive = read_rows(repo/FREEZE/"smd_entities.csv")
    data = {(r["entity"],r["arm"]): r for r in archive if r["panel"]=="mixed"}
    replay = {entity for entity,arm in data if arm=="replay"}
    weighted = {entity for entity,arm in data if arm=="window"}
    assert replay==weighted and len(replay)==28, (sorted(data)[:8],len(replay),len(weighted))
    series = sorted(((100*(float(data[e,"window"]["AP"])-float(data[e,"replay"]["AP"])),e)
                     for e in replay))
    x, entities = zip(*series)
    fig,ax=plt.subplots(figsize=(7,6.1),layout="constrained")
    ax.barh(range(len(x)),x,color=[COLORS["R"] if v>=0 else COLORS["W"] for v in x],height=0.72)
    ax.set_yticks(range(len(x)),entities,fontsize=7.5)
    ax.axvline(0,color="#374151",linewidth=0.8)
    ax.axvline(sum(x)/len(x),color="#332a52",linestyle="--",label="Equal-weight macro")
    ax.set_xlabel("Weighted replay − replay AP (percentage points)")
    ax.legend(loc="lower right",fontsize=9,frameon=False)
    save(fig,output,"smd28_heterogeneity")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    paired={(r["domain"],int(r["seed"])):r for r in read_rows(args.repository/REPEAT/"paired.csv")
            if r["metric"]=="AP"}
    assert len(paired)==6
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"svg.hashsalt":"left-controlled-2026","font.family":"DejaVu Sans",
                         "font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.grid":True,"grid.alpha":0.16,"axes.axisbelow":True})
    natural(args.repository,paired,args.output)
    exposure(paired,args.output)
    smd(args.repository,args.output)


if __name__=="__main__":
    main()
