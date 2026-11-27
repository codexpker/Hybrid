"""Render the first TimeRadar history-context architecture from the fixed design."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw(output):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set(xlim=(0,12), ylim=(0,6))
    ax.axis("off")
    boxes = {
        "history": (0.1,4.5,2.0,0.7,"Past 256 / 512\nsame record, before X"),
        "patch": (2.6,4.5,2.3,0.7,"Per-variable patch32\ncumulative level + spread"),
        "past": (5.4,4.5,2.4,0.7,"Shared causal time blocks\nscales 1 / 2 / 4"),
        "vars": (8.3,4.5,2.8,0.7,"Variable attention\nretain channel and patch axes"),
        "current": (0.1,2.7,2.0,0.7,"Current W100\nsource-FIT scaler"),
        "native": (2.6,2.7,2.3,0.7,"Official order, FRFT\n10 complementary masks"),
        "complex": (5.4,2.7,2.4,0.7,"Official complex encoder\n10 x C x 20 x 256"),
        "context": (8.3,2.7,2.8,0.7,"Cross-attention from copies\nzero-init complex residual"),
        "decoder": (8.3,0.9,2.8,0.7,"Official decoder + iFRFT + abs\n10 complete reconstructions"),
        "loss": (4.6,0.9,2.9,0.7,"Train: full-window copy MSE\nhistory path + native decoder"),
        "score": (0.4,0.9,3.4,0.7,"Deploy: original endpoint variance V\ncorrection=1, channel mean"),
    }
    for key,(x,y,w,h,text) in boxes.items():
        color = "#e7eef3" if key in ("native","complex","decoder","score") else "#e8f1e9"
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04",facecolor=color,
                                   edgecolor="#345366",linewidth=1.1))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=9)
    def arrow(a,b):
        ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1.4,color="#345366"))
    for y in (4.85,3.05):
        for x1,x2 in ((2.1,2.6),(4.9,5.4),(7.8,8.3)):
            arrow((x1,y),(x2,y))
    arrow((9.7,4.5),(9.7,3.4))
    arrow((9.7,2.7),(9.7,1.6))
    arrow((8.3,1.25),(7.5,1.25))
    ax.plot([9.7, 9.7, 2.1], [.9, .57, .57], color="#345366", linewidth=1.4)
    arrow((2.1,.57),(2.1,.9))
    ax.text(0.15,5.65,"TimeRadar with trainable past multivariate multiscale context",
            fontsize=15,weight="bold",color="#193c50")
    ax.text(0.15,.25,"Prototype only. No separate history anomaly score. Past path trained jointly; official score retained.",
            fontsize=10,color="#345366")
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(output,metadata={"Date":None})
    plt.close(fig)
    output.write_text("\n".join(x.rstrip() for x in output.read_text().splitlines())+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    draw(args.output)
