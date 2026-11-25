"""Assemble the advisor manuscript from immutable archived metric exports."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
DOC=REPO/"docs/experiments"
D=("PSM","SMD","MSL","SMAP")
M=("AP","ROC_AUC","VUS_PR","VUS_ROC")
USED={}


def csv(path):
    file=DOC/path
    USED[path]=hashlib.sha256(file.read_bytes()).hexdigest()
    return pd.read_csv(file)


def md(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|",
                       *["| "+" | ".join(map(str,row))+" |" for row in rows]])


def paired_tables():
    frame=csv("native_endpoint_external_20261125/domain_metrics.csv")
    full=frame[frame.panel=="causal"]
    repeat=[]
    paired=[]
    seed_rows=[]
    for domain in D:
        sub=full[full.domain==domain]
        for method,label in [("official_V","Frozen V"),("Source reconstruction continuation","Reconstruction"),
                             ("Source-supervised endpoint V","Direct V")]:
            a=sub[sub.method==method].sort_values("seed")
            n=len(a)
            assert n==(1 if method=="official_V" else 3)
            repeat.append([domain,label,str(n),*[
                f"{a[k].mean():.5f}"+(f" ± {a[k].std(ddof=1):.5f}" if n==3 else "") for k in M]])
        cand=sub[sub.method=="Source-supervised endpoint V"].set_index("seed")
        for method,label in [("official_V","Direct V − frozen V"),("Source reconstruction continuation","Direct V − reconstruction")]:
            ref=sub[sub.method==method].set_index("seed")
            delta=cand[list(M)].subtract(ref.iloc[0][list(M)],axis=1) if method=="official_V" else cand[list(M)]-ref[list(M)]
            assert len(delta)==3 and delta.notna().all().all()
            paired.append([domain,label,*[f"{delta[k].mean():+.5f} ± {delta[k].std(ddof=1):.5f}" for k in M]])
            for seed,row in delta.iterrows():
                seed_rows.append([domain,label,int(seed),*[f"{row[k]:+.6f}" for k in M]])
    return frame,md(["Domain","Method","Training seeds",*M],repeat),md(["Domain","Within-seed contrast",*M],paired),md(["Domain","Contrast","Seed",*M],seed_rows)


def main_tables(frame):
    offline=frame[frame.panel=="common_offline"]
    labels=[("official_V","Frozen TimeRadar V"),("official_E","Frozen TimeRadar E"),
            ("official_V_plus_E","Frozen fixed V + E"),("Source reconstruction continuation","Source reconstruction (3 seeds)"),
            ("Source-supervised endpoint V","Direct V (3 seeds)"),("Complete LEFT","Complete LEFT (1 seed)"),
            ("DADA official V","Complete DADA"),
            ("TimeRCD official weights, FIT-only adapter","TimeRCD, FIT-only"),
            ("CATCH W192 seed0","CATCH seed0"),("CATCH W192 seed1","CATCH seed1"),
            ("CATCH W192 seed2","CATCH seed2"),("PCA residual","PCA")]
    panels=[]
    for metrics in (M[:2],M[2:]):
        rows=[]
        for method,name in labels:
            values=[]
            for domain in D:
                a=offline[(offline.domain==domain)&(offline.method==method)]
                values.append("NA" if a.empty else " / ".join(
                    f"{a[k].mean():.4f}"+(f" ± {a[k].std(ddof=1):.4f}" if len(a)==3 else "") for k in metrics))
            rows.append([name,*values])
        panels.append(md(["Method","PSM","SMD28","MSL27","SMAP53"],rows))
    aux=frame[frame.panel=="causal_history"]
    rows=[]
    for method in ["Source-supervised endpoint V","History Mean512","PCA residual"]:
        a=aux[(aux.method==method)&aux.seed.isin([0,2025])]
        for _,r in a.iterrows():
            rows.append([r.domain,method,*[f"{r[k]:.5f}" for k in M]])
    return *panels,md(["Domain","Method",*M],rows)


def ablation():
    f=csv("native_supervised_mlp_20261123/methods.csv")
    cross=csv("native_e_vs_v_20261123/methods.csv")
    f=pd.concat([f,cross[cross.method.isin(["Vtrain_E","Etrain_V"])]],ignore_index=True)
    rows=[]
    for key,label in [("copy_V","Normal reconstruction / V"),("Vtrain_V","Direct V / V"),
                      ("Etrain_E","Direct E / E"),("MLP_full","Frozen features + supervised MLP"),
                      ("Vtrain_E","V-trained / E (auxiliary)"),("Etrain_V","E-trained / V (auxiliary)")]:
        a=f[f.method==key].set_index("dataset")
        rows.append([label,*[" / ".join(f"{a.loc[d,k]:.4f}" for k in M) for d in D]])
    return md(["Training / formal readout (seed2025)","PSM","SMD28","MSL27","SMAP53"],rows)


def boundaries():
    rows=[]
    for path,method,ref,label in [
        ("native_tep_support_20261125/paired_distribution.csv",None,"single_V","Expanded support V − matched single-source V"),
        ("native_source_full_20261125/paired_distribution.csv","Full V supervision / V","Source-supervised endpoint V","Full update V − decoder-only V")]:
        f=csv(path)
        f=f[(f.panel=="causal")&(f.reference==ref)]
        if method:f=f[f.method==method]
        a=f.set_index(["domain","metric"])
        rows.append([label,*[" / ".join(f"{a.loc[(d,k),'mean']:+.4f}" for k in M) for d in D]])
    return md(["Boundary comparison (AP / ROC / VUS-PR / VUS-ROC)","PSM","SMD28","MSL27","SMAP53"],rows)


def operations():
    rows=[]
    f=csv("native_event_rank_repeats_20261122/operating_methods.csv")
    g=csv("native_event_smd28_20261123/native/operating.csv")
    for d in D:
        for method,label in [("frozen_native_V","Frozen V"),("copy","Reconstruction"),("point_weighted","Direct V")]:
            for stream in ("strict","confirmed3"):
                if d!="SMD":
                    a=f[(f.dataset==d)&(f.method==method)&(f.stream==stream)].iloc[0]
                    get=lambda k:a[k+"_mean"]
                    n=int(a["n"])
                else:
                    a=g[(g.stream==stream)&(g.key.str.contains(method) if method!="frozen_native_V" else g.key.eq(method))]
                    get=lambda k:a[k].mean()
                    n=len(a)
                rows.append([d,label,stream,str(n),f"{get('normal_FPR'):.4f}",f"{get('Recall'):.4f}",
                    f"{get('M'):.2f} / {get('Ntrue'):.0f}",f"{get('auxiliary_any_hit_events'):.2f}",
                    f"{get('event_F1'):.4f}",f"{get('normal_alarm_fragments'):.1f}",
                    f"{get('long_steady_macro_coverage'):.4f} / {get('long_steady_pooled_coverage'):.4f}",
                    f"{get('longest_event_miss'):.0f}",f"{get('hit_only_median_delay'):.1f}"])
    for row in rows:
        if row[1]=="Reconstruction":row[1]="Recon."
    rates=md(["Domain","Method","Stream","Seeds","Normal FPR","Recall","Event F1","False fragments"],
             [[row[i] for i in (0,1,2,3,4,5,8,9)] for row in rows])
    coverage=md(["Domain","Method","Stream","Matched / eligible","Any-hit","Long macro / pooled","Longest miss","Hit delay"],
                [[row[i] for i in (0,1,2,6,7,10,11,12)] for row in rows])
    return rates+"\n\nEvent and coverage values for the same models and streams follow.\n\n"+coverage


def entities():
    f=csv("native_endpoint_external_20261125/entity_pairs.csv")
    rows=[]
    for (d,seed,ref),group in f[f.panel=="causal"].groupby(["domain","seed","reference"]):
        if d=="PSM":continue
        a=group.AP
        rows.append([d,int(seed),ref,f"{a.mean():+.5f}",f"{a.median():+.7f}",f"{(a>0).sum()}/{(a==0).sum()}/{(a<0).sum()}"])
    return md(["Domain","Seed","Reference","Mean ΔAP","Median ΔAP","Win/tie/loss"],rows)


def extensions():
    f=csv("native_event_stage2_u_20261123/methods.csv")
    f=f[f.stream=="strict"]
    rows=[]
    names={"noU":"Frozen candidate","replay_copy":"Replay reconstruction","replay_pair":"Replay + source pair","source_pair":"Source-only continuation"}
    for _,r in f.iterrows():
        rows.append([r.domain,names[r.arm],*[f"{r[k]:.4f}" for k in M]])
    u=md(["Separate mixed-task axis","Procedure",*M],rows)
    f=csv("native_event_stage2_f_20261123/methods.csv")
    rows=[]
    for _,r in f[f.stream=="strict"].iterrows():
        rows.append([r.domain,r.initialization,r.phase,*[f"{r[k]:.4f}" for k in M]])
    return u,md(["Separate support-task axis","Initialization","Before/after",*M],rows)


def selections():
    first=csv("native_event_rank_20261122/source_select.csv")
    first["seed"]=2025
    other=csv("native_event_rank_repeats_20261122/source/source_select.csv")
    rows=[]
    for _,r in pd.concat([first,other],ignore_index=True).iterrows():
        if str(r.selected).lower()!="true" or r.arm not in ("copy","point_weighted"):continue
        steps=int(r["update"])
        rows.append([int(r.seed),"Direct V" if r.arm=="point_weighted" else "Reconstruction",steps,
                     f"{r.risk:.7f}",steps*32,steps*15 if r.arm=="point_weighted" else 0])
    assert len(rows)==6
    return md(["Seed","Procedure","Selected update","Source risk","Nominal exposures","Positive / negative pair exposures each"],rows)


def diagram():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42})
    fig,ax=plt.subplots(figsize=(12,6.1))
    ax.set_xlim(0,12);ax.set_ylim(0,6.1);ax.axis("off")
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.07",facecolor=color,edgecolor="#304256",lw=1))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)
    def arrow(x,y,u,v,color="#304256"):
        ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle="-|>",mutation_scale=13,lw=1.3,color=color))
    box(.2,4.5,2.5,.85,"SOURCE windows\nnominal + event pairs","#edf2f6")
    box(3.2,4.35,3.5,1.2,"Frozen official TimeRadar\ninput-dependent order + FRFT\npatch embeddings + encoder","#dce9f5")
    box(7.15,4.35,2.35,1.2,"Native decoder\nonly updated weights","#ffe7c2")
    box(9.95,4.35,1.8,1.2,"iFRFT + magnitude\n10 reconstructions","#dce9f5")
    arrow(2.77,4.92,3.1,4.92);arrow(6.77,4.92,7.05,4.92);arrow(9.57,4.92,9.86,4.92)
    box(6.9,2.65,2.7,.85,"All-position per-copy MSE\non nominal windows","#edf4e8")
    box(9.85,2.35,1.95,1.15,"Endpoint 99\ncopy V, correction=1\nchannel mean","#edf4e8")
    arrow(10.85,4.27,10.85,3.58)
    arrow(10.5,4.27,8.25,3.58)
    box(2.1,2.45,3.8,1.15,"Weighted source-pair softplus\nlog(1 + V/a), margin 1\n+ nominal reconstruction MSE","#ffe7c2")
    arrow(6.82,3.08,5.98,3.08);arrow(9.77,2.75,5.98,2.75)
    arrow(4.0,3.68,8.3,4.23,"#b86b16")
    ax.text(5.15,4.03,"decoder gradients",fontsize=9,color="#92530c")
    ax.plot([.2,11.8],[1.85,1.85],color="#97a8b8",lw=1)
    box(.2,.45,2.5,.9,"TARGET x[t−99:t]\nPermitted target FIT scaler","#edf2f6")
    box(3.2,.45,5.95,.9,"Selected frozen detector\nsame ten masks and independent B=1 forward","#dce9f5")
    box(9.85,.45,1.95,.9,"Raw endpoint V\navailable at t","#edf4e8")
    arrow(2.77,.9,3.1,.9);arrow(9.23,.9,9.76,.9)
    ax.text(.2,5.85,"Source training and source-only checkpoint selection",weight="bold",fontsize=12)
    ax.text(.2,1.58,"Target detection with zero parameter updates",weight="bold",fontsize=12)
    folder=ROOT/"figures";folder.mkdir(exist_ok=True)
    for ext in ("pdf","png"):
        fig.savefig(folder/f"method.{ext}",bbox_inches="tight",dpi=220)
    plt.close(fig)


def build():
    ROOT.mkdir(exist_ok=True)
    snapshot=REPO/"manuscript/snapshots/before_endpoint_research_20261126"
    snapshot.mkdir(parents=True,exist_ok=True)
    for name in ("draft.md","method_definition.md","references.bib"):
        source=REPO/"manuscript/next_generation_20261115"/name
        target=snapshot/name
        if target.exists():assert target.read_bytes()==source.read_bytes()
        else:shutil.copy2(source,target)
    frame,repeat,paired,seeds=paired_tables()
    ap,vu,history=main_tables(frame)
    u,f=extensions()
    values={"MAIN_AP":ap,"MAIN_VUS":vu,"REPEATS":repeat,"PAIRED":paired,"SEEDS":seeds,
            "ABLATION":ablation(),"BOUNDARIES":boundaries(),"OPERATIONS":operations(),"ENTITIES":entities(),
            "U":u,"F":f,"HISTORY":history,"SELECTION":selections()}
    for template,output in (("main.template.md","manuscript.md"),("supplement.template.md","supplement.md")):
        text=(ROOT/template).read_text(encoding="utf8")
        text=re.sub(r"(?<=[a-z])(?=\d)"," ",text)
        text=re.sub(r"@([A-Za-z]+) (?=\d)",r"@\1",text)
        text=re.sub(r"(?<=[A-Za-z]),(?=\d)",", ",text)
        for key,value in values.items():text=text.replace("{{"+key+"}}",value)
        assert not re.search(r"\{\{[A-Z_]+\}\}",text)
        (ROOT/output).write_text(text,encoding="utf8")
    usedbib=(REPO/"manuscript/next_generation_20261115/references.bib").read_text()
    keys=["timeradar2026","left2026","catch2025","shentu2025dada","lan2026timercd"]
    entries=[re.search(r"@[^@]+\{"+k+r",.*?(?=\n@|\Z)",usedbib,re.S).group().strip() for k in keys]
    extra=(ROOT/"additional_references.bib").read_text()
    (ROOT/"references.bib").write_text("\n\n".join(entries)+"\n\n"+extra,encoding="utf8")
    diagram()
    provenance=["native_event_rank_20261122/PROVENANCE.json",
        "native_event_rank_repeats_20261122/source/source_provenance.json",
        "native_endpoint_external_20261125/AXES_AND_INPUTS.json",
        "native_endpoint_external_20261125/RESULTS.md",
        "native_event_rank_public_20261122/MSL_NATIVE_MEMBERS.json"]
    lineage={name:hashlib.sha256((DOC/name).read_bytes()).hexdigest() for name in provenance}
    (ROOT/"evidence_sources.json").write_text(json.dumps(dict(metric_files_sha256=USED,
        provenance_files_sha256=lineage,
        source_git="d92740aec208017b56723055f365318f19159f48",
        snapshots=str(snapshot.relative_to(REPO)),statistics="Entity-first original exports; seed sample SD ddof=1",
        experiment_execution=False),indent=2),encoding="utf8")


if __name__=="__main__":
    build()
