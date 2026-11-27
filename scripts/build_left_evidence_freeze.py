"""Consolidate archived LEFT evidence by protocol without loading scores or models."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

ARCHIVE = "a1ada3751e8840212b75400e3291422a302a9ed0"
BASE, WEIGHT, SMD = ("left_multisource_20261126", "left_local_reliability_20261126",
                     "left_smd_reliability_20261126")
MEMORY, PROJECT = "left_memory_exposure_20261126", "left_source_constraints_20261126"
METRICS = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC")
DOMAINS = ("PSM", "SWAT", "SMD28")
AXES = {
    "public": {"PSM": (0, 87744), "SWAT": (495000, 944856)},
    "external_common": {"PSM": (99, 87744)},
    "mixed": {"PSM": (22656, 87744), "SWAT": (608535, 944856)},
}
METHOD = {
    "A": "Native LEFT, target-normal from scratch",
    "B": "Frozen multisource LEFT (Z)",
    "C": "Multisource ordinary target U",
    "D": "Multisource source replay U",
    "E": "Window-weighted source replay U",
    "F": "Local-weighted source replay U",
    "N": "Multisource target-normal training (N)",
    "G": "Permuted-window-weight source replay U",
    "H": "Source-only EMA write restriction with replay",
    "I": "Random-initialized target U",
    "J": "Normal source continuation",
}
NAME_TO_ID = {
    "Target-normal LEFT from scratch": "A",
    "Frozen multisource LEFT": "B",
    "Multisource LEFT + normal training": "N",
}
MIXED_KEYS = dict(B="source", C="ordinary", D="replay", E="window", F="local",
                  I="random_u", J="source_only")
SMD_NAMES = dict(zip("BCDEIJ", ("Frozen multisource LEFT", "Multisource target U",
                              "Multisource source replay", "Window-weighted source replay",
                              "Random-init target U", "Source continuation")))
PUBLIC_CODES = {"public": "public_native_grid", "external_common": "public_external_intersection"}


def read_csv(root, file):
    with (root / file).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(root, file):
    return json.loads((root/file).read_text(encoding="utf-8"))


def index(rows, *fields):
    result = {tuple(r[f] for f in fields): r for r in rows}
    assert len(result) == len(rows), fields
    return result


def near(row, saved):
    for m in METRICS:
        assert math.isclose(float(row[m]), float(saved[m]), abs_tol=1e-12, rel_tol=0), (m,row,saved)


def mean(values):
    return math.fsum(values)/len(values)


def save(path, rows):
    fields = list(dict.fromkeys(key for r in rows for key in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="NA", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metrics(row):
    return {k: row[k] if row and row[k] not in ("", None) else "NA" for k in METRICS}


def source_row(file, selector, **hashes):
    return dict(metric_source="docs/experiments/"+file, source_selector=selector, **hashes)


def identity(receipt):
    bank, = [b for b in receipt["banks"] if b["role"] == "QUERY"]
    assert receipt["sha256"] == bank["checkpoint_sha256"]
    return dict(parent_sha256=receipt["parent"], checkpoint_sha256=receipt["sha256"],
                bank_sha256=bank["sha256"], plan_sha256=receipt.get("plan_sha256",receipt.get("numeric_plan","NA")),
                weights_sha256=receipt.get("weights_sha256","NA"))


def permissions(panel, domain, method, present):
    if not present:
        return dict(target_parameter_fit="NA", scaler="NA", source_pool="NA", readout="NA")
    if method in METHOD:
        mixed = panel in ("mixed_suffix","controlled_exposure")
        target_fit = ("target normal" if method in ("A","N") else
                      "none" if method in ("B","J") else "unlabeled target U-FIT")
        source = ("none" if method in ("A","I") else
                  "TEP FaultFree + HAI 21.03" if domain=="SMD28" else "TEP FaultFree + SMD28 train")
        return dict(target_parameter_fit=target_fit, source_pool=source,
                    scaler="target train FIT-only per machine" if mixed and domain=="SMD28" else
                    "target normal prefix including SELECT",
                    readout="W192 nonoverlapping; available at window end")
    lookup = {
        "CATCH": ("target normal", "none", "target FIT", "W192 reconstruction; offline"),
        "PCA residual": ("target normal", "none", "target FIT", "current standardized observation"),
        "Pointwise level deviation": ("none", "none", "target normal prefix", "current observation"),
        "Pretrained TimeRadar (native block grid)": ("none", "official upstream", "normal prefix", "W100 archived block grid"),
        "Pretrained TimeRadar V (causal endpoint)": ("none", "official upstream", "see fixed endpoint archive",
                                                    "causal W100 trailing endpoint"),
        "Pretrained TimeRCD (FIT-only)": ("none", "official upstream", "target FIT", "W5000 offline context"),
        "Pretrained DADA": ("none", "official upstream", "target normal prefix", "W100 block readout"),
    }
    return dict(zip(("target_parameter_fit","source_pool","scaler","readout"),lookup[method]))


def integrate(root):
    """
    Inputs are immutable CSVs and receipts. Aggregate SMD only after verifying
    equal entity identities and windows within each panel.
    """
    pub_path = f"{SMD}/public_methods.csv"
    s_path = f"{SMD}/smd_entity_metrics.csv"
    smd_review = read_json(root,f"{SMD}/training_review.json")["entities"]
    smd_parent = read_json(root,f"{SMD}/source_training.json")["sha256"]
    smd_rows = read_csv(root,s_path)
    entities = []
    for r in smd_rows:
        who, arm, panel = (r[k] for k in ("entity","arm","panel"))
        receipt = smd_review[who].get(arm)
        role = "QUERY" if panel=="mixed" else "TEST"
        if receipt:
            assert receipt["banks"][role] == r["bank_sha256"]
        entities.append(dict(**r, seed=2025, domain="SMD28",
            checkpoint_sha256=receipt["checkpoint"] if receipt else "NA",
            parent_sha256=smd_parent if arm in ("ordinary","replay","window","N","source_only") else "NA",
            plan_sha256=(receipt or {}).get("plan") or "NA",
            identity_receipt=f"docs/experiments/{SMD}/training_review.json",
            metric_source=f"docs/experiments/{s_path}"))
    assert len({r["entity"] for r in entities}) == 28
    members = {}
    for r in entities:
        members.setdefault((r["panel"],r["arm"]),[]).append(r)
    axes = {}
    macros = {}
    for (panel,arm), rows in members.items():
        assert len(rows)==len({r["entity"] for r in rows})==28
        footprint = {r["entity"]:(r["start"],r["stop"],r["points"],r["positives"]) for r in rows}
        axes.setdefault(panel,footprint)
        assert footprint==axes[panel],(panel,arm)
        assert all(0<int(r["positives"])<int(r["points"]) for r in rows)
        macros[panel,arm] = {m:mean([float(r[m]) for r in rows]) for m in METRICS}
    shperm = index(read_csv(root,f"{SMD}/permutation_review.csv"),"domain","arm")
    smd_original = index(read_csv(root,f"{SMD}/smd_mixed_macro.csv"),"arm")
    b_banks = {(r["domain"],Path(r["file"]).stem):r for r in read_json(root,f"{BASE}/banks.json")}
    b_fit = index(read_csv(root,f"{BASE}/training.csv"),"domain","arm")
    pub_catch = index(read_csv(root,f"{SMD}/psm_catch_seeds.csv"),"panel","metric")
    ordinary = index(read_csv(root,f"{WEIGHT}/methods.csv"),"domain","method","condition")
    controlled = index(read_csv(root,f"{MEMORY}/methods.csv"),"domain","method","condition")
    shuffled = index(read_csv(root,f"{SMD}/permutation_methods.csv"),"domain","method","condition")
    exposure = read_json(root,f"{MEMORY}/receipts.json")
    weights = read_json(root,f"{WEIGHT}/receipts.json")
    project = read_json(root,f"{PROJECT}/receipts.json")
    main, abc, origins = [], [], []

    def put(table, panel, domain, method_id, name, row, origin, condition="NA",
            seed="2025", n_runs=1, **extra):
        key = "/".join((panel,domain,method_id,condition))
        axis = AXES.get("mixed" if panel in ("mixed_suffix","controlled_exposure") else
                        "public" if panel=="public_native_grid" else "external_common",{}).get(domain)
        status = "NA" if row is None else "entity_macro" if "entity_source" in origin else "archived"
        table.append(dict(key=key,panel=panel,domain=domain,method_id=method_id,method=name,condition=condition,
            seed=seed,n_runs=n_runs,entities=28 if domain=="SMD28" else 1,
            start=axis[0] if axis else "per_entity",stop=axis[1] if axis else "per_entity",
            status=status,**metrics(row),**permissions(panel,domain,method_id,bool(row)),**extra))
        origins.append(dict(key=key,**origin))

    for r in read_csv(root,pub_path):
        d = "SMD28" if r["domain"]=="SMD" else r["domain"]
        p,name = r["panel"],r["method"]
        method_id = NAME_TO_ID.get(name,name)
        source = source_row(pub_path,f"{p}/{r['domain']}/{name}")
        seed,n_runs,extra = "NA","NA",{}
        if method_id in ("A","B","N"):
            seed,n_runs = "2025",1
            arm={"A":"scratch_N","B":"source","N":"N"}[method_id]
            if d=="SMD28":
                near(macros[p,arm],r)
                r=macros[p,arm]
                source.update(entity_source=f"docs/experiments/{s_path}",
                              identity_receipt=f"docs/experiments/{SMD}/training_review.json")
            else:
                saved = b_banks.get((d,arm+"_TEST"),{})
                source.update(checkpoint_sha256=b_fit[d,arm]["sha256"],
                              bank_sha256=saved.get("sha256","NA"),
                              identity_receipt=f"docs/experiments/{BASE}/{'training.csv' if arm=='scratch_N' else 'banks.json'}",
                              parent_sha256=b_fit[d,"source"]["sha256"] if arm=="N" else "NA")
                if p=="public":
                    old = index(read_csv(root,f"{BASE}/{d}/public.csv"),"arm")
                    near(r,old[arm,])
        elif name=="CATCH":
            seed,n_runs = "see archived CATCH rows",3
            for m in METRICS:
                a=pub_catch[p,m]
                assert int(a["n"])==n_runs and math.isclose(float(a["mean"]),float(r[m]),abs_tol=1e-13)
                extra[m+"_sample_sd"]=a["sample_sd"]
            source["identity_receipt"]=f"docs/experiments/{SMD}/psm_catch_seeds.csv"
        put(main,PUBLIC_CODES[p],d,method_id,METHOD.get(method_id,name),r,source,seed=seed,n_runs=n_runs,**extra)

    for d in DOMAINS:
        for code,arm in MIXED_KEYS.items():
            if d=="SMD28":
                row=macros.get(("mixed",arm))
                if row:
                    near(row,smd_original[SMD_NAMES[code],])
                link=source_row(s_path,f"mixed/{arm}",entity_source=f"docs/experiments/{s_path}",
                                identity_receipt=f"docs/experiments/{SMD}/training_review.json") if row else {}
            else:
                cond="A" if code in "CDEF" else "ref"
                row=ordinary[d,arm,cond]
                assert (int(row["start"]),int(row["stop"]))==AXES["mixed"][d]
                if code in "BCDI":
                    path=f"{BASE}/{d}/mixed.csv"
                    old=index(read_csv(root,path),"arm")
                    near(row,old[arm,])
                    b=b_banks[d,arm+"_QUERY"]
                    link=source_row(path,arm,checkpoint_sha256=b["checkpoint_sha256"],bank_sha256=b["sha256"],
                        parent_sha256=b_fit[d,"source"]["sha256"] if code in "CD" else "NA",
                        identity_receipt=f"docs/experiments/{BASE}/banks.json")
                    if code in "CD":
                        link["plan_sha256"]=exposure[d]["qualification"]["A_sha256"]
                else:
                    family=PROJECT if code=="J" else WEIGHT
                    receipt=project[d]["arms"][arm] if code=="J" else weights[d]["arms"][arm+"_A"]
                    link=source_row(f"{WEIGHT}/methods.csv",f"{d}/{row['arm']}",
                                    identity_receipt=f"docs/experiments/{family}/receipts.json",**identity(receipt))
            put(main,"mixed_suffix",d,code,METHOD[code],row,link,condition="natural" if code in "CDEF" else "ref",
                seed="2025" if row else "NA",n_runs=1 if row else "NA")
        put(main,"mixed_suffix",d,"A",METHOD["A"],None,{},seed="NA",n_runs="NA")

    for d in DOMAINS[:2]:
        for c in "ABC":
            plan=exposure[d]["qualification"][c+"_sha256"]
            for code,arm in dict(C="ordinary",D="replay",E="window",F="local",G="permuted",H="restricted").items():
                family = MEMORY if code in "CDH" else WEIGHT if code in "EF" else SMD
                row = (controlled if family==MEMORY else ordinary if family==WEIGHT else shuffled)[d,arm,c]
                if code in "CD":
                    near(row,ordinary[d,arm,c])
                assert (int(row["start"]),int(row["stop"]))==AXES["mixed"][d]
                if code=="G":
                    r=shperm[d,arm+"_"+c]
                    binding=dict(checkpoint_sha256=r["checkpoint"],parent_sha256=r["source"],
                                 plan_sha256=r["input_plan"],weights_sha256=r["weights"])
                    receipt_file=f"{SMD}/permutation_review.csv"
                elif code in "CD" and c=="A":
                    b=b_banks[d,arm+"_QUERY"]
                    binding=dict(checkpoint_sha256=b["checkpoint_sha256"],bank_sha256=b["sha256"],
                                 parent_sha256=b_fit[d,"source"]["sha256"],plan_sha256=plan)
                    receipt_file=f"{BASE}/banks.json"
                else:
                    r=(exposure if family==MEMORY else weights)[d]["arms"][arm+"_"+c]
                    binding=identity(r)
                    receipt_file=f"{family}/receipts.json"
                assert binding["parent_sha256"]==b_fit[d,"source"]["sha256"]
                assert binding["plan_sha256"]==plan
                file="permutation_methods.csv" if family==SMD else "methods.csv"
                put(abc,"controlled_exposure",d,code,METHOD[code],row,
                    source_row(f"{family}/{file}",f"{d}/{row['arm']}",
                               identity_receipt="docs/experiments/"+receipt_file,**binding),condition=c)
    table=index(abc,"domain","method_id","condition")
    contrasts=[]
    archive_deltas=index(read_csv(root,f"{WEIGHT}/contrasts.csv"),"domain","comparison")
    for d in DOMAINS[:2]:
        for m in "CDEFGH":
            for a,b in ("BA","CA","BC"):
                val={k:float(table[d,m,a][k])-float(table[d,m,b][k]) for k in METRICS}
                contrasts.append(dict(domain=d,method=m,reference="NA",contrast=f"{a}-{b}",**val))
        for m,ref in (("D","C"),("E","D"),("F","D"),("F","E"),("E","G"),("H","D")):
            def difference(k,c):
                return float(table[d,m,c][k])-float(table[d,ref,c][k])
            for c in "ABC":
                contrasts.append(dict(domain=d,method=m,reference=ref,contrast=c,**{k:difference(k,c) for k in METRICS}))
            interaction={k:difference(k,"B")-difference(k,"C") for k in METRICS}
            if (m,ref)==("E","D"):
                near(interaction,archive_deltas[d,"window-replay interaction"])
            contrasts.append(dict(domain=d,method=m,reference=ref,contrast="interaction_B-C",**interaction))
    return main, abc, contrasts, entities, origins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repository/"docs/experiments"
    out = args.output or root/"left_evidence_freeze_20261128"
    out.mkdir(parents=True, exist_ok=True)
    main_rows,abc,contrasts,entities,origins=integrate(root)
    for name, data in zip(("main_metrics","abc_metrics","abc_contrasts","smd_entities","source_manifest"),
                          (main_rows,abc,contrasts,entities,origins)):
        save(out/f"{name}.csv",data)
    files = {r[k] for r in origins for k in ("metric_source","entity_source","identity_receipt") if k in r}
    files.update(f"docs/experiments/{d}/{f}" for d,f in (
        (BASE,"training.csv"),(SMD,"source_training.json"),(WEIGHT,"contrasts.csv"),
        (MEMORY,"receipts.json"),(PROJECT,"RESULTS.md"),("left_proxy_rank_20261127","methods.csv"),
        ("left_input_correction_20261127","selection.json")))
    digests = {f:hashlib.sha256((args.repository/f).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
               for f in sorted(files)}
    (out/"inputs.json").write_text(json.dumps(dict(archive_commit=ARCHIVE,archived_lf_sha256=digests),
                                           indent=2)+"\n",encoding="utf-8")
    print(f"{len(main_rows)} main rows, {len(abc)} exposure rows, {len(contrasts)} contrasts, "
          f"{len(entities)} SMD entity rows")


if __name__=="__main__":
    main()
