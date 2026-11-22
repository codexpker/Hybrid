"""Compare cached causal NASA scores at the historical 512-point warmup."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from .left_native_swat_metrics import continuous_metrics, load_author_metric


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_entity(args):
    root, banks, metric_file, phase, entity = args
    folder = (root / "results/hybrid_v1_public_native_zs_20261105_v1/evaluation/entities/SMAP" /
              entity)
    with np.load(folder / "history_mean512/calculation_inputs.npz") as z:
        qi = z["index"]
        truth = z["full_labels"]
    np.testing.assert_array_equal(qi, np.arange(512, len(truth)))
    refs = {}
    if phase == "external":
        for method in ("history_mean512", "persistence", "RoCA_FULL_T"):
            base = folder
            if method == "RoCA_FULL_T":
                base = root / "results/roca_public_unified_evaluation_20261106_v1/entities/SMAP" / entity
            path = base / method / "calculation_inputs.npz"
            with np.load(path) as z:
                for key in ("available_index", "index"):
                    np.testing.assert_array_equal(z[key], qi)
                np.testing.assert_array_equal(z["full_labels"], truth)
                assert z["valid"].all()
                refs[method] = (z["score"], sha(path), base/method/"result.json")
    else:
        path = banks/f"{entity}.npz"
        info = json.loads(path.with_suffix(".json").read_text())
        digest = sha(path)
        assert digest == info["sha256"]
        with np.load(path) as z:
            np.testing.assert_array_equal(z["query_index"], np.arange(99,len(truth)))
            for i,method in enumerate(info["arms"]):
                refs[method] = (z["query"][512-99:,i], digest, None)
    rows = []
    m = load_author_metric(metric_file)
    for method, (scores, bank_sha, rank_path) in refs.items():
        assert np.isfinite(scores).all()
        ranking,_ = continuous_metrics(truth[qi],scores,m)
        if rank_path:
            old = json.loads(rank_path.read_text(encoding="utf-8"))["ranking"]
            for key in ("AP","ROC_AUC"):
                np.testing.assert_allclose(ranking[key],old[key],rtol=0,atol=1e-12)
        rows.append(dict(entity=entity, method=method, start=512, stop=len(truth), n=len(qi),
                         positives=int(truth[qi].sum()), bank_sha256=bank_sha,
                         **{k:ranking[k] for k in ("AP","ROC_AUC","VUS_PR","VUS_ROC")}))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("external","frozen"))
    for key in ("root","metric","out"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--banks",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    archive=args.root/"results/hybrid_v1_public_native_zs_20261105_v1/evaluation/entities/SMAP"
    ids=sorted(p.name for p in archive.iterdir() if p.is_dir())
    if len(ids)!=53:
        raise ValueError("Expected the 53 previously fixed SMAP entities")
    if args.phase=="frozen" and (not args.banks or not (args.banks/"methods.csv").is_file()):
        raise ValueError("Complete frozen metric archive required")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        groups=list(pool.map(rank_entity,[(args.root,args.banks,args.metric,args.phase,i) for i in ids]))
    rows=[r for group in groups for r in group]
    with args.out.open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__=="__main__":
    main()
