"""Verify and summarize the complete source-only LEFT correction matrix."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from hybrid_tsad import left_input_correction as dev
from hybrid_tsad import left_multisource as base


def verify_roles(root, width):
    preflight, plan = dev.plan_files(root)
    roles = base.read(root/"roles.json")
    for a, b in itertools.combinations(roles, 2):
        for left, right in zip(roles[a], roles[b]):
            for x, y in itertools.product(left, right):
                if x["record"] == y["record"]:
                    assert x["stop"] <= y["start"] or y["stop"] <= x["start"]
    for role, group in roles.items():
        rows = plan[role].reshape(-1, width+2)
        assert np.all(np.diff(np.sort(rows[:, 2:], axis=1), axis=1) > 0)
        for j, records in enumerate(group):
            r, start = rows[j::2, :2].T
            lengths = np.array([x["stop"]-x["start"] for x in records])
            assert np.all((r >= 0) & (r < len(lengths)))
            assert np.all((start >= 0) & (start+dev.WINDOW <= lengths[r]))
            assert np.all((rows[j::2, 2:] >= 0) & (rows[j::2, 2:] < (52 if j==0 else 38)))
    return preflight, roles


def audit(w):
    summary, panels, receipts = [], [], {}
    for d, width in base.WIDTH.items():
        root = w/dev.OUT/d
        prep, roles = verify_roles(root,width)
        gate = base.read(root/"check.json")
        assert gate["raw_native_readout"] and gate["teacher_frozen"] and gate["source_unchanged"]
        assert gate["rng_equal"] and gate["teacher_optimizer_disjoint"]
        receipts[d] = dict(prepare=prep, check=gate, roles=roles, arms={})
        for arm in dev.CONFIG:
            p = root/arm
            r = base.read(p/"receipt.json")
            assert r["checkpoint_sha256"]==base.sha(p/"selected.pt")
            assert r["plan"]==prep["hashes"]["plan.npz"] and r["query"]==prep["hashes"]["query.npz"]
            assert r["parent"]==prep["parent"] and r["updates"]==2000 and r["model_step"]==6000
            assert r["source_exposure"]==r["target_exposure"]==32000 and not r["labels_read"]
            assert r["edited_proxy_exposure"]==8000
            proxy, diag = r["proxy"], r["diagnostics"]
            raw = [x["AP"] for x in proxy["panels"]]
            np.testing.assert_allclose(np.mean(raw),proxy["macro_AP"],rtol=0,atol=1e-15)
            summary.append(dict(
                width=width, arm=arm, AP=np.mean(raw),
                ROC_AUC=np.mean([x["ROC_AUC"] for x in proxy["panels"]]),
                TEP_risk=proxy["normal_risk"]["TEP"], SMD_risk=proxy["normal_risk"]["SMD"],
                clean_changed=diag["clean_changed"], clean_rms=diag["clean_rms"],
                amp_edited_corr=diag["eligible_edit_fraction"][0],
                shift_edited_corr=diag["eligible_edit_fraction"][1],
                amp_relative_rms=diag["clean_relative_edit_rms"][0],
                shift_relative_rms=diag["clean_relative_edit_rms"][1],
                amp_off_edit_rms=diag["off_edit_rms"][0], shift_off_edit_rms=diag["off_edit_rms"][1],
                teacher_train_seconds=r["teacher_train_seconds"], job_seconds=r["seconds"],
            ))
            panels.extend(dict(width=width,arm=arm,**q) for q in proxy["panels"])
            receipts[d]["arms"][arm] = r
    return summary, panels, receipts


def select(rows):
    table = {(r["width"],r["arm"]):r for r in rows}
    metrics = {}
    for arm in dev.CONFIG:
        cells = [table[width,arm] for width in base.WIDTH.values()]
        ref = [table[width,"R0"] for width in base.WIDTH.values()]
        risk = {str(a["width"]):{s:a[s]/b[s] for s in ("TEP_risk","SMD_risk")}
                for a,b in zip(cells,ref)}
        ap_delta = [a["AP"]-b["AP"] for a,b in zip(cells,ref)]
        improves = np.mean(ap_delta) >= .005 and min(ap_delta) >= -.005
        safe = max(v for r in risk.values() for v in r.values()) <= 1.05
        metrics[arm] = dict(mean_AP=float(np.mean([a["AP"] for a in cells])),
                           mean_AP_minus_R0=float(np.mean(ap_delta)), fold_AP_minus_R0=ap_delta,
                           risk_ratio=risk, safe=bool(safe),
                           eligible=bool(arm in ("R1","R2","R3","R4") and improves and safe))
    eligible = [k for k,v in metrics.items() if v["eligible"]]
    selected = min(eligible,key=lambda k:(-metrics[k]["mean_AP"],dev.CONFIG[k][0],-dev.CONFIG[k][1])) if eligible else "R0"
    return dict(selected=selected, arms=metrics, target_query_opened=False)


def save_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def teacher_copy(w):
    """Measure paired teacher response and whether matching the edit suppresses repair."""
    import time
    import torch

    started = time.monotonic()
    rows = []
    for domain, width in base.WIDTH.items():
        root = w/dev.OUT/domain
        prep, _ = dev.plan_files(root)
        model = base.initialize(w, domain, device="cpu").eval().requires_grad_(False)
        parent = base.restore(w, model, domain)
        assert parent["sha256"] == prep["parent"]

        def predict(x):
            return torch.cat([dev.reconstruction(model, torch.from_numpy(v))
                              for v in np.split(x, len(x)//16)]).numpy()

        with np.load(root/"query.npz") as q:
            normal = q["clean"]
            clean_hat = predict(normal)
            for name, edited in zip(("amplitude", "shift"), q["edited"]):
                edited_hat = predict(edited)
                for i, source in enumerate(("TEP", "SMD")):
                    x, r = edited[i::2], edited_hat[i::2]
                    delta = x-normal[i::2]
                    changed = delta != 0
                    response = (r-clean_hat[i::2])*changed
                    relative = np.linalg.norm((response-delta).reshape(len(delta), -1), axis=1)
                    size = np.linalg.norm(delta.reshape(len(delta), -1), axis=1)
                    valid = size > 0
                    untriggered = {q:float(((np.abs(x-r) <= prep["stats"][q]) & changed).sum()
                                           / changed.sum()) for q in ("0.95", "0.99")}
                    copy_like = valid & (relative <= .25*size)
                    copied = changed & copy_like[:,None,None]
                    rows.append(dict(
                        width=width, source=source, family=name, nonzero_edit_windows=int(valid.sum()),
                        copy_like_windows=int(copy_like.sum()),
                        median_relative_teacher_response_error=float(np.median(relative[valid]/size[valid])),
                        untriggered_actual_edit_coordinates=untriggered,
                        copy_like_coordinates=int(copied.sum()),
                        copy_like_coordinates_below_q95=(float(((np.abs(x-r)<=prep["stats"]["0.95"]) & copied).sum()
                                                               / copied.sum()) if copied.any() else None),
                    ))
        assert model.step == parent["selected"] and base.fingerprint(model.state_dict()) == parent["fingerprint"]
    return dict(rows=rows, seconds=time.monotonic()-started, device="cpu", target_query_opened=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("workspace",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--copy-only",action="store_true")
    args = p.parse_args()
    if args.copy_only:
        base.dump(args.output/"teacher_copy.json",teacher_copy(args.workspace))
        return
    rows, panels, receipts = audit(args.workspace)
    decision = select(rows)
    dest = args.output
    save_csv(dest/"development.csv",rows)
    save_csv(dest/"proxy_panels.csv",panels)
    base.dump(dest/"receipts.json",receipts)
    base.dump(dest/"selection.json",decision)
    print(json.dumps(decision,indent=2))


if __name__=="__main__":
    main()
