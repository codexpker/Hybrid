"""Label-free training and scoring entry points for the two fixed public tasks."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import trajectory_subspace as ts
from .left_smd28_author import bindings, checked_data

OUT = Path("results/trajectory_subspace_20261126_v1")
TEP = Path("results/native_tep_support_20261125_v1/fault_free_training.npy")
PSM = Path("data/common_psm_raw_20261029_v1")
SWAT = Path("preparations/left_multievent_mixed_u_20261111_v1/data/ALL_VALUES.npy")
SOURCE = "source"
NORMAL = ("Z","source_normal","target_normal")
MIXED = ("Z","source_quadratic","target_robust","source_robust")
ROLES = {
    "PSM": dict(FIT=(0,105984), SELECT=(105984,132481), TEST=(0,87841),
                U=(0,17568), CAL=(18112,21960), QUERY=(22504,87841)),
    "SWAT": dict(FIT=(0,396000), SELECT=(396000,495000), TEST=(495000,944919),
                 U=(495000,585000), CAL=(585544,607479), QUERY=(608023,944919)),
}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def checked(path, digest):
    if sha(path) != digest:
        raise ValueError(f"Input digest mismatch: {path.name}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def blocks(x):
    n = len(x)//ts.W
    if n == 0:
        raise ValueError("No complete window in role")
    return np.ascontiguousarray(x[:n*ts.W], dtype=np.float64).reshape(n,ts.W,-1)


def standardizer(x):
    mean, std = x.mean(0), x.std(0)
    return mean,np.where(std>0,std,1.)


def source(w, out):
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    tep = checked(w/TEP,"2c4e24dc41b14a7a521de71a8bbac87cef3788fcf46c9dd321aab5a14f38c698")
    assert tep.shape == (1,25,500,52)
    cov = {}
    rows = []
    for role, indices in (("FIT",range(20)),("SELECT",range(20,25))):
        records = []
        for i in indices:
            x = blocks(tep[0,i])
            records.append(ts.covariance(x))
            rows.append(dict(domain="TEP",record=i+1 if i<20 else i+41,role=role,blocks=len(x)))
        cov["TEP",role] = np.mean(records,0)
    smd = {"FIT":[],"SELECT":[]}
    for name, record in sorted(bindings(w).items()):
        raw = checked_data(w,record,"train")
        cut = len(raw)*4//5
        mean,std = standardizer(raw[:cut])
        for role,part in (("FIT",raw[:cut]),("SELECT",raw[cut:])):
            x = blocks((part-mean)/std)
            smd[role].append(ts.covariance(x))
            rows.append(dict(domain="SMD",record=name,role=role,blocks=len(x),cut=cut))
    for role in smd:
        cov["SMD",role] = np.mean(smd[role],0)
    g = (cov["TEP","FIT"]+cov["SMD","FIT"])/2
    p = ts.planes(g)
    np.save(out/"planes.npy",p)
    risks = {"/".join(k):float(np.sum((np.eye(8)-p)*v)/2) for k,v in cov.items()}
    eig = np.linalg.eigvalsh(g)
    dump(out/"fit.json",dict(records=rows,risk=risks,eigengaps=(eig[:,-2]-eig[:,-3]).tolist(),
                              seconds=time.monotonic()-start,sha256=sha(out/"planes.npy")))


def target(w, domain, role):
    if domain=="PSM":
        is_train = role in ("FIT","SELECT")
        name = "prefix.npy" if is_train else "test.npy"
        digest = ("f722bce0b19324d85c7b5691efe73453eb56022e867ccc1ba47bb3406787df38" if is_train else
                  "6592518615d690ab153592b6dd0220c424441a0d4ee7bac87c3f89fe942893f5")
        raw = checked(w/PSM/name,digest)
    else:
        raw = checked(w/SWAT,"d2304906b800ea47365b864886bd83fa79de9202190f75dccb6ef7f22d00cafa")
    lo,hi = ROLES[domain][role]
    return raw[lo:hi]


def jobs(args):
    chunk,p,robust = args
    sol = ts.Solver(p,chunk.shape[-1],robust)
    ys,diag = zip(*(sol.solve(x) for x in chunk))
    return np.stack(ys), np.array(diag)


def decompose(x, p, robust, pool):
    chunks = [x[s:s+24] for s in range(0,len(x),24)]
    result = list(pool.map(jobs, ((chunk,p,robust) for chunk in chunks)))
    return np.concatenate([v[0] for v in result]),np.concatenate([v[1] for v in result])


def train(w, out, domain, arm, pool):
    dest = out/domain/arm
    dest.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    fit = target(w,domain,"FIT")
    m,s = standardizer(fit)
    np.savez(dest/"scaler.npz",mean=m,scale=s)
    source_p = np.load(out/SOURCE/"planes.npy")
    p = source_p
    rows = []
    ref_seconds = 0.
    if arm!="Z":
        role = "FIT" if arm in NORMAL else "U"
        part = fit if role=="FIT" else target(w,domain,role)
        x = blocks((part-m)/s)
        if arm.startswith("target"):
            t = time.monotonic()
            p = ts.planes(ts.covariance(x))
            ref_seconds = time.monotonic()-t
        anchor = p.copy()
        for sweep in range(5):
            t = time.monotonic()
            y,diagnostic = decompose(x,p,arm!="source_quadratic",pool)
            g = ts.covariance(y)
            new = ts.planes(g,anchor)
            objective = diagnostic[:,3].mean()+ts.ETA/2*np.sum((p-anchor)**2)
            decrease = np.sum((new-p)*(g/2+ts.ETA*anchor))
            if decrease < -1e-12:
                raise ArithmeticError("Projector step raised anchored objective")
            rows.append(dict(sweep=sweep+1,blocks=len(y),mean_objective=objective,
                             updated_objective=objective-decrease,projector_change=float(np.linalg.norm(new-p)),
                             capped=int(np.sum(diagnostic[:,1]>=ts.TOL)),seconds=time.monotonic()-t))
            np.save(dest/f"sweep{sweep+1}_diagnostics.npy",diagnostic)
            p = new
    np.save(dest/"planes.npy",p)
    dump(dest/"train.json",dict(domain=domain,arm=arm,reference_covariance_seconds=ref_seconds,
         seconds=time.monotonic()-start,sweeps=rows,model_sha256=sha(dest/"planes.npy"),
         scaler_sha256=sha(dest/"scaler.npz"),role_bounds=ROLES[domain],labels_read=False))


def score(w,out,domain,arm,role,pool):
    dest = out/domain/arm
    receipt = json.loads((dest/"train.json").read_text())
    assert receipt["model_sha256"] == sha(dest/"planes.npy") and receipt["scaler_sha256"] == sha(dest/"scaler.npz")
    t = time.monotonic()
    p = np.load(dest/"planes.npy")
    with np.load(dest/"scaler.npz") as s:
        x = blocks((target(w,domain,role)-s["mean"])/s["scale"])
    y,d = decompose(x,p,True,pool)
    idx = ROLES[domain][role][0]+np.arange(len(x)*ts.W)
    path = dest/(role+".npz")
    if path.exists():
        raise FileExistsError(path)
    np.savez(path,score=np.abs(x-y).mean(2).ravel(),index=idx,available=np.repeat(idx[ts.W-1::ts.W],ts.W),
             start=idx[::ts.W],diagnostics=d)
    if arm=="source_robust" and role=="QUERY":
        # Keep decompositions until the evaluator selects label-defined examples.
        np.save(dest/"QUERY_normal.npy",y)
    dump(dest/(role+".json"),dict(blocks=len(x),tail=len(target(w,domain,role))-len(idx),
             capped=int((d[:,1]>=ts.TOL).sum()),seconds=time.monotonic()-t,sha256=sha(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["check","source","train","score"])
    parser.add_argument("workspace",type=Path)
    parser.add_argument("--domain",choices=ROLES)
    parser.add_argument("--arm",choices=sorted(set(NORMAL+MIXED)))
    parser.add_argument("--role",choices=["TEST","CAL","QUERY"])
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    w=args.workspace
    out=w/OUT
    with threadpool_limits(1):
        if args.action=="source":
            source(w,out/SOURCE)
        elif args.action=="check":
            from .trajectory_checks import check
            out.mkdir(parents=True,exist_ok=True)
            dump(out/"checks.json",check(target(w,"PSM","FIT")))
        else:
            if args.domain is None or args.arm is None:
                parser.error("train/score require domain and arm")
            with ProcessPoolExecutor(args.workers) as pool:
                if args.action=="train":
                    train(w,out,args.domain,args.arm,pool)
                elif args.role is None:
                    parser.error("score requires role")
                else:
                    score(w,out,args.domain,args.arm,args.role,pool)


if __name__=="__main__":
    main()
