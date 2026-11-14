"""Source metadata/observed-mask preparation, no Torch/model/fitting/query."""
import hashlib
import json
import sys
import time
import resource
import datetime
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"snapshot/src"))
from hybrid_tsad.hybrid_v1.data import load_manifest, fixed_plan, resolve_plan

def write(path, value):
    with path.open("x", encoding="utf-8") as f: json.dump(value, f, indent=2)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
started = time.monotonic()
runs = load_manifest(HERE/"SOURCE_MANIFEST.json")
selection, cal = [], []
for d in ("SMD", "Sunspot"):
    subset = [r for r in runs if r.domain == d]
    selection += fixed_plan(subset, 512, 32, 32, 2718, "SELECT", replace=False)
    cal += fixed_plan(subset, 512, 32, 512, 31415, "CAL", arrived=True, replace=False)
resolve_plan([r for r in runs if r.role == "SELECT"], selection, 512, 32)
resolve_plan([r for r in runs if r.role == "CAL"], cal, 512, 32, arrived=True)
outputs = []
for seed in (2025, 2026, 2027):
    fit = fixed_plan(runs, 512, 32, 10000, seed, "FIT")
    resolve_plan([r for r in runs if r.role == "FIT"], fit, 512, 32)
    plan = dict(history=512, horizon=32, seed=seed, FIT=fit, SELECT=selection, CAL=cal,
        selection_seed=2718, calibration_seed=31415, FIT_sampling="domain uniform; run uniform; observation-eligible grid uniform with replacement",
        SELECT_CAL_sampling="fixed unique observation-eligible positions per domain, identical across training seeds",
        eligibility="all current channels observed, >=2 past observations/channel; no value/loss/label-based selection",
        natural_missing_teacher="observed=False and zero student placeholder, never LOCF teacher",
        source_manifest_sha256=sha(HERE/"SOURCE_MANIFEST.json"))
    out = HERE/("PLANS_seed%d.json"%seed); write(out, plan)
    outputs.append(dict(seed=seed, path=str(out), sha256=sha(out), FIT_attempts=len(fit),
        FIT_domain_counts={d: sum(e["domain"]==d for e in fit) for d in ("SMD", "Sunspot")},
        SELECT_unique_per_domain=32, CAL_unique_per_domain=512, SELECT_total=64, CAL_total=1024))
summary = dict(status="SOURCE_PLANS_PREPARED_NOT_TRAINED", created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    files=outputs, runs=[dict(domain=r.domain,entity=r.entity,run=r.run,role=r.role,start=r.start,
        end=r.start+len(r.values),shape=list(r.values.shape),missing=int((~r.observed).sum()),
        values_float32_sha256=hashlib.sha256(r.values.tobytes()).hexdigest(),
        observed_sha256=hashlib.sha256(r.observed.tobytes()).hexdigest()) for r in runs],
    SELECT_forward_pairs_per_seed=6400, CAL_forward_pairs_per_seed=1024,
    total_3seed_SELECT_forward_pairs=19200,total_3seed_CAL_forward_pairs=3072,
    H_plus_P=544, no_cross_role_or_run_footprint=True,
    wall_seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    model_or_fit_or_query_or_labels=False, torch_imported="torch" in sys.modules)
write(HERE/"SOURCE_PLAN_SUMMARY.json",summary)
print(json.dumps(summary))
