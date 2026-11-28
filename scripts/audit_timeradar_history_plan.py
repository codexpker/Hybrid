"""Focused CPU audit of the fixed plan; no target label access or model forward."""
import argparse
import json
from pathlib import Path

import numpy as np

from hybrid_tsad.timeradar_history_experiment import pools
from hybrid_tsad.trajectory_experiment import target, dump, ROLES
from hybrid_tsad.native_v_conditioning import sha


def run(workspace, out):
    data = pools(workspace)
    with np.load(out / "plans.npz") as archive:
        plans = {k: archive[k] for k in archive.files}
    checks = []
    for name, role in (("source", "fit"), ("select", "select"), ("replay", "fit")):
        plan = plans[name]
        flat = plan.reshape(-1, 3)
        for domain, record, start in flat:
            h = 256 if domain == 0 else 512
            assert h <= start and start + 100 <= len(data[role][domain][record])
        for batch in plan.reshape(-1, plan.shape[-2], 3):
            assert np.sum(batch[:, 0] == 0) == np.sum(batch[:, 0] == 1)
        checks.append(dict(name=name, role=role, windows=len(flat),
            unique_TEP_windows=len(np.unique(flat[flat[:, 0] == 0], axis=0)),
            unique_SMD_windows=len(np.unique(flat[flat[:, 0] == 1], axis=0)),
            footprint="356 points in TEP; 612 points in SMD; one record and role"))
    for domain in ("PSM", "SWAT"):
        starts = plans[domain]
        assert starts.shape == (2000, 16)
        assert starts.min() >= 512
        assert starts.max() + 100 <= len(target(workspace, domain, "U"))
        checks.append(dict(name=domain, role="U-FIT", windows=starts.size,
                           unique_current_windows=len(np.unique(starts)),
                           bounds=ROLES[domain]["U"], labels_read=False))
    copy_identity = {}
    for arm in ("native", "history"):
        rows = json.loads((out / "source" / arm / "select.json").read_text())
        error = max(abs(r["copy"] - r["E_full"] - .9 * r["V_full"]) for r in rows)
        assert error < 1e-6
        copy_identity[arm] = error
    zero = [json.loads((out / "source" / arm / "select.json").read_text())[0]
            for arm in ("native", "history")]
    assert zero[0] == zero[1]
    negative_fraction=[]; absolute_output_floor=[]
    for domain,record,start in plans["select"]:
        x=data["select"][domain][record][start:start+100]
        negative_fraction.append(float((x<0).mean()))
        absolute_output_floor.append(float(np.square(np.minimum(x,0).astype(np.float64)).mean()))
    result = dict(plan_sha256=sha(out / "plans.npz"), panels=checks,
                  initial_SELECT_exact_equal=True,
                  maximum_copy_minus_E_minus_point9V=copy_identity,
                  SELECT_mean_negative_input_fraction=float(np.mean(negative_fraction)),
                  SELECT_nonnegative_reconstruction_MSE_lower_bound=float(np.mean(absolute_output_floor)),
                  lower_bound_reason="norm0 retains standardized signed inputs; official abs output remains nonnegative; shared by both methods",
                  source_records_and_roles_checked=True, target_labels_read=False,
                  audit_scope="CPU footprint, exposure, initial SELECT and copy decomposition; no efficacy test")
    dump(out / "plan_audit.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.workspace, args.output)
