"""Render the controlled mechanism checks and independently verify core ROC-AUC."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from hybrid_tsad.left_history_control import DOMAINS, GROUPS, OUT, digest, summarize

DOC = Path("docs/experiments/left_history_relation_20261126")
PANEL = {
    ("source", "native"): "Frozen source LEFT",
    ("continue", "native"): "Matched continuation LEFT",
    ("joint", "simple"): "Historical correlation reference",
    ("continue", "native_simple"): "Continuation + historical correlation",
    ("joint", "joint"): "Joint LEFT + learned relation",
    ("joint", "relation"): "Learned relation component",
}
TESTS = {
    ("weak", "clean"),
    ("weak", "weak_control"),
    ("lag", "clean"),
    ("lag", "lag_amplitude"),
    ("lag", "synchronous"),
}


def report(w):
    summarize(w, DOC)
    diff = pd.read_csv(DOC / "controlled_pairs.csv")
    records = pd.read_csv(DOC / "controlled_records.csv.gz")
    panel = []
    for r in diff[diff.region.eq("core")].itertuples():
        name = PANEL.get((r.arm, r.readout))
        if name and (r.target, r.reference) in TESTS:
            panel.append(
                dict(
                    group=r.group,
                    domain=r.domain,
                    method=name,
                    comparison=f"{r.target} vs {r.reference}",
                    ROC_AUC=r.ROC_AUC,
                )
            )
    panel = pd.DataFrame(panel)
    costs, diagnostics, identities = [], [], {}
    maximum_error = 0.0
    for domain in DOMAINS:
        dest = w / OUT / domain / "controlled"
        ids = json.loads((dest / "input_receipt.json").read_text())
        assert digest(dest / "input_diagnostics.csv") == ids["input_diagnostics_sha256"]
        assert all(digest(dest / k) == v for k, v in ids["arrays"].items())
        identities[domain] = ids
        diagnostics.append(pd.read_csv(dest / "input_diagnostics.csv"))
        cost = pd.read_csv(dest / "inference.csv")
        for r in cost.itertuples():
            assert digest(dest / f"{r.arm}_{r.group}_scores.npz") == r.sha256
        costs.append(cost)
        for group, n in GROUPS.items():
            s = records.query(
                "domain==@domain and group==@group and region=='core' and arm=='joint' and readout=='joint'"
            )
            assert len(s) == n
            delta = s.lag.to_numpy()[:, None] - s.lag_amplitude.to_numpy()
            check = (np.sum(delta > 0) + 0.5 * np.sum(delta == 0)) / n**2
            val = panel.query(
                "domain==@domain and group==@group and method=='Joint LEFT + learned relation' and comparison=='lag vs lag_amplitude'"
            ).ROC_AUC.item()
            maximum_error = max(maximum_error, abs(check - val))
    assert maximum_error < 1e-12
    costs = pd.concat(costs)
    diag = pd.concat(diagnostics)
    costs.to_csv(DOC / "controlled_cost.csv", index=False)
    diag.to_csv(DOC / "controlled_inputs.csv", index=False)
    (DOC / "controlled_identities.json").write_text(json.dumps(identities, indent=2))
    energy = diag.pivot(
        index=["domain", "group", "record"], columns="edit", values="energy"
    )
    checks = [
        (energy[t] - energy[c]).abs().max()
        for t, c in (("weak", "weak_control"), ("lag", "lag_amplitude"))
    ]
    cols = [
        "energy",
        "correlation_lag4",
        "correlation_lag16",
        "receiver_core_abs_mean_shift",
        "receiver_core_abs_sd_shift",
    ]
    inputs = diag.groupby(["group", "domain", "edit"])[cols].mean()
    paragraphs = [
        "# Controlled weak amplitude and relative delay",
        "The positive class is an edited record. Scores are the mean over the two current 32-point blocks at positions 96–159. For every lag up to 64 the observations used in these blocks fall inside the full-strength edit. A separate transition comparison uses 0–31 and 160–191. Full-window records include all six blocks. Native offline LEFT may use the entire current window.",
        "For each target width, 64 independently seeded unit-variance AR(1) records have AR coefficient 0.85. In the first 2 floor(C/4) columns, each receiver equals 0.8 times its driver at lag 4 plus 0.6 times independent AR noise. Remaining channels retain independent AR trajectories. In a separate transfer check, 16 evenly spaced nonoverlapping 448-point windows come from the released target normal SELECT. They share one underlying time series per domain. The archived target normal-prefix scaler includes SELECT. No target test scores or annotations entered this construction. The pairing and channel choices are identical on real and simulated inputs. A known lag of four applies only to the simulation.",
        "The weak shift adds ±0.25 standardized units to each receiver under a smooth 32-point rise and fall. Its same-support bipolar control uses a cosine profile and matches the total edit energy per receiver. The delay edit gradually replaces each receiver by its own value 12 steps earlier. In the steady synthetic core this moves the known relative delay from 4 to 16 and preserves each channel's stationary marginal distribution in expectation. A synchronous sham shifts both partners, with its own energy. A smooth same-support amplitude control matches the receiver-specific total delay-edit energy. All controls retain the original history and both outer endpoint samples. The four-condition comparison tests block-level discrimination in these fixtures. It does not establish physical fault timing or point-level localization.",
        "## Core record discrimination",
        "Cells report ROC-AUC on equal counts of edited and reference records. Pairing enters the separate mean and median differences and sign counts; AP here has balanced artificial prevalence.",
    ]
    for group in GROUPS:
        table = panel[panel.group.eq(group)].pivot(
            index=["domain", "method"], columns="comparison", values="ROC_AUC"
        )
        paragraphs.extend([f"### {group}", table.to_markdown(floatfmt=".4f")])
    paragraphs += [
        "The lag edit produced less core separation from its equal-energy amplitude control for the learned relation component than for simple historical correlation on both synthetic widths. Adding native scores reduced synthetic discrimination further. On the real nominal blocks, joint core ROC-AUC was near 0.5 on both widths. Weak shifts did not consistently separate from the matched bipolar control. These findings limit claims of either lag specificity or a weak-amplitude advantage.",
        "## Manipulation checks",
        inputs.to_markdown(floatfmt=".4f"),
        f"Maximum absolute differences in total stored float32 edit energy were {checks[0]:.3g} (weak) and {checks[1]:.3g} (delay). Each channel pair was also checked against float32 roundoff bounds. Finite-window SD and mean shifts quantify residual marginal differences. Constant pairs were excluded from the input correlation summary.",
        f"An independent pair-count ROC calculation reproduced four joint lag-versus-amplitude core comparisons with maximum absolute discrepancy {maximum_error:.3g}. Batched bank generation for all twelve score banks took {costs.seconds.sum():.3f} A40 seconds, including CPU transfers and extraction; weights were frozen.",
        "[All core, transition and full-window comparisons](controlled_pairs.csv) · [Record-level means](controlled_records.csv.gz) · [Input diagnostics](controlled_inputs.csv) · [Input identities](controlled_identities.json) · [Cost and bank hashes](controlled_cost.csv).",
    ]
    (DOC / "controlled_results.md").write_text(
        "\n\n".join(paragraphs) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    report(p.parse_args().workspace)
