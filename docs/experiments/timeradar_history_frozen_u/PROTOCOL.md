# Frozen-History U

This is one update-scope comparison of the same TimeRadar history model. The
sealed reference is Git commit `59ea8402272080aa95392355cee2b587e14dda53`
and its author-local run `results/timeradar_history_development_v1`.

| Row | Initial state and target update |
| --- | --- |
| Z | Source-trained history model, no target update |
| A | Same history checkpoint, updating its history path and the native decoder in U |
| B | Same history checkpoint, active frozen history path and decoder-only U |
| Native U | Independently trained no-history control, native decoder updates in U |

B leaves every non-decoder parameter and buffer unchanged. The history path
includes the entire PastContext module, query projection, cross-attention
and complex output projection. All of these still run for every history
forward, feeding the original decoder. The only changed optimizer membership
is exclusion of these parameters. The first logical batch checks exact
history-hook call counts, frozen hashes, finite decoder gradients, actual
decoder movement and equality of the copy loss to archived A.

Both B runs use seed 2025, the original source parent, the same fixed replay
plan and target normal-prefix scaler as A. Each of 2,000 steps presents
8 TEP, 8 SMD and 16 unlabeled mixed target windows. A fresh AdamW has
learning rate 1e-4, betas (0.9, 0.95), epsilon 1e-8, decay 0.1 outside
bias/LayerNorm groups and 200 warmup steps. The final update is the
endpoint. The native model stays in eval mode, and mask RNG is isolated
with seed 84551 for each B1 forward. Copy reconstruction MSE over the ten
copies is the only objective. Full source footprints are 356 points on TEP
and 612 on SMD. Target footprints are 612 points. All remain within their
record and role. There is no learner access to target labels.

The primary paired comparison is B minus A. Report B minus Z and B minus
Native U, and retain the original A minus Z. The QUERY grids are PSM
[23115, 87744) and SWaT [608634, 944856), each with the same causal
time index and history as A. Only native terminal V, the channel mean of
unbiased sample variance over ten reconstruction copies at position 99,
is saved. The new banks are produced by full B1 forwards in four
disjoint shards per domain. Existing Z/A/Native U values are checked on
the same axis. Normal source SELECT uses the original 32 windows
diagnostically, with Copy MSE, mean reconstruction error E and sample
variance V. It does not select a U checkpoint. There is no new CAL.

## Reproduction

Use the author workspace and environment documented in the
[parent round](../timeradar_history_development/REPRODUCE.md). Keep the
archived run read-only and prepare a separate output directory. The
following commands assume they are run in a clone with the original
workspace assets and appropriate GPU placement.

```bash
export PYTHONPATH=/path/to/checkout/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
w=/path/to/author_workspace
py="$w/envs/timeradar/bin/python"
ref="$w/results/timeradar_history_development_v1"
out="$w/results/timeradar_history_frozen_u_v1"
mkdir "$out"
mkdir -p "$out/source/history"
cp "$ref/plans.npz" "$ref/contract.json" "$out/"
cp "$ref/source/history/selected.pt" "$ref/source/history/receipt.json" "$out/source/history/"
CUDA_VISIBLE_DEVICES=0 "$py" -m hybrid_tsad.timeradar_history_experiment U "$w" "$out" --reference "$ref" --arm frozen_history --domain PSM
# Repeat for SWAT on another GPU. For each domain score all four parts, 0-3.
CUDA_VISIBLE_DEVICES=0 "$py" -m hybrid_tsad.timeradar_history_evaluation score "$w" "$out" --arm frozen_history --domain PSM --stage U --part 0 --parts 4
# After both complete banks, run independent endpoint checks and ranking.
CUDA_VISIBLE_DEVICES=0 "$py" /path/to/checkout/scripts/evaluate_timeradar_frozen_history_u.py "$w" "$out" "$ref"
```

The evaluator checks the source array identities, saved parameters, score
hashes, indices, footprints, three fresh forwards per bank and independent
AP/ROC calculations. VUS uses the same pinned implementation.

## Revised F support eligibility

Use the chronological first complete support event as previously sealed.
Choose the earliest 512-point label-normal block for which *every* W100
current window has 512 preceding numeric U-FIT points. The normal block is
therefore required to start at least 512 points into U-FIT. Eligibility
uses input bounds and support labels only. Numeric history can include
unlabeled abnormal values and need not itself be declared normal.

| Task | Anomaly support (half-open) | Normal support (half-open) | Legal abnormal / normal endpoints | Numeric footprint union (half-open) |
| --- | --- | --- | ---: | --- |
| PSM | [3248,3439) | [512,1024) | 92 / 413 | [0,1024), [2736,3439) |
| SWaT | [496754,497694) | [495512,496024) | 841 / 413 | [495000,496024), [496242,497694) |

The 413 normal windows per domain overlap within one normal block.
The label budget remains one complete event and 512 normal points.
Each footprint union has a further 1,024 numeric history points outside
the support annotation budget. No CAL or QUERY data are needed.

The versioned support directory is selected with `--support-contract
full-normal-block` on F preparation, training, scoring and evaluation.
Preparation was executed on CPU. The old support files retain their original
identities. Formal F fitting and F scoring are deferred. When a final model
is fixed, matched F starts from the appropriate source endpoint and reuses the
240-step copy/pair procedure and shared scale construction on the revised
normal panel. A no-source initialization resets the full native and history
state. The U scope choice must not be inferred to apply to F automatically.
