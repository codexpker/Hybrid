# TimeRadar History Context on all 28 SMD devices

Reference commit `d9ad4cb9c2c9d491f4c6ad1226ebb6c0b80791e9`.
Project sources are TEP FaultFree Training and HAI 21.03 normal records.
All of SMD is excluded from project source fitting and selection. The
original official TimeRadar checkpoint retains its upstream data policy.
SMD is a held-out project source fold with prior public development exposure.

## Data and information permissions

| Domain | Records and roles | Input map | Footprint |
| --- | --- | --- | --- |
| TEP | FIT runs 1–20, SELECT runs 61–65. FIT-pooled scaler including no SELECT rows. | All 52 ordered process channels | 256 historical + 100 current |
| HAI 21.03 | FIT train1 and train3, SELECT train2. FIT records use own scaler, SELECT uses pooled FIT scaler. | 79 ordered process columns. Timestamp and four attack indicators excluded. | 512 historical + 100 current |
| SMD | All 28 bound machines. Per-machine scaler from the first 60% of the released normal training record. Mixed support uses the first third of the released test record. | All 38 original ordered channels | 512 historical + 100 current |

The HAI normalized arrays and column identities reuse the
[official-version asset binding](../left_smd_reliability_20261126/source_inputs.json).
Model parameters do not depend on the number of channels. Historical
and current observations use the same native variable order within a
single record and role. There is no cross-run or cross-role concatenation.
SMD normal scaler data are target statistics only.

For a SMD record with test length q, the established QUERY role begins at
floor(q/3)+544. The first scored endpoint is floor(q/3)+1155.
Every endpoint t is available at t and consumes [t−611,t+1).
The right bound is q. The 611-point delay at the start enforces the
complete pre-current history and current window within the QUERY role.
Z and U share the full eligible suffix on each device. CAL remains in a
separate later segment of the normal training record. No thresholds are
needed for the four ranking metrics.

## Matched training and evaluation

Seed 2025 and per-window mask seed 84551. Within each source domain, draw
the record uniformly and then the legal window start uniformly within
that record. Each target machine has its own deterministic sampling
stream, shared by A/B/C. Both source arms use the same
official parent, logical batch 32, 16 windows per source domain, 240 updates,
24-step warmup, and normal source SELECT at 0/40/80/120/160/200/240.
Minimum native copy MSE selects the earliest minimum. The native arm trains
its decoder. The history arm trains its history and interaction paths along
with that decoder. All paths use the established eval runtime mode.

The U queue has 84 endpoints: A joint history/decoder U, B active frozen
history with decoder-only U, and C native decoder-only U per machine.
A and B share the selected history checkpoint. C starts from the
matched native checkpoint. Each endpoint uses a fresh AdamW, 2,000 updates,
learning rate 1e-4, 200-step warmup and fixed last endpoint. Every update
receives 16 mixed target windows, 8 TEP windows and 8 HAI windows. The
three arms share each machine's plan. Training uses the original copy
reconstruction loss. B checks optimizer membership, first-step active
history calls, and exact frozen hashes after all updates.

The 56 frozen Z banks are completed and evaluated before U starts.
The first sorted device's A endpoint is the throughput pilot and is part
of the 84-endpoint matrix. Independent tasks run on four A40 lanes.
Every score uses the full B1 forward, ten fixed masks, norm 0, and
channel-mean terminal V with correction=1. Each bank records indices,
timing, model/scaler binding and input footprint hashes. The pinned
VUS implementation and the existing independent AP/ROC recomputation
are used. Degenerate metrics are reported as NA with effective coverage.

All four metrics are computed per device, followed by equal-device
aggregation. Report sample SD, median, and the raw positive/zero/negative
counts for all paired contrasts. Device counts are not training seeds.
If included, historical LEFT remains an offline W192 background reference
and is kept out of the paired native-V contrasts.

Formal F is outside this run. After the endpoint matrix is sealed, the
offline qualifier reports eligible U-FIT events and normal 512-point
blocks with full 612-point footprints. Future F would begin at a source
endpoint. CAL is in a separate record and QUERY is excluded from support.

## Reproduction and storage

Use the project Python/CUDA environment and registered public assets.
The commands below accept local absolute author-workspace paths.

```bash
export PYTHONPATH=/path/to/checkout/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
w=/path/to/author_workspace
out="$w/results/timeradar_history_smd28_v1"
py="$w/envs/timeradar/bin/python"
mod=hybrid_tsad.timeradar_history_smd
"$py" -m "$mod" prepare "$w" "$out"
CUDA_VISIBLE_DEVICES=0 "$py" -m "$mod" source "$w" "$out" --arm native
CUDA_VISIBLE_DEVICES=1 "$py" -m "$mod" source "$w" "$out" --arm history
"$py" /path/to/checkout/scripts/run_timeradar_history_smd28.py "$w" "$out"
```

The run directory holds plans, source selections, checkpoints, score banks,
logs and review receipts. For recovery, run the same scheduler against
the same sealed run. Verified stored endpoints and banks are reused.
The training entry automatically resumes from the last saved update.
Public Git carries code, metrics and hashes, while the arrays and
weights remain in the author workspace.

Upon completion, `scripts/build_timeradar_history_smd28_report.py`
exports full-precision tables, summaries, and audit receipts from that
run. The existing PSM/SWaT source weights play no role in this fold.
