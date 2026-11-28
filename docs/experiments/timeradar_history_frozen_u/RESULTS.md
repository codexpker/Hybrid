# Frozen-History U on PSM and SWaT

Both 2,000-step decoder-only endpoints and their 400,851 QUERY scores are complete.
Freezing the active source history path improved AP relative to joint U by
3.45 percentage points on PSM and 0.33 on SWaT. Both are still below their
same-axis source Z. PSM shows a clear recovery relative to A within this
single run. SWaT trades small AP and ROC gains for lower VUS.

The eligible axes are PSM [23115,87744) and SWaT [608634,944856).
Source Z is restricted to each QUERY grid. All rows keep the native endpoint
sample variance, and the two history U rows share the parent, scaler,
sample order, mask stream, copy objective, and fixed endpoint.
See the [protocol](PROTOCOL.md) and [full-precision table](metrics.csv).

| Domain | Method | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | ---: | ---: | ---: | ---: |
| PSM | History Z | 0.494804 | 0.711088 | 0.456147 | 0.633278 |
| PSM | A - joint-update U | 0.445443 | 0.693170 | 0.450203 | 0.671977 |
| PSM | B - Frozen-History U | 0.479989 | 0.706731 | 0.476365 | 0.672431 |
| PSM | Native U | 0.478992 | 0.704771 | 0.470963 | 0.661983 |
| SWaT | History Z | 0.787593 | 0.855493 | 0.563266 | 0.702569 |
| SWaT | A - joint-update U | 0.756319 | 0.849155 | 0.627997 | 0.794651 |
| SWaT | B - Frozen-History U | 0.759666 | 0.849667 | 0.612149 | 0.772340 |
| SWaT | Native U | 0.752374 | 0.846517 | 0.621046 | 0.797047 |

Differences are computed before rounding. [Full-precision contrasts](contrasts.csv)

| Domain | Comparison | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | ---: | ---: | ---: | ---: |
| PSM | A - Z | -0.049361 | -0.017918 | -0.005944 | +0.038700 |
| PSM | B - Z | -0.014815 | -0.004357 | +0.020219 | +0.039154 |
| PSM | B - A | +0.034547 | +0.013561 | +0.026162 | +0.000454 |
| PSM | B - Native U | +0.000997 | +0.001960 | +0.005402 | +0.010448 |
| SWaT | A - Z | -0.031273 | -0.006338 | +0.064731 | +0.092082 |
| SWaT | B - Z | -0.027927 | -0.005826 | +0.048884 | +0.069771 |
| SWaT | B - A | +0.003346 | +0.000511 | -0.015848 | -0.022311 |
| SWaT | B - Native U | +0.007291 | +0.003150 | -0.008897 | -0.024707 |

## Interpretation

On PSM, B improves every measured ranking statistic relative to A and
reduces the AP loss from Z by about 70%. Versus Native U, the gains are
+0.10 AP, +0.20 ROC-AUC, +0.54 VUS-PR and +1.04 VUS-ROC percentage
points. This supports an adverse effect of allowing the historical path
to update under this fixed replay contract. The comparison changes the
joint optimization trajectory and does not identify which history
parameter or interaction produced the difference.

On SWaT, B has +0.73 AP and +0.31 ROC-AUC percentage points over Native U,
alongside −0.89 VUS-PR and −2.47 VUS-ROC points. B's AP remains 1.48 and
2.79 points below its own Z on PSM and SWaT, respectively. B improves
both VUS measures over Z in both domains. There is one project seed and
two historically used development targets. The results support a
candidate update scope, with an explicit ranking trade-off.

Retain Frozen-History U as the U candidate for the next complete-model
validation, with A, matched Native U and same-axis Z as references.
Prioritize the fixed full-SMD scope next using an SMD-excluded source fold
and a matched native branch. Keep the source budget fixed for that
comparison. A source-budget extension should be assessed separately
with matched source training, since minimum SELECT loss at step 240
alone does not establish a detection limitation. Formal F remains a
separate evaluation.

## Parameter checks and normal source SELECT

Both first logical batches ran PastContext, query, cross-attention and
the complex output projection 32 times. The two first-batch copy losses
agree with A within the measured absolute differences of 8.85e-9 and
1.12e-8. Decoder gradients were finite, and the fresh AdamW parameter
groups contained only decoder tensors. Each domain has nine changed
decoder tensors. Decoder L2 displacement is 14.998054 on PSM and
14.030663 on SWaT. History displacement is exactly zero, with all 143
non-decoder state tensor hashes unchanged. This also covers native
persistent state. The first-update bypass was an engineering diagnostic
only. Both B training and its official QUERY bank use the active history
path throughout. [Saved state and score review](sealed_bank_review.json)

| Domain | Method | Copy MSE | E_full | V_full |
| --- | --- | ---: | ---: | ---: |
| PSM | History Z | 0.919520 | 0.855925 | 0.070661 |
| PSM | A - joint-update U | 0.817838 | 0.788131 | 0.033009 |
| PSM | Native U | 0.818677 | 0.787124 | 0.035058 |
| PSM | B - Frozen-History U | 0.815386 | 0.785009 | 0.033752 |
| SWaT | History Z | 0.919520 | 0.855925 | 0.070661 |
| SWaT | A - joint-update U | 0.791289 | 0.758706 | 0.036203 |
| SWaT | Native U | 0.806449 | 0.770888 | 0.039512 |
| SWaT | B - Frozen-History U | 0.798188 | 0.764713 | 0.037194 |

The 32 normal SOURCE SELECT windows are fixed and diagnostic only. Compared
with A, B has slightly lower Copy MSE on PSM and higher Copy MSE on SWaT.
B's V_full is slightly higher in both cases. All trained endpoints retain
the identity Copy MSE = E_full + 0.9 V_full to float precision. The
normal-source reductions do not establish changed target anomaly separation.
[Full-precision normal diagnostics](normal_select.csv)

## Execution, verification and support

- Training took 1549.55 s on PSM and 1614.58 s on SWaT. Peak allocated training memory was 302.17 and 311.06 MiB.
- Four-shard scoring used 889.80 and 4953.65 A40 process-seconds. Training and scoring total 2.502 A40 process-hours, measured from worker times. The elapsed GPU phase was 51.53 min. Six fresh readbacks and CPU metrics are metered separately. [Cost record](cost.json)
- All eight AP/ROC rows pass independent rank recomputation on the same QUERY indices. Six fresh B1 forwards match the saved score exactly. B VUS uses the existing pinned implementation, and A/Z/Native U VUS are the corresponding archived same-axis values. [Metric review](metric_review.json)
- The revised F qualifier retains the same anomaly events and 512 labeled normal points per domain. By selecting the first normal block with full past coverage, it supplies 413 overlapping normal windows on each task. The qualifier uses public U-FIT annotations to locate supports, and all numeric footprints stay within U-FIT. Formal F fits and scores remain zero. [F support and commands](PROTOCOL.md#revised-f-support-eligibility) - [CPU eligibility review](F_support_eligibility_review.json)
- The full B checkpoints, resume states and eight score shards are stored in the persistent author workspace at `results/timeradar_history_frozen_u_v1`, with model and score backups on the local workstation. Git carries code, full-precision metrics and hashes. The A/Z/Native U archive retains its original files. [Exact runtime identity](contract.json)
