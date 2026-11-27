# Static proxy ranking for multi-source LEFT

**Decision: local gains only. End this fixed setting.** On the independent mixed QUERY axes, the proxy-ranked model after full replay U did not beat both references in either domain. Mean U AP differences were −0.001540 versus matched continuation and −0.007654 versus archived M0. The larger SWaT VUS gains appeared in frozen Z and were substantially smaller after U. This single-seed result concerns the specified proxy tasks, loss and training budget. [Frozen protocol](PROTOCOL.md) · [Full-precision decision](decision.json)

## Native-score performance

M0 is the archived source model and its ordinary replay endpoint. C and M1 both start from M0, run 2,000 source updates, then use the identical full 2,000-update ordinary replay U program with fresh Adam. C uses the native normal loss. M1 also ranks clean and edited pairs with the native score and coefficient 0.1. Z and U each use one unchanged retrospective W192 readout, available at the window end. PSM uses the test-relative QUERY interval `[22656,87744)` and SWaT the absolute interval `[608535,944856)`. Both retain their prior score masks and statistics permissions. Neither domain was assigned a new threshold. The prior PSM all-attack CAL remains unsuitable for certifying normal false-alarm rates.

| Domain | Arm | Stage | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| PSM | M0 archived | Z | 0.504296 | 0.720009 | 0.515853 | 0.731294 |
| PSM | M0 archived | U | 0.591946 | 0.755556 | 0.592012 | 0.771655 |
| PSM | Normal continuation C | Z | 0.520141 | 0.729079 | 0.530849 | 0.741466 |
| PSM | Normal continuation C | U | 0.581910 | 0.742328 | 0.579708 | 0.756648 |
| PSM | Proxy ranking M1 | Z | 0.513869 | 0.721916 | 0.521246 | 0.730134 |
| PSM | Proxy ranking M1 | U | 0.578548 | 0.755406 | 0.566917 | 0.756230 |
| SWaT | M0 archived | Z | 0.791359 | 0.842321 | 0.630690 | 0.734086 |
| SWaT | M0 archived | U | 0.786150 | 0.841968 | 0.633839 | 0.740272 |
| SWaT | Normal continuation C | Z | 0.789383 | 0.842681 | 0.629396 | 0.736364 |
| SWaT | Normal continuation C | U | 0.783959 | 0.842556 | 0.635060 | 0.742795 |
| SWaT | Proxy ranking M1 | Z | 0.773091 | 0.845680 | 0.659020 | 0.781154 |
| SWaT | Proxy ranking M1 | U | 0.784240 | 0.842677 | 0.635979 | 0.743254 |

[Full-precision scores and AP/ROC checks](methods.csv)

## Matched comparisons

All differences use the original metric units, taking differences before rounding.

| Domain | Stage | Contrast | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| PSM | Z | M1 − C | −0.006273 | −0.007163 | −0.009603 | −0.011332 |
| PSM | Z | M1 − M0 | +0.009573 | +0.001907 | +0.005394 | −0.001160 |
| PSM | U | M1 − C | −0.003363 | +0.013078 | −0.012790 | −0.000418 |
| PSM | U | M1 − M0 | −0.013398 | −0.000150 | −0.025094 | −0.015425 |
| PSM | U − Z | M0 | +0.087650 | +0.035548 | +0.076159 | +0.040361 |
| PSM | U − Z | C | +0.061769 | +0.013249 | +0.048858 | +0.015182 |
| PSM | U − Z | M1 | +0.064679 | +0.033490 | +0.045671 | +0.026096 |
| SWaT | Z | M1 − C | −0.016293 | +0.002999 | +0.029624 | +0.044791 |
| SWaT | Z | M1 − M0 | −0.018268 | +0.003359 | +0.028331 | +0.047068 |
| SWaT | U | M1 − C | +0.000282 | +0.000120 | +0.000919 | +0.000459 |
| SWaT | U | M1 − M0 | −0.001909 | +0.000709 | +0.002140 | +0.002982 |
| SWaT | U − Z | M0 | −0.005209 | −0.000353 | +0.003150 | +0.006186 |
| SWaT | U − Z | C | −0.005425 | −0.000124 | +0.005664 | +0.006431 |
| SWaT | U − Z | M1 | +0.011150 | −0.003003 | −0.023041 | −0.037900 |

The PSM U ROC-AUC gain versus C coexists with lower AP and VUS-PR. Relative to M0, both PSM U VUS metrics declined by more than 0.01. The SWaT Z VUS gains coexisted with lower AP. After replay, SWaT gains against C were less than 0.001 on all four metrics and AP remained below M0. Both M1 U jobs executed all 2,000 updates. Their within-arm U−Z AP gains therefore describe the net movement from their own initial states and do not establish an advantage over either control. [Full-precision contrasts](comparisons.csv)

## Optimization and implementation checks

Each new source arm consumed the same 64,000 normal windows within fold and one Adam step per batch. M1 additionally presented 64,000 clean windows and 64,000 edited windows to the differentiable scoring path. The 64,000 pairs included 32,000 amplitude and 32,000 shift edits per fold. Each new U endpoint consumed 32,000 target and 32,000 normal source windows. New Z and U states ended at curriculum steps 6,000 and 8,000, respectively. The scale medians were 0.5662378073 (PSM) and 0.2236603051 (SWaT) on A40.

From the first 400-update block to the last, M1's mean pair loss fell from 1.28166 to 1.27625 on PSM and from 1.25620 to 1.04813 on SWaT. Normal SOURCE SELECT losses at the final step were 1.11854 (M1) and 1.11858 (C) on PSM, and 0.82444 (M1) and 0.38379 (C) on SWaT. SELECT was diagnostic. All four final weights were used regardless of the curve. The larger proxy learning improvement on SWaT therefore accompanied a normal validation cost.

CPU and A40 checks gave zero score difference against native `infer`, finite nonzero pair gradients, an unchanged pair-path parameter/buffer fingerprint and RNG, and exactly one native curriculum tick. Under SWaT's fixed native score, the cycle score coefficient is zero. The score-only gradients reach both encoders and the multi-scale decoder. The time/frequency reconstruction decoders and prototype tensors received zero from the auxiliary loss in that check. The normal learning and EMA paths were retained. The initial checker had required a nonzero auxiliary gradient in those inactive groups on SWaT. That assertion was corrected to respect the native score. The initial PSM probes wrote their valid receipts before the CLI attempted a duplicate exclusive write. Their receipts were retained and the probes were not repeated. No formal run required a retry. A separate CPU check confirmed 32 bit-exact comparisons for the archived editor's default behavior. [Engineering receipts](preflight.json) · [Training curves, hashes and timings](receipts.json)

## Cost and verification

| Domain | Arm | Source job (s) | Replay U job (s) | Z and U readouts (s) | Pair forward / backward (s) | CPU editing inside source job (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PSM | C | 79.49 | 78.21 | 2.68 | 0 / 0 | 0 |
| PSM | M1 | 154.48 | 73.40 | 2.63 | 28.63 / 45.58 | 6.40 |
| SWaT | C | 82.26 | 79.18 | 5.90 | 0 / 0 | 0 |
| SWaT | M1 | 150.61 | 75.74 | 5.92 | 28.95 / 42.09 | 5.76 |

The successful job bodies total 790.50 A40 seconds (0.2196 A40·hours). Summed lane time including process startup and cleanup is 841 seconds (0.2336 A40·hours), with the longest lane finishing in 245 seconds. The pair forward and backward columns measure in-stream elapsed time inside the source jobs and are included in those totals. Peak allocated memory in the formal training jobs was 781 MiB. A40 preparation/debug log intervals total about 24.83 seconds, including the initial SWaT check failure and the PSM post-receipt metadata error. CPU evaluation took 132.20 summed per-row seconds on four workers.

After all endpoints and scores were sealed, the evaluator checked AP and ROC-AUC for all 12 rows using independent sorted/ranked-score calculations. The largest absolute difference was 3.33e−16. New VUS values use the existing LEFT evaluator, with window size 100 and 250 thresholds. There was no separate VUS implementation. All eight new checkpoint hashes and paired bank hashes, checkpoint lineage, normal-input-plan hashes, and score index/availability arrays were checked after local transfer. M0 VUS comes directly from the archived table.

Raw weights, point-score banks, window-end indices and logs are in `research_workspace/results/left_proxy_rank_20261127_v1` and the corresponding `results/left_proxy_rank_20261127_v1` directory of the data workspace. Both fixed source arms, unchanged replay U, native scoring, and CPU evaluation are callable from the [runner](../../../scripts/run_left_proxy_rank.sh) and [evaluator](../../../scripts/evaluate_left_proxy_rank.py). [Asset and execution identities](receipts.json)
