# Stage 2 mixed-unlabeled adaptation of the point-weighted native variance detector

## Four methods on the fixed adaptation suffixes

| Task | Method | AP | ROC-AUC | VUS-PR | VUS-ROC |
|---|---|---|---|---|---|
| PSM | Frozen point-weighted checkpoint | 0.61342 | 0.78640 | 0.53270 | 0.67210 |
| PSM | Replay reconstruction | 0.52483 | 0.73360 | 0.52774 | 0.71323 |
| PSM | Pair-preserving mixed adaptation | 0.56986 | 0.78498 | 0.50937 | 0.66592 |
| PSM | Source-only continuation | 0.57886 | 0.80148 | 0.52102 | 0.67872 |
| machine-3-2 | Frozen point-weighted checkpoint | 0.03174 | 0.51506 | 0.06593 | 0.70779 |
| machine-3-2 | Replay reconstruction | 0.03924 | 0.52744 | 0.12982 | 0.75989 |
| machine-3-2 | Pair-preserving mixed adaptation | 0.03349 | 0.52047 | 0.07367 | 0.72483 |
| machine-3-2 | Source-only continuation | 0.03197 | 0.51145 | 0.07360 | 0.71196 |
| machine-3-3 | Frozen point-weighted checkpoint | 0.05126 | 0.63099 | 0.08253 | 0.74337 |
| machine-3-3 | Replay reconstruction | 0.08263 | 0.63108 | 0.11592 | 0.75429 |
| machine-3-3 | Pair-preserving mixed adaptation | 0.04307 | 0.58097 | 0.06975 | 0.68830 |
| machine-3-3 | Source-only continuation | 0.04223 | 0.57494 | 0.07092 | 0.67327 |

## Paired differences

Each difference is the pair-preserving mixed model minus the named reference on identical target indices. Each target has one seed-2025 observation per arm.

| Task | Reference | AP | ROC-AUC | VUS-PR | VUS-ROC |
|---|---|---|---|---|---|
| PSM | Replay reconstruction | +0.04503 | +0.05138 | -0.01837 | -0.04731 |
| PSM | Frozen point-weighted checkpoint | -0.04356 | -0.00142 | -0.02333 | -0.00618 |
| PSM | Source-only continuation | -0.00900 | -0.01650 | -0.01165 | -0.01279 |
| machine-3-2 | Replay reconstruction | -0.00576 | -0.00697 | -0.05616 | -0.03506 |
| machine-3-2 | Frozen point-weighted checkpoint | +0.00175 | +0.00541 | +0.00773 | +0.01703 |
| machine-3-2 | Source-only continuation | +0.00152 | +0.00902 | +0.00007 | +0.01286 |
| machine-3-3 | Replay reconstruction | -0.03956 | -0.05011 | -0.04617 | -0.06599 |
| machine-3-3 | Frozen point-weighted checkpoint | -0.00819 | -0.05002 | -0.01278 | -0.05507 |
| machine-3-3 | Source-only continuation | +0.00084 | +0.00604 | -0.00117 | +0.01502 |

## Fixed CAL operating points

| Task | Method | Stream | Normal FPR | False fragments | Recall | Matched/eligible | Any-hit | Event F1 | Long macro | Long pooled | Longest miss | Hit-only median delay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PSM | Frozen point-weighted checkpoint | strict | 0.06724 | 251 | 0.40060 | 19/48 | 20 | 0.09291 | 0.55067 | 0.74797 | 874.00000 | 0.50000 |
| PSM | Frozen point-weighted checkpoint | confirmed3 | 0.05919 | 93 | 0.37333 | 10/48 | 11 | 0.10638 | 0.53069 | 0.73561 | 874.00000 | 3.00000 |
| PSM | Replay reconstruction | strict | 0.04031 | 246 | 0.22315 | 24/48 | 26 | 0.10169 | 0.33205 | 0.32130 | 732.00000 | 0.00000 |
| PSM | Replay reconstruction | confirmed3 | 0.03293 | 89 | 0.18382 | 10/48 | 11 | 0.09302 | 0.30477 | 0.29171 | 1228.00000 | 5.00000 |
| PSM | Pair-preserving mixed adaptation | strict | 0.00698 | 17 | 0.07274 | 8/48 | 8 | 0.08840 | 0.14901 | 0.13740 | 1228.00000 | 5.00000 |
| PSM | Pair-preserving mixed adaptation | confirmed3 | 0.00636 | 13 | 0.05953 | 5/48 | 5 | 0.09174 | 0.12193 | 0.11821 | 1228.00000 | 45.00000 |
| PSM | Source-only continuation | strict | 0.00700 | 17 | 0.06757 | 8/48 | 8 | 0.10811 | 0.12532 | 0.12146 | 1228.00000 | 4.00000 |
| PSM | Source-only continuation | confirmed3 | 0.00643 | 10 | 0.05638 | 5/48 | 5 | 0.11236 | 0.10636 | 0.10911 | 1245.00000 | 45.00000 |
| machine-3-2 | Frozen point-weighted checkpoint | strict | 0.00034 | 2 | 0.01626 | 1/8 | 1 | 0.18182 | NA | NA | 123.00000 | 0.00000 |
| machine-3-2 | Frozen point-weighted checkpoint | confirmed3 | 0.00014 | 1 | 0.00813 | 1/8 | 1 | 0.20000 | NA | NA | 123.00000 | 2.00000 |
| machine-3-2 | Replay reconstruction | strict | 0.00034 | 2 | 0.02033 | 1/8 | 1 | 0.20000 | NA | NA | 123.00000 | 0.00000 |
| machine-3-2 | Replay reconstruction | confirmed3 | 0.00028 | 1 | 0.01220 | 1/8 | 1 | 0.22222 | NA | NA | 123.00000 | 2.00000 |
| machine-3-2 | Pair-preserving mixed adaptation | strict | 0.00007 | 1 | 0.01626 | 1/8 | 1 | 0.20000 | NA | NA | 123.00000 | 1.00000 |
| machine-3-2 | Pair-preserving mixed adaptation | confirmed3 | 0.00000 | 0 | 0.00813 | 1/8 | 1 | 0.22222 | NA | NA | 123.00000 | 3.00000 |
| machine-3-2 | Source-only continuation | strict | 0.00007 | 1 | 0.01220 | 1/8 | 1 | 0.20000 | NA | NA | 123.00000 | 1.00000 |
| machine-3-2 | Source-only continuation | confirmed3 | 0.00000 | 0 | 0.00407 | 1/8 | 1 | 0.22222 | NA | NA | 123.00000 | 3.00000 |
| machine-3-3 | Frozen point-weighted checkpoint | strict | 0.03657 | 43 | 0.02759 | 1/16 | 1 | 0.03279 | NA | NA | 412.00000 | 412.00000 |
| machine-3-3 | Frozen point-weighted checkpoint | confirmed3 | 0.03163 | 22 | 0.02069 | 1/16 | 1 | 0.05000 | NA | NA | 414.00000 | 414.00000 |
| machine-3-3 | Replay reconstruction | strict | 0.01511 | 97 | 0.08448 | 13/16 | 13 | 0.19118 | NA | NA | 96.00000 | 0.00000 |
| machine-3-3 | Replay reconstruction | confirmed3 | 0.00565 | 23 | 0.02241 | 2/16 | 2 | 0.09302 | NA | NA | 302.00000 | 27.50000 |
| machine-3-3 | Pair-preserving mixed adaptation | strict | 0.03904 | 35 | 0.00000 | 0/16 | 0 | 0.00000 | NA | NA | 481.00000 | NA |
| machine-3-3 | Pair-preserving mixed adaptation | confirmed3 | 0.03487 | 19 | 0.00000 | 0/16 | 0 | 0.00000 | NA | NA | 481.00000 | NA |
| machine-3-3 | Source-only continuation | strict | 0.03763 | 34 | 0.00000 | 0/16 | 0 | 0.00000 | NA | NA | 481.00000 | NA |
| machine-3-3 | Source-only continuation | confirmed3 | 0.03374 | 17 | 0.00000 | 0/16 | 0 | 0.00000 | NA | NA | 481.00000 | NA |

## Interpretation and cost

Only PSM showed higher AP and ROC under the retained pair objective than under replay reconstruction. Its AP recovered 0.04503 of the 0.08859 lost with replay reconstruction relative to noU. Both VUS measures fell compared with replay reconstruction. All four rankings fell compared with both noU and source-only continuation. At its own CAL threshold, three-confirmation normal FPR fell to 0.64%, alongside 5/48 matched events and 12.19% long steady macro coverage; noU gave 10/48 and 53.07%. The 512 attack-labeled PSM CAL points do not certify a normal false-positive budget.

On machine-3-2 the pair-preserving mixed model exceeded noU and source-only continuation on all four rankings, but replay reconstruction had the highest values of all four. All methods matched one of eight events after confirmation. Machine-3-3 replay reconstruction improved all four rankings relative to noU. Retaining the pair term gave lower values on all four; it missed all 16 events at the CAL threshold despite giving normal alarms. The shared source-only endpoint also missed all 16. Both SMD suffixes have zero eligible long events. Record their long steady coverage as NA.

Keep the established source-supervised checkpoint and its published target-independent readout. This fixed U comparison does not establish a consistent net benefit from retaining the source pair objective during target adaptation. The two remaining seeds are not recommended for this recipe on the strength of these results. Comparisons concern natural mixed data; contamination effects have not been identified.

## Permissions, exposure and reproducibility

Each run begins from the same seed-2025 stage-1 selected decoder (SHA-256 ec572d31d5612ff0d7ebdbe58b133ec631a2430700e3bd449907307de798ddf1), resets AdamW and trains only the decoder for 2,000 updates. The original native endpoint variance is used throughout. Each mixed run exposes the learner to 64,000 target and 64,000 nominal source reconstruction windows. Each retained-pair run also uses 30,000 positive and 30,000 paired nominal source-window exposures from the original SWaT source task. The single shared source-only run uses the identical pair draws and 128,000 nominal source reconstruction windows. Exposures include repeats. [Fixed training configuration](source/training_config.json) and [costs and identities](source/training_cost.csv).

SOURCE_FIT uses nominal SWaT and SMD28 intervals; the two SMD machines are seen normal-source devices. Target U-FIT, CAL and QUERY remain isolated. PSM uses 64,728 evaluated query points. The two SMD suffixes each use 14,746. SMD CAL uses nominal training-side data without independent point labels. Each model uses the same 512 CAL locations within a task, its own q99/higher threshold and strict exceedance. The three targets share the source-only checkpoint, with each target's own scaler and CAL. SMD noU required fresh readout because the previous full SMD benchmark used different scalers and CAL positions. PSM noU scores were reused. [Role and index identities](target_permissions.json).

Seven training jobs consumed 5.8821 A40-hours. Frozen readout occupied 0.5323 A40-hours across four cards, including worker startup and shutdown; total 6.4144 A40-hours. [Score bank identities](SCORES_FROZEN.json), [event-level results](events.csv), [all operating contrasts](paired.csv) and [independent CPU check](review.json).
