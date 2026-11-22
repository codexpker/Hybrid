# Frozen public causal benchmark, seed 2025

PSM development results selected the point-weighted checkpoint before the NASA evaluation. All three detectors use the same native terminal variance. Frozen V uses official TimeRadar weights. Reconstruction continuation uses reconstruction gradients and labeled SWaT event SELECT. Point weighting additionally uses matched abnormal and normal source windows during training. Both trained decoders use the previously frozen SWaT/SMD28 fold. All target parameter updates are zero.

| Dataset | Method | AP | ROC-AUC | VUS-PR | VUS-ROC |
|---|---|---:|---:|---:|---:|
| PSM | frozen_native_V | 0.535209 | 0.730648 | 0.483912 | 0.664568 |
| PSM | copy | 0.539030 | 0.731520 | 0.491796 | 0.666652 |
| PSM | point_weighted | 0.577169 | 0.765796 | 0.513778 | 0.679864 |
| MSL | frozen_native_V | 0.335593 | 0.628740 | 0.405696 | 0.701477 |
| MSL | copy | 0.341520 | 0.632249 | 0.413882 | 0.704681 |
| MSL | point_weighted | 0.354118 | 0.665732 | 0.422976 | 0.735645 |
| SMAP | frozen_native_V | 0.365602 | 0.641929 | 0.402673 | 0.688702 |
| SMAP | copy | 0.368300 | 0.638931 | 0.398075 | 0.678984 |
| SMAP | point_weighted | 0.382719 | 0.626800 | 0.429495 | 0.676176 |

PSM uses the pinned 25-channel raw test release, [99,87841), 87,742 points. Each NASA entity starts at its own test endpoint 99. MSL uses all 27 native entities (71,056 points) and SMAP uses all 53 entities in the existing admitted release (422,370 points). The earlier SMAP admission excluded P-2 under its conflict rule. NASA uses sensor column zero only. Ranking is calculated per entity and averaged equally within spacecraft. Inputs come from the version-1 NASA archive and the archived SMAP value mapping. [MSL member identities](MSL_NATIVE_MEMBERS.json) and [spacecraft source identities](space/input_sha256.csv).

## Interpretation

Point weighting improved mean AP over both controls on each of the two held-out spacecraft tasks. On MSL all four mean ranking metrics improved. On SMAP the AP gain over frozen V had a paired median of 0.000000409, with 28 wins, three ties, and 22 losses. Both ROC metrics decreased. These are single-checkpoint transfers on single-sensor entities. The evidence warrants a fixed-seed-repeat study of source-supervised ranking. Persistent detection at a matched normal alarm budget, multivariate interaction and robustness to target contamination remain open.

## Operating points and peer comparisons

At each NASA entity's own fixed CAL threshold, three-confirmation one-to-one hits for frozen / copy / point were 19/19/24 of 35 complete MSL events and 28/29/36 of 57 complete SMAP events. Point-level normal FPR under three-confirmation increased over frozen V by 0.90 and 1.01 percentage points respectively. Long steady macro coverage improved on MSL against frozen V but fell below reconstruction continuation. SMAP long steady macro coverage was near 20% in all three arms. Empirical CAL q99/higher equals the largest eligible CAL score on each entity. [Operating results](space/operating.csv), [per-entity pairs](space/ranking_paired.csv), [eligibility and cache identities](space/banks.csv).

On PSM, LEFT and History Mean512 have a joint causal axis [512,87841). Point-weighted AP was 0.577618 versus LEFT 0.569918 and History Mean 0.322624. LEFT had higher VUS-PR and VUS-ROC at 0.579000 and 0.780386. LEFT used target training. [PSM common-axis panels](psm/PSM_RANKING.csv). The 64,728-point development suffix retains its own operating table and 512 attack-only CAL points. [Archived operating results](../native_event_rank_20261122/RESULTS.md).

SMAP external comparisons use a separate [512,N) axis per entity, covering 400,481 points. [External comparison and independent verification](review/). The archived MSL anonymous flattened series has uncertified entity boundaries for this comparison. SMD28 and SWaT served as source datasets. This batch evaluated only PSM, MSL and SMAP.

Target inference used approximately 1.426 A40-hours measured by scoring wall time on the occupied devices, including per-entity I/O. All prior weights were reused. CPU verification used the sealed score banks.

On the 53-entity external SMAP intersection, point-weighted AP/ROC/VUS-PR/VUS-ROC were .387089/.626108/.434722/.677951. History Mean512 gave .318134/.633858/.336666/.686247. Native V gave .373687/.643164/.410779/.691587. [Full six-method comparison](review/SMAP_COMMON_MACRO.csv).
