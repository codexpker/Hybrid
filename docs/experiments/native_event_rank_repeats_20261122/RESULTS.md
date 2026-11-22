# Source-training repeats of point-weighted native endpoint variance

Three project-training seeds (2025, 2026, 2027) share one official TimeRadar initialization. Values with ± give a mean and sample SD across seeds (ddof=1). The official detector has one fixed result. All target parameter updates are zero.

## Full causal ranking axes

| Dataset | Method | AP | ROC-AUC | VUS-PR | VUS-ROC |
|---|---|---|---|---|---|
| PSM | Official frozen V | 0.53521 | 0.73065 | 0.48391 | 0.66457 |
| PSM | Reconstruction continuation | 0.53874 ± 0.00208 | 0.73136 ± 0.00094 | 0.49047 ± 0.00519 | 0.66623 ± 0.00197 |
| PSM | Point-weighted source supervision | 0.57307 ± 0.00467 | 0.76382 ± 0.00275 | 0.50889 ± 0.00487 | 0.67571 ± 0.00386 |
| MSL | Official frozen V | 0.33559 | 0.62874 | 0.40570 | 0.70148 |
| MSL | Reconstruction continuation | 0.34139 ± 0.00138 | 0.63255 ± 0.00164 | 0.41301 ± 0.00162 | 0.70444 ± 0.00166 |
| MSL | Point-weighted source supervision | 0.35267 ± 0.00162 | 0.66251 ± 0.00281 | 0.42019 ± 0.00249 | 0.73057 ± 0.00468 |
| SMAP | Official frozen V | 0.36560 | 0.64193 | 0.40267 | 0.68870 |
| SMAP | Reconstruction continuation | 0.36623 ± 0.00213 | 0.63499 ± 0.00417 | 0.39735 ± 0.00220 | 0.67737 ± 0.00170 |
| SMAP | Point-weighted source supervision | 0.38287 ± 0.00213 | 0.63104 ± 0.00462 | 0.42961 ± 0.00357 | 0.68028 ± 0.00497 |

PSM uses 87,742 scored endpoints [99,87841) of the 25-channel release. The 27 MSL entities and all 53 admitted SMAP entities use their individual [99,N) axes, with 71,056 and 422,370 total points. NASA ranking gives each entity equal weight after calculating its four metrics individually. The version, sensor-zero mapping, SMAP admission rule and scaler identities are those of the [fixed public benchmark](../native_event_rank_public_20261122/RESULTS.md).

## Paired ranking differences

Each row subtracts the named reference from point weighting at the same seed. The frozen reference is shared.

| Dataset | Reference | Seed | AP | ROC-AUC | VUS-PR | VUS-ROC |
|---|---|---|---|---|---|---|
| PSM | Official frozen V | 2025 | +0.04196 | +0.03515 | +0.02987 | +0.01530 |
| PSM | Reconstruction continuation | 2025 | +0.03814 | +0.03428 | +0.02198 | +0.01321 |
| PSM | Official frozen V | 2026 | +0.03278 | +0.03003 | +0.02013 | +0.00767 |
| PSM | Reconstruction continuation | 2026 | +0.03146 | +0.03033 | +0.01929 | +0.00815 |
| PSM | Official frozen V | 2027 | +0.03885 | +0.03434 | +0.02493 | +0.01047 |
| PSM | Reconstruction continuation | 2027 | +0.03340 | +0.03277 | +0.01397 | +0.00708 |
| MSL | Official frozen V | 2025 | +0.01852 | +0.03699 | +0.01728 | +0.03417 |
| MSL | Reconstruction continuation | 2025 | +0.01260 | +0.03348 | +0.00909 | +0.03096 |
| MSL | Official frozen V | 2026 | +0.01532 | +0.03243 | +0.01250 | +0.02815 |
| MSL | Reconstruction continuation | 2026 | +0.01097 | +0.03009 | +0.00706 | +0.02696 |
| MSL | Official frozen V | 2027 | +0.01737 | +0.03188 | +0.01369 | +0.02495 |
| MSL | Reconstruction continuation | 2027 | +0.01027 | +0.02630 | +0.00537 | +0.02047 |
| SMAP | Official frozen V | 2025 | +0.01712 | -0.01513 | +0.02682 | -0.01253 |
| SMAP | Reconstruction continuation | 2025 | +0.01442 | -0.01213 | +0.03142 | -0.00281 |
| SMAP | Official frozen V | 2026 | +0.01522 | -0.01157 | +0.02343 | -0.00985 |
| SMAP | Reconstruction continuation | 2026 | +0.01678 | -0.00027 | +0.02700 | +0.00325 |
| SMAP | Official frozen V | 2027 | +0.01947 | -0.00596 | +0.03056 | -0.00289 |
| SMAP | Reconstruction continuation | 2027 | +0.01874 | +0.00055 | +0.03835 | +0.00829 |

| Dataset | Reference | AP Δ | ROC-AUC Δ | VUS-PR Δ | VUS-ROC Δ |
|---|---|---|---|---|---|
| PSM | Official frozen V | 0.03786 ± 0.00467 | 0.03317 ± 0.00275 | 0.02497 ± 0.00487 | 0.01115 ± 0.00386 |
| PSM | Reconstruction continuation | 0.03433 ± 0.00343 | 0.03246 ± 0.00199 | 0.01841 ± 0.00408 | 0.00948 ± 0.00328 |
| MSL | Official frozen V | 0.01707 ± 0.00162 | 0.03377 ± 0.00281 | 0.01449 ± 0.00249 | 0.02909 ± 0.00468 |
| MSL | Reconstruction continuation | 0.01128 ± 0.00119 | 0.02996 ± 0.00359 | 0.00718 ± 0.00186 | 0.02613 ± 0.00530 |
| SMAP | Official frozen V | 0.01727 ± 0.00213 | -0.01089 ± 0.00462 | 0.02694 ± 0.00357 | -0.00842 ± 0.00497 |
| SMAP | Reconstruction continuation | 0.01665 ± 0.00217 | -0.00395 ± 0.00710 | 0.03226 ± 0.00572 | 0.00291 ± 0.00556 |

## NASA AP differences by entity

| Dataset | Reference | Seed | Win / tie / loss | Median ΔAP |
|---|---|---|---|---|
| MSL | Reconstruction continuation | 2025 | 16 / 0 / 11 | +0.003732589 |
| MSL | Reconstruction continuation | 2026 | 17 / 0 / 10 | +0.008429359 |
| MSL | Reconstruction continuation | 2027 | 17 / 0 / 10 | +0.010930757 |
| MSL | Official frozen V | 2025 | 18 / 0 / 9 | +0.019389527 |
| MSL | Official frozen V | 2026 | 17 / 0 / 10 | +0.014737373 |
| MSL | Official frozen V | 2027 | 16 / 0 / 11 | +0.016343577 |
| SMAP | Reconstruction continuation | 2025 | 29 / 4 / 20 | +0.000262067 |
| SMAP | Reconstruction continuation | 2026 | 28 / 3 / 22 | +0.000223101 |
| SMAP | Reconstruction continuation | 2027 | 30 / 3 / 20 | +0.000242882 |
| SMAP | Official frozen V | 2025 | 28 / 3 / 22 | +0.000000409 |
| SMAP | Official frozen V | 2026 | 24 / 3 / 26 | +0.000000000 |
| SMAP | Official frozen V | 2027 | 25 / 3 / 25 | +0.000000000 |

Exact score-metric ties are counted separately. [Entity distributions for all four metrics](entity_paired_distribution.csv).

## Fixed CAL operating points

PSM operating metrics use the original 64,728-point development suffix [23016,87744) with all 512 CAL points labeled anomalous. NASA uses the same per-entity CAL indices as the prior study (14–30 MSL, 4–29 SMAP). For these CAL sizes q99/higher equals the sample maximum. Each model uses its own threshold with strict >. These operating points omit any guarantee about the normal false-positive rate.

Three-consecutive confirmation, with no backfill. Counts and rates below average over the three training seeds for trained models. Event F1 is one-to-one. Delay is the median in points among detected eligible events within each run. Long steady coverage omits the first 512 points of each long event. The MSL and SMAP long-event populations contain four and ten events respectively.

| Dataset | Method | Normal FPR | False fragments | Point recall | Matched / eligible | Any-hit | Event F1 | Long macro / pooled | Max miss | Delay |
|---|---|---|---|---|---|---|---|---|---|---|
| PSM | Official frozen V | 0.2910 | 88.0 | 0.4888 | 6.0 / 48 | 17.0 | 0.0902 | 0.8645 / 0.9211 | 694.0 | 0.0 |
| PSM | Reconstruction continuation | 0.2901 | 76.3 | 0.4842 | 6.0 / 48 | 17.0 | 0.1006 | 0.8458 / 0.9102 | 717.3 | 0.0 |
| PSM | Point-weighted source supervision | 0.0623 | 97.0 | 0.3725 | 10.3 / 48 | 11.7 | 0.1110 | 0.5333 / 0.7440 | 856.0 | 3.3 |
| MSL | Official frozen V | 0.2696 | 162.0 | 0.2408 | 19.0 / 35 | 20.0 | 0.1343 | 0.1408 / 0.1075 | 1141.0 | 6.5 |
| MSL | Reconstruction continuation | 0.2436 | 161.3 | 0.2608 | 19.0 / 35 | 20.0 | 0.1483 | 0.2434 / 0.1442 | 1141.0 | 6.5 |
| MSL | Point-weighted source supervision | 0.2800 | 145.7 | 0.2708 | 24.3 / 35 | 26.3 | 0.1841 | 0.2102 / 0.1201 | 1141.0 | 11.0 |
| SMAP | Official frozen V | 0.1121 | 607.0 | 0.3933 | 28.0 / 57 | 28.0 | 0.0773 | 0.2046 / 0.4449 | 2812.0 | 45.0 |
| SMAP | Reconstruction continuation | 0.1138 | 599.7 | 0.3933 | 28.7 / 57 | 28.7 | 0.0809 | 0.2043 / 0.4448 | 2812.0 | 45.5 |
| SMAP | Point-weighted source supervision | 0.1191 | 821.7 | 0.4287 | 36.3 / 57 | 36.3 | 0.0786 | 0.2023 / 0.4439 | 2401.0 | 34.7 |

In the original seed-2025 SMAP run the confirmed normal false-alarm fragment count rose from 607 to 808 and event F1 from 0.07735 to 0.07895. [Strict and confirmed operating results by seed](operating_by_seed.csv), [paired differences](operating_paired_seeds.csv) and [paired operating means and SD](operating_paired_summary.csv).

## Training and scope

Nominal source windows came from SWaT and SMD28. The point-weighted arm saw 7,200 additional labeled, paired-window exposures per run from the fixed SWaT source design. Both arms used 7,680 shared nominal-window exposures and the same source-event SELECT comparison. New reconstruction checkpoints were selected at updates 40 and 80, and new supervised checkpoints at 240 and 240. [Selection and identity records](source/). PSM development informed the choice of point weighting. The NASA projection to one sensor per entity limits interpretation of multivariate dependence. This source-supervised experiment measures reproducibility of target-label-free scoring under project-training randomness.

## Research decision

Keep point-weighted source supervision as a reproducible ranking candidate. Point weighting beat both references in AP and VUS-PR in all nine domain-by-seed comparisons. PSM and MSL also improved on both ROC measures in every seed. SMAP lost both ROC measures against the official detector in each seed. Its median entity AP difference against the official detector was effectively zero across the three runs.

The operating trade-off depends on the task. On PSM the confirmed normal point rate fell markedly while long-event steady coverage and any-hit fell. MSL gained one-to-one matches at increased normal point FPR and longer median detection delay. SMAP gained matches and incurred substantially more false alarm episodes, leaving average event F1 close to the official detector and below reconstruction continuation. These targets share one set of source-trained weights per method and seed. Evidence concerns ranking with true public source supervision. Evidence for target adaptation, contamination robustness and multivariate lag mechanisms requires separate experiments.

## Compute and verification

Four additional source training runs consumed 0.2817 A40-hours. PSM scoring used 0.5682 A40-hours; NASA per-entity scoring used 1.4486 A40-hours including entity I/O. The sum is approximately 2.30 A40-hours, excluding CPU metrics and process startup outside the recorded intervals. Seed 2025 and official frozen outputs were reused. The [independent verification](review.json) checked source selection, full-axis PSM AP and ROC-AUC, and MSL entity-macro AP and confirmed normal alarm fragments from sealed scores.
