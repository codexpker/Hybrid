# Fixed three-seed repetition of complete-window source supervision

[Complete method and paired tables](TABLES.md) | [per-seed paired effects](paired_seed.csv) | [paired mean and sample SD](paired_seed_summary.csv) | [all entity metrics](entity_metrics.csv) | [entity differences](paired_entity.csv) | [wins, ties, losses and medians](paired_distribution.csv)

## Fixed training and source selection

Six new trajectories were completed: seed2026/2027 x nominal reconstruction, endpoint supervision and all-position supervision. All seed2025 artifacts and the official/LEFT baselines are reused. No additional source events, target gradients, losses, scales or score choices were introduced. The unchanged all-position objective computes each position's softplus before the mean over 100 positions. Native V, B1 fractional order, ten complementary masks, eval mode, isolated RNG and input scalers are unchanged. Only the official decoder updates.

The archived seed-specific FIT plans differ. All five SOURCE SELECT index arrays and original scale a=0.02649214304983616 are identical across seeds. Actual process seeds are set, while native forward RNG is restored to the fixed mask schedule. Thus this tests source-sampling/training randomness from one official pretrained initialization, not independent pretrained models or new randomized masks. Seven full states and seven per-position SELECT banks are saved per trajectory; earliest minimum source risk selects a checkpoint, including step0. Nominal SELECT MSE is separate. Source-event labels select even the reconstruction-only control.

| Seed | Arm | Executed updates | Selected step | SELECT risk | Normal SELECT MSE | Selected nominal exposures | Selected positive / pair-normal exposures (each) |
|---|---|---|---|---|---|---|---|
| 2025 | copy | 240 | 120 | 0.893636504 | 0.613082 | 3840 | 0 |
| 2025 | terminal | 240 | 200 | 0.888699353 | 0.628327 | 6400 | 3000 |
| 2025 | all_positions | 240 | 200 | 0.770134859 | 0.603891 | 6400 | 3000 |
| 2026 | copy | 240 | 200 | 0.904836809 | 0.595774 | 6400 | 0 |
| 2026 | terminal | 240 | 240 | 0.901679405 | 0.623752 | 7680 | 3600 |
| 2026 | all_positions | 240 | 240 | 0.798873166 | 0.600369 | 7680 | 3600 |
| 2027 | copy | 240 | 160 | 0.876633371 | 0.589141 | 5120 | 0 |
| 2027 | terminal | 240 | 240 | 0.870799602 | 0.620054 | 7680 | 3600 |
| 2027 | all_positions | 240 | 200 | 0.758627005 | 0.602987 | 6400 | 3000 |

The selected exposures above are per trajectory and per category, not independent examples or events. Each complete supervised 240-step trajectory has 7,680 nominal, 3,600 positive and 3,600 pair-normal presentations. The complete copy trajectory has 7,680 nominal presentations. The source pool retains five FIT and four SELECT events. Seed2027 selects all-position step200 but endpoint step240; these selected models differ in exposure as well as objective. No terminal checkpoint was forced. [Seed2025 original source receipts](../native_position_aligned_20261124/RESULTS.md) | [New source ledger and exposure](source/source_training.csv) | [all source validation curves](source/source_select_curves.csv) | [42 snapshots and 42 banks: identities](source/source_assets_sha256.csv)

## Actual four-domain results and decision

All109 objects were evaluated on the unchanged W100/stride100 grid. Common points total 87,645 PSM, 702,996 SMD, 69,495 MSL and 417,265 SMAP. Each entity is evaluated first, then equally weighted inside its domain. Native full axes remain separate. Scores are available at window end and are offline; no missing tail is padded. Official V/E, fixed V+E and complete LEFT retain distinct permissions. No new CAL or causal bank was created.

The supervision-range AP increment is repeatable against endpoint supervision on PSM (+0.01465 +/- 0.00134), SMD (+0.02101 +/- 0.00169) and MSL (+0.01250 +/- 0.00137); all three seeds are positive in each task. SMAP has a small mean increment (+0.00205 +/- 0.00241) with seed2025 negative. These are paired mean and sample SD, n=3, ddof=1, not confidence intervals or independent pretraining repetitions.

Net benefit over nominal reconstruction is narrower. PSM improves AP (+0.00756 +/- 0.00052), ROC and both VUS metrics in all three seeds. SMD has a small positive mean AP (+0.00273 +/- 0.00293) but seed2026 is negative (-0.000264), and both VUS metrics are lower than reconstruction at every seed. MSL AP is essentially unchanged on average (-0.00016 +/- 0.00331), with a negative seed2025 and two small positive seeds; its VUS-PR improves at every seed, while ROC decreases. SMAP AP is variable (+0.00063 +/- 0.00405), with two negative seeds; VUS-PR improves at every seed, while ROC is lower on average. Positive VUS-PR does not replace AP, ROC or event evidence.

Relative to endpoint supervision, PSM loses both VUS metrics in all three repeats despite its AP/ROC gains. SMD loses VUS-ROC in every repeat and has no stable positive ROC increment. MSL has ROC/VUS-ROC costs despite its AP/VUS-PR gains. SMAP retains ROC costs and varying VUS-ROC differences. Full per-seed values and all negative entities remain in the tables.

All-position supervision exceeds official V in domain-average AP at every seed, but complete LEFT has higher AP, ROC and both VUS metrics in every domain and seed. On both NASA tasks, official E and the fixed V+E also remain higher in AP. LEFT uses target-normal training and a different native window/context; the aligned timestamps do not isolate architecture or label efficiency. NASA sensor0 results cannot establish multivariate interaction benefits. SMD is a seen-normal-source held-out test. PSM remains an exposed development task.

The supported result is a bounded, source-supervised supervision-range effect with the clearest net matched-control improvement on PSM. It is not a broadly superior complete detector or an established mixed-unlabeled method. Different selected steps, labeled-pair exposure and reconstruction gradients are components of the compared training procedures; their entire effect cannot be attributed solely to the position-average formula. No new U/F, calibration, contamination, causal-coverage or delay claim is inferred.

## Per-seed entity AP distribution

| Domain | Seed | Reference | AP wins / ties / losses | AP median difference |
|---|---|---|---|---|
| SMD | 2025 | terminal | 21 / 0 / 7 | +0.014020 |
| SMD | 2025 | copy | 17 / 0 / 11 | +0.002487 |
| SMD | 2026 | terminal | 20 / 0 / 8 | +0.011628 |
| SMD | 2026 | copy | 14 / 0 / 14 | +0.000118 |
| SMD | 2027 | terminal | 21 / 0 / 7 | +0.010224 |
| SMD | 2027 | copy | 13 / 0 / 15 | -0.000403 |
| MSL | 2025 | terminal | 14 / 0 / 13 | +0.001474 |
| MSL | 2025 | copy | 11 / 0 / 16 | -0.002064 |
| MSL | 2026 | terminal | 14 / 0 / 13 | +0.001769 |
| MSL | 2026 | copy | 11 / 0 / 16 | -0.000645 |
| MSL | 2027 | terminal | 15 / 0 / 12 | +0.008139 |
| MSL | 2027 | copy | 11 / 0 / 16 | -0.000757 |
| SMAP | 2025 | terminal | 31 / 0 / 22 | +0.000158 |
| SMAP | 2025 | copy | 26 / 0 / 27 | -0.000063 |
| SMAP | 2026 | terminal | 29 / 0 / 24 | +0.000218 |
| SMAP | 2026 | copy | 28 / 0 / 25 | +0.000036 |
| SMAP | 2027 | terminal | 24 / 0 / 29 | -0.000105 |
| SMAP | 2027 | copy | 24 / 0 / 29 | -0.000209 |

These entities are not independent training repetitions. PSM has one record; its per-seed differences are in TABLES.md. The CSV includes all four metrics and all references, including official V and LEFT, with no entity removed.

## Compute, preserved assets and independent check

Six new source trajectories consumed 1,561.581 measured A40-seconds (0.433773 GPU-hours), including seven source validations and saves per trajectory. Successful four-lane readout consumed 340.210 measured A40-seconds (0.094503 GPU-hours). The combined recorded total is **0.528275 GPU-hours**. Minimal engineering checks were not separately GPU-metered. Output-before-start failures from a missing runtime module and a shell-argument mismatch are preserved in READOUT_BOOTSTRAP.json; their extra loading cost was not measured separately. The stated <=60 single-GPU seconds is a conservative planning bound, not measured consumption. No failed training trajectory, data/score redesign or query-driven change occurred. CPU metric workers consumed 1,512.142 summed worker wall-seconds; driver elapsed420.696 seconds includes waiting/overlap, and is not GPU time. [Readout cost](READOUT_COST.json) | [source cost](source/SOURCE_COMPLETION.json)

Two endpoint-supervised states match archived same-seed models tensor-for-tensor, so their scores and metrics are reused; the other four selected states receive necessary new readout. All109 new banks (28,137,365 bytes) were downloaded and SHA-verified. Score indices, footprints, available times, identities and result bindings are retained. The42 source states and42 position banks are also locally verified. Large models and numeric banks remain local/server assets, not Git payloads. Source assets: D:/code/bst/seal_detection_project/research_workspace/hybrid_ad/results/native_position_aligned_repeats_20261124_v1. Target banks: D:/code/bst/seal_detection_project/research_workspace/results/native_position_aligned_repeats_20261124_v1. [Frozen identities](RESULTS_FROZEN.json) | [implementation](READOUT_IMPLEMENTATION.json) | [independent review](review.json)

The CPU-only independent check verified42 state SHA values,42 full-position SOURCE risks/banks and earliest-minimum selections, fixed SELECT/scale, actual seed-specific plans, all109-object macro/paired/sample-SD/entity-distribution calculations, and historical/baseline reuse. Raw AP/ROC recomputation covers seed2026 x three arms x four fixed entities (PSM, SMD1-1, MSL C-1, SMAP A-1) x two axes:24 rows. Maximum raw AP difference is8.33e-17 and ROC difference0; aggregate maximum difference is3.33e-16. VUS was independently checked for aggregation, pairing and SD only; its integrals were not rerun. No review GPU or model forward was used.

The existing method definition and draft contain the current formulas, actual repeated tables and the bounded decision above. The seed2025-only original result and all prior causal/U/F/full-window results remain archived. Public data only; first-paper laboratory, weak-label and enterprise assets are excluded. This batch ends without another seed, variant or adaptation task.
