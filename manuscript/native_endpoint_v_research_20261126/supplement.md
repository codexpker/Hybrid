---
title: "Supplement to Direct Source Supervision of Native Reconstruction Disagreement"
date: "26 November 2026"
lang: en
geometry: margin=16mm
fontsize: 9pt
header-includes:
  - \usepackage{etoolbox}
  - \AtBeginEnvironment{longtable}{\footnotesize}
  - \setlength{\tabcolsep}{3pt}
---

# S1 Scope and implementation correspondence

This supplement contains material needed to understand and reproduce the endpoint method and its direct comparisons. Broader historical candidates are maintained in the repository experiment index. They do not supply additional components or pooled performance for this method.

| Formal component | Existing implementation | Fixed behavior |
|---|---|---|
| Score and pair objective | src/hybrid_tsad/native_event_rank.py | native_endpoint_and_copy_loss and event_pair_loss |
| Source roles, draws, fitting and selection | src/hybrid_tsad/native_event_rank_training.py | prepare, train_step, validate, training |
| Seed repeats | src/hybrid_tsad/native_event_rank_repeats.py | original initialization and fixed SELECT panel |
| PSM complete causal axis | src/hybrid_tsad/native_rank_psm_extension.py and native_rank_psm_repeats.py | original terminal V on legal windows |
| NASA single sensor | src/hybrid_tsad/native_event_space.py | read_roles and per-entity metrics |
| SMD28 | src/hybrid_tsad/native_event_smd 28.py | independent machine records |
| E and cross-readouts | native_e_vs_v_psm.py, native_e_vs_v_space.py, native_e_vs_v_smd.py | same archived target inputs |
| Standard head | src/hybrid_tsad/native_supervised_mlp.py | frozen encoder and shared nonlinear per-copy/channel head |
| External common axis | src/hybrid_tsad/external_endpoint_metrics.py | immutable per-entity intersections |

Official TimeRadar source SHA256 is 881e 8912e 4c 5b 298b 202e 5b 7ac 2fe 58107c 7dcf 1b 56acf 733e 28c 99f 2ffc 0317. Its initial checkpoint SHA256 is 528aea 6e 803f 9e 6b 1ea 5d 07e 64e 44162309bdb 016cc 7721a 8ac 3753d 942e 6c 7c. The seed 2025 plan hash isdf 7e 74047bb 0fd 49d 25a 2da 0bb 58e 54a 2fbe 3327fd 9f 61b 6e 07ceca 6422543e 9. The official source model is loaded with seq_len 100, hidden_dim 64, d_model 256, depth 10, patch_len 5 and mask_mode c. The existing raw source and checkpoint identities are preserved. This writing task produces no model artifacts.

Masks are generated with isolated seed 84551. One tensor of shape \([5C, 20]\) contains five masks of shape \([C, 20]\). Appending its complement yields a bank reshaped to \([10,C, 20]\), restored for each window with channel count C. Batch membership therefore cannot change the order of a scored window. Source and target scalers are applied before the native forward, whose norm option is 0. Eval mode disables dropout throughout the project decoder fitting. No ordinary no_grad inference call is used for the differentiable pair term.

The registered official model has 1,636,043 parameters. The method updates the 793,610 parameters of the original decoder and adds zero parameters. All other native parameters and buffers are fixed. Scores retain full observed current windows and are causal reconstruction scores. They are not forecasts from strictly past-only inputs.

# S2 Exact source roles and selection

All indices are half-open in the archived SWaT record with 31 channels.

| Role | Interval or event |
|---|---|
| Nominal reconstruction FIT | [0,396000) |
| Nominal reconstruction SELECT | [396000,495000) |
| Labeled event FIT role | [495000,502578) |
| FIT event 1 | [496754,497694) |
| FIT event 2 | [498068,498511) |
| FIT event 3 | [499920,500303) |
| FIT event 4 | [501459,501849) |
| FIT event 5 | [502255,502451) |
| Labeled event SELECT role | [502578,585000) |
| SELECT event 1 | [502705,503134) |
| SELECT event 2 | [506410,507374) |
| SELECT event 3 | [510380,511101) |
| SELECT event 4 | [568800,569521) |

FIT positive stratum sizes are(281,280,280),(115,115,114),(95,95,94),(97,97,97),(33,32,32). Positive windows must be completely inside their event. Paired negatives have 100 labeled-normal points in the authorized role. The four SELECT events contribute three strata each, with 16 fixed paired draws per stratum. Training and validation risk thus have different, explicit aggregation rules.

The complete 240-step supervised trajectory presents 7,680 nominal windows, 3,600 positive windows and 3,600 paired-negative windows. Each of the 15 positive strata is presented 240 times, with repeat sampling. Counts describe exposure, not distinct source events or independent original points. Reconstruction continuation has the nominal stream only. It shares the source-label-assisted selection criterion.

The six original selected trajectories have the following source-only choices. Actual selected exposure follows the saved update, while all trajectories retain the same 240-update cap.

| Seed | Procedure | Selected update | Source risk | Nominal exposures | Positive / negative pair exposures each |
|---|---|---|---|---|---|
| 2025 | Reconstruction | 80 | 0.8438232 | 2560 | 0 |
| 2025 | Direct V | 240 | 0.6027993 | 7680 | 3600 |
| 2026 | Reconstruction | 40 | 0.8558579 | 1280 | 0 |
| 2026 | Direct V | 240 | 0.6332353 | 7680 | 3600 |
| 2027 | Reconstruction | 80 | 0.8329712 | 2560 | 0 |
| 2027 | Direct V | 240 | 0.6070388 | 7680 | 3600 |

The seed 2025 source-training job for the V candidate occupied 286.7547 A40-seconds under the archived process accounting. Scale generation, repeated source runs and target readout have their own cost records. Timing is tied to actual task granularity and cannot be inferred from the throughput of full-window external methods. Complete controls do not share identical source-label exposure or inference cost.

The source roles, all seed plans, selected checkpoint identities and dependency versions are recorded in the original PROVENANCE.json and source_provenance.json exports. Their repository paths are linked through evidence_sources.json in this directory. Original weights and raw score banks remain on the research server. Obtaining SWaT requires the dataset provider's access procedure. NASA and SMD follow their original public distributions and the recorded project admission rules. The repository contains no first-paper laboratory or enterprise data in this manuscript's evidence.

# S3 Per-seed differences and entity distributions

The following tables use complete causal axes, with each difference calculated before domain aggregation. Exact ties use the existing numerical comparison rule.

| Domain | Contrast | Seed | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|---|
| PSM | Direct V − frozen V | 2025 | +0.041959 | +0.035148 | +0.029866 | +0.015297 |
| PSM | Direct V − frozen V | 2026 | +0.032781 | +0.030033 | +0.020130 | +0.007669 |
| PSM | Direct V − frozen V | 2027 | +0.038848 | +0.034337 | +0.024927 | +0.010472 |
| PSM | Direct V − reconstruction | 2025 | +0.038139 | +0.034276 | +0.021982 | +0.013212 |
| PSM | Direct V − reconstruction | 2026 | +0.031462 | +0.030327 | +0.019294 | +0.008150 |
| PSM | Direct V − reconstruction | 2027 | +0.033400 | +0.032771 | +0.013966 | +0.007081 |
| SMD | Direct V − frozen V | 2025 | -0.002172 | +0.014099 | -0.032836 | -0.008732 |
| SMD | Direct V − frozen V | 2026 | -0.004391 | +0.012121 | -0.036607 | -0.010209 |
| SMD | Direct V − frozen V | 2027 | -0.007233 | +0.012350 | -0.043578 | -0.011728 |
| SMD | Direct V − reconstruction | 2025 | -0.008158 | +0.006805 | -0.042569 | -0.014118 |
| SMD | Direct V − reconstruction | 2026 | -0.006293 | +0.010784 | -0.038953 | -0.011195 |
| SMD | Direct V − reconstruction | 2027 | -0.013915 | +0.004374 | -0.053862 | -0.017649 |
| MSL | Direct V − frozen V | 2025 | +0.018525 | +0.036992 | +0.017280 | +0.034168 |
| MSL | Direct V − frozen V | 2026 | +0.015319 | +0.032425 | +0.012505 | +0.028151 |
| MSL | Direct V − frozen V | 2027 | +0.017373 | +0.031879 | +0.013693 | +0.024953 |
| MSL | Direct V − reconstruction | 2025 | +0.012599 | +0.033483 | +0.009094 | +0.030964 |
| MSL | Direct V − reconstruction | 2026 | +0.010966 | +0.030085 | +0.007062 | +0.026957 |
| MSL | Direct V − reconstruction | 2027 | +0.010275 | +0.026300 | +0.005374 | +0.020467 |
| SMAP | Direct V − frozen V | 2025 | +0.017117 | -0.015129 | +0.026822 | -0.012526 |
| SMAP | Direct V − frozen V | 2026 | +0.015225 | -0.011569 | +0.023428 | -0.009854 |
| SMAP | Direct V − frozen V | 2027 | +0.019470 | -0.005960 | +0.030563 | -0.002890 |
| SMAP | Direct V − reconstruction | 2025 | +0.014419 | -0.012131 | +0.031420 | -0.002808 |
| SMAP | Direct V − reconstruction | 2026 | +0.016776 | -0.000268 | +0.027000 | +0.003255 |
| SMAP | Direct V − reconstruction | 2027 | +0.018744 | +0.000554 | +0.038350 | +0.008288 |

| Domain | Seed | Reference | Mean ΔAP | Median ΔAP | Win/tie/loss |
|---|---|---|---|---|---|
| MSL | 2025 | Source reconstruction continuation | +0.01260 | +0.0037326 | 16/0/11 |
| MSL | 2025 | official_V | +0.01852 | +0.0193895 | 18/0/9 |
| MSL | 2026 | Source reconstruction continuation | +0.01097 | +0.0084294 | 17/0/10 |
| MSL | 2026 | official_V | +0.01532 | +0.0147374 | 17/0/10 |
| MSL | 2027 | Source reconstruction continuation | +0.01027 | +0.0109308 | 17/0/10 |
| MSL | 2027 | official_V | +0.01737 | +0.0163436 | 16/0/11 |
| SMAP | 2025 | Source reconstruction continuation | +0.01442 | +0.0002621 | 29/4/20 |
| SMAP | 2025 | official_V | +0.01712 | +0.0000004 | 28/3/22 |
| SMAP | 2026 | Source reconstruction continuation | +0.01678 | +0.0002231 | 28/3/22 |
| SMAP | 2026 | official_V | +0.01522 | +0.0000000 | 24/3/26 |
| SMAP | 2027 | Source reconstruction continuation | +0.01874 | +0.0002429 | 30/3/20 |
| SMAP | 2027 | official_V | +0.01947 | +0.0000000 | 25/3/25 |
| SMD | 2025 | Source reconstruction continuation | -0.00816 | -0.0077321 | 12/0/16 |
| SMD | 2025 | official_V | -0.00217 | -0.0065297 | 12/0/16 |
| SMD | 2026 | Source reconstruction continuation | -0.00629 | -0.0072194 | 12/0/16 |
| SMD | 2026 | official_V | -0.00439 | -0.0082576 | 12/0/16 |
| SMD | 2027 | Source reconstruction continuation | -0.01392 | -0.0077444 | 10/0/18 |
| SMD | 2027 | official_V | -0.00723 | -0.0085786 | 12/0/16 |

# S4 Operational evaluation

Every operating result uses its original legal CAL locations and each model's own threshold. No threshold was chosen from these tables. PSM evaluates [23016,87744), separate from the full causal ranking axis. Its 512 CAL points all have attack labels. NASA q 99/higher equals the maximum of the small eligible CAL sample. SMD CAL records are nominally normal. These are empirical operation points without certified normal error budgets.

Strict and three-confirmation rows remain distinct. Three-confirmation has no backfill. The same absolute indices are used to preserve continuity. One-to-one event matching uses the archived chronological interval matching implementation and requires overlap of predicted and true intervals with each interval used at most once. Any-hit counts true events with any alarm and need not agree with matching. Only complete eligible events enter event coverage. Long steady coverage begins 512 points after onset. No-eligibility entries remain NA.

The following values average the original per-seed operating summaries. For multi-entity tasks, point and event counts are first pooled across entities within a seed, while long-event macro coverage weights eligible long events equally. These are different from the equal-entity ranking means. Displayed fractional matched counts and false-fragment counts arise from averaging training seeds.

| Domain | Method | Stream | Seeds | Normal FPR | Recall | Event F1 | False fragments |
|---|---|---|---|---|---|---|---|
| PSM | Frozen V | strict | 1 | 0.2967 | 0.5087 | 0.1446 | 185.0 |
| PSM | Frozen V | confirmed3 | 1 | 0.2910 | 0.4888 | 0.0902 | 88.0 |
| PSM | Recon. | strict | 3 | 0.2953 | 0.5045 | 0.1474 | 178.0 |
| PSM | Recon. | confirmed3 | 3 | 0.2901 | 0.4842 | 0.1006 | 76.3 |
| PSM | Direct V | strict | 3 | 0.0699 | 0.3981 | 0.0904 | 233.3 |
| PSM | Direct V | confirmed3 | 3 | 0.0623 | 0.3725 | 0.1110 | 97.0 |
| SMD | Frozen V | strict | 1 | 0.1933 | 0.5298 | 0.0938 | 4534.0 |
| SMD | Frozen V | confirmed3 | 1 | 0.1841 | 0.4977 | 0.1770 | 1189.0 |
| SMD | Recon. | strict | 3 | 0.1951 | 0.5346 | 0.0945 | 4404.7 |
| SMD | Recon. | confirmed3 | 3 | 0.1861 | 0.5019 | 0.1735 | 1204.7 |
| SMD | Direct V | strict | 3 | 0.1911 | 0.5420 | 0.0874 | 3307.3 |
| SMD | Direct V | confirmed3 | 3 | 0.1838 | 0.5181 | 0.1449 | 1276.7 |
| MSL | Frozen V | strict | 1 | 0.2886 | 0.2891 | 0.0461 | 881.0 |
| MSL | Frozen V | confirmed3 | 1 | 0.2696 | 0.2408 | 0.1343 | 162.0 |
| MSL | Recon. | strict | 3 | 0.2616 | 0.2972 | 0.0512 | 834.0 |
| MSL | Recon. | confirmed3 | 3 | 0.2436 | 0.2608 | 0.1483 | 161.3 |
| MSL | Direct V | strict | 3 | 0.2943 | 0.3148 | 0.0632 | 657.7 |
| MSL | Direct V | confirmed3 | 3 | 0.2800 | 0.2708 | 0.1841 | 145.7 |
| SMAP | Frozen V | strict | 1 | 0.1206 | 0.4039 | 0.0349 | 2277.0 |
| SMAP | Frozen V | confirmed3 | 1 | 0.1121 | 0.3933 | 0.0773 | 607.0 |
| SMAP | Recon. | strict | 3 | 0.1222 | 0.4027 | 0.0346 | 2239.3 |
| SMAP | Recon. | confirmed3 | 3 | 0.1138 | 0.3933 | 0.0809 | 599.7 |
| SMAP | Direct V | strict | 3 | 0.1320 | 0.4401 | 0.0261 | 3185.7 |
| SMAP | Direct V | confirmed3 | 3 | 0.1191 | 0.4287 | 0.0786 | 821.7 |

Event and coverage values for the same models and streams follow.

| Domain | Method | Stream | Matched / eligible | Any-hit | Long macro / pooled | Longest miss | Hit delay |
|---|---|---|---|---|---|---|---|
| PSM | Frozen V | strict | 18.00 / 48 | 28.00 | 0.8696 / 0.9241 | 486 | 0.0 |
| PSM | Frozen V | confirmed3 | 6.00 / 48 | 17.00 | 0.8645 / 0.9211 | 694 | 0.0 |
| PSM | Recon. | strict | 18.00 / 48 | 28.00 | 0.8522 / 0.9140 | 693 | 0.0 |
| PSM | Recon. | confirmed3 | 6.00 / 48 | 17.00 | 0.8458 / 0.9102 | 717 | 0.0 |
| PSM | Direct V | strict | 17.00 / 48 | 18.33 | 0.5531 / 0.7564 | 855 | 0.3 |
| PSM | Direct V | confirmed3 | 10.33 / 48 | 11.67 | 0.5333 / 0.7440 | 856 | 3.3 |
| SMD | Frozen V | strict | 250.00 / 327 | 278.00 | 0.5822 / 0.6992 | 631 | 0.0 |
| SMD | Frozen V | confirmed3 | 149.00 / 327 | 180.00 | 0.5744 / 0.6952 | 837 | 2.0 |
| SMD | Recon. | strict | 246.00 / 327 | 277.33 | 0.5856 / 0.7008 | 333 | 0.0 |
| SMD | Recon. | confirmed3 | 148.33 / 327 | 182.33 | 0.5772 / 0.6963 | 837 | 2.0 |
| SMD | Direct V | strict | 172.33 / 327 | 205.33 | 0.5792 / 0.7004 | 837 | 0.0 |
| SMD | Direct V | confirmed3 | 123.00 / 327 | 158.00 | 0.5730 / 0.6982 | 838 | 2.0 |
| MSL | Frozen V | strict | 26.00 / 35 | 27.00 | 0.2735 / 0.1815 | 1141 | 6.0 |
| MSL | Frozen V | confirmed3 | 19.00 / 35 | 20.00 | 0.1408 / 0.1075 | 1141 | 6.5 |
| MSL | Recon. | strict | 26.33 / 35 | 27.33 | 0.3146 / 0.1933 | 1141 | 5.8 |
| MSL | Recon. | confirmed3 | 19.00 / 35 | 20.00 | 0.2434 / 0.1442 | 1141 | 6.5 |
| MSL | Direct V | strict | 27.67 / 35 | 29.67 | 0.2786 / 0.1647 | 1129 | 5.3 |
| MSL | Direct V | confirmed3 | 24.33 / 35 | 26.33 | 0.2102 / 0.1201 | 1141 | 11.0 |
| SMAP | Frozen V | strict | 46.00 / 57 | 46.00 | 0.2288 / 0.4595 | 2701 | 29.5 |
| SMAP | Frozen V | confirmed3 | 28.00 / 57 | 28.00 | 0.2046 / 0.4449 | 2812 | 45.0 |
| SMAP | Recon. | strict | 44.33 / 57 | 44.33 | 0.2259 / 0.4576 | 2701 | 32.3 |
| SMAP | Recon. | confirmed3 | 28.67 / 57 | 28.67 | 0.2043 / 0.4448 | 2812 | 45.5 |
| SMAP | Direct V | strict | 45.00 / 57 | 45.00 | 0.2280 / 0.4535 | 2368 | 28.0 |
| SMAP | Direct V | confirmed3 | 36.33 / 57 | 36.33 | 0.2023 / 0.4439 | 2401 | 34.7 |

Long-event populations are five for the PSM suffix, four for MSL and ten for SMAP. The SMD population and all entity-specific eligibility records are retained in the source files. Hit delay is a detected-event-only median within each seed, then averaged across seeds. Missed events are also represented by event recall, right-censoring records and longest missed spans. A short hit-only delay does not establish fast discovery of all events.

# S5 Simple references and temporal permissions

History Mean uses 512 preceding observations. The following panel uses its separate compatible axis, beginning at 512, with the prior endpoint and PCA values restricted to that same axis. These numbers do not enter differences with the external panel beginning at 99. Candidate entries here use seed 2025.

| Domain | Method | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| PSM | Source-supervised endpoint V | 0.57763 | 0.76547 | 0.51410 | 0.67959 |
| SMAP | Source-supervised endpoint V | 0.38709 | 0.62611 | 0.43472 | 0.67795 |
| PSM | History Mean512 | 0.32104 | 0.53004 | 0.34987 | 0.57301 |
| SMAP | History Mean512 | 0.31813 | 0.63386 | 0.33667 | 0.68625 |
| PSM | PCA residual | 0.53463 | 0.79223 | 0.52910 | 0.76589 |

Complete TimeRadar endpoint controls use archived target FIT statistics and no target parameter updates. LEFT uses target normal training/selection and statistics from the complete normal prefix. DADA uses its official model and normal target train-array scaling. TimeRCD's primary B track uses only archived FIT statistics, with no second normalization, and its actual input block determines availability. Its A track normalizes the passed QUERY sequence and is separately archived. CATCH uses target FIT training and W192 context. PCA is a current-point residual fit on permitted target FIT values. An absent compatible cache is NA.

All method tables are local computed scores. Published paper values are not inserted into local-result cells. PSM, SMD and NASA version mappings, all 109 object identities, common-axis endpoints and external weight hashes are recorded in the immutable AXES_AND_INPUTS.json from the archived external comparison. Full precision external domain and entity rows are linked by evidence_sources.json.

# S6 Independent mixed-update and one-event extensions

These completed experiments use distinct suffix tasks and sometimes different archived scalers from the public full-axis benchmark. Their frozen reference values must therefore be read within these tables. They are not replacements for the formal source-supervised benchmark.

Mixed updates start from the original seed 2025 candidate. Replay reconstruction trains on nominal source and mixed target windows. Pair-preserving replay adds the original source pair objective. Source-only continuation replaces target slots with additional source normals. All use the fixed update cap and endpoint V. Source-only trains once and is evaluated on all targets. PSM QUERY has 64,728 points, and the two SMD suffixes each 14,746. Their scales and roles remain isolated. Neither SMD suffix has an eligible long event.

| Separate mixed-task axis | Procedure | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| PSM | Frozen candidate | 0.6134 | 0.7864 | 0.5327 | 0.6721 |
| PSM | Replay reconstruction | 0.5248 | 0.7336 | 0.5277 | 0.7132 |
| PSM | Replay + source pair | 0.5699 | 0.7850 | 0.5094 | 0.6659 |
| PSM | Source-only continuation | 0.5789 | 0.8015 | 0.5210 | 0.6787 |
| machine-3-2 | Frozen candidate | 0.0317 | 0.5151 | 0.0659 | 0.7078 |
| machine-3-2 | Replay reconstruction | 0.0392 | 0.5274 | 0.1298 | 0.7599 |
| machine-3-2 | Replay + source pair | 0.0335 | 0.5205 | 0.0737 | 0.7248 |
| machine-3-2 | Source-only continuation | 0.0320 | 0.5115 | 0.0736 | 0.7120 |
| machine-3-3 | Frozen candidate | 0.0513 | 0.6310 | 0.0825 | 0.7434 |
| machine-3-3 | Replay reconstruction | 0.0826 | 0.6311 | 0.1159 | 0.7543 |
| machine-3-3 | Replay + source pair | 0.0431 | 0.5810 | 0.0697 | 0.6883 |
| machine-3-3 | Source-only continuation | 0.0422 | 0.5749 | 0.0709 | 0.6733 |

The source-pair-preserving mixed update improves PSM AP relative to replay reconstruction but stays below frozen and source-only references on all four ranking measures. On SMD3-2 replay reconstruction is stronger than pair-preserving replay, while SMD3-3 retains negative results. These outcomes do not establish a stable mixed-update gain.

The F experiment uses one predetermined full abnormal support event plus 512 labeled-normal support points per target. PSM's event is[3248,3439), with 92 legal positive endpoints; SMD3-2's is[3080,3917), with 738. Normal blocks are[92,604) and[161,673), respectively. Each supplies 413 legal normal endpoints. Three initializations share the same 240-step terminal update rule and official-normal support V scale, without source replay. They do not update from a U endpoint. SMD3-3 remains ineligible and is not replaced.

| Separate support-task axis | Initialization | Before/after | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|---|
| PSM | official | Z | 0.5510 | 0.7350 | 0.4846 | 0.6482 |
| PSM | copy | Z | 0.5561 | 0.7357 | 0.4948 | 0.6508 |
| PSM | point_weighted | Z | 0.6134 | 0.7864 | 0.5327 | 0.6721 |
| PSM | official | F | 0.5472 | 0.7530 | 0.4748 | 0.6632 |
| PSM | copy | F | 0.5437 | 0.7534 | 0.4733 | 0.6615 |
| PSM | point_weighted | F | 0.5342 | 0.7521 | 0.4718 | 0.6602 |
| machine-3-2 | official | Z | 0.0360 | 0.5038 | 0.0685 | 0.7284 |
| machine-3-2 | copy | Z | 0.0348 | 0.5060 | 0.0694 | 0.7306 |
| machine-3-2 | point_weighted | Z | 0.0317 | 0.5151 | 0.0659 | 0.7078 |
| machine-3-2 | official | F | 0.0272 | 0.7001 | 0.1022 | 0.7620 |
| machine-3-2 | copy | F | 0.0269 | 0.6899 | 0.1013 | 0.7590 |
| machine-3-2 | point_weighted | F | 0.0267 | 0.6918 | 0.1002 | 0.7522 |

Candidate F has lower AP than either matched F control on both eligible tasks. Some ROC, VUS and coverage effects are favorable, accompanied by false alarms or ranking loss. PSM candidate AP decreases from 0.613415 to 0.534156 under its suffix preprocessing. Its long coverage gains coexist with increased false-positive rate. On SMD3-2 all three F models have zero QUERY abnormal recall at the fixed CAL threshold despite ROC/VUS improvements. These are bounded negative extension results, without an implication that all few-shot or mixed-unlabeled adaptation is ineffective.

# S7 Traceability and reproducible document assembly

| Evidence | Immutable repository export |
|---|---|
| Original source roles, fixed scale and initial model identity | [Source provenance](../../docs/experiments/native_event_rank_20261122/PROVENANCE.json) |
| Repeated source plans and checkpoint selection | [Three-seed source provenance](../../docs/experiments/native_event_rank_repeats_20261122/source/source_provenance.json) |
| PSM/NASA per-seed ranking | [Original ranking rows](../../docs/experiments/native_event_rank_repeats_20261122/ranking_by_seed.csv) |
| SMD28 entity metrics and operations | [Entity rows](../../docs/experiments/native_event_smd 28_20261123/native/entity_methods.csv) |
| Complete external comparison and timestamps | [Axis and input manifest](../../docs/experiments/native_endpoint_external_20261125/AXES_AND_INPUTS.json) |
| External entity-level comparisons | [All entity/panel rows](../../docs/experiments/native_endpoint_external_20261125/entity_metrics.csv) |
| Same-label V/E and cross-readouts | [Fixed statistic controls](../../docs/experiments/native_e_vs_v_20261123/methods.csv) |
| Standard supervised head | [Fixed-head comparison](../../docs/experiments/native_supervised_mlp_20261123/RESULTS.md) |
| Original operating data | [Three-seed operations](../../docs/experiments/native_event_rank_repeats_20261122/operating_by_seed.csv) |
| Independent U/F extensions | [U](../../docs/experiments/native_event_stage 2_u_20261123/RESULTS.md), [F](../../docs/experiments/native_event_stage 2_f_20261123/RESULTS.md) |

The build script reads archived metric CSVs, computes descriptive seed means and sample SDs, and renders tables and the method figure. It does not invoke an experiment runner, model forward or historical metric recomputation. Its output preserves axis identity. evidence_sources.json records source file hashes. This supplement links original results rather than reproducing the unrelated historical exploration archive.

The direct method source files, per-seed source selection, fixed V/E/head comparisons, external comparison and the U/F records provide the evidence necessary to interpret the paper. TEP, all-position and full-update studies appear only as bounded comparisons summarized in the main text. Older Hybrid, density, conditional-energy and LEFT-modulation experiments remain outside this paper's scientific object.
