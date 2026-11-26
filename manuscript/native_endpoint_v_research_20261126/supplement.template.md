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
| Score and pair objective | [native_event_rank.py](../../src/hybrid_tsad/native_event_rank.py) | native_endpoint_and_copy_loss and event_pair_loss |
| Source roles, draws, fitting and selection | [native_event_rank_training.py](../../src/hybrid_tsad/native_event_rank_training.py) | prepare, train_step, validate, training |
| Seed repeats | [native_event_rank_repeats.py](../../src/hybrid_tsad/native_event_rank_repeats.py) | original initialization and fixed SELECT panel |
| PSM complete causal axis | [native_rank_psm_extension.py](../../src/hybrid_tsad/native_rank_psm_extension.py) and native_rank_psm_repeats.py | original terminal V on legal windows |
| NASA single sensor | [native_event_space.py](../../src/hybrid_tsad/native_event_space.py) | read_roles and per-entity metrics |
| SMD28 | [native_event_smd28.py](../../src/hybrid_tsad/native_event_smd28.py) | independent machine records |
| E and cross-readouts | native_e_vs_v_psm.py, native_e_vs_v_space.py, native_e_vs_v_smd.py | same archived target inputs |
| Standard head | [native_supervised_mlp.py](../../src/hybrid_tsad/native_supervised_mlp.py) | frozen encoder and shared nonlinear per-copy/channel head |
| External common axis | [external_endpoint_metrics.py](../../src/hybrid_tsad/external_endpoint_metrics.py) | immutable per-entity intersections |

Official TimeRadar source SHA-256 is `881e8912e4c5b298b202e5b7ac2fe58107c7dcf1b56acf733e28c99f2ffc0317`. Its initial checkpoint SHA-256 is `528aea6e803f9e6b1ea5d07e64e44162309bdb016cc7721a8ac3753d942e6c7c`. The seed 2025 plan hash is `df7e74047bb0fd49d25a2da0bb58e54a2fbe3327fd9f61b6e07ceca6422543e9`. The official source model is loaded with `seq_len=100, hidden_dim=64, d_model=256, depth=10, patch_len=5, mask_mode='c'`. The existing raw source and checkpoint identities are preserved. This writing task produces no model artifacts.

Masks are generated with isolated seed 84551. One tensor of shape \([5C,20]\) contains five masks of shape \([C,20]\). Appending its complement yields a bank reshaped to \([10,C,20]\), restored for each window with channel count C. Batch membership therefore cannot change the order of a scored window. Source and target scalers are applied before the native forward, whose norm option is 0. Eval mode disables dropout throughout the project decoder fitting. No ordinary no_grad inference call is used for the differentiable pair term.

The registered official model has 1,636,043 parameters. The method updates the 793,610 parameters of the original decoder and adds zero parameters. All other native parameters and buffers are fixed. Scores retain full observed current windows and are causal reconstruction scores. They are not forecasts from strictly past-only inputs.

## Data origins, project versions and metric software

Primary distributions are cited in the main text. Project identity is additionally fixed by archived array hashes. PSM uses the 25-channel arrays with 132,481 normal-prefix points and 87,841 test points recorded in AXES_AND_INPUTS.json. SMD uses all 28 original machines and 38 channels, with entity roles and scaler hashes in its archived binding. NASA uses the version-1 archive whose download URL, member hashes and shapes are retained in MSL_NATIVE_MEMBERS.json, together with the archived SMAP mapping and pre-existing P-2 exclusion. The primary telemetry publication does not by itself specify this project's single-sensor or admission choices. SWaT uses the archived physical-process arrays and fixed 31-channel map.

TEP appears only in the completed source-support extension. Its DOI is `10.7910/DVN/6C3JR1`, Version 1. The two files are `TEP_FaultFree_Training.RData` and `TEP_Faulty_Training.RData`; their identities, column mapping, run split and injection timing are in the [TEP source manifest](../../docs/experiments/native_tep_support_20261125/INPUT.json). They do not enter the formal endpoint method.

The original VUS article defines the measure. The VUS repository is its reference software distribution, while this study called the metric file bundled with the archived LEFT implementation. The wrapper is [left_native_swat_metrics.py](../../src/hybrid_tsad/left_native_swat_metrics.py). Its metric-file SHA-256 is `808a07460a84d31c87edce1d803e1df231111ec53ffa702ee32eb54a79897dfc`, with invocation `RangeAUC_volume_opt(truth, score, windowSize=100, thre=250)`. The maximum buffer and threshold discretization are fixed software parameters. Neither the dataset paper nor current VUS package defaults determine these recorded results. [The completed evaluation receipt](../../docs/experiments/native_endpoint_external_20261125/EVALUATION.json) records the metric settings; the wrapper binds the metric-file hash. This revision aggregates existing exports only.

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

The complete 240-step supervised trajectory presents 7,680 nominal windows,3,600 positive windows and 3,600 paired-negative windows. Each of the 15 positive strata is presented 240 times, with repeat sampling. Counts describe exposure, not distinct source events or independent original points. Reconstruction continuation has the nominal stream only. It shares the source-label-assisted selection criterion.

The six original selected trajectories have the following source-only choices. Actual selected exposure follows the saved update, while all trajectories retain the same 240-update cap.

{{SELECTION}}

The seed 2025 source-training job for the V candidate occupied 286.7547 A40-seconds under the archived process accounting. Scale generation, repeated source runs and target readout have their own cost records. Timing is tied to actual task granularity and cannot be inferred from the throughput of full-window external methods. Complete controls do not share identical source-label exposure or inference cost.

The source roles, all seed plans, selected checkpoint identities and dependency versions are recorded in the original PROVENANCE.json and source_provenance.json exports. Their repository paths are linked through evidence_sources.json in this directory. Original weights and raw score banks remain on the research server. Obtaining SWaT requires the dataset provider's access procedure. NASA and SMD follow their original public distributions and the recorded project admission rules. The repository contains no first-paper laboratory or enterprise data in this manuscript's evidence.

# S3 Per-seed differences and entity distributions

The following tables use complete causal axes, with each difference calculated before domain aggregation. Exact ties use the existing numerical comparison rule.

{{REPEATS}}

{{SEEDS}}

{{ENTITIES}}

# S4 Operational evaluation

Every operating result uses its original legal CAL locations and each model's own threshold. No threshold was chosen from these tables. PSM evaluates [23016,87744), separate from the full causal ranking axis. Its 512 CAL points all have attack labels. NASA q99/higher equals the maximum of the small eligible CAL sample. SMD CAL records are nominally normal. These are empirical operation points without certified normal error budgets.

Strict and three-confirmation rows remain distinct. Three-confirmation has no backfill. The same absolute indices are used to preserve continuity. One-to-one event matching uses the archived chronological interval matching implementation and requires overlap of predicted and true intervals with each interval used at most once. Any-hit counts true events with any alarm and need not agree with matching. Only complete eligible events enter event coverage. Long steady coverage begins 512 points after onset. No-eligibility entries remain NA.

The following values average the original per-seed operating summaries. For multi-entity tasks, point and event counts are first pooled across entities within a seed, while long-event macro coverage weights eligible long events equally. These are different from the equal-entity ranking means. Displayed fractional matched counts and false-fragment counts arise from averaging training seeds.

{{OPERATIONS}}

Long-event populations are five for the PSM suffix, four for MSL and ten for SMAP. The SMD population and all entity-specific eligibility records are retained in the source files. Hit delay is a detected-event-only median within each seed, then averaged across seeds. Missed events are also represented by event recall, right-censoring records and longest missed spans. A short hit-only delay does not establish fast discovery of all events.

# S5 Auxiliary readouts, CATCH seeds and temporal permissions

The two cross-readouts are explanatory outputs from the fixed seed2025 decoders. They do not select a target-specific method.

{{CROSS_READOUT}}

CATCH's individual archived metric rows follow. Main-table means and sample SDs are calculated from these full-precision records, with n=3 and ddof=1.

{{CATCH_SEEDS}}

History Mean uses 512 preceding observations. The following panel uses its separate compatible axis, beginning at 512, with the prior endpoint and PCA values restricted to that same axis. These numbers do not enter differences with the external panel beginning at 99. Candidate entries here use seed 2025.

{{HISTORY}}

Complete TimeRadar endpoint controls use archived target FIT statistics and no target parameter updates. LEFT uses target normal training/selection and statistics from the complete normal prefix. DADA uses its official model and normal target train-array scaling. TimeRCD's primary B track uses only archived FIT statistics, with no second normalization, and its actual input block determines availability. Its A track normalizes the passed QUERY sequence and is separately archived. CATCH uses target FIT training and W192 context. PCA is a current-point residual fit on permitted target FIT values. An absent compatible cache is NA.

All method tables are local computed scores. Published paper values are not inserted into local-result cells. PSM, SMD and NASA version mappings, all 109 object identities, common-axis endpoints and external weight hashes are recorded in the immutable AXES_AND_INPUTS.json from the archived external comparison. Full precision external domain and entity rows are linked by evidence_sources.json.

# S6 Independent mixed-update and one-event extensions

These completed experiments use distinct suffix tasks and sometimes different archived scalers from the public full-axis benchmark. Their frozen reference values must therefore be read within these tables. They are not replacements for the formal source-supervised benchmark.

Mixed updates start from the original seed 2025 candidate. Replay reconstruction trains on nominal source and mixed target windows. Pair-preserving replay adds the original source pair objective. Source-only continuation replaces target slots with additional source normals. All use the fixed update cap and endpoint V. Source-only trains once and is evaluated on all targets. PSM QUERY has 64,728 points, and the two SMD suffixes each 14,746. Their scales and roles remain isolated. Neither SMD suffix has an eligible long event.

{{U}}

The source-pair-preserving mixed update improves PSM AP relative to replay reconstruction but stays below frozen and source-only references on all four ranking measures. On SMD3-2 replay reconstruction is stronger than pair-preserving replay, while SMD3-3 retains negative results. These outcomes do not establish a stable mixed-update gain.

The F experiment uses one predetermined full abnormal support event plus 512 labeled-normal support points per target. PSM's event is[3248,3439), with 92 legal positive endpoints; SMD3-2's is[3080,3917), with 738. Normal blocks are[92,604) and[161,673), respectively. Each supplies 413 legal normal endpoints. Three initializations share the same 240-step terminal update rule and official-normal support V scale, without source replay. They do not update from a U endpoint. SMD3-3 remains ineligible and is not replaced.

{{F}}

Candidate F has lower AP than either matched F control on both eligible tasks. Some ROC, VUS and coverage effects are favorable, accompanied by false alarms or ranking loss. PSM candidate AP decreases from 0.613415 to 0.534156 under its suffix preprocessing. Its long coverage gains coexist with increased false-positive rate. On SMD3-2 all three F models have zero QUERY abnormal recall at the fixed CAL threshold despite ROC/VUS improvements. These are bounded negative extension results, without an implication that all few-shot or mixed-unlabeled adaptation is ineffective.

# S7 Traceability and reproducible document assembly

| Evidence | Immutable repository export |
|---|---|
| Original source roles, fixed scale and initial model identity | [Source provenance](../../docs/experiments/native_event_rank_20261122/PROVENANCE.json) |
| Repeated source plans and checkpoint selection | [Three-seed source provenance](../../docs/experiments/native_event_rank_repeats_20261122/source/source_provenance.json) |
| PSM/NASA per-seed ranking | [Original ranking rows](../../docs/experiments/native_event_rank_repeats_20261122/ranking_by_seed.csv) |
| SMD28 entity metrics and operations | [Entity rows](../../docs/experiments/native_event_smd28_20261123/native/entity_methods.csv) |
| Complete external comparison and timestamps | [Axis and input manifest](../../docs/experiments/native_endpoint_external_20261125/AXES_AND_INPUTS.json) |
| External entity-level comparisons | [All entity/panel rows](../../docs/experiments/native_endpoint_external_20261125/entity_metrics.csv) |
| Same-label V/E and cross-readouts | [Fixed statistic controls](../../docs/experiments/native_e_vs_v_20261123/methods.csv) |
| Standard supervised head | [Fixed-head comparison](../../docs/experiments/native_supervised_mlp_20261123/RESULTS.md) |
| Original operating data | [Three-seed operations](../../docs/experiments/native_event_rank_repeats_20261122/operating_by_seed.csv) |
| Independent U/F extensions | [U](../../docs/experiments/native_event_stage2_u_20261123/RESULTS.md), [F](../../docs/experiments/native_event_stage2_f_20261123/RESULTS.md) |

The build script reads archived metric CSVs, computes descriptive seed means and sample SDs, and renders tables and the method figure. It does not invoke an experiment runner, model forward or historical metric recomputation. Its output preserves axis identity. evidence_sources.json records source file hashes. This supplement links original results rather than reproducing the unrelated historical exploration archive.

The direct method source files, per-seed source selection, fixed V/E/head comparisons, external comparison and the U/F records provide the evidence necessary to interpret the paper. TEP, all-position and full-update studies appear only as bounded comparisons summarized in the main text. Older Hybrid, density, conditional-energy and LEFT-modulation experiments remain outside this paper's scientific object.
