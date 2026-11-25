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
| SMD28 | src/hybrid_tsad/native_event_smd28.py | independent machine records |
| E and cross-readouts | native_e_vs_v_psm.py, native_e_vs_v_space.py, native_e_vs_v_smd.py | same archived target inputs |
| Standard head | src/hybrid_tsad/native_supervised_mlp.py | frozen encoder and shared nonlinear per-copy/channel head |
| External common axis | src/hybrid_tsad/external_endpoint_metrics.py | immutable per-entity intersections |

Official TimeRadar source SHA256 is881e8912e4c5b298b202e5b7ac2fe58107c7dcf1b56acf733e28c99f2ffc0317. Its initial checkpoint SHA256 is528aea6e803f9e6b1ea5d07e64e44162309bdb016cc7721a8ac3753d942e6c7c. The seed2025 plan hash isdf7e74047bb0fd49d25a2da0bb58e54a2fbe3327fd9f61b6e07ceca6422543e9. The official source model is loaded with seq_len100, hidden_dim64, d_model256, depth10, patch_len5 and mask_mode c. The existing raw source and checkpoint identities are preserved. This writing task produces no model artifacts.

Masks are generated with isolated seed84551. One tensor of shape \([5C,20]\) contains five masks of shape \([C,20]\). Appending its complement yields a bank reshaped to \([10,C,20]\), restored for each window with channel count C. Batch membership therefore cannot change the order of a scored window. Source and target scalers are applied before the native forward, whose norm option is0. Eval mode disables dropout throughout the project decoder fitting. No ordinary no_grad inference call is used for the differentiable pair term.

The registered official model has1,636,043 parameters. The method updates the793,610 parameters of the original decoder and adds zero parameters. All other native parameters and buffers are fixed. Scores retain full observed current windows and are causal reconstruction scores. They are not forecasts from strictly past-only inputs.

# S2 Exact source roles and selection

All indices are half-open in the archived SWaT record with31 channels.

| Role | Interval or event |
|---|---|
| Nominal reconstruction FIT | [0,396000) |
| Nominal reconstruction SELECT | [396000,495000) |
| Labeled event FIT role | [495000,502578) |
| FIT event1 | [496754,497694) |
| FIT event2 | [498068,498511) |
| FIT event3 | [499920,500303) |
| FIT event4 | [501459,501849) |
| FIT event5 | [502255,502451) |
| Labeled event SELECT role | [502578,585000) |
| SELECT event1 | [502705,503134) |
| SELECT event2 | [506410,507374) |
| SELECT event3 | [510380,511101) |
| SELECT event4 | [568800,569521) |

FIT positive stratum sizes are(281,280,280),(115,115,114),(95,95,94),(97,97,97),(33,32,32). Positive windows must be completely inside their event. Paired negatives have100 labeled-normal points in the authorized role. The four SELECT events contribute three strata each, with16 fixed paired draws per stratum. Training and validation risk thus have different, explicit aggregation rules.

The complete240-step supervised trajectory presents7,680 nominal windows,3,600 positive windows and3,600 paired-negative windows. Each of the15 positive strata is presented240 times, with repeat sampling. Counts describe exposure, not distinct source events or independent original points. Reconstruction continuation has the nominal stream only. It shares the source-label-assisted selection criterion.

The six original selected trajectories have the following source-only choices. Actual selected exposure follows the saved update, while all trajectories retain the same240-update cap.

{{SELECTION}}

The seed2025 source-training job for the V candidate occupied286.7547 A40-seconds under the archived process accounting. Scale generation, repeated source runs and target readout have their own cost records. Timing is tied to actual task granularity and cannot be inferred from the throughput of full-window external methods. Complete controls do not share identical source-label exposure or inference cost.

The source roles, all seed plans, selected checkpoint identities and dependency versions are recorded in the original PROVENANCE.json and source_provenance.json exports. Their repository paths are linked through evidence_sources.json in this directory. Original weights and raw score banks remain on the research server. Obtaining SWaT requires the dataset provider's access procedure. NASA and SMD follow their original public distributions and the recorded project admission rules. The repository contains no first-paper laboratory or enterprise data in this manuscript's evidence.

# S3 Per-seed differences and entity distributions

The following tables use complete causal axes, with each difference calculated before domain aggregation. Exact ties use the existing numerical comparison rule.

{{SEEDS}}

{{ENTITIES}}

# S4 Operational evaluation

Every operating result uses its original legal CAL locations and each model's own threshold. No threshold was chosen from these tables. PSM evaluates [23016,87744), separate from the full causal ranking axis. Its512 CAL points all have attack labels. NASA q99/higher equals the maximum of the small eligible CAL sample. SMD CAL records are nominally normal. These are empirical operation points without certified normal error budgets.

Strict and three-confirmation rows remain distinct. Three-confirmation has no backfill. The same absolute indices are used to preserve continuity. One-to-one event matching uses the archived chronological interval matching implementation and requires overlap of predicted and true intervals with each interval used at most once. Any-hit counts true events with any alarm and need not agree with matching. Only complete eligible events enter event coverage. Long steady coverage begins512 points after onset. No-eligibility entries remain NA.

The following values average the original per-seed operating summaries. For multi-entity tasks, point and event counts are first pooled across entities within a seed, while long-event macro coverage weights eligible long events equally. These are different from the equal-entity ranking means. Displayed fractional matched counts and false-fragment counts arise from averaging training seeds.

{{OPERATIONS}}

Long-event populations are five for the PSM suffix, four for MSL and ten for SMAP. The SMD population and all entity-specific eligibility records are retained in the source files. Hit delay is a detected-event-only median within each seed, then averaged across seeds. Missed events are also represented by event recall, right-censoring records and longest missed spans. A short hit-only delay does not establish fast discovery of all events.

# S5 Simple references and temporal permissions

History Mean uses512 preceding observations. The following panel uses its separate compatible axis, beginning at512, with the prior endpoint and PCA values restricted to that same axis. These numbers do not enter differences with the external panel beginning at99. Candidate entries here use seed2025.

{{HISTORY}}

Complete TimeRadar endpoint controls use archived target FIT statistics and no target parameter updates. LEFT uses target normal training/selection and statistics from the complete normal prefix. DADA uses its official model and normal target train-array scaling. TimeRCD's primary B track uses only archived FIT statistics, with no second normalization, and its actual input block determines availability. Its A track normalizes the passed QUERY sequence and is separately archived. CATCH uses target FIT training and W192 context. PCA is a current-point residual fit on permitted target FIT values. An absent compatible cache is NA.

All method tables are local computed scores. Published paper values are not inserted into local-result cells. PSM, SMD and NASA version mappings, all109 object identities, common-axis endpoints and external weight hashes are recorded in the immutable AXES_AND_INPUTS.json from the archived external comparison. Full precision external domain and entity rows are linked by evidence_sources.json.

# S6 Independent mixed-update and one-event extensions

These completed experiments use distinct suffix tasks and sometimes different archived scalers from the public full-axis benchmark. Their frozen reference values must therefore be read within these tables. They are not replacements for the formal source-supervised benchmark.

Mixed updates start from the original seed2025 candidate. Replay reconstruction trains on nominal source and mixed target windows. Pair-preserving replay adds the original source pair objective. Source-only continuation replaces target slots with additional source normals. All use the fixed update cap and endpoint V. Source-only trains once and is evaluated on all targets. PSM QUERY has64,728 points, and the two SMD suffixes each14,746. Their scales and roles remain isolated. Neither SMD suffix has an eligible long event.

{{U}}

The source-pair-preserving mixed update improves PSM AP relative to replay reconstruction but stays below frozen and source-only references on all four ranking measures. On SMD3-2 replay reconstruction is stronger than pair-preserving replay, while SMD3-3 retains negative results. These outcomes do not establish a stable mixed-update gain.

The F experiment uses one predetermined full abnormal support event plus512 labeled-normal support points per target. PSM's event is[3248,3439), with92 legal positive endpoints; SMD3-2's is[3080,3917), with738. Normal blocks are[92,604) and[161,673), respectively. Each supplies413 legal normal endpoints. Three initializations share the same240-step terminal update rule and official-normal support V scale, without source replay. They do not update from a U endpoint. SMD3-3 remains ineligible and is not replaced.

{{F}}

Candidate F has lower AP than either matched F control on both eligible tasks. Some ROC, VUS and coverage effects are favorable, accompanied by false alarms or ranking loss. PSM candidate AP decreases from0.613415 to0.534156 under its suffix preprocessing. Its long coverage gains coexist with increased false-positive rate. On SMD3-2 all three F models have zero QUERY abnormal recall at the fixed CAL threshold despite ROC/VUS improvements. These are bounded negative extension results, without an implication that all few-shot or mixed-unlabeled adaptation is ineffective.

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
