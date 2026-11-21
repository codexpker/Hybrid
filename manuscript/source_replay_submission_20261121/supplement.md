---
title: "Supplementary Material: Source Replay for Mixed-Unlabeled Adaptation"
lang: en-US
---

This supplement accompanies [the main manuscript](manuscript.md). All numerical tables, plots, and decisions identified below refer to the existing archived runs.

| Main text | Supplement |
|:--|:--|
| 1–2. Introduction and related work | E. Reference verification and metadata |
| 3. Problem formulation and experimental protocol | A. Exact split, score availability, and reproduction details |
| 4. Natural mixed-unlabeled adaptation | B. Full three-seed outcomes and natural adaptation contrasts |
| 5. Controlled abnormal exposure | B. Exposure counts, all pairing contrasts, curves, event-level coverage, and archived costs |
| 6. Comparison with existing methods | C. Expanded external comparisons, source permissions, and unavailable comparisons |
| 7–8. Discussion, limitations, and conclusion | D. Selected historical links. The full prior manuscript remains a separate archived record. |

## A. Protocol and reproduction details

All intervals use half-open indexing. The SWaT timeline concatenates the 495,000-point source prefix and the published 449,919-point test segment. In PSM, target positions are relative to the released 87,841-point test record. The PSM source prefix contains 132,481 points. Source scaler statistics for the primary LEFT detector follow its native protocol over the entire released source prefix, including its validation interval. The source-only and replay draws are restricted to SOURCE_FIT. The model's adaptation input is limited to legal complete windows from the prescribed fit intervals.

| Task | Grid anchor | Legal CAL score axis | Raw query role | Legal query score support | Final scored crop |
|:--|--:|:--|:--|:--|:--|
| SWaT | 495000 | [585624,607320) | [608023,944919) | [608472,944856) | [608535,944856) |
| PSM | 0 | [18240,21888) | [22504,87841) | [22656,87744) | [22656,87744) |

The nonoverlapping readout windows have length and step 192. Each emitted score's availability index is the window's final observed index. SWaT additionally discards the first 63 legally scored points to retain the historical common-axis start at 608535, which is inside its first legal window. The last 63 SWaT and last 97 PSM test points have no full-window score. The first 152 positions in the PSM raw query role lie outside complete legal query windows. The final SWaT crop contains 288,544 normal and 47,777 anomalous points, with 22 complete events, nine of them long. The PSM crop contains 47,309 normal and 17,779 anomalous points, with 48 complete events, five of them long. Two boundary-truncated PSM events contribute 4,196 of 8,861 and 3,029 of 3,120 points to pointwise metrics and are excluded from the complete-event analysis. Calibration selects 512 points via NumPy `linspace(first, last, 512, dtype=int64)` over legal CAL score support. Post hoc attack counts are 38 and 512 at the selected SWaT and PSM locations. The entire PSM CAL role, spanning 3,848 points, is also attack-labeled.

Each task has one fixed LEFT initialization selected using the native source validation procedure. The endpoint adaptation budget is 2,000 steps with a new Adam optimizer at learning rate 0.0001 and batch size 256. Sampling is uniform with replacement over legal window starts in the respective fit interval. Matching draw positions cross arms but never cross tasks. The original checkpoints include the native prototypes and running state. In SWaT, noU is cropped from the native archived scores. In PSM, the source endpoint is restored with its actual runtime state for the frozen readout. The 2025 ordinary U128 and earlier restricted-window comparisons remain separate. The implementation and record give the exact source window roles and seeds. [LEFT native SWaT](../../docs/experiments/left_native_swat_20261118/RESULTS.md), [SWaT natural U and readout](../../docs/experiments/left_swat_natural_u_20261118/RESULTS.md), [SWaT matched replay](../../docs/experiments/left_swat_source_replay_20261118/RESULTS.md), [PSM matched replay](../../docs/experiments/left_psm_replay_20261118/RESULTS.md).

LEFT is pinned to upstream commit [3fadb4811797075e233076b9faceb0d8ec59b697](https://github.com/DezhengWang/Left/tree/3fadb4811797075e233076b9faceb0d8ec59b697). The [SWaT task configuration](../../configs/left-exposure-swat-20261118.json) and [PSM task configuration](../../configs/left-author-psm-source-replay-20261118.json) fix asset fingerprints, tensor shapes, roles, and initialization. The [replay entry point](../../src/hybrid_tsad/left_domain_replay.py) uses `default_rng(seed)` for $T$ and `SeedSequence(seed).spawn(3)` for $S$, $T'$, and $S'$ in that order. In the [replacement generator](../../src/hybrid_tsad/left_exposure_bundle.py), `SeedSequence([seed, 20261118, 10]).spawn(3)` supplies separate streams for slots, abnormal donors, and nominally normal donors. Slots are drawn without replacement and donors with replacement. Original numeric arrays, weights, optimizers, and complete score banks reside in the separate experiment archive. Public files contain code, fixed configurations, asset fingerprints, and derived tables. Recreating the numeric input tensors and restoring the archived artifacts require those assets.

The range metrics use the archived LEFT `metricor.RangeAUC_volume_opt` path with `windowSize=100` and `thre=250`. For AP and ROC-AUC the full scored axis retains the original labels without point adjustment. Strict and three-point confirmed streams are evaluated independently. Any-hit measures episode overlap, whereas the chronological greedy one-to-one rule limits each alarm fragment and each event to one match. Normal-region fragments count contiguous alarm portions within normal labels and can include fragments clipped at anomaly boundaries. Long steady macro coverage averages the coverage within each eligible event after its first 512 samples. Pooled steady coverage weights events by their steady durations. All reported offsets are in sample positions. Full-window availability and hit-only medians limit their use as latency comparisons. [Metric definitions and full archived rows](../../docs/experiments/left_replay_exposure_repeat_20261118/comparison.csv).

## B. Full natural adaptation and controlled exposure results

The archived natural-condition rows are noU, TT-A, SS, and R-A within each task. The B and C rows retain both updating policies and all designated seeds. Sample standard deviation is calculated across the three paired differences. Frozen noU remains a single result. The strict and C3 streams preserve rank metrics, point precision/recall/F1, false-positive counts and rates, false-positive fragments, event precision/recall/F1, one-to-one matches, any-hits, complete-event coverage, steady macro and pooled coverage, longest missed spans, and availability offsets.

| Supplementary record | Contents |
|:--|:--|
| [Method means and original study report](../../docs/experiments/left_replay_exposure_repeat_20261118/RESULTS.md) | Absolute natural, B and C results. Paired natural and exposure responses with sample variation. |
| [Every fixed seed](../../docs/experiments/left_replay_exposure_repeat_20261118/PER_SEED_TABLE.md) and [machine-readable streams](../../docs/experiments/left_replay_exposure_repeat_20261118/comparison.csv) | Absolute outcomes, strict/C3 distinctions, event definitions, and unrounded quantities. |
| Natural paired summaries for [SWaT](../../docs/experiments/left_swat_replay_repeat_20261118/paired_summary.csv) and [PSM](../../docs/experiments/left_psm_replay_20261118/paired_summary.csv) | R−TT, R−SS, and R−noU. |
| [Exposure contrasts](../../docs/experiments/left_replay_exposure_repeat_20261118/contrasts.csv), [paired summaries](../../docs/experiments/left_replay_exposure_repeat_20261118/paired_summary.csv), and [interactions](../../docs/experiments/left_replay_exposure_repeat_20261118/interaction.csv) | B−C, B−A, C−A, R-B−TT-B, and the difference of B−C differences. |
| [Exposure counts](../../docs/experiments/left_replay_exposure_repeat_20261118/exposure_by_seed.csv) | Selected slots, actual label exposure, window reuse, and task/seed identifiers. Label-derived fields are evaluation and construction metadata. |
| [PR and low-FPR ROC](../../docs/experiments/left_replay_exposure_repeat_20261118/curves/PR_lowFPR_ROC.png), [original operating points](../../docs/experiments/left_replay_exposure_repeat_20261118/curves/operating_points.csv), and [long-event distributions](../../docs/experiments/left_replay_exposure_repeat_20261118/all_seed_nine_long_events.csv) | Frozen seed-2025 figures with CAL points, and all three seeds' per-event coverage for the nine SWaT long events. |
| [Exposure compute receipts](../../docs/experiments/left_replay_exposure_repeat_20261118/resource_summary.json) and [archived result verification](../../docs/experiments/left_replay_exposure_repeat_20261118/verification_summary.json) | Historical cost and verification for the exposure repetitions. |

### Historical computing budget

The per-arm adaptation budget is fixed across TT, SS, and R. B and C retain the same update budget. The following A40 GPU-hour values are the archived occupancies for each phase. They include whatever scoring or setup is specified in that phase's receipt. They are not converted into comparable efficiency estimates and no study-wide total is reported for reused assets with incomplete cost coverage.

| Phase | Archived scope | A40 GPU-hours |
|:--|:--|--:|
| [SWaT source initialization](../../docs/experiments/left_native_swat_20261118/RESULTS.md) | Original native training and readout | 1.081298 |
| [SWaT natural, initial](../../docs/experiments/left_swat_source_replay_20261118/RESULTS.md) | Three matched seed-2025 jobs | 0.228534 |
| [SWaT natural, repeats](../../docs/experiments/left_swat_replay_repeat_20261118/RESULTS.md) | Six additional matched jobs | 0.455570 |
| [PSM natural](../../docs/experiments/left_psm_replay_20261118/RESULTS.md) | Nine matched jobs plus one frozen noU readout. Source initialization reused. | 0.628679 |
| [Exposure, initial](../../docs/experiments/left_replay_exposure_20261118/RESULTS.md) | Eight seed-2025 jobs in the two tasks, including setup, readout, and cleanup | 0.584922 |
| [Exposure, repeats](../../docs/experiments/left_replay_exposure_repeat_20261118/resource_summary.json) | Sixteen successful jobs. One guard wall-time entry without CUDA adds only to the conservative ledger. | 1.190616 |

## C. Expanded external comparisons

The local MLP and CANDI comparisons share point indices with the main tasks but use 10-point endpoint scores and their own fitted source scalers and calibration scores. The MLP noU, TT-A, and R-A results separate source initialization and learning policy within that backbone. Full CANDI uses its own lightweight adaptation and is frozen after its authorized U-FIT stream. Its normal reference and internal screening thresholds derive from source validation. Its external alarm threshold derives from the project CAL points. The [cross-backbone comparison](../../docs/experiments/mlp_replay_20261119/RESULTS.md), [full table](../../docs/experiments/mlp_replay_20261119/comparison.csv), and [two-task curves](../../docs/experiments/mlp_replay_20261119/curves/SWAT_PR_ROC.png) preserve seed 2025 and the scores' own CAL operating points. See also the separate [PSM curves](../../docs/experiments/mlp_replay_20261119/curves/PSM_PR_ROC.png) and [CANDI permissions and costs](../../docs/experiments/candi_native_a_20261118/RESULTS.md).

The [separate native TimeRadar comparison](../../docs/experiments/timeradar_native_v_replay_20261120/RESULTS.md) preserves the official initial weights, its 100-point context and variance readout, and the distinct PSM query endpoint axis. It uses its own fixed thresholds. [TimeRCD](../../docs/experiments/timercd_fixed_20261117/logit/RESULTS.md) is also archived with distinct source training, normalization, and context permissions. The [TSAD-C compatibility record](../../docs/research_scope.md) documents the published interval-score versus project point-score mismatch and local NA. No published headline number fills an unavailable local result.

## D. Historical exploration index

| Historical material | Archived evidence |
|:--|:--|
| Hybrid V1 and six technical lines | [Parameter and operation mapping](../../docs/hybrid_v1/paper_module_map.md), [public evaluation](../../docs/hybrid_v1/public_evaluation.md), [review experiment index](../../review_artifacts/EXPERIMENT_INDEX.md), and [public history index](../../docs/provenance/history_index.md). |
| TimeRadar shallow adaptation and diagnostics | [Fixed full-SMD extension](../../docs/experiments/timeradar_smd_full_left_matched_20261116/RESULTS.md), [selective interaction](../../docs/experiments/timeradar_selective_context_20261116/RESULTS.md), [readout decomposition](../../docs/experiments/timeradar_error_decomposition_20261117/RESULTS.md), and [frozen decoder](../../docs/experiments/timeradar_adapter_only_20261117/RESULTS.md). |
| MeanT | [Initial comparison](../../docs/experiments/timeradar_target_mean_20261120/RESULTS.md), [three seeds](../../docs/experiments/timeradar_target_mean_repeat_20261120/RESULTS.md), and [the fixed SMD subset](../../docs/experiments/timeradar_target_mean_smd_20261120/RESULTS.md). |
| TimeRadar and TimesFM investigations | [Conditioned energy](../../docs/experiments/tf_conditional_energy_20261119/RESULTS.md), [V conditioning](../../docs/experiments/timeradar_native_v_conditioning_20261119/RESULTS.md), [full frozen predictor benchmark](../../docs/experiments/timesfm3_frozen_20261121/RESULTS.md), and [cached event attribution](../../docs/experiments/timesfm3_v_cached_attribution_20261121/RESULTS.md). |
| The controlled complementarity study | [Predefined comparisons and full 238-stream outcome](../../docs/experiments/controlled_complementarity_20261121/RESULTS.md). The fixed controls did not substantiate anomaly-specific complementarity. |
| Other stopped adaptations and TimeRCD variants | [Window-loss trimming](../../docs/experiments/left_selective_swat_20261118/RESULTS.md), [source-response preservation](../../docs/experiments/left_source_preservation_20261118/RESULTS.md), [TimeRCD FRFT](../../docs/experiments/timercd_source_frft_20261117/RESULTS.md), [reference matching](../../docs/experiments/timercd_reference_repeats_20261117/RESULTS.md), and [frozen SWaT follow-up](../../docs/experiments/timercd_reference_swat_20261118/RESULTS.md). |
| Previous paper direction and other negative results | [Previous manuscript snapshot](https://github.com/codexpker/Hybrid/blob/733667a627cf8da16a1546853c6def7b66917b92/manuscript/next_generation_20261115/draft.md) and [public version lineage and closed branches](../../docs/research_scope.md). |

The preceding entries provide context and distinct external results. The core evidence in Sections 4 and 5 uses only the stated public SWaT and PSM tasks and their archived roles.

## E. Citation records and public data access

The [bibliographic source ledger](citation_sources.md) records publication metadata, methods checked, and classification of related entries carried over from the previous bibliography. The paper cites the authors of the methods, evaluation tools, and datasets directly. SWaT data are obtained through the [iTrust data access information](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/). PSM is documented in the [eBay authors' release](https://github.com/eBay/RANSynCoders). The supplement points to the public experiment records. Access to the original data follows the providers' terms.
