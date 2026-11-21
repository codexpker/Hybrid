# Supplementary Material

This supplement accompanies [the main manuscript](manuscript.md). All numerical tables, plots, and decisions identified below refer to the existing archived runs.

| Main text | Supplement |
|:--|:--|
| 1–2. Introduction and related work | E. Reference verification and metadata |
| 3. Problem formulation and experimental protocol | A. Exact split, score availability, and reproduction details |
| 4. Natural mixed-unlabeled adaptation | B. Full three-seed outcomes and natural adaptation contrasts |
| 5. Controlled abnormal exposure | B. Exposure counts, all pairing contrasts, curves, event-level coverage, and archived costs |
| 6. Comparison with existing methods | C. Expanded external comparisons, source permissions, and unavailable comparisons |
| 7–8. Discussion, limitations, and conclusion | D. Historical experiments and prior manuscript |

## A. Protocol and reproduction details

All intervals use half-open indexing. The SWaT timeline concatenates the 495,000-point source prefix and the published 449,919-point test segment. In PSM, target positions are relative to the released 87,841-point test record. The PSM source prefix contains 132,481 points. Source scaler statistics for the primary LEFT detector follow its native protocol over the entire released source prefix, including its validation interval. The source-only and replay draws are restricted to SOURCE_FIT. The model's adaptation input is limited to legal complete windows from the prescribed fit intervals.

| Task | Grid anchor | Legal full-window CAL score axis | Scored query axis | Normal / anomalous query points | Complete events / long events |
|:--|--:|:--|:--|--:|--:|
| SWaT | 495000 | [585624,607320) | [608535,944856) | 288,544 / 47,777 | 22 / 9 |
| PSM | 0 | [18240,21888) | [22656,87744) | 47,309 / 17,779 | 48 / 5 |

The nonoverlapping readout windows have length and step 192. Each emitted score's availability index is the window's final observed index. Score evaluation uses the fixed query interval intersected with the legal full-window score axis. The last 63 SWaT and last 97 PSM test points have no full-window score in this comparison. The PSM raw query role begins at 22504, with its first 152 positions outside complete legal query windows. The PSM eligible event count excludes two boundary-truncated events, retaining 4,196 of 8,861 and 3,029 of 3,120 points from those events in pointwise metrics. Calibration uses 512 equally spaced integer indices between the first and last legal CAL score positions, inclusive. The archived SWaT and PSM implementations use NumPy's `linspace(..., dtype=int64)` for this selection. At the chosen locations, post hoc attack-label counts are 38 and 512, respectively.

Each task has one fixed LEFT initialization selected using the native source validation procedure. The endpoint adaptation budget is 2,000 steps with a new Adam optimizer at learning rate 0.0001 and batch size 256. Sampling is uniform with replacement over legal window starts in the respective fit interval. Matching draw positions cross arms but never cross tasks. The original checkpoints include the native prototypes and running state. In SWaT, noU is cropped from the native archived scores. In PSM, the source endpoint is restored with its actual runtime state for the frozen readout. The 2025 ordinary U128 and earlier restricted-window comparisons remain separate. The implementation and record give the exact source window roles and seeds. [LEFT native SWaT](../../docs/experiments/left_native_swat_20261118/RESULTS.md), [SWaT natural U and readout](../../docs/experiments/left_swat_natural_u_20261118/RESULTS.md), [SWaT matched replay](../../docs/experiments/left_swat_source_replay_20261118/RESULTS.md), [PSM matched replay](../../docs/experiments/left_psm_replay_20261118/RESULTS.md).

The range metrics use the archived LEFT `metricor.RangeAUC_volume_opt` path with `windowSize=100` and `thre=250`. For AP and ROC-AUC the full scored axis retains the original labels without point adjustment. Strict and three-point confirmed streams are evaluated independently. Any-hit measures episode overlap, whereas the chronological greedy one-to-one rule limits each alarm fragment and each event to one match. Normal-region fragments count contiguous alarm portions within normal labels and can include fragments clipped at anomaly boundaries. Long steady macro coverage averages the coverage within each eligible event after its first 512 samples. Pooled steady coverage weights events by their steady durations. All reported offsets are in sample positions. Full-window availability and hit-only medians limit their use as latency comparisons. [Metric definitions and full archived rows](../../docs/experiments/left_replay_exposure_repeat_20261118/comparison.csv).

## B. Full natural adaptation and controlled exposure results

The primary tables preserve task and seed identities. The first four rows per task in the natural-condition comparison are noU and the TT-A, SS, and R-A method means. The B and C rows retain both updating policies and all designated seeds. Sample standard deviation is based on the three values of a paired contrast, not on the distributions of scores or events. Frozen noU remains a single result. The archived strict and C3 streams include AP, ROC-AUC, VUS-PR, VUS-ROC, point precision/recall/F1, false-positive counts and rates, false-positive fragments, event precision/recall/F1, one-to-one matches, any-hits, all-event coverage, steady macro and pooled coverage, longest missed spans, and availability offsets.

| Supplementary record | Contents |
|:--|:--|
| [Method means and original study report](../../docs/experiments/left_replay_exposure_repeat_20261118/RESULTS.md) | Absolute natural, B and C results. Paired natural and exposure responses with sample variation. |
| [Every fixed seed](../../docs/experiments/left_replay_exposure_repeat_20261118/PER_SEED_TABLE.md) and [machine-readable streams](../../docs/experiments/left_replay_exposure_repeat_20261118/comparison.csv) | Absolute outcomes, strict/C3 distinctions, event definitions, and unrounded quantities. |
| [Paired effects](../../docs/experiments/left_replay_exposure_repeat_20261118/contrasts.csv), [paired summaries](../../docs/experiments/left_replay_exposure_repeat_20261118/paired_summary.csv), and [interactions](../../docs/experiments/left_replay_exposure_repeat_20261118/interaction.csv) | R−TT, R−SS, R−noU, B−C, B−A, C−A, and the difference of B−C differences. |
| [Exposure counts](../../docs/experiments/left_replay_exposure_repeat_20261118/exposure_by_seed.csv) | Selected slots, actual label exposure, window reuse, and task/seed identifiers. Label-derived fields are evaluation and construction metadata. |
| [PR and low-FPR ROC](../../docs/experiments/left_replay_exposure_repeat_20261118/curves/PR_lowFPR_ROC.png), [original operating points](../../docs/experiments/left_replay_exposure_repeat_20261118/curves/operating_points.csv), and [long-event distributions](../../docs/experiments/left_replay_exposure_repeat_20261118/all_seed_nine_long_events.csv) | Frozen seed-2025 figures with CAL points, and all three seeds' per-event coverage for the nine SWaT long events. |
| [Compute receipts](../../docs/experiments/left_replay_exposure_repeat_20261118/resource_summary.json) and [archived result verification](../../docs/experiments/left_replay_exposure_repeat_20261118/verification_summary.json) | Actual cost and verification for the exposure repetitions. Reads and model training are documented in the original experiment records. |

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
