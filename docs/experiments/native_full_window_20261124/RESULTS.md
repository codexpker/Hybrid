# Fixed-weight full-window native V/E comparison

[Table 1: complete-window offline methods](TABLE_OFFLINE.md) · [Table 2: matched-checkpoint endpoint/full-window and source-training effects](TABLE_READOUT.md) · [native full axes](TABLE_NATIVE_AXES.md) · [compatible external panels](TABLE_EXTERNAL.md)

## Fixed computation and assets

One official TimeRadar checkpoint, three reconstruction-only continuation checkpoints, three source-supervised V checkpoints and the seed2025 E-supervised checkpoint were read without training or new selection. Each entity's test sequence starts its own W100, stride100 grid at zero. Every complete window yields ten fixed complementary masked reconstructions with B1 order computation, eval mode, isolated mask RNG, correction1 and channel-mean reduction. All100 positions retain native V. The same reconstructions yield E as squared copy-mean reconstruction error, averaged across channels. The existing official V+E uses its fixed unit coefficients. MLP was not broadcast and remains in the earlier causal panel.

A score's index is the real observed timestamp; input footprint is its containing [100b,100b+100) window; available index is100b+99. No final short window is padded. Full native scoring covers 87,800 PSM points, 707,000 SMD points, 72,600 MSL points and425,400 SMAP points. Unscored tails total41/1420/1129/2217 respectively. NASA uses the existing sensor-zero arrays and all27/53 admitted entities. SMD keeps all28 distinct 38-channel machines and its seen-normal-source status. Input normalization, models and original causal score banks retain their identities.

The small normal SOURCE window check compared grouped decoding against each of the eight complete native forwards exactly. FullV at99 equals the existing terminal function. Parameters and buffers are unchanged and RNG is restored. The same independent-window condition is checked at the first window of each entity. [Check](CHECKS.json) · [109 score-bank identities](scoring/banks.csv) · [readout implementation](../../../src/hybrid_tsad/native_full_window_readout.py)

The original score banks remain on the server and have been copied to the user's existing local asset area, D:/code/bst/seal_detection_project/research_workspace/results/native_full_window_20261124_v1/. The109 npz banks and their receipts match remote SHA256 values. Each npz stores V and E for all eight fixed weights, real index, window start/end and available index. Raw arrays and weights were not added to Git. The repository contains source, metric tables, configuration identities and score receipts.

## Readout effect at the same weights

All differences below use a newly evaluated per-entity intersection of complete W100 outputs, existing rolling causal endpoints and LEFT output coverage. The first common index is99. Its stop is the smaller of the W100 and LEFT complete-window stops. No historical full-axis metric is substituted into this comparison. Entity metrics precede equal-domain means; the seed SD concerns three project-training seeds with one official initialization, n=3 and ddof=1.

| Domain | Candidate endpoint AP | Candidate full-window AP | Full−endpoint AP |
|---|---|---|---|
| PSM | .57306±.00469 | .44922±.00150 | −.12384±.00380 |
| SMD28 | .44023±.00255 | .43614±.00139 | −.00409±.00344 |
| MSL27 | .36592±.00188 | .35562±.00159 | −.01030±.00260 |
| SMAP53 | .38265±.00218 | .33422±.00185 | −.04843±.00109 |

The average AP change is negative in each task. SMD's same-weight full-window readout improves ROC-AUC and both VUS measures. PSM's VUS-ROC improves while AP, ROC and VUS-PR decrease. MSL/SMAP ROC increases coexist with other metric costs. These changes concern context and output position at fixed parameters. They do not create a source-learning benefit or an online detection advantage.

The source pair term directly trained endpoint99. Earlier block positions did not receive that direct supervision. Their complete-window scores also use observations after their assigned timestamps. These two differences are not separately identified by this fixed comparison. No position, mask, window, threshold or score weight was selected to improve the result.

## Source-training effect under a fixed readout

| Domain | Full-window candidate−official V AP | Full-window candidate−reconstruction continuation AP | Candidate−LEFT AP |
|---|---|---|---|
| PSM | +.00098±.00150 | −.00361±.00182 | −.12768±.00150 |
| SMD28 | +.00898±.00139 | −.00426±.00349 | −.06194±.00139 |
| MSL27 | +.00712±.00159 | −.01249±.00393 | −.15272±.00159 |
| SMAP53 | +.03257±.00185 | +.00518±.00965 | −.15894±.00185 |

Paired seed SD is calculated from the actual seed differences. Candidate mean full-window AP is above official V in all four tasks, but is below reconstruction continuation in three. SMAP's positive candidate-minus-continuation mean varies materially across seeds. All four LEFT macro metrics exceed the candidate's full-window values. LEFT uses target-normal training and a different window/context; matching scored points does not isolate architecture or training effects. Original strong NASA E results, SMD limitations and prior causal positive results remain in the previous panels.

The complete tables retain AP, ROC-AUC and both VUS metrics. Full-window source-learning contrasts are separate from same-weight changes. Source-learning contrasts also retain the causal readout, with its newly cropped axis. [All seed-specific macro values](macro.csv) · [raw entity metrics](entity_metrics.csv) · [paired entity values](paired_entity.csv) · [wins/ties/losses and medians](paired_distribution.csv) · [paired seed summaries](paired_seed_summary.csv)

## External permissions and calibration scope

The main comparison retains native LEFT offline coverage (W192 for PSM/SMD/SMAP and W96 for MSL). Extra PSM [512,87744) panels reuse CATCH, PCA, HistoryMean512 and LEFT caches. TimeRCD uses original saved logit differences in its W5000 offline context; A uses whole-input standardization and B uses its prior FIT-only standardization. The compatible PSM and SMD rows are separate from the W100 source-learning contrasts. NASA's corresponding TimeRCD rows are NA. External model training and standardization permissions are retained. No new external training or inference took place.

This study adds no CAL scores, thresholds, binary operating points or online delay estimates. Historical PSM attack-only CAL and NASA small-sample CAL records remain unchanged. Existing U/F results stay separate, and no adaptation or contamination experiment was started. MLP remains solely in the existing causal comparison.

## Technical decision and formal text

The fixed complete-window readout does not rescue a generally competitive offline method. The existing source-supervised causal V evidence remains valid on its original axes, but its gains cannot be expanded to all window positions. Some ROC/VUS benefits and small full-window improvements over frozen V survive; they coexist with weaker AP against reconstruction continuation and a gap to LEFT. The comparison provides a readout-dependent boundary for the current source-supervised candidate. It does not establish an independent FRFT contribution, weak/lag sensitivity, target U/F efficacy or contamination robustness. No successor readout is selected or launched.

The existing [draft](../../../manuscript/next_generation_20261115/draft.md) now presents the complete-window public panel, matched-weight readout panel, training-target/readout ablations and Z/U/F extensions as distinct sections. The prior causal common-axis panel is retained in an appendix with its own boundaries. The [method definition](../../../manuscript/next_generation_20261115/method_definition.md) defines both mappings and explicitly limits source supervision to endpoint99. Historical text and user edits remain preserved.

## Compute and independent review

Readout receipts sum to469.24 occupied A40-seconds, **0.130344 GPU-hours**. This includes per-entity load, checking, scoring and score-bank writing; process startup and the minimal normal-input check were not separately timed. There were no failed score jobs. Main CPU metrics took approximately1089seconds with six workers. External cache metrics took557seconds with two workers and overlapped the main queue; these are not additive GPU costs or CPU core-hour profiles. [GPU cost](scoring/COST.json) · [CPU timing](METRIC_EXECUTION.json)

One independent CPU review used PSM, SMD machine-1-1, MSL C-1 and SMAP A-1, ten formal readouts and their full/common/causal mappings. It checked grids, footprints, available indices and omitted tails; independent AP/ROC maximum discrepancy was8.33e-17. Whole-CSV macro aggregation, three-seed means/sample SD and pair distributions were checked with maximum discrepancy3.10e-16. VUS was independently aggregated from production entity metrics, not recomputed from raw scores. Current full-window99 versus current terminal-function checks are exact. Archived PSM official E reconstructed in NumPy differs from current CUDA endpoint reduction by up to3.81e-6, and V+E by4.50e-6. Those saved scores are preserved rather than rewritten. [Review scope and numerical details](review.json)
