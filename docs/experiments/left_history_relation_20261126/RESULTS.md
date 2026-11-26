# Detecting departures from a history-conditioned lag reference

## Matched public comparison

The source checkpoints at widths 25, 31 and 38 each initialized two complete LEFT models. Both models received the same 64,000 source windows over 2,000 Adam updates. The joint arm also learned to predict six blocks of self- and ordered cross-correlations from eight preceding patches. Native and relation gradients reached shared parameters. Four GPUs ran the matched jobs for one development seed. Fixed endpoints determined all deployments. Calibration of component scales used only the designated source SELECT windows. Target normal-prefix statistics follow the archived public Z protocol. [Method and equations](../../../manuscript/next_generation_20261115/method_definition.md) · [Checkpoints, exposure hashes and cost](source_training.csv)

The full detector scores the 192 current observations when the last arrives. Each relation score covers 32 observations. On the original nonoverlapping grid the first eligible start is test-relative 384, so all 256 preceding observations belong to the same test record. The primary panel uses the legal W192 endpoint. External panels use their stated intersections, including a separate shorter axis where a comparator needs 512 past samples. SMD values are equal-weight means of 28 per-machine metrics. All folds have prior project development exposure. Target fitting and future context differ across external methods. AP denotes average precision. VUS uses LEFT's `metricor.RangeAUC_volume_opt` with windowSize 100 and 250 thresholds.

The joint minus matched-continuation AP differences were −0.09062 on PSM, −0.00662 on SWaT and −0.03882 on SMD28. The corresponding native-only readout of the very same joint checkpoints reached AP 0.51531, 0.72319 and 0.50314, close to the three continuation models. The prescribed addition of relation error explains most of this AP trade-off at evaluation time. SWaT gained ROC-AUC and VUS-ROC while losing PR performance. The project retains these scope-specific results. The evidence favors checking whether a predicted relationship can separate a known changed pair from its matched control before considering any new adaptation objective. U/F, contamination and operating-point claims remain open.

[Complete public metric panels and entity comparisons](PUBLIC_RESULTS.md) · [Paper results](../../../manuscript/next_generation_20261115/draft.md)

## What the source encoder learned

The mean relation regression loss over the first and last 200 training steps fell from 0.1221 to 0.0892 in the PSM fold, from 0.1235 to 0.0888 in the SWaT fold and from 0.1852 to 0.1267 in the SMD fold. On held-out normal source windows, the mean valid block errors used for scaling were 0.0839, 0.0848 and 0.1202, respectively. Shared-encoder gradients and a lower source error establish that the regression target influenced learning. Its value for detection is measured separately on the target records and matched edits.

## Mechanism test

The lag manipulation moved the planted peak on both independent synthetic panels. On the PSM-width panel the driver–receiver correlation at lag 4 fell from 0.788 to −0.031, while the correlation at lag 16 rose from −0.022 to 0.785. The corresponding SWaT-width values changed from 0.794 to −0.026 and from −0.023 to 0.796. Each width used 64 independent AR records. Transfer checks used 16 fixed nonoverlapping nominal windows on each public normal validation segment. These windows are repeated observations from a series. Known link identity and planted delay apply to the simulation only.

| Core ROC-AUC, intervention versus same-energy amplitude control | PSM-width AR | SWaT-width AR | PSM normal windows | SWaT normal windows |
| --- | ---: | ---: | ---: | ---: |
| Delay edit, simple historical correlation | 0.5554 | 0.5330 | 0.5859 | 0.4766 |
| Delay edit, learned relation alone | 0.5293 | 0.5186 | 0.4922 | 0.5117 |
| Delay edit, complete joint model | 0.2957 | 0.4802 | 0.4922 | 0.5117 |
| Weak shift versus smooth bipolar control, complete joint model | 0.5017 | 0.4944 | 0.2617 | 0.0000 |

Preserving the unedited history and matching the per-receiver energy made it possible to separate two questions. Known relationships changed on the input side. The deployed detector nevertheless assigned the synthetic delay edit a larger core score than the amplitude control in 0 of 64 PSM-width pairs and 22 of 64 SWaT-width pairs. Centered Pearson targets are unchanged by a constant offset over the aligned samples of a block. Consequently, detection of a weak plateau can come from the native path or its transitions. A single affected pair also supplies only a fraction of a dense cross-pair mean. Those properties delimit the test and suggest explanations for low sensitivity. This experiment does not isolate pooling as its cause. [Clean, energy-matched and synchronous controls, core and transitions](controlled_results.md) · [Paired differences and counts](controlled_pairs.csv)

## Artifact and verification record

The six fixed endpoints are mirrored under `research_workspace/results/left_history_relation_20261126_v1/{PSM,SWAT,SMD}/{continue,joint}/selected.pt` alongside training receipts. The compute workspace stores the same relative run directory under `results/`. The Git archive includes their SHA-256 identities and the actual source training traces. The paired native RNG hashes agree at the start and finish in every fold. [Training receipts](source_receipts.json) · [Matched training entry point](../../../scripts/run_left_history_training.sh)

Six training runs, including their recorded source checks and scale estimation, used 491.11 A40-seconds in aggregate. Sixty new public banks used 13.94 A40-seconds to score 1,232,064 timestamps per arm across 30 targets. Controlled score generation used 5.77 A40-seconds. The sum is about 0.142 A40-hours of measured job time across lanes. Public metric evaluation additionally took 4,364.10 aggregate CPU worker-seconds. Transfers and archival are outside these measures. Source checkpoints and reused external results retain their original costs.

An independent CPU implementation verified the history and availability indices across all 30 joint score banks and reconstructed the persistence coefficient errors on 72 real blocks in PSM, SWaT and the first and last SMD machines. The largest absolute discrepancy for persistence was below 2.1×10⁻⁸. Native-plus-relation score recomposition agreed within 2.4×10⁻⁷ on these blocks. Four controlled core AUCs were reproduced by pair counting. Public AP and ROC were separately recalculated from archived scores. VUS is from the pinned metric implementation. [Readout review](readout_review.json) · [Control checks and inference times](controlled_results.md)
