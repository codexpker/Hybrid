# SWaT31: seven frozen TimeRCD models, original B-only protocol

No model is trained or selected in this evaluation. The official zero-update model and the already source-selected batch/reference checkpoints for seeds2025/2026/2027 are scored once on the existing exposed SWaT roles. These are model-comparison results, not a new Hybrid architecture, lag mechanism or contamination-robustness claim.

## Fixed common QUERY axis

| Model | AP | ROC | Strict normal FPR (%) | C3 normal FPR (%) | C3 any-hit | C3 one-to-one matches | C3 event F1 (%) | C3 anomaly point coverage (%) | Long9 steady macro (%) | Long9 steady pooled (%) | C3 longest miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official_zero | 0.199793 | 0.613073 | 0.488554 | 0.394308 | 5 | 5 | 6.849315 | 0.937690 | 0.045527 | 0.374010 | 31669 |
| batch2025 | 0.205991 | 0.676077 | 0.028412 | 0.015939 | 2 | 2 | 13.793103 | 0.334889 | 0.000628 | 0.005159 | 31961 |
| reference2025 | 0.104429 | 0.185188 | 0.077961 | 0.056132 | 5 | 5 | 19.607843 | 1.433744 | 0.000628 | 0.005159 | 31961 |
| batch2026 | 0.178620 | 0.602515 | 0.029452 | 0.022868 | 5 | 5 | 22.727273 | 0.707453 | 0.120748 | 0.010318 | 31961 |
| reference2026 | 0.111141 | 0.197688 | 0.127856 | 0.096671 | 5 | 5 | 15.873016 | 1.540490 | 0.004396 | 0.036111 | 31883 |
| batch2027 | 0.277454 | 0.746354 | 0.074149 | 0.043658 | 4 | 4 | 14.285714 | 1.067459 | 0.003768 | 0.030953 | 31897 |
| reference2027 | 0.096609 | 0.187033 | 0.128549 | 0.098057 | 5 | 5 | 16.393443 | 1.167926 | 0.000628 | 0.005159 | 31961 |
| LEFT_native | 0.151321 | 0.546587 | 1.380424 | 1.213761 | 11 | 11 | 8.730159 | 1.387697 | 1.167543 | 0.257938 | 31842 |
| HistoryMean512 | 0.087649 | 0.227811 | 1.207871 | 1.125406 | 5 | 5 | 6.896552 | 0.230236 | 1.534654 | 0.123810 | 35900 |
| TimeRadar | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

TimeRadar is NA because no cached bank was qualified with this exact QUERY/CAL role identity; the old reproduction is retained and is not rerun. LEFT is a target-normal FIT/SELECT-trained external comparison, whereas the seven TimeRCD models have no target training. History Mean512 additionally fits per-channel median residual scales on the same512 CAL points before setting its threshold. Its operating results are not represented as equal calibration permission. No cross-permission superiority claim follows from this table.

## Three source-seed summaries (mean ± sample SD)

| Arm | AP | ROC | Strict normal FPR (%) | C3 normal FPR (%) | C3 event F1 (%) | C3 anomaly point coverage (%) | Long9 steady macro (%) | Long9 steady pooled (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batch | 0.220688 ± 0.051030 | 0.674982 ± 0.071926 | 0.044004 ± 0.026111 | 0.027488 ± 0.014426 | 16.935363 ± 5.021984 | 0.703267 ± 0.366303 | 0.041715 ± 0.068463 | 0.015476 ± 0.013649 |
| reference | 0.104060 ± 0.007273 | 0.189969 ± 0.006748 | 0.111455 ± 0.029009 | 0.083620 ± 0.023816 | 17.291434 ± 2.022875 | 1.380720 ± 0.191859 | 0.001884 ± 0.002175 | 0.015476 ± 0.017870 |

The official frozen model appears once per stream and receives no artificial three-seed SD. Training-seed summaries use ddof1 on the same fixed recording and events, not three independent SWaT domains.

## Fixed source-seed paired differences

| Reference arm minus comparator | Metric / stream | Seed2025 | Seed2026 | Seed2027 | Mean ± sample SD |
|---|---|---:|---:|---:|---:|
| reference − batch | AP / strict | -0.101561 | -0.067479 | -0.180845 | -0.116629 ± 0.058166 |
| reference − batch | ROC_AUC / strict | -0.490889 | -0.404827 | -0.559322 | -0.485013 ± 0.077415 |
| reference − batch | normal_FPR / confirmed3 | 0.000402 | 0.000738 | 0.000544 | 0.000561 ± 0.000169 |
| reference − batch | event_point_coverage / confirmed3 | 0.010989 | 0.008330 | 0.001005 | 0.006775 ± 0.005171 |
| reference − batch | long_steady_macro_coverage / confirmed3 | 0.000000 | -0.001164 | -0.000031 | -0.000398 ± 0.000663 |
| reference − official_zero | AP / strict | -0.095364 | -0.088652 | -0.103184 | -0.095734 ± 0.007273 |
| reference − official_zero | ROC_AUC / strict | -0.427885 | -0.415386 | -0.426041 | -0.423104 ± 0.006748 |
| reference − official_zero | normal_FPR / confirmed3 | -0.003382 | -0.002976 | -0.002963 | -0.003107 ± 0.000238 |
| reference − official_zero | event_point_coverage / confirmed3 | 0.004961 | 0.006028 | 0.002302 | 0.004430 ± 0.001919 |
| reference − official_zero | long_steady_macro_coverage / confirmed3 | -0.000449 | -0.000411 | -0.000449 | -0.000436 ± 0.000022 |

Pair differences above use raw fractions, not percentage-point scaling. Positive false-alarm differences are costs. Reference-minus-batch is the matched source-training comparison; reference-minus-official tests change from frozen pretrained weights, with different source-training exposure.

## Preserved SWaT definitions and permissions

- Original31 channel identities/order and raw FIT/CAL/QUERY roles are retained. Each model uses exactly the original FIT population mean/std once, with the official zero-std safeguard. No QUERY statistic or target training is used.
- Full input roles remain CAL445661:495000 and QUERY608023:944919. Native5000-point windows are anchored at those original input starts. Only after forward inference are scores cropped to the old QUERY608535:944919 axis (336384 points). This is not a new window origin.
- All seven use raw abnormal-minus-normal head logits and the same pre-existing512 CAL indices linspace(446173,494999,512,int64), q99/higher and strict greater-than. Full CAL is read out but is not substituted as a new calibration pool. CAL and QUERY never share context.
- Main event evaluation retains22 fully covered, non-boundary-censored QUERY events. Long events are duration>512 (nine events), with steady coverage from original event start+512. This is the existing SWaT definition, not the PSM/SMD steady-after192 definition.
- Strict and current-point C3 confirmation, any-hit and one-to-one event matching remain distinct. The full CSVs retain point counts, false-alarm fragments, event metrics and long missed spans. Only the fixed22 complete QUERY events are exported in per_event_complete_query; outside-QUERY events remain unobserved/NA in raw coverage records, never false misses.
- TimeRCD uses noncausal full-window context: first-hit and availability offsets are offline descriptions, not online detection-delay claims. LEFT/H512 native endpoint permissions are disclosed separately. Processed sample positions do not establish a physical sampling clock.

## Cost and provenance

New frozen inference cost is 0.157028835 GPUh, with zero optimizer updates. parameters_and_compute.csv records each model cost and peak allocated memory. No historical training cost is charged again. Safe provenance identifies raw seals and accepted baseline tables without exporting raw observations, labels, weights or private host paths. This fixed evaluation does not automatically authorize a new model, normalization, window, score or threshold variant.

## Fixed scientific outcome

**The PSM source-learning positive result remains preserved, but its ranking benefit does not transfer to this frozen SWaT setting.** This is a completed fixed comparison, not a trigger for score reversal, normalization changes or another local candidate.

Reference AP is0.104429 /0.111141 /0.096609 for seeds2025/2026/2027, below each matching batch result0.205991 /0.178620 /0.277454 and below official zero-update0.199793. Mean paired reference-minus-batch AP is -0.116629 with sample SD0.058166. Reference ROC mean is0.189969±0.006748, versus batch0.674982±0.071926 and official0.613073. These are descriptive fixed-development differences; no statistical-significance or independent-domain claim is made. The same predeclared abnormal-minus-normal logit direction is retained despite the unfavorable ordering.

At the fixed original CAL thresholds, reference C3 normal FPR averages0.083620%, lower than official0.394308% but higher than matching batch0.027488% in all three pairs. Reference anomaly point coverage averages1.380720%, versus batch0.703267% and official0.937690%. This modest total-point coverage increment is not a long-steady gain: nine-event steady macro coverage is only0.001884% for reference, below batch0.041715% and official0.045527%; steady pooled coverage averages0.015476% for both trained arms, versus official0.374010%. Long event miss spans remain approximately31883–31961 points. Low false-alarm counts cannot on their own establish effective detection when sustained faults are almost entirely missed.

Reference hits five events in each seed (any-hit and one-to-one both5); batch hits2 /5 /4 and official5. Event F1 is mixed relative to batch: reference improves in2025 and2027 but decreases in2026. Compared with official weights, event F1 is higher, but ranking worsens and long-steady coverage is lower. These operating tradeoffs are reported together; none is expanded into a broad source-learning, persistent-detection, mixed-unlabeled or Hybrid novelty claim.

Historical LEFT and History Mean512 results remain visible under their different training/calibration permissions, not as matched source-learning controls. TimeRadar has no qualified bank for this exact role protocol and remains NA. The current fixed evaluation is complete, with no new training, checkpoint selection, threshold tuning, score reversal or automatic follow-up variant.

## Verification and exact execution identity

The primary agent independently reconstructed the original labels/events and checked all14 raw-bank hashes, seven head-logit differences, original512 CAL thresholds,14 strict/C3 metric rows, any-hit, all22 complete events, nine long-event macro/pooled coverage, and paired three-seed means/sample SD. All matched. This is primary verification, not an external independent audit or another experiment. The exact executed evaluator source was preserved before evaluation and matches its raw RESULT source hash.

All seven frozen inference tasks exited successfully, with zero training updates and no failed model job. Actual additional GPU cost is0.157028835 GPUh including process startup. Earlier source training and external baseline computation were not rerun or charged again.
