# Public multivariate anomaly detection after multi-source pretraining

## Research question

Can public source records teach a detector which local dynamics recur across sensors, and can the detector learn new normal dynamics from a target record that also contains anomalies? We study this question in two stages. A complete public detection benchmark establishes the performance that a proposed method must reach. Separate experiments examine zero target updates (Z), mixed-unlabeled learning (U), a fixed labeled support set (F), controlled extra abnormal exposure and the mechanism of the update.

The current study asks whether complete LEFT detectors pretrained on two public normal sources provide a useful initialization for a new multivariate target. The comparison separates frozen detection, normal-target training and mixed-unlabeled updates. Each source model sees the target number of variables through window-wise ordered channel sampling from TEP and SMD. Normal-only and mixed experiments retain their respective training and calibration permissions. [Method and data permissions](method_definition.md)

## Public benchmark

Each method occupies one row within its training group. Panels A and B use different archived timestamp intersections. Panel C in the [supplement](README.md) covers scores available at the observation on fixed suffixes. Only compare values within a dataset, panel and metric; future-window context and target-normal training are additional information even on identical timestamps. Blank coverage is marked NA. All numerical cells in A–C come from local evaluations. The trajectory-subspace rows use target FIT-only standardization.

### A. Complete-window comparison on the five public tasks

SMD includes all 28 machines at 38 channels. PSM uses 25 channels and SWaT the archived 31-channel mapping. ¹MSL27 and SMAP53 use only the first continuous sensor per entity, including all entities on their respective frozen lists. Compute metrics per entity first and weight each entity equally. SMD has one LEFT model per machine; each NASA domain has one shared LEFT model. The PSM and SWaT columns cover [0,87744) test-relative and [495000,944856) absolute. The SMD, MSL and SMAP archived external intersections cover 705,768, 72,168 and 395,376 points; SMAP starts at 512 per eligible record for its history comparator. Original windows and indices determine each intersection. All datasets have prior development exposure.

LEFT uses normal FIT and SELECT, a normal-prefix scaler including SELECT and W192 (W96 on MSL). PSM/SWaT LEFT summarizes seeds 2025/2026 from the matched implementation. The other three LEFT columns each have seed2025. CATCH uses target FIT scaling/training, normal SELECT and W192 (three seeds). TimeRadar block0 uses a full normal-prefix scaler, W100 and a batch-shared order (B64 PSM, B128 SWaT); the separate W100 row uses FIT scaling and one window per order. Treat batched order as batch-context access. TimeRCD uses frozen synthetic-pretrained weights, FIT-only statistics and W5000 contexts. Complete-window scores use the end of the input window or later batch context. Level deviation is mean channel SmoothL1 to zero in the LEFT scaler; PCA reads the current point after target FIT; History Mean uses 512 past observations. Each repeated cell is the mean ± sample SD across training seeds, calculated from archived full-precision metrics. Single observations and fixed weights have no SD. [Metric rows and permissions](../../docs/experiments/left_decoder_context_20261124/public_extension/methods.csv) · [Per-entity results](../../docs/experiments/left_decoder_context_20261124/public_extension/entity_metrics.csv)

**AP**

| Training group | Method | SMD28 | PSM | SWaT | MSL27¹ | SMAP53¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target normal | CATCH | NA | 0.4364 ± 0.0084 | NA | NA | NA |
| Target normal | LEFT | 0.4975 | 0.5734 ± 0.0046 | 0.5650 ± 0.0299 | 0.5043 | 0.4967 |
| Pretrained | TimeRCD, FIT-only | 0.4723 | NA | NA | NA | NA |
| Pretrained | TimeRadar, W100 individual-window order | 0.4261 | NA | NA | 0.3436 | 0.3072 |
| Pretrained | TimeRadar, block0 batch order | NA | 0.4333 | 0.6393 | NA | NA |
| Pretrained | TimeRadar, FIT-only batch order | NA | 0.4554 | NA | NA | NA |
| Statistical | History Mean512 | NA | NA | NA | NA | 0.3178 |
| Statistical | Level deviation | 0.4265 | 0.4383 | 0.7448 | 0.2880 | 0.2966 |
| Statistical | PCA residual | NA | 0.5344 | NA | NA | NA |
| Project multi-source, frozen | Frozen multi-source subspaces | NA | 0.3294 | 0.2002 | NA | NA |
| Source + normal target | Source planes → normal target fit | NA | 0.3278 | 0.1113 | NA | NA |
| Normal target only | Normal target covariance → normal target fit | NA | 0.3278 | 0.0981 | NA | NA |

**ROC-AUC**

| Training group | Method | SMD28 | PSM | SWaT | MSL27¹ | SMAP53¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target normal | CATCH | NA | 0.6474 ± 0.0060 | NA | NA | NA |
| Target normal | LEFT | 0.8787 | 0.7739 ± 0.0030 | 0.8177 ± 0.0045 | 0.7660 | 0.7610 |
| Pretrained | TimeRCD, FIT-only | 0.8594 | NA | NA | NA | NA |
| Pretrained | TimeRadar, W100 individual-window order | 0.8274 | NA | NA | 0.6709 | 0.6624 |
| Pretrained | TimeRadar, block0 batch order | NA | 0.6707 | 0.8220 | NA | NA |
| Pretrained | TimeRadar, FIT-only batch order | NA | 0.6838 | NA | NA | NA |
| Statistical | History Mean512 | NA | NA | NA | NA | 0.6344 |
| Statistical | Level deviation | 0.8034 | 0.6542 | 0.8318 | 0.5530 | 0.5723 |
| Statistical | PCA residual | NA | 0.7933 | NA | NA | NA |
| Project multi-source, frozen | Frozen multi-source subspaces | NA | 0.5347 | 0.5534 | NA | NA |
| Source + normal target | Source planes → normal target fit | NA | 0.5325 | 0.4264 | NA | NA |
| Normal target only | Normal target covariance → normal target fit | NA | 0.5326 | 0.3983 | NA | NA |

**VUS-PR**

| Training group | Method | SMD28 | PSM | SWaT | MSL27¹ | SMAP53¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target normal | CATCH | NA | 0.5275 ± 0.0087 | NA | NA | NA |
| Target normal | LEFT | 0.5416 | 0.5674 ± 0.0073 | 0.5286 ± 0.0084 | 0.5486 | 0.5257 |
| Pretrained | TimeRCD, FIT-only | 0.5146 | NA | NA | NA | NA |
| Pretrained | TimeRadar, W100 individual-window order | 0.4508 | NA | NA | 0.3697 | 0.3055 |
| Pretrained | TimeRadar, block0 batch order | NA | 0.4413 | 0.5141 | NA | NA |
| Pretrained | TimeRadar, FIT-only batch order | NA | 0.4581 | NA | NA | NA |
| Statistical | History Mean512 | NA | NA | NA | NA | 0.3360 |
| Statistical | Level deviation | 0.4255 | 0.4460 | 0.4645 | 0.3069 | 0.3021 |
| Statistical | PCA residual | NA | 0.5290 | NA | NA | NA |
| Project multi-source, frozen | Frozen multi-source subspaces | NA | 0.3624 | 0.2108 | NA | NA |
| Source + normal target | Source planes → normal target fit | NA | 0.3607 | 0.1163 | NA | NA |
| Normal target only | Normal target covariance → normal target fit | NA | 0.3607 | 0.1019 | NA | NA |

**VUS-ROC**

| Training group | Method | SMD28 | PSM | SWaT | MSL27¹ | SMAP53¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target normal | CATCH | NA | 0.6977 ± 0.0065 | NA | NA | NA |
| Target normal | LEFT | 0.9186 | 0.7817 ± 0.0047 | 0.7799 ± 0.0415 | 0.7965 | 0.7832 |
| Pretrained | TimeRCD, FIT-only | 0.9097 | NA | NA | NA | NA |
| Pretrained | TimeRadar, W100 individual-window order | 0.8825 | NA | NA | 0.7191 | 0.7045 |
| Pretrained | TimeRadar, block0 batch order | NA | 0.6784 | 0.7668 | NA | NA |
| Pretrained | TimeRadar, FIT-only batch order | NA | 0.6892 | NA | NA | NA |
| Statistical | History Mean512 | NA | NA | NA | NA | 0.6865 |
| Statistical | Level deviation | 0.8667 | 0.6543 | 0.6443 | 0.6055 | 0.6136 |
| Statistical | PCA residual | NA | 0.7667 | NA | NA | NA |
| Project multi-source, frozen | Frozen multi-source subspaces | NA | 0.5812 | 0.5665 | NA | NA |
| Source + normal target | Source planes → normal target fit | NA | 0.5790 | 0.4419 | NA | NA |
| Normal target only | Normal target covariance → normal target fit | NA | 0.5790 | 0.4139 | NA | NA |

### B. Fixed pretrained detectors on the external intersection

Here every entity starts at 99 and ends at the earlier complete DADA/LEFT window boundary. Coverage in SMD/PSM/MSL/SMAP order is 702,996 / 87,645 / 69,495 / 417,265. DADA retains its official complementary-copy variance, multidomain pretraining with synthetic anomalies and full normal-array scaler. TimeRCD uses the original anomaly-minus-normal logit after FIT-only standardization (multivariate weights on SMD/PSM, univariate on NASA). Both require future context within each block. TimeRadar's released L=1 variance is computed at the causal endpoint with FIT-only statistics. Group labels identify the access difference. LEFT here uses its single archived target-trained reference per task. The two-seed LEFT results in A apply to that panel only. SWaT has no four-way external-axis receipt in this archive. [Full-precision metrics](../../docs/experiments/native_endpoint_external_20261125/domain_metrics.csv) · [Source and input identities](../../docs/experiments/native_endpoint_external_20261125/AXES_AND_INPUTS.json) · [Entity results](../../docs/experiments/native_endpoint_external_20261125/entity_metrics.csv)

**AP**

| Training group | Method | SMD28 | PSM | MSL27¹ | SMAP53¹ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target normal | CATCH | NA | 0.4368 ± 0.0084 | NA | NA |
| Target normal | LEFT | 0.4981 | 0.5769 | 0.5083 | 0.4932 |
| Pretrained, offline | DADA | 0.4146 | 0.3957 | 0.3957 | 0.4288 |
| Pretrained, offline | TimeRCD, FIT-only | 0.4729 | 0.3019 | 0.4827 | 0.4427 |
| Pretrained, causal | TimeRadar endpoint V | 0.4448 | 0.5353 | 0.3426 | 0.3656 |
| Statistical | PCA residual | NA | 0.5344 | NA | NA |

**ROC-AUC**

| Training group | Method | SMD28 | PSM | MSL27¹ | SMAP53¹ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target normal | CATCH | NA | 0.6477 ± 0.0060 | NA | NA |
| Target normal | LEFT | 0.8785 | 0.7762 | 0.7633 | 0.7629 |
| Pretrained, offline | DADA | 0.8514 | 0.6050 | 0.7336 | 0.6459 |
| Pretrained, offline | TimeRCD, FIT-only | 0.8594 | 0.4762 | 0.7631 | 0.6749 |
| Pretrained, causal | TimeRadar endpoint V | 0.8132 | 0.7310 | 0.6327 | 0.6427 |
| Statistical | PCA residual | NA | 0.7931 | NA | NA |

**VUS-PR**

| Training group | Method | SMD28 | PSM | MSL27¹ | SMAP53¹ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target normal | CATCH | NA | 0.5272 ± 0.0087 | NA | NA |
| Target normal | LEFT | 0.5418 | 0.5741 | 0.5531 | 0.5217 |
| Pretrained, offline | DADA | 0.4747 | 0.4395 | 0.4298 | 0.4459 |
| Pretrained, offline | TimeRCD, FIT-only | 0.5151 | 0.3485 | 0.5298 | 0.4280 |
| Pretrained, causal | TimeRadar endpoint V | 0.4672 | 0.4836 | 0.4147 | 0.4027 |
| Statistical | PCA residual | NA | 0.5287 | NA | NA |

**VUS-ROC**

| Training group | Method | SMD28 | PSM | MSL27¹ | SMAP53¹ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target normal | CATCH | NA | 0.6970 ± 0.0066 | NA | NA |
| Target normal | LEFT | 0.9185 | 0.7858 | 0.7942 | 0.7849 |
| Pretrained, offline | DADA | 0.8997 | 0.6520 | 0.7747 | 0.6826 |
| Pretrained, offline | TimeRCD, FIT-only | 0.9098 | 0.5320 | 0.8065 | 0.7038 |
| Pretrained, causal | TimeRadar endpoint V | 0.8700 | 0.6648 | 0.7048 | 0.6902 |
| Statistical | PCA residual | NA | 0.7670 | NA | NA |

### What the observed comparisons establish

The target-trained LEFT reference has the highest AP of the listed methods on each of the four B intersections. On SMD its AP is 0.4981, compared with 0.4729 for FIT-only TimeRCD and 0.4146 for DADA. On SWaT the pointwise level baseline already attains 0.7448 AP on panel A. A source-transfer detector must therefore address both a target-trained reconstruction reference and a simple level shift. These comparisons measure complete systems with different information and supervision. Neither comparative result identifies a time-frequency mechanism. Local AP is unadjusted average precision. VUS uses the pinned LEFT evaluator with windowSize 100 and 250 thresholds. SMD pooled-paper results, corrected alarms and target-adaptation suffixes have their own protocols.

### Published reference panel

Published VUS-PR and VUS-ROC are provided independently of local AP, preprocessing and entity weighting. CrossAD is the authors' result. ModernTCN is the comparator reported by the LEFT authors under their benchmark. Both lack local five-domain scores on A or B. No repetition count or SD was inferred. [CrossAD, Tables 2–3](https://arxiv.org/html/2510.12489) · [LEFT, Table 2](https://arxiv.org/html/2602.08638v2#S4.T2)

| Published method | Metric | SMD | PSM | SWaT | MSL | SMAP |
|---|---|---:|---:|---:|---:|---:|
| CrossAD | VUS-PR | .2344 | .5596 | .4767 | .3144 | .1443 |
| CrossAD | VUS-ROC | .8580 | .7302 | .7865 | .8091 | .5779 |
| ModernTCN (LEFT benchmark) | VUS-PR | .1596 | .4668 | .0941 | .3010 | .1395 |
| ModernTCN (LEFT benchmark) | VUS-ROC | .7707 | .6480 | .2735 | .7747 | .5470 |

### Complete LEFT after normal-source pretraining

Two source models trained on fault-free TEP and normal SMD records were transferred intact at the matching width. On the complete PSM axis, frozen transfer, target-normal continuation and target-normal training from scratch achieved AP of 0.4964, 0.5544 and 0.5766. Target-normal continuation fell below the from-scratch model on all four metrics. On SWaT, the corresponding AP values were 0.7239, 0.5425 and 0.5439. The frozen model exceeded the archived pretrained TimeRadar AP of 0.6393, while pointwise level deviation reached 0.7448. At matched target-normal training, the source initialization improved SWaT ROC-AUC from 0.8209 to 0.8556 and VUS-ROC from 0.8093 to 0.8431, with small decreases in both PR measures. Transfer value therefore depends on both the target task and the operating question. [Complete external benchmark, training contrasts and costs](../../docs/experiments/left_multisource_20261126/RESULTS.md)

### Archived synthetic supervision of the pretrained variance detector

On the complete public axes, frozen official TimeRadar, source reconstruction and source intervention training attained AP of 0.4481, 0.4557 and 0.4556 on PSM, and 0.6423, 0.6558 and 0.6562 on SWaT. Most of the AP improvement over the official checkpoint was already present after normal reconstruction. Source pairing improved SWaT VUS-PR by 0.0030 and VUS-ROC by 0.0045 over reconstruction, with near-identical AP. On PSM, target-trained LEFT and PCA exceeded the paired model in AP. On SWaT its AP exceeded LEFT but remained below pointwise level deviation, and LEFT retained higher VUS-PR.

On the separate mixed-target suffixes the local intervention reference changed AP by −0.01587 (PSM) and −0.01799 (SWaT) from the frozen source-pair detector. Against the matched absolute reference, AP changed by −0.00145 and +0.00090. The gains in sensitivity to artificial amplitude edits thus did not yield a consistent advantage on real events. SWaT's three-confirmation FPR decreased from 37.65% to 27.10%, while long-event macro coverage decreased from 40.01% to 29.18% and the number of normal alarm fragments increased. PSM's three-confirmation recall increased slightly with the same number of one-to-one matches. Its 512 mixed CAL points were all attack-labeled. Results are offline and conditional on the fixed edits and one training seed. [Public and U tables, full ablation and operating costs](../../docs/experiments/intervention_reference_20261126/RESULTS.md)

## Use conditions, contamination and mechanism

Establish the complete detector on the public benchmark before attributing gains. The source-excluded Z, separate U and labeled F lanes share one score. Report the zero-update comparator again on the same untouched target suffix where a comparison requires it. Controlled abnormal exposure compares fixed abnormal and nominal replacements of the same slots. Seek a net improvement in absolute detection and measure false alarms, matched events, delay and sustained coverage. Sparse deviation estimates are statistical residuals. Their existence supplies no physical fault-channel labels. The prior PSM mixed CAL was entirely attack-labeled; NASA small-sample empirical q99 can equal the maximum. The actual use of each CAL governs what can be inferred about normal false alarms.

### Mixed learning from a cross-domain LEFT initialization

On the isolated PSM suffix the source model scored 0.5043 AP. Ordinary mixed-target training raised this to 0.5770. The random-initialization control reached 0.5809 with exactly the same target windows, while source replay reached 0.5919. Ordinary U improved ROC-AUC and both VUS measures relative to random U. Replay traded some ROC-AUC for higher AP and VUS-PR than ordinary U. Its confirmed-three normal FPR fell from 16.27% for Z to 12.89%, alongside a decline from 23 to 21 one-to-one event matches and from 74.27% to 67.09% long-event macro coverage. All 512 PSM CAL points were labeled as attacks in the post-training evaluation.

On SWaT, Z reached 0.7914 AP. Ordinary U, replay U and random U reached 0.7800, 0.7861 and 0.7882, respectively. Adaptation produced small improvements over Z on the VUS measures. Ordinary U increased confirmed event matches from 9/22 to 13/22 and reduced the normal point FPR from 72.30% to 53.27%, while long-event macro coverage fell from 60.52% to 27.27%. Replay recovered some of that coverage (57.83%) at a higher FPR (62.91%). The data support studying this initialization and replay under identifiable task and operating conditions. They provide no general advantage of pretrained U over random U. These comparisons evaluate complete training procedures under fixed window budgets on two exposed tasks and one development seed. [Four-metric matched results, offline availability and individual events](../../docs/experiments/left_multisource_20261126/RESULTS.md)

### Does within-window reliability improve on window weighting?

We held each replayed window's total loss weight fixed while varying the resolution of that weight. A frozen source LEFT model assigned a native score to each position of each target window. Low percentile ranks received higher coefficients. The local arm used these coefficients on the original time grid; the matched window arm used their window mean throughout. Both used that same mean for coarse-scale and spectral losses. Independent post-training annotation of the 7,181 partially anomalous window presentations under natural PSM mixing showed that the average coefficient on their nominal positions increased from 0.954 to 0.984 and that on their anomalous positions fell from 0.785 to 0.715. Across 1,221 partial SWaT presentations, the corresponding changes were 0.817 to 0.843 and 0.776 to 0.752. These position-specific coefficients describe loss allocation. They cannot quantify the parameter-gradient contribution of a labeled point.

On naturally mixed PSM, local weighting reached AP 0.59774, ahead of frozen transfer (0.50430), ordinary replay (0.59195) and window weighting (0.59485). Local weighting improved all four ranking measures over window weighting. At their respective CAL thresholds, confirmed normal FPR fell from 12.38% to 11.67% but recall also fell from 44.59% to 43.60%. Both matched 20/48 events; long-event macro coverage fell from 66.28% to 64.87%. All PSM CAL points had attack labels. Under abnormal replacement B, both weighting schemes improved all four ranking measures over ordinary replay. Window weighting reached higher B AP (0.59574 versus 0.59426) and a smaller B−C AP loss (0.00449 versus 0.00848). Local weighting retained a B advantage on the two ROC measures.

On SWaT, local and window weighting differed by only 0.000005 AP under A. Frozen transfer remained ahead by 0.00414 AP. In B, local weighting improved AP by 0.00022 over window weighting, with lower VUS-PR and VUS-ROC. Both matched 12/22 events after confirmation and attained 56.77% long-event macro coverage. Thus score-derived weighting has a single-seed PSM positive relative to ordinary replay. Finer temporal allocation gives nominal positions more weight within mixed windows on both tasks, but does not establish a consistent detection or exposure-protection advantage over uniform within-window weighting. We retain the allocation finding and the PSM natural-mixing result without advancing this fixed local mechanism as a complete method. [Full adaptation, interactions, operating costs and example windows](../../docs/experiments/left_local_reliability_20261126/RESULTS.md)

### Abnormal exposure and the two prototype update pathways

We replaced the same 3,200 target slots with wholly anomalous or wholly nominal windows from U-FIT, holding the second half-batch fixed within each method. The 3,200 anomaly presentations came from five distinct PSM events or nine SWaT events. Labels determined the controlled replacement outside the learner. All models used the same official target normal scaler, native retrospective readout and isolated query suffix within a domain. The comparison covers one development seed.

On PSM the B−C AP change was −0.05103 for ordinary target updates and −0.02062 for source replay. Replay also reached higher absolute AP in B (0.57537 versus 0.52646). This attenuation was specific to the PR measures: the ROC-AUC declines were 0.02362 and 0.02419, respectively. Under replay's own mixed-CAL thresholds, B matched 24/48 events after three consecutive exceedances compared with 21/48 in C, while the normal false-positive rate rose from 12.56% to 15.25%. All 512 PSM CAL points carried attack annotations.

On SWaT, replay again had a higher absolute B AP than ordinary target updates (0.78061 versus 0.77841), but the replay B−C decline was larger (−0.00548 versus −0.00155). Under replay, one-to-one matches increased from 10/22 in C to 14/22 in B, whereas any-hit remained 19/22. The confirmed normal point false-positive rate fell from 65.54% to 53.87%, while long-event macro coverage fell from 57.98% to 28.54%. SWaT CAL included 38 attack points among 512. Relative ranking, absolute detection and the coverage at a model's calibrated operating point support different conclusions.

Explicit prototype writes did not explain these exposure responses. Across the 14 newly instrumented trajectories (28,000 updates), neither the full-batch native filter nor the source-restricted rule admitted a single window. Both prototype banks nevertheless moved through Adam. Under natural mixing, restricting write permission changed AP by less than 0.00001 on either task. Under controlled exposure the AP interactions against ordinary replay were −0.000068 on PSM and −0.000022 on SWaT. The experiment identifies sensitivity to the mixture supplied to learning, but contains no observed explicit write for the proposed restriction to prevent. The earlier natural ordinary-U and replay checkpoints have no archived per-window selection masks. We retain these detection results and close this write restriction at its fixed settings. [Methods, interactions, event costs and prototype measurements](../../docs/experiments/left_memory_exposure_20261126/RESULTS.md)

### Do source-specific update constraints improve mixed-target detection?

We kept LEFT's training batch, replay schedule, initialization and detection score fixed and constrained the realized Adam displacement using either one mean source gradient or two separate gradients. A source-only control replaced the 16 target slots with 16 additional normal source windows. All six new jobs used one seed, 2,000 updates and the fixed endpoint. The two constraints were satisfied at every stored parameter update. They activated on 315 of 2,000 PSM steps and 225 of 2,000 SWaT steps, versus 71 and 25 for the mean constraint. The minimum retained fractions of the proposed step norm were 99.76% and 99.92% under separate constraints. The effect was primarily directional. This experiment does not isolate shortening with a separate length-matched control. Nor did the constraints guarantee monotonic reference losses: the TEP loss rose after 25.20% and 26.80% of the respective parameter steps.

On PSM the separately constrained model reached AP 0.59226, up 0.08797 from frozen transfer and 0.06813 from source continuation. Ordinary replay already reached 0.59195. The additional AP gain over replay was 0.00032, and over the mean constraint it was 0.00049. At their respective mixed-CAL thresholds, both ordinary and separately constrained replay matched 21 of 48 events after three consecutive exceedances. Separate constraints reduced normal FPR from 12.89% to 12.83% and reduced long-event macro coverage from 67.09% to 66.74%. All PSM CAL points were annotated as attacks.

On SWaT separate constraints produced AP 0.78593, below frozen transfer (0.79136), source continuation (0.78972), ordinary replay (0.78615) and the mean constraint (0.78602). VUS-PR improved by 0.00040 over replay. Confirmed normal FPR fell from 62.91% to 62.72%, but normal alarm fragments increased from 270 to 289. One-to-one matches remained 10/22, compared with 11/22 under the mean constraint, and long-event macro coverage remained near 57.8%. The two source gradients carry different information about update conflicts, yet enforcing both added little to ordinary replay on the two target suffixes. These results support retaining ordinary replay as the comparator. They do not establish contamination robustness. [Complete adaptation table, paired contrasts and update traces](../../docs/experiments/left_source_constraints_20261126/RESULTS.md)

## Archived fixed trajectory-subspace mixed-target results

| Method | PSM AP | PSM ROC-AUC | PSM VUS-PR | PSM VUS-ROC | SWaT AP | SWaT ROC-AUC | SWaT VUS-PR | SWaT VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen multi-source subspaces | 0.3292 | 0.5442 | 0.3608 | 0.5875 | 0.2348 | 0.5675 | 0.2447 | 0.5770 |
| Source planes → quadratic U | 0.3273 | 0.5415 | 0.3589 | 0.5850 | 0.1394 | 0.4571 | 0.1460 | 0.4688 |
| Mixed target covariance → robust U | 0.3275 | 0.5417 | 0.3591 | 0.5852 | 0.1207 | 0.4415 | 0.1251 | 0.4535 |
| Source planes → robust U | 0.3273 | 0.5415 | 0.3589 | 0.5850 | 0.1374 | 0.4550 | 0.1438 | 0.4668 |

Both target-initialized references use the covariance of the permitted target fit. On the public comparison, the frozen source model exceeded the two updated subspace models on both datasets, while falling below LEFT and the pointwise level baseline. On the mixed suffix, replacing quadratic fitting with localized sparse separation changed AP by −0.000001 on PSM and −0.00195 on SWaT. On SWaT both fits matched 17 of 22 events after confirmation, but the source robust U model reached just 0.68% point recall compared with 6.24% for frozen Z. Source-initialized robust U exceeded target-only robust U there by 0.01671 AP. This is evidence of a limited reference-subspace advantage alongside a loss from adaptation. No substantive benefit of separating deviations during U was established. All readouts are retrospective. [Axes, costs, convergence, decompositions and event results](../../docs/experiments/trajectory_subspace_20261126/RESULTS.md).

## Prior evidence and its limits

The [frozen endpoint V study](../native_endpoint_v_research_20261126/manuscript.md) used five SWaT attack events for source gradients, four for source selection, and nominal SWaT/SMD windows. Its three-seed paired AP increments over reconstruction continuation on the full causal axis were +.03433 ± .00343 (PSM), −.00946 ± .00397 (SMD), +.01128 ± .00119 (MSL) and +.01665 ± .00217 (SMAP), given as mean ± sample SD. SMD contributed normal source data. SWaT supplied real labels. NASA used one sensor. Its V/E, supervised head, reconstruction and parameter-scope comparisons belong to that detector's [ablation and operating results](../native_endpoint_v_research_20261126/supplement.md); its U/F extensions did not establish stable net gains. These results remain a stage finding. No score transfers to the proposed row. Temporal lag, weak anomalies and robustness to mixed training remain distinct empirical questions.
