# Public multivariate anomaly detection after multi-source pretraining

## Research question

Can public source records teach a detector which local dynamics recur across sensors, and can the detector learn new normal dynamics from a target record that also contains anomalies? We study this question in two stages. A complete public detection benchmark establishes the performance that a proposed method must reach. Separate experiments examine zero target updates (Z), mixed-unlabeled learning (U), a fixed labeled support set (F), controlled extra abnormal exposure and the mechanism of the update.

The proposed detector learns short, frequency-specific trajectory subspaces across public source domains. During mixed target adaptation it estimates a normal multichannel trajectory and a sparse departure on the original timeline, then updates the subspaces from the estimated normal trajectory. At test time the residual from this trajectory produces a point score. The [method definition](method_definition.md) specifies the source folds, objective, solver, permissions and identifying controls. Its status is a design. The tables below contain measured external comparators.

## Public benchmark

Each method occupies one row within its training group. Panels A and B use different archived timestamp intersections. Panel C in the [supplement](README.md) covers scores available at the observation on fixed suffixes. Only compare values within a dataset, panel and metric; future-window context and target-normal training are additional information even on identical timestamps. Blank coverage is marked NA. All numerical cells in A–C come from local evaluations. Hybrid's proposed row has no scores.

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
| Statistical | History Mean512 | NA | NA | NA | NA | 0.3178 |
| Statistical | Level deviation | 0.4265 | 0.4383 | 0.7448 | 0.2880 | 0.2966 |
| Statistical | PCA residual | NA | 0.5344 | NA | NA | NA |
| Proposed | Adapted trajectory subspaces (design) | Pending | Pending | Pending | Pending | Pending |

**ROC-AUC**

| Training group | Method | SMD28 | PSM | SWaT | MSL27¹ | SMAP53¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target normal | CATCH | NA | 0.6474 ± 0.0060 | NA | NA | NA |
| Target normal | LEFT | 0.8787 | 0.7739 ± 0.0030 | 0.8177 ± 0.0045 | 0.7660 | 0.7610 |
| Pretrained | TimeRCD, FIT-only | 0.8594 | NA | NA | NA | NA |
| Pretrained | TimeRadar, W100 individual-window order | 0.8274 | NA | NA | 0.6709 | 0.6624 |
| Pretrained | TimeRadar, block0 batch order | NA | 0.6707 | 0.8220 | NA | NA |
| Statistical | History Mean512 | NA | NA | NA | NA | 0.6344 |
| Statistical | Level deviation | 0.8034 | 0.6542 | 0.8318 | 0.5530 | 0.5723 |
| Statistical | PCA residual | NA | 0.7933 | NA | NA | NA |
| Proposed | Adapted trajectory subspaces (design) | Pending | Pending | Pending | Pending | Pending |

**VUS-PR**

| Training group | Method | SMD28 | PSM | SWaT | MSL27¹ | SMAP53¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target normal | CATCH | NA | 0.5275 ± 0.0087 | NA | NA | NA |
| Target normal | LEFT | 0.5416 | 0.5674 ± 0.0073 | 0.5286 ± 0.0084 | 0.5486 | 0.5257 |
| Pretrained | TimeRCD, FIT-only | 0.5146 | NA | NA | NA | NA |
| Pretrained | TimeRadar, W100 individual-window order | 0.4508 | NA | NA | 0.3697 | 0.3055 |
| Pretrained | TimeRadar, block0 batch order | NA | 0.4413 | 0.5141 | NA | NA |
| Statistical | History Mean512 | NA | NA | NA | NA | 0.3360 |
| Statistical | Level deviation | 0.4255 | 0.4460 | 0.4645 | 0.3069 | 0.3021 |
| Statistical | PCA residual | NA | 0.5290 | NA | NA | NA |
| Proposed | Adapted trajectory subspaces (design) | Pending | Pending | Pending | Pending | Pending |

**VUS-ROC**

| Training group | Method | SMD28 | PSM | SWaT | MSL27¹ | SMAP53¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target normal | CATCH | NA | 0.6977 ± 0.0065 | NA | NA | NA |
| Target normal | LEFT | 0.9186 | 0.7817 ± 0.0047 | 0.7799 ± 0.0415 | 0.7965 | 0.7832 |
| Pretrained | TimeRCD, FIT-only | 0.9097 | NA | NA | NA | NA |
| Pretrained | TimeRadar, W100 individual-window order | 0.8825 | NA | NA | 0.7191 | 0.7045 |
| Pretrained | TimeRadar, block0 batch order | NA | 0.6784 | 0.7668 | NA | NA |
| Statistical | History Mean512 | NA | NA | NA | NA | 0.6865 |
| Statistical | Level deviation | 0.8667 | 0.6543 | 0.6443 | 0.6055 | 0.6136 |
| Statistical | PCA residual | NA | 0.7667 | NA | NA | NA |
| Proposed | Adapted trajectory subspaces (design) | Pending | Pending | Pending | Pending | Pending |

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

## Use conditions, contamination and mechanism

Establish the complete detector on the public benchmark before attributing gains. The source-excluded Z, separate U and labeled F lanes share one score. Report the zero-update comparator again on the same untouched target suffix where a comparison requires it. Controlled abnormal exposure compares fixed abnormal and nominal replacements of the same slots. Seek a net improvement in absolute detection and measure false alarms, matched events, delay and sustained coverage. Sparse deviation estimates are statistical residuals. Their existence supplies no physical fault-channel labels. The prior PSM mixed CAL was entirely attack-labeled; NASA small-sample empirical q99 can equal the maximum. The actual use of each CAL governs what can be inferred about normal false alarms.

## Prior evidence and its limits

The [frozen endpoint V study](../native_endpoint_v_research_20261126/manuscript.md) used five SWaT attack events for source gradients, four for source selection, and nominal SWaT/SMD windows. Its three-seed paired AP increments over reconstruction continuation on the full causal axis were +.03433 ± .00343 (PSM), −.00946 ± .00397 (SMD), +.01128 ± .00119 (MSL) and +.01665 ± .00217 (SMAP), given as mean ± sample SD. SMD contributed normal source data. SWaT supplied real labels. NASA used one sensor. Its V/E, supervised head, reconstruction and parameter-scope comparisons belong to that detector's [ablation and operating results](../native_endpoint_v_research_20261126/supplement.md); its U/F extensions did not establish stable net gains. These results remain a stage finding. No score transfers to the proposed row. Temporal lag, weak anomalies and robustness to mixed training remain distinct empirical questions.
