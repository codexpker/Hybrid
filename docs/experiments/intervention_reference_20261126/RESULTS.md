# Synthetic intervention supervision of native TimeRadar variance

Source pairing raises AP over the released checkpoint by 0.00752 on PSM and 0.01388 on SWaT. Against the matched normal reconstruction control, the corresponding differences are −0.00006 and +0.00043. SWaT retains small gains in both VUS measures over reconstruction. PSM loses ROC-AUC and VUS-PR. The mixed-target local reference lowers AP relative to its frozen source model on both tasks. Its direct comparison with the absolute reference changes sign between tasks and stays within 0.0015 AP. This single development run provides little support for a detection advantage attributable to the local reference.

## Complete public test

| Detector | PSM AP | PSM ROC-AUC | PSM VUS-PR | PSM VUS-ROC | SWAT AP | SWAT ROC-AUC | SWAT VUS-PR | SWAT VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Official TimeRadar | 0.4481 | 0.6785 | 0.4501 | 0.6806 | 0.6423 | 0.8227 | 0.5061 | 0.7550 |
| Source reconstruction | 0.4557 | 0.6857 | 0.4490 | 0.6718 | 0.6558 | 0.8269 | 0.5133 | 0.7510 |
| Source intervention pairs (Z) | 0.4556 | 0.6814 | 0.4476 | 0.6720 | 0.6562 | 0.8278 | 0.5162 | 0.7555 |
| Native LEFT | 0.5734 ± 0.0046 | 0.7739 ± 0.0030 | 0.5674 ± 0.0073 | 0.7817 ± 0.0047 | 0.5650 ± 0.0299 | 0.8177 ± 0.0045 | 0.5286 ± 0.0084 | 0.7799 ± 0.0415 |
| Pointwise level deviation | 0.4383 | 0.6542 | 0.4460 | 0.6543 | 0.7448 | 0.8318 | 0.4645 | 0.6443 |
| CATCH W192 | 0.4364 ± 0.0084 | 0.6474 ± 0.0060 | 0.5275 ± 0.0087 | 0.6977 ± 0.0065 | NA | NA | NA | NA |
| PCA residual | 0.5344 | 0.7933 | 0.5290 | 0.7667 | NA | NA | NA | NA |

The public rows use PSM [0,87744) and SWaT [495000,944856). New TimeRadar banks use 25 or 31 channels, target normal FIT statistics and W100 offline scores available at the window end. The source corpus consists of TEP fault-free training and SMD normal training. Both domains receive equal window counts. LEFT uses normal target training, normal validation and a scaler including validation. LEFT and CATCH use W192 offline outputs. PCA fits target-normal data. Pointwise level deviation uses the LEFT scaler and current observation. Archived LEFT and CATCH show mean ± sample SD over two and three training seeds respectively. All new results use seed 2025. [External cache provenance](../left_decoder_context_20261124/public_extension/methods.csv)

## Independent mixed-target comparison

| Adaptation | PSM AP | PSM ROC-AUC | PSM VUS-PR | PSM VUS-ROC | SWAT AP | SWAT ROC-AUC | SWAT VUS-PR | SWAT VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Source intervention pairs (Z) | 0.4613 | 0.6982 | 0.4532 | 0.6804 | 0.7229 | 0.8528 | 0.6532 | 0.8228 |
| Ordinary reconstruction replay | 0.4670 | 0.7069 | 0.4520 | 0.6674 | 0.7061 | 0.8599 | 0.6638 | 0.8440 |
| Absolute intervention reference | 0.4469 | 0.6940 | 0.4470 | 0.6821 | 0.7040 | 0.8522 | 0.6488 | 0.8244 |
| Local intervention reference | 0.4454 | 0.6935 | 0.4465 | 0.6824 | 0.7049 | 0.8522 | 0.6484 | 0.8240 |

U-FIT uses the contiguous target records PSM [0,17568) and SWaT [495000,585000). The suffix evaluation uses PSM [22656,87744) and SWaT [608535,944856). Z has byte-identical scores on the public and isolated role banks at shared positions. All U arms start from the same source-pair decoder and use fixed step 120. Relative versus absolute is the matched reference ablation. The comparison with replay also changes losses and synthetic exposure. [Role intervals and scaler values](inputs.json)

## Within-axis differences

| Difference | PSM AP | PSM ROC-AUC | PSM VUS-PR | PSM VUS-ROC | SWAT AP | SWAT ROC-AUC | SWAT VUS-PR | SWAT VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pair − official | 0.0075 | 0.0030 | -0.0025 | -0.0087 | 0.0139 | 0.0051 | 0.0101 | 0.0006 |
| pair − reconstruction | -0.0001 | -0.0043 | -0.0014 | 0.0002 | 0.0004 | 0.0008 | 0.0030 | 0.0045 |
| replay − pair | 0.0056 | 0.0088 | -0.0011 | -0.0129 | -0.0168 | 0.0072 | 0.0106 | 0.0213 |
| absolute − pair | -0.0144 | -0.0041 | -0.0061 | 0.0018 | -0.0189 | -0.0006 | -0.0044 | 0.0016 |
| relative − pair | -0.0159 | -0.0046 | -0.0066 | 0.0020 | -0.0180 | -0.0005 | -0.0048 | 0.0013 |
| relative − absolute | -0.0014 | -0.0005 | -0.0005 | 0.0003 | 0.0009 | 0.0001 | -0.0004 | -0.0003 |
| relative − replay | -0.0215 | -0.0134 | -0.0055 | 0.0150 | -0.0012 | -0.0077 | -0.0154 | -0.0200 |

## CAL operating points

| Task | Method | Stream | FPR | Recall | Normal fragments | Matched | Events | Any hit | Event F1 | Coverage | Long macro | Long pooled | Longest miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PSM | Source intervention pairs (Z) | strict | 0.1357 | 0.3321 | 961 | 23 | 48 | 24 | 0.0307 | 0.5205 | 0.5487 | 0.6672 | 473 |
| PSM | Source intervention pairs (Z) | confirmed3 | 0.0984 | 0.2697 | 704 | 17 | 48 | 18 | 0.0311 | 0.4322 | 0.4453 | 0.5634 | 656 |
| PSM | Ordinary reconstruction replay | strict | 0.1581 | 0.3603 | 1082 | 25 | 48 | 25 | 0.0318 | 0.5767 | 0.5920 | 0.7280 | 399 |
| PSM | Ordinary reconstruction replay | confirmed3 | 0.1171 | 0.3036 | 681 | 16 | 48 | 16 | 0.0305 | 0.4940 | 0.4890 | 0.6340 | 675 |
| PSM | Absolute intervention reference | strict | 0.1416 | 0.3458 | 1029 | 25 | 48 | 26 | 0.0317 | 0.5365 | 0.5676 | 0.6891 | 473 |
| PSM | Absolute intervention reference | confirmed3 | 0.1025 | 0.2825 | 743 | 17 | 48 | 18 | 0.0300 | 0.4478 | 0.4620 | 0.5852 | 555 |
| PSM | Local intervention reference | strict | 0.1396 | 0.3402 | 1023 | 24 | 48 | 25 | 0.0303 | 0.5274 | 0.5575 | 0.6779 | 473 |
| PSM | Local intervention reference | confirmed3 | 0.1007 | 0.2757 | 721 | 17 | 48 | 18 | 0.0305 | 0.4373 | 0.4525 | 0.5712 | 555 |
| SWAT | Source intervention pairs (Z) | strict | 0.4432 | 0.8449 | 10945 | 21 | 22 | 21 | 0.0036 | 0.8449 | 0.4613 | 0.9114 | 721 |
| SWAT | Source intervention pairs (Z) | confirmed3 | 0.3765 | 0.8196 | 7305 | 21 | 22 | 21 | 0.0054 | 0.8196 | 0.4001 | 0.8954 | 721 |
| SWAT | Ordinary reconstruction replay | strict | 0.4428 | 0.8580 | 13087 | 22 | 22 | 22 | 0.0032 | 0.8580 | 0.5878 | 0.9119 | 127 |
| SWAT | Ordinary reconstruction replay | confirmed3 | 0.3644 | 0.8319 | 7495 | 22 | 22 | 22 | 0.0055 | 0.8319 | 0.5109 | 0.8966 | 170 |
| SWAT | Absolute intervention reference | strict | 0.3528 | 0.8068 | 14764 | 21 | 22 | 21 | 0.0027 | 0.8068 | 0.3431 | 0.8868 | 721 |
| SWAT | Absolute intervention reference | confirmed3 | 0.2616 | 0.7769 | 10146 | 21 | 22 | 21 | 0.0039 | 0.7769 | 0.2889 | 0.8677 | 1016 |
| SWAT | Local intervention reference | strict | 0.3602 | 0.8088 | 14435 | 21 | 22 | 21 | 0.0028 | 0.8088 | 0.3458 | 0.8882 | 721 |
| SWAT | Local intervention reference | confirmed3 | 0.2710 | 0.7796 | 10060 | 21 | 22 | 21 | 0.0040 | 0.7796 | 0.2918 | 0.8693 | 1016 |


Each threshold uses the method's own 512 historic mixed-CAL scores, q99/higher and strict exceedance. PSM has 512 labeled attack points and SWaT has 38 on this CAL grid. These empirical operating points provide no normal-FPR guarantee. Confirmed-three alarms require consecutive exceedances without backfilling. Coverage pools alarmed points over complete eligible events; long macro and pooled coverage count positions after the first 512 of a long event. Matching uses chronological one-to-one overlap. Scores have offline availability. [Full operating metrics](operating.csv) · [Individual events](events.csv)

## What the intervention changed

Source reconstruction and source pairing ran for 120 updates on identical original source windows. The minimum held-out synthetic pair risk was 1.27231 at step 100 for reconstruction and 1.23979 at step 120 for pairing, from 1.27967 at initialization. Each run saw 480 nominal source-window presentations. The source-pair arm additionally used their edited counterparts. Each target job saw 480 original source and 480 original target windows. The two intervention arms also evaluated the corresponding edits and the frozen reference. This adds computation compared with ordinary replay. [Source selection](source_select.csv) · [Optimization traces](learning.csv)

A uniform raw-score translation of seven FIT scale units changed the paired loss by zero to double precision in the fixed numerical checks; the original-window anchor incurred positive cost. On the fixed unlabeled target probe the local arm's mean edited-minus-original variance for amplitude offsets increased from 0.08608 to 0.23773 on PSM and from 0.01908 to 0.05973 on SWaT. For local time shifts the corresponding adapted mean differences were +0.00008 and −0.00020. All time-shift probe windows had changed input values. On real QUERY windows local U shifted scores upward on average (+0.02417 and +0.02652), with SDs of the within-point changes of 0.08058 and 0.11728. Rank correlations with Z were 0.9919 and 0.9877. The model learned a differential response to artificial offsets alongside a drift in absolute score level. These observations explain why success on edited inputs alone is insufficient evidence of transfer to real anomalies. [Paired input and score measurements](edits.csv) · [Shift checks](checks.json) · [QUERY scores summary](QUERY.csv)

Against Z, local U raised PSM three-confirmation recall from 0.2697 to 0.2757, with 17 matched events in both cases and more normal alarm fragments (704 to 721). On SWaT, three-confirmation FPR fell from 0.3765 to 0.2710, with 21 of 22 matched events in both cases. Recall fell from 0.8196 to 0.7796, long-event macro coverage fell from 0.4001 to 0.2918 and normal fragments rose from 7,305 to 10,060. There is an operating-point trade-off as well as a loss in AP. No controlled contamination effect was measured.

## Execution and verification

On A40 hardware the eight training jobs consumed 316.8 summed device-job seconds, including initialization and source validation. The six role-bank jobs consumed a further 249.9 seconds, and three input/translation checks took 22.4 seconds. The sum is 0.164 A40 job-hours. These are elapsed times within jobs that can run concurrently. The decoder and score archives are available in the project data workspace under `results/intervention_reference_20261126_v1`. All eight decoder hashes and six score-bank hashes were verified after copying. Independent sorted-score AP and rank-based ROC reproduced every new ranking row within 1e-11. VUS used the pinned native LEFT metric with windowSize 100 and 250 thresholds, with no independent VUS implementation in this review. [Weights and costs](weights.csv) · [Bank hashes](banks.json) · [Public metric check](TEST_review.json) · [Mixed metric check](QUERY_review.json)

Run the fixed queue from the data workspace with PYTHONPATH pointing to this project's src directory, OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Pass the pinned metric source path as the first argument to [the runner](../../../scripts/run_intervention_reference.sh). All predictions precede label-based evaluation for their comparison; external method scores were reused from the archived common axis.
