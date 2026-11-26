# Cross-domain pretraining of the complete LEFT detector

The 31-channel frozen transfer model achieved 0.7239 AP on the complete SWaT benchmark, ahead of target-normal LEFT (0.5439) and below pointwise level deviation (0.7448). Normal target training after transfer traded AP for ROC-AUC and VUS-ROC. On PSM, normal training improved the frozen model, yet the native from-scratch run remained ahead in all four metrics. On the independent mixed PSM suffix, replay from the source checkpoint ranked first among the four U comparisons. Random initialization was competitive on both suffixes. This is development evidence from one project training seed.

## Complete public test

| Method | PSM AP | PSM ROC-AUC | PSM VUS-PR | PSM VUS-ROC | SWAT AP | SWAT ROC-AUC | SWAT VUS-PR | SWAT VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Target-normal LEFT from scratch | 0.5766 | 0.7761 | 0.5726 | 0.7851 | 0.5439 | 0.8209 | 0.5346 | 0.8093 |
| Frozen multi-source LEFT (Z) | 0.4964 | 0.7011 | 0.5010 | 0.7065 | 0.7239 | 0.8115 | 0.4976 | 0.6611 |
| Multi-source LEFT + target-normal training (N) | 0.5544 | 0.7567 | 0.5520 | 0.7663 | 0.5425 | 0.8556 | 0.5310 | 0.8431 |
| Pretrained TimeRadar (native block grid) | 0.4333 | 0.6707 | 0.4413 | 0.6784 | 0.6393 | 0.8220 | 0.5141 | 0.7668 |
| CATCH (3 seeds) | 0.4364 ± 0.0084 | 0.6474 ± 0.0060 | 0.5275 ± 0.0087 | 0.6977 ± 0.0065 | NA | NA | NA | NA |
| PCA residual | 0.5344 | 0.7933 | 0.5290 | 0.7667 | NA | NA | NA | NA |
| Pointwise level deviation | 0.4383 | 0.6542 | 0.4460 | 0.6543 | 0.7448 | 0.8318 | 0.4645 | 0.6443 |

The public axes are [0,87744) on PSM (25 channels) and [495000,944856) on SWaT (31 channels). LEFT uses the target normal-prefix scaler including normal SELECT; Z uses those target statistics with zero target parameter updates. All LEFT scores use 192 real positions in a nonoverlapping window and become available at its end. Pointwise level deviation uses the same prefix scaler and the current observation. TimeRadar uses the released weights, archived W100 batched order and normal-prefix statistics. CATCH uses learned W192 target-normal reconstruction with FIT statistics. PCA fits target-normal data. CATCH reports mean ± sample SD over three archived training seeds. The local LEFT comparison uses seed 2025. [External results and permissions](../left_decoder_context_20261124/public_extension/methods.csv)

## Pretraining and target-normal learning

| Method | PSM AP | PSM ROC-AUC | PSM VUS-PR | PSM VUS-ROC | SWAT AP | SWAT ROC-AUC | SWAT VUS-PR | SWAT VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N − scratch_N | -0.0223 | -0.0194 | -0.0206 | -0.0188 | -0.0014 | 0.0347 | -0.0036 | 0.0339 |
| N − source | 0.0580 | 0.0556 | 0.0511 | 0.0598 | -0.1814 | 0.0441 | 0.0334 | 0.1820 |
| source − scratch_N | -0.0803 | -0.0750 | -0.0717 | -0.0786 | 0.1801 | -0.0093 | -0.0370 | -0.1482 |

Both source models were selected at step 4,000 on the independent source validation panel. Each consumed 64,000 TEP and 64,000 SMD windows (128,000 total). The 25-channel model ran 1,001 updates at full prototype mixing and the 31-channel model ran 2,501. FIT covered all 20 assigned TEP runs and all 28 SMD machines; SELECT used five separate TEP runs and disjoint subsequent SMD time intervals. The models have 4,519,985 and 3,742,202 registered parameters, respectively, reflecting the two existing domain-specific native configurations. No variable mapping assumes homologous sensors across datasets. Width-specific weights, including the original Et/Ef/Dt/Df projections, transfer directly to the matching target model. Source learning used numeric normal records alone.

N and target-normal training from scratch share data, archived scaler, optimizer, sampler, the 20-epoch ceiling and validation rules. Both N models completed 18 epochs and selected epoch 15. The archived from-scratch selections were epoch 20 on PSM and 13 on SWaT. Normal validation determines the actual update count. The inherited prototype state and curriculum clock belong to the complete initialization. [Training costs and checkpoint identities](training.csv) · [Source record accounting](source_exposure.csv) · [Method and roles](../../../manuscript/next_generation_20261115/method_definition.md)

## Natural mixed-target learning

| Method | PSM AP | PSM ROC-AUC | PSM VUS-PR | PSM VUS-ROC | SWAT AP | SWAT ROC-AUC | SWAT VUS-PR | SWAT VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen multi-source LEFT (Z) | 0.5043 | 0.7200 | 0.5159 | 0.7313 | 0.7914 | 0.8423 | 0.6307 | 0.7341 |
| Multi-source LEFT + ordinary U | 0.5770 | 0.7579 | 0.5850 | 0.7715 | 0.7800 | 0.8417 | 0.6332 | 0.7447 |
| Multi-source LEFT + source replay U | 0.5919 | 0.7556 | 0.5920 | 0.7717 | 0.7861 | 0.8420 | 0.6338 | 0.7403 |
| Random LEFT + ordinary U | 0.5809 | 0.7392 | 0.5743 | 0.7502 | 0.7882 | 0.8422 | 0.6338 | 0.7414 |

| Method | PSM AP | PSM ROC-AUC | PSM VUS-PR | PSM VUS-ROC | SWAT AP | SWAT ROC-AUC | SWAT VUS-PR | SWAT VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary − source | 0.0727 | 0.0379 | 0.0692 | 0.0402 | -0.0114 | -0.0007 | 0.0025 | 0.0106 |
| replay − source | 0.0876 | 0.0355 | 0.0762 | 0.0404 | -0.0052 | -0.0004 | 0.0031 | 0.0062 |
| ordinary − random_u | -0.0039 | 0.0186 | 0.0108 | 0.0213 | -0.0082 | -0.0006 | -0.0006 | 0.0033 |
| replay − ordinary | 0.0150 | -0.0023 | 0.0070 | 0.0001 | 0.0062 | 0.0003 | 0.0007 | -0.0044 |
| replay − random_u | 0.0111 | 0.0163 | 0.0178 | 0.0215 | -0.0020 | -0.0002 | 0.0000 | -0.0011 |

The U axes are PSM [22656,87744) and SWaT [608535,944856). Each U job uses 2,000 fresh Adam updates, batch 32. Ordinary U and random U each see the same 64,000 target-window presentations. Replay replaces the second half with source data, exposing 32,000 target windows and 32,000 source windows, equally split between TEP and SMD. Replay and ordinary U share the first 16 target slots at every step. Pretraining itself is isolated by the ordinary-versus-random U comparison, subject to the full initialization including its prototype curriculum state. Replay versus ordinary U measures substitution at fixed total window exposure. Comparing replay to random U combines both changes.

## Operating costs on the isolated mixed suffix

| Domain | Method | Stream | Normal FPR | Recall | Normal fragments | Matched/eligible | Any hit | Event F1 | Long macro | Long pooled | Longest miss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PSM | Frozen multi-source LEFT (Z) | strict | 0.1820 | 0.5030 | 566 | 32/48 | 36 | 0.0882 | 0.7643 | 0.8759 | 399 |
| PSM | Frozen multi-source LEFT (Z) | confirmed3 | 0.1627 | 0.4715 | 267 | 23/48 | 27 | 0.1237 | 0.7427 | 0.8659 | 399 |
| PSM | Multi-source LEFT + ordinary U | strict | 0.1539 | 0.5124 | 569 | 36/48 | 37 | 0.0923 | 0.7215 | 0.8797 | 397 |
| PSM | Multi-source LEFT + ordinary U | confirmed3 | 0.1339 | 0.4707 | 311 | 21/48 | 23 | 0.0938 | 0.6742 | 0.8595 | 397 |
| PSM | Multi-source LEFT + source replay U | strict | 0.1486 | 0.4981 | 566 | 37/48 | 38 | 0.0966 | 0.7022 | 0.8564 | 397 |
| PSM | Multi-source LEFT + source replay U | confirmed3 | 0.1289 | 0.4586 | 295 | 21/48 | 22 | 0.0977 | 0.6709 | 0.8424 | 397 |
| PSM | Random LEFT + ordinary U | strict | 0.2174 | 0.6035 | 702 | 38/48 | 39 | 0.0875 | 0.9172 | 0.9535 | 397 |
| PSM | Random LEFT + ordinary U | confirmed3 | 0.1924 | 0.5626 | 384 | 26/48 | 27 | 0.1032 | 0.8888 | 0.9387 | 397 |
| SWAT | Frozen multi-source LEFT (Z) | strict | 0.7242 | 0.9212 | 216 | 9/22 | 21 | 0.0706 | 0.6140 | 0.9695 | 721 |
| SWAT | Frozen multi-source LEFT (Z) | confirmed3 | 0.7230 | 0.9200 | 111 | 9/22 | 21 | 0.1385 | 0.6052 | 0.9684 | 721 |
| SWAT | Multi-source LEFT + ordinary U | strict | 0.5407 | 0.8023 | 1755 | 14/22 | 19 | 0.0135 | 0.3067 | 0.8624 | 2016 |
| SWAT | Multi-source LEFT + ordinary U | confirmed3 | 0.5327 | 0.7948 | 541 | 13/22 | 18 | 0.0434 | 0.2727 | 0.8585 | 2018 |
| SWAT | Multi-source LEFT + source replay U | strict | 0.6332 | 0.9178 | 912 | 10/22 | 19 | 0.0216 | 0.5813 | 0.9656 | 998 |
| SWAT | Multi-source LEFT + source replay U | confirmed3 | 0.6291 | 0.9175 | 270 | 10/22 | 19 | 0.0712 | 0.5783 | 0.9654 | 1000 |
| SWAT | Random LEFT + ordinary U | strict | 0.6759 | 0.9181 | 1007 | 10/22 | 20 | 0.0196 | 0.5939 | 0.9663 | 994 |
| SWAT | Random LEFT + ordinary U | confirmed3 | 0.6709 | 0.9177 | 193 | 10/22 | 20 | 0.0976 | 0.5885 | 0.9660 | 999 |


Operating points use the fixed 512 mixed-CAL positions per dataset and a separate q99/higher threshold for each model, with strict exceedance. PSM has 512 attack-labeled CAL points; SWaT has 38. Neither threshold certifies a 1% normal FPR. Confirmed-three requires three consecutive exceedances without backfilling. A long event exceeds 512 points and its steady coverage starts after point 512. One-to-one matching and any hit have separate columns. High point recall can coexist with few matched events when a long alarm spans several events. All results have window-end retrospective availability.

**Decision.** Complete cross-domain LEFT initialization is useful for selected outcomes, particularly frozen SWaT AP and the ranking benefit of the PSM replay procedure. The ordinary-U comparison gives no AP advantage over random U in either domain. PSM replay improves AP against random U (+0.0111) but matches 21/48 events after confirmation, compared with 26/48 for random U and 23/48 for Z. SWaT replay improves AP against ordinary U (+0.0062) but trails Z and random U. Ordinary SWaT U improves the number of confirmed event matches from 9 to 13 of 22 and reduces FPR, while long-event macro coverage falls from 0.6052 to 0.2727. The baseline warrants retention for future matched learning research. A general gain from pretrained mixed adaptation, robustness to contamination, and a new detection mechanism remain unestablished.

**Execution.** The two source fits, two target-normal continuation fits and six mixed fits took 5,095.9 summed A40 job-seconds. Twenty public or role-specific inference banks took 25.8 further seconds. This totals 1.42 A40 job-hours across jobs; CPU evaluation and prior native training are excluded. The previously reproduced native from-scratch checkpoints and external comparisons were reused. A target-scaler check in the first SWaT random-U launch encountered last-bit rounding differences before its first update; the successful job loaded the archived scaler values. The PSM random-U values were independently confirmed identical in float32 to the archived-scaler inputs. [Bank identities and inference costs](banks.json) · [Independent selection and overlap checks](checks.json)

All fourteen method/domain ranking rows were checked by a second sorted-score AP and rank-based ROC calculation (absolute tolerance 1e-11). VUS used the pinned LEFT implementation with windowSize 100 and 250 thresholds and was not recomputed independently. All ten new checkpoint hashes and twenty bank hashes were checked after local transfer. Full scores, timestamps, checkpoints and logs reside under `results/left_multisource_20261126_v1` in the data workspace. [PSM operations](PSM/operations.csv) · [PSM events](PSM/events.csv) · [SWaT operations](SWAT/operations.csv) · [SWaT events](SWAT/events.csv)

The [runner](../../../scripts/run_left_multisource_target.sh) accepts the dataset name and two GPU indices from the data workspace, with PYTHONPATH set to the project's src directory. Source learning, normal-target training, mixed fitting and scoring are independently callable. [Full-precision within-axis differences](contrasts.csv)
