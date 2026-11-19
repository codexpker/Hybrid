# Frozen TimeRadar response geometry: completed fixed A/B verification

Status: **completed, independently recomputed, no expansion of this candidate**. Public SWaT31 / PSM25, seed2025. A is the implemented Hybrid candidate (raw + frozen official response energy); B is its matched raw-only conditional generator. A/B here are model arms, not historical contamination conditions. All weights, source selection and16 score banks were frozen before opening QUERY labels. No first-paper/private data, new dose, third arm, SMD or additional seed was used.

## Technical decision

The frozen-response increment does **not** establish independent ranking benefits that survive natural mixed adaptation. SWaT A−B AP is −1.0594pp before adaptation and −7.8658pp after R. PSM is +0.0165pp and −0.5707pp. PSM's negligible noU difference brings no C3 event or complete-event coverage gain. This fixed candidate is not promoted to a paper contribution or expanded automatically.

Retain the bounded SWaT noU operating trade-off: A lowers strict normal FPR from31.2462% to1.4192%, and C3 FPR from16.8674% to0.8917%, while C3 complete-event coverage falls74.7933%→63.8676% and matched events17→14/22. Long steady coverage is9.9598% macro /78.0624% pooled for A, versus21.3215% /85.3775% for B. These are their own mixed-CAL operating points, **not equal normal-FPR budgets or a proved calibration mechanism**; high pooled coverage does not resolve the weak event macro coverage.

After R, SWaT A AP falls.651273→.330814 (B:.661866→.409471). A C3 coverage falls63.8676%→1.0486%, long macro9.9598%→.1809%, pooled78.0624%→.1238%, longest event miss3931→31850 points. A_R matches13 events versus B_R11, but has lower event F1, slightly more C3 false alarms and almost-zero sustained coverage. This limited event-count positive cannot override ranking and duration losses. PSM R modestly improves both arms' ranking but reduces coverage; A does not improve on B.

Threshold-free AP/ROC deterioration in SWaT means a CAL threshold alone cannot explain the loss. The experiment does not identify whether altered learned conditional distributions, abnormal history becoming predictable, or another representation/learning property is responsible. It has no new TT control, so R−noU cannot isolate replay from additional updates. Historical LEFT R−TT or pollution protection is not inherited by this generator. Do not resume stopped scoring, masking, shallow adapters or tune this geometry from QUERY results. No new candidate is launched here; the general public-data/two-stage research goal is unchanged.

## Table1a: complete common-axis ranking comparison

AP, ROC and VUS are fractions, unadjusted. SWaT QUERY[608535,944856),336321 points; PSM[23016,87744),64728 points. The360 additional PSM points were removed only for513-point input legality, identically for all methods. Old full-axis tables are preserved. VUS uses the pinned official LEFT implementation `RangeAUC_volume_opt(windowSize=100,thre=250)`; AP is not trapezoidal PR-AUC or VUS-PR.

| Domain | Method | AP | ROC-AUC | VUS-PR | VUS-ROC | Permission |
| --- | --- | --- | --- | --- | --- | --- |
| SWAT | A_noU | 0.6513 | 0.8311 | 0.6539 | 0.8326 | CAUSAL_ENDPOINT |
| SWAT | B_noU | 0.6619 | 0.8324 | 0.6640 | 0.8346 | CAUSAL_ENDPOINT |
| SWAT | A_R | 0.3308 | 0.7896 | 0.3387 | 0.7970 | CAUSAL_ENDPOINT |
| SWAT | B_R | 0.4095 | 0.8138 | 0.4201 | 0.8211 | CAUSAL_ENDPOINT |
| SWAT | LEFT_noU | 0.6076 | 0.8423 | 0.6063 | 0.8296 | OFFLINE_CONTEXT |
| SWAT | LEFT_TT | 0.6103 | 0.8846 | 0.6183 | 0.8816 | OFFLINE_CONTEXT |
| SWAT | LEFT_R | 0.6293 | 0.8855 | 0.6393 | 0.8845 | OFFLINE_CONTEXT |
| SWAT | MLP_noU | 0.7839 | 0.8562 | 0.5436 | 0.6969 | CAUSAL_ENDPOINT |
| SWAT | MLP_TT | 0.7761 | 0.8452 | 0.5145 | 0.6657 | CAUSAL_ENDPOINT |
| SWAT | MLP_R | 0.7820 | 0.8633 | 0.5418 | 0.7026 | CAUSAL_ENDPOINT |
| SWAT | CANDI_A | 0.7839 | 0.8570 | 0.5417 | 0.6957 | CAUSAL_ENDPOINT |
| SWAT | TimeRadar_native | NA | NA | NA | NA | No compatible31-channel bank |
| SWAT | HistoryMean512 | 0.0877 | 0.2278 | 0.0909 | 0.2327 | CAUSAL_ENDPOINT |
| PSM | A_noU | 0.3574 | 0.5777 | 0.3922 | 0.6200 | CAUSAL_ENDPOINT |
| PSM | B_noU | 0.3572 | 0.5775 | 0.3920 | 0.6199 | CAUSAL_ENDPOINT |
| PSM | A_R | 0.3641 | 0.5931 | 0.3974 | 0.6343 | CAUSAL_ENDPOINT |
| PSM | B_R | 0.3698 | 0.6023 | 0.4029 | 0.6430 | CAUSAL_ENDPOINT |
| PSM | LEFT_noU | 0.5866 | 0.7770 | 0.5777 | 0.7808 | OFFLINE_CONTEXT |
| PSM | LEFT_TT | 0.5778 | 0.7444 | 0.5670 | 0.7506 | OFFLINE_CONTEXT |
| PSM | LEFT_R | 0.5932 | 0.7595 | 0.5907 | 0.7719 | OFFLINE_CONTEXT |
| PSM | MLP_noU | 0.6065 | 0.8151 | 0.6278 | 0.8233 | CAUSAL_ENDPOINT |
| PSM | MLP_TT | 0.4924 | 0.7630 | 0.5389 | 0.7836 | CAUSAL_ENDPOINT |
| PSM | MLP_R | 0.6358 | 0.8281 | 0.6855 | 0.8522 | CAUSAL_ENDPOINT |
| PSM | CANDI_A | 0.6065 | 0.8151 | 0.6278 | 0.8233 | CAUSAL_ENDPOINT |
| PSM | TimeRadar_native | 0.4305 | 0.6884 | 0.4377 | 0.6775 | OFFLINE_CONTEXT |
| PSM | HistoryMean512 | 0.3037 | 0.5256 | 0.3324 | 0.5679 | CAUSAL_ENDPOINT |

## Table1b: complete method operating results

All rates and coverage are percentages. Each method uses its **own newly computed** q99/higher/strict> threshold on the same512 legal mixed-CAL points. SWaT34/512 and PSM512/512 selected CAL labels are anomalous (post-evaluation description only). These are not certified1% normal-FPR points. Old thresholds are never inherited. Rank-only TimeRadar PSM and HistoryMean SWaT have **NA** operating metrics; no compatible SWaT TimeRadar bank exists. Missing values are not zeros.

`M` is chronological greedy one-to-one matched events; any-hit is separate. `cov` is coverage of complete eligible events, distinct from recall on every positive QUERY point. Long events last>512; steady starts512 points after the original start. Both macro and pooled denominators are retained (SWaT9 events/38769 steady points; PSM5/6150). Delay is hit-only median in points with censored-event count, not an all-event average. For retrospective LEFT it is the score-availability offset, **not an online discovery delay**. C3 requires three consecutive above-threshold arrivals; no point adjustment.

| Domain | Method | Stream | Normal FPR% | FP points | Normal fragments | Recall% | M/events | Any-hit/events | Event P% | Event R% | Event F1% | Complete-event cov% | Long macro% | Long pooled% | Longest miss(points) | Hit-only offset(points) | Censored | Own threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWAT | A_noU | strict | 1.4192 | 4095 | 1230 | 66.1071 | 21/22 | 21/22 | 1.0978 | 95.4545 | 2.1705 | 66.1071 | 11.0996 | 80.2316 | 2042 | 38.0000 | 1 | 0.1838 |
| SWAT | A_noU | confirmed3 | 0.8917 | 2573 | 164 | 63.8676 | 14/22 | 14/22 | 3.4568 | 63.6364 | 6.5574 | 63.8676 | 9.9598 | 78.0624 | 3931 | 42.0000 | 8 | 0.1838 |
| SWAT | B_noU | strict | 31.2462 | 90159 | 27985 | 76.5724 | 22/22 | 22/22 | 0.0769 | 100.0000 | 0.1538 | 76.5724 | 24.0341 | 85.7747 | 991 | 0.0000 | 0 | 0.1222 |
| SWAT | B_noU | confirmed3 | 16.8674 | 48670 | 7796 | 74.7933 | 17/22 | 17/22 | 0.2146 | 77.2727 | 0.4280 | 74.7933 | 21.3215 | 85.3775 | 2036 | 1.0000 | 5 | 0.1222 |
| SWAT | A_R | strict | 0.7812 | 2254 | 1423 | 1.2684 | 21/22 | 21/22 | 1.3926 | 95.4545 | 2.7451 | 1.2684 | 0.9987 | 0.2270 | 31473 | 38.0000 | 1 | 0.1274 |
| SWAT | A_R | confirmed3 | 0.1688 | 487 | 196 | 1.0486 | 13/22 | 13/22 | 6.1611 | 59.0909 | 11.1588 | 1.0486 | 0.1809 | 0.1238 | 31850 | 43.0000 | 9 | 0.1274 |
| SWAT | B_R | strict | 1.1028 | 3182 | 2461 | 1.3668 | 21/22 | 21/22 | 0.8092 | 95.4545 | 1.6049 | 1.3668 | 1.0708 | 0.2683 | 13482 | 30.0000 | 1 | 0.1051 |
| SWAT | B_R | confirmed3 | 0.1456 | 420 | 106 | 1.0402 | 11/22 | 11/22 | 9.3220 | 50.0000 | 15.7143 | 1.0402 | 0.0257 | 0.1161 | 31849 | 40.0000 | 11 | 0.1051 |
| SWAT | LEFT_noU | strict | 0.9884 | 2852 | 223 | 1.5133 | 13/22 | 13/22 | 5.3942 | 59.0909 | 9.8859 | 1.5133 | 1.0415 | 0.3250 | 31840 | 152.0000 | 9 | 0.3693 |
| SWAT | LEFT_noU | confirmed3 | 0.8418 | 2429 | 194 | 1.4317 | 12/22 | 12/22 | 5.6872 | 54.5455 | 10.3004 | 1.4317 | 0.8550 | 0.2915 | 31840 | 151.0000 | 10 | 0.3693 |
| SWAT | LEFT_TT | strict | 0.9395 | 2711 | 219 | 1.2684 | 13/22 | 13/22 | 5.4852 | 59.0909 | 10.0386 | 1.2684 | 1.0233 | 0.1754 | 31861 | 152.0000 | 9 | 0.3764 |
| SWAT | LEFT_TT | confirmed3 | 0.7964 | 2298 | 187 | 1.1930 | 12/22 | 12/22 | 5.8824 | 54.5455 | 10.6195 | 1.1930 | 0.8374 | 0.1470 | 31861 | 151.0000 | 10 | 0.3764 |
| SWAT | LEFT_R | strict | 0.9375 | 2705 | 226 | 1.3207 | 13/22 | 13/22 | 5.3498 | 59.0909 | 9.8113 | 1.3207 | 1.0227 | 0.1702 | 31862 | 152.0000 | 9 | 0.3729 |
| SWAT | LEFT_R | confirmed3 | 0.7888 | 2276 | 197 | 1.2433 | 12/22 | 12/22 | 5.6604 | 54.5455 | 10.2564 | 1.2433 | 0.8368 | 0.1419 | 31862 | 151.0000 | 10 | 0.3729 |
| SWAT | MLP_noU | strict | 97.1862 | 280425 | 63 | 98.0242 | 4/22 | 21/22 | 8.1633 | 18.1818 | 11.2676 | 98.0242 | 88.8889 | 99.4609 | 721 | 0.0000 | 1 | 0.0771 |
| SWAT | MLP_noU | confirmed3 | 97.1564 | 280339 | 60 | 98.0032 | 4/22 | 21/22 | 8.6957 | 18.1818 | 11.7647 | 98.0032 | 88.8889 | 99.4609 | 721 | 0.0000 | 1 | 0.0771 |
| SWAT | MLP_TT | strict | 97.0573 | 280053 | 63 | 98.0535 | 5/22 | 21/22 | 10.4167 | 22.7273 | 14.2857 | 98.0535 | 88.8889 | 99.4609 | 721 | 0.0000 | 1 | 0.0509 |
| SWAT | MLP_TT | confirmed3 | 97.0275 | 279967 | 59 | 98.0367 | 5/22 | 21/22 | 11.3636 | 22.7273 | 15.1515 | 98.0367 | 88.8889 | 99.4609 | 721 | 0.0000 | 1 | 0.0509 |
| SWAT | MLP_R | strict | 96.6116 | 278767 | 80 | 98.0262 | 5/22 | 21/22 | 7.6923 | 22.7273 | 11.4943 | 98.0262 | 88.8889 | 99.4609 | 721 | 0.0000 | 1 | 0.0539 |
| SWAT | MLP_R | confirmed3 | 96.5700 | 278647 | 71 | 98.0095 | 5/22 | 21/22 | 8.9286 | 22.7273 | 12.8205 | 98.0095 | 88.8889 | 99.4609 | 721 | 0.0000 | 1 | 0.0539 |
| SWAT | CANDI_A | strict | 97.5016 | 281335 | 61 | 98.0137 | 4/22 | 21/22 | 8.5106 | 18.1818 | 11.5942 | 98.0137 | 88.8889 | 99.4609 | 721 | 0.0000 | 1 | 0.0588 |
| SWAT | CANDI_A | confirmed3 | 97.4742 | 281256 | 54 | 97.9928 | 4/22 | 21/22 | 10.2564 | 18.1818 | 13.1148 | 97.9928 | 88.8889 | 99.4609 | 721 | 0.0000 | 1 | 0.0588 |
| PSM | A_noU | strict | 2.3632 | 1118 | 946 | 7.5492 | 41/48 | 41/48 | 3.1179 | 85.4167 | 6.0161 | 5.7703 | 4.5320 | 3.3171 | 399 | 1.0000 | 7 | 0.2928 |
| PSM | A_noU | confirmed3 | 0.1057 | 50 | 31 | 1.4410 | 17/48 | 17/48 | 25.3731 | 35.4167 | 29.5652 | 1.1749 | 0.4997 | 0.3902 | 3137 | 8.0000 | 31 | 0.2928 |
| PSM | B_noU | strict | 2.3759 | 1124 | 951 | 7.5435 | 41/48 | 41/48 | 3.1037 | 85.4167 | 5.9898 | 5.7798 | 4.5412 | 3.3496 | 399 | 1.0000 | 7 | 0.2913 |
| PSM | B_noU | confirmed3 | 0.1057 | 50 | 31 | 1.4352 | 17/48 | 17/48 | 25.3731 | 35.4167 | 29.5652 | 1.1749 | 0.4997 | 0.3902 | 3137 | 8.0000 | 31 | 0.2913 |
| PSM | A_R | strict | 2.0038 | 948 | 824 | 6.5561 | 41/48 | 41/48 | 3.5621 | 85.4167 | 6.8390 | 4.8418 | 3.3948 | 2.6016 | 477 | 2.0000 | 7 | 0.3247 |
| PSM | A_R | confirmed3 | 0.0613 | 29 | 19 | 1.1654 | 14/48 | 14/48 | 29.1667 | 29.1667 | 29.1667 | 0.8528 | 0.3317 | 0.2276 | 3693 | 9.0000 | 34 | 0.3247 |
| PSM | B_R | strict | 2.0208 | 956 | 832 | 6.6077 | 41/48 | 41/48 | 3.5254 | 85.4167 | 6.7713 | 4.9555 | 3.5103 | 2.7480 | 477 | 2.0000 | 7 | 0.3217 |
| PSM | B_R | confirmed3 | 0.0613 | 29 | 19 | 1.2056 | 14/48 | 14/48 | 28.0000 | 29.1667 | 28.5714 | 0.9096 | 0.3455 | 0.2764 | 3693 | 9.0000 | 34 | 0.3217 |
| PSM | LEFT_noU | strict | 18.6920 | 8843 | 489 | 49.5953 | 34/48 | 38/48 | 5.3797 | 70.8333 | 10.0000 | 73.5361 | 80.0022 | 89.7398 | 399 | 86.0000 | 10 | 0.4873 |
| PSM | LEFT_noU | confirmed3 | 17.0264 | 8055 | 251 | 45.4676 | 20/48 | 24/48 | 6.0241 | 41.6667 | 10.5263 | 70.8926 | 76.4067 | 88.1301 | 399 | 94.0000 | 24 | 0.4873 |
| PSM | LEFT_TT | strict | 26.8744 | 12714 | 380 | 67.2369 | 36/48 | 42/48 | 8.1633 | 75.0000 | 14.7239 | 83.8734 | 98.1564 | 98.9268 | 363 | 94.0000 | 6 | 0.4288 |
| PSM | LEFT_TT | confirmed3 | 25.6336 | 12127 | 201 | 64.7109 | 22/48 | 28/48 | 9.2050 | 45.8333 | 15.3310 | 82.4427 | 97.3743 | 98.4715 | 399 | 86.0000 | 20 | 0.4288 |
| PSM | LEFT_R | strict | 17.0475 | 8065 | 557 | 53.9698 | 35/48 | 36/48 | 4.8276 | 72.9167 | 9.0556 | 75.3837 | 84.2260 | 92.0325 | 399 | 82.0000 | 12 | 0.4417 |
| PSM | LEFT_R | confirmed3 | 15.1515 | 7168 | 281 | 50.0144 | 21/48 | 22/48 | 5.4404 | 43.7500 | 9.6774 | 72.3043 | 79.9751 | 89.9675 | 399 | 86.0000 | 26 | 0.4417 |
| PSM | MLP_noU | strict | 4.3269 | 2047 | 137 | 32.5679 | 29/48 | 31/48 | 16.0221 | 60.4167 | 25.3275 | 50.2653 | 25.5283 | 73.0081 | 1000 | 1.0000 | 17 | 0.2697 |
| PSM | MLP_noU | confirmed3 | 3.8830 | 1837 | 99 | 31.3566 | 20/48 | 21/48 | 15.5039 | 41.6667 | 22.5989 | 49.1662 | 23.8174 | 72.3902 | 1000 | 4.0000 | 27 | 0.2697 |
| PSM | MLP_TT | strict | 3.4940 | 1653 | 196 | 11.6884 | 31/48 | 33/48 | 8.1365 | 64.5833 | 14.4522 | 15.4254 | 16.5123 | 14.8293 | 475 | 1.0000 | 15 | 0.2229 |
| PSM | MLP_TT | confirmed3 | 2.8113 | 1330 | 142 | 9.0820 | 20/48 | 21/48 | 7.8431 | 41.6667 | 13.2013 | 12.1281 | 11.7962 | 11.3984 | 1019 | 6.0000 | 27 | 0.2229 |
| PSM | MLP_R | strict | 0.9956 | 471 | 60 | 5.2816 | 29/48 | 31/48 | 27.3585 | 60.4167 | 37.6623 | 4.9365 | 3.5490 | 3.2358 | 1482 | 2.0000 | 17 | 0.2207 |
| PSM | MLP_R | confirmed3 | 0.8223 | 389 | 52 | 4.1047 | 20/48 | 21/48 | 23.8095 | 41.6667 | 30.3030 | 3.8469 | 2.6972 | 2.4390 | 1621 | 7.0000 | 27 | 0.2207 |
| PSM | CANDI_A | strict | 4.3269 | 2047 | 137 | 32.5679 | 29/48 | 31/48 | 16.0221 | 60.4167 | 25.3275 | 50.2653 | 25.5283 | 73.0081 | 1000 | 1.0000 | 17 | 0.2697 |
| PSM | CANDI_A | confirmed3 | 3.8830 | 1837 | 99 | 31.3566 | 20/48 | 21/48 | 15.5039 | 41.6667 | 22.5989 | 49.1662 | 23.8174 | 72.3902 | 1000 | 4.0000 | 27 | 0.2697 |
| PSM | HistoryMean512 | strict | 3.0185 | 1428 | 636 | 5.3677 | 41/48 | 41/48 | 4.6591 | 85.4167 | 8.8362 | 6.8031 | 3.3134 | 4.1138 | 615 | 1.0000 | 7 | 1.9654 |
| PSM | HistoryMean512 | confirmed3 | 1.1140 | 527 | 155 | 2.4628 | 20/48 | 20/48 | 9.4340 | 41.6667 | 15.3846 | 3.4489 | 1.9353 | 2.3740 | 1324 | 13.0000 | 28 | 1.9654 |

## Table2: matched A/B and adaptation differences

All rate/ranking differences below are **percentage points**; event and miss differences are counts/points. n=1 fixed training seed; sample SD=NA. No uncertainty interval or independent-dataset replication is implied. Full strict/C3 and numerical precision are in `paired.csv`.

| Domain | Stream | Contrast | ΔAP pp | ΔROC pp | ΔVUS-PR pp | ΔVUS-ROC pp | ΔFPR pp | ΔRecall pp | ΔM | Δevent cov pp | Δlong macro pp | Δlong pooled pp | Δlongest miss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWAT | strict | A_noU−B_noU | -1.0594 | -0.1286 | -1.0113 | -0.1989 | -29.8270 | -10.4653 | -1 | -10.4653 | -12.9346 | -5.5431 | 1051 |
| SWAT | strict | A_R−B_R | -7.8658 | -2.4206 | -8.1375 | -2.4122 | -0.3216 | -0.0984 | 0 | -0.0984 | -0.0721 | -0.0413 | 17991 |
| SWAT | strict | A_R−A_noU | -32.0459 | -4.1504 | -31.5195 | -3.5605 | -0.6380 | -64.8387 | 0 | -64.8387 | -10.1009 | -80.0046 | 29431 |
| SWAT | strict | B_R−B_noU | -25.2395 | -1.8584 | -24.3933 | -1.3472 | -30.1434 | -75.2056 | -1 | -75.2056 | -22.9634 | -85.5065 | 12491 |
| SWAT | confirmed3 | A_noU−B_noU | -1.0594 | -0.1286 | -1.0113 | -0.1989 | -15.9757 | -10.9258 | -3 | -10.9258 | -11.3617 | -7.3151 | 1895 |
| SWAT | confirmed3 | A_R−B_R | -7.8658 | -2.4206 | -8.1375 | -2.4122 | +0.0232 | +0.0084 | 2 | +0.0084 | +0.1552 | +0.0077 | 1 |
| SWAT | confirmed3 | A_R−A_noU | -32.0459 | -4.1504 | -31.5195 | -3.5605 | -0.7229 | -62.8189 | -1 | -62.8189 | -9.7789 | -77.9386 | 27919 |
| SWAT | confirmed3 | B_R−B_noU | -25.2395 | -1.8584 | -24.3933 | -1.3472 | -16.7219 | -73.7531 | -6 | -73.7531 | -21.2958 | -85.2614 | 29813 |
| PSM | strict | A_noU−B_noU | +0.0165 | +0.0156 | +0.0135 | +0.0080 | -0.0127 | +0.0057 | 0 | -0.0095 | -0.0092 | -0.0325 | 0 |
| PSM | strict | A_R−B_R | -0.5707 | -0.9249 | -0.5509 | -0.8708 | -0.0169 | -0.0517 | 0 | -0.1137 | -0.1155 | -0.1463 | 0 |
| PSM | strict | A_R−A_noU | +0.6658 | +1.5439 | +0.5259 | +1.4343 | -0.3593 | -0.9932 | 0 | -0.9286 | -1.1372 | -0.7154 | 78 |
| PSM | strict | B_R−B_noU | +1.2529 | +2.4844 | +1.0903 | +2.3131 | -0.3551 | -0.9358 | 0 | -0.8243 | -1.0309 | -0.6016 | 78 |
| PSM | confirmed3 | A_noU−B_noU | +0.0165 | +0.0156 | +0.0135 | +0.0080 | +0.0000 | +0.0057 | 0 | +0.0000 | +0.0000 | +0.0000 | 0 |
| PSM | confirmed3 | A_R−B_R | -0.5707 | -0.9249 | -0.5509 | -0.8708 | +0.0000 | -0.0402 | 0 | -0.0569 | -0.0138 | -0.0488 | 0 |
| PSM | confirmed3 | A_R−A_noU | +0.6658 | +1.5439 | +0.5259 | +1.4343 | -0.0444 | -0.2756 | -3 | -0.3222 | -0.1680 | -0.1626 | 556 |
| PSM | confirmed3 | B_R−B_noU | +1.2529 | +2.4844 | +1.0903 | +2.3131 | -0.0444 | -0.2296 | -3 | -0.2653 | -0.1542 | -0.1138 | 556 |

## Whole-method comparison limits

LEFT192-point full-window scores remain offline; candidates use512 strictly past inputs plus the current value only to score. MLP/CANDI use native10-point endpoint MSE. LEFT's normal-prefix scaler includes its source SELECT, whereas the candidate/MLP use SOURCE_FIT statistics. Historical LEFT/MLP updates useB256; candidates use logicalB16,32000 examples per phase. Different native objectives, pretrained priors and exposures prevent treating external differences as a pure time-frequency ablation. TimeRadar PSM is the archived complete native offline L=1 result, ranking only because its batch-shared order has no compatible new-CAL footprint. HistoryMean residual scaling is retained unchanged. No model is selected per domain or metric.

| Domain | Method | Actual training/context permissions | Operating status |
| --- | --- | --- | --- |
| SWAT | A_noU | FIT-only scaler; current-value conditional generation from strictly past512; own same-domain source init; logicalB16/2000updates | new own mixed-CAL threshold |
| SWAT | B_noU | FIT-only scaler; current-value conditional generation from strictly past512; own same-domain source init; logicalB16/2000updates | new own mixed-CAL threshold |
| SWAT | A_R | FIT-only scaler; current-value conditional generation from strictly past512; own same-domain source init; logicalB16/2000updates | new own mixed-CAL threshold |
| SWAT | B_R | FIT-only scaler; current-value conditional generation from strictly past512; own same-domain source init; logicalB16/2000updates | new own mixed-CAL threshold |
| SWAT | LEFT_noU | native192-point nonoverlapping whole-window retrospective scores; normal-prefix scaler includes SOURCE_SELECT; existing noU or2000xB256 native updates; not same-context/compute ablation | new own mixed-CAL threshold |
| SWAT | LEFT_TT | native192-point nonoverlapping whole-window retrospective scores; normal-prefix scaler includes SOURCE_SELECT; existing noU or2000xB256 native updates; not same-context/compute ablation | new own mixed-CAL threshold |
| SWAT | LEFT_R | native192-point nonoverlapping whole-window retrospective scores; normal-prefix scaler includes SOURCE_SELECT; existing noU or2000xB256 native updates; not same-context/compute ablation | new own mixed-CAL threshold |
| SWAT | MLP_noU | native10-point MLP normal source initialization, FIT-only scaler, author normal SELECT; no target updates | new own mixed-CAL threshold |
| SWAT | MLP_TT | native10-point MLP; FIT-only scaler; existing2000xB256 replay/TT; different context/objective/exposure from candidate | new own mixed-CAL threshold |
| SWAT | MLP_R | native10-point MLP; FIT-only scaler; existing2000xB256 replay/TT; different context/objective/exposure from candidate | new own mixed-CAL threshold |
| SWAT | CANDI_A | native10-point whole-window MSE at endpoint; SOURCE_FIT-only scaler; normal source MLP; CANDI U-only FPM/SANA then frozen | new own mixed-CAL threshold |
| SWAT | HistoryMean512 | Accepted strictly-past512 historical predictor; original source residual scaling retained | NA_NO_COMPATIBLE_MIXED_CAL_BANK |
| PSM | A_noU | FIT-only scaler; current-value conditional generation from strictly past512; own same-domain source init; logicalB16/2000updates | new own mixed-CAL threshold |
| PSM | B_noU | FIT-only scaler; current-value conditional generation from strictly past512; own same-domain source init; logicalB16/2000updates | new own mixed-CAL threshold |
| PSM | A_R | FIT-only scaler; current-value conditional generation from strictly past512; own same-domain source init; logicalB16/2000updates | new own mixed-CAL threshold |
| PSM | B_R | FIT-only scaler; current-value conditional generation from strictly past512; own same-domain source init; logicalB16/2000updates | new own mixed-CAL threshold |
| PSM | LEFT_noU | native192-point nonoverlapping whole-window retrospective scores; normal-prefix scaler includes SOURCE_SELECT; existing noU or2000xB256 native updates; not same-context/compute ablation | new own mixed-CAL threshold |
| PSM | LEFT_TT | native192-point nonoverlapping whole-window retrospective scores; normal-prefix scaler includes SOURCE_SELECT; existing noU or2000xB256 native updates; not same-context/compute ablation | new own mixed-CAL threshold |
| PSM | LEFT_R | native192-point nonoverlapping whole-window retrospective scores; normal-prefix scaler includes SOURCE_SELECT; existing noU or2000xB256 native updates; not same-context/compute ablation | new own mixed-CAL threshold |
| PSM | MLP_noU | native10-point MLP normal source initialization, FIT-only scaler, author normal SELECT; no target updates | new own mixed-CAL threshold |
| PSM | MLP_TT | native10-point MLP; FIT-only scaler; existing2000xB256 replay/TT; different context/objective/exposure from candidate | new own mixed-CAL threshold |
| PSM | MLP_R | native10-point MLP; FIT-only scaler; existing2000xB256 replay/TT; different context/objective/exposure from candidate | new own mixed-CAL threshold |
| PSM | CANDI_A | native10-point whole-window MSE at endpoint; SOURCE_FIT-only scaler; normal source MLP; CANDI U-only FPM/SANA then frozen | new own mixed-CAL threshold |
| PSM | TimeRadar_native | Official pretrained zero-update TimeRadar; original100-window/batched offline L1; old FIT/scoring permissions | NA_NO_COMPATIBLE_MIXED_CAL_BANK |
| PSM | HistoryMean512 | cached causal history512 with unchanged legacy source-calibrated residual scaling; new own mixed-CAL threshold; not FIT-only candidate normalization | new own mixed-CAL threshold |
