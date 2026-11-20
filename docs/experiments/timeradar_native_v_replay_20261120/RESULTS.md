# Native TimeRadar V: matched head-only natural-mixture adaptation

Completed 20 November 2026. Public SWaT31/PSM25, seed2025; all six fixed endpoint trainings and twelve new score banks completed. The previous A/noU banks and metrics are reused. No new noU inference, model selection or threshold rescue was performed.

## Complete method table

| Task | Method | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- | --- |
| SWAT | noU | 0.795725 | 0.857852 | 0.576600 | 0.708990 |
| SWAT | TT | 0.767373 | 0.840286 | 0.635916 | 0.795740 |
| SWAT | SS | 0.759642 | 0.840201 | 0.629941 | 0.793154 |
| SWAT | R | 0.761478 | 0.838195 | 0.638054 | 0.789673 |
| PSM | noU | 0.550952 | 0.735048 | 0.484578 | 0.648161 |
| PSM | TT | 0.497243 | 0.712915 | 0.520744 | 0.714204 |
| PSM | SS | 0.534776 | 0.731992 | 0.520077 | 0.697853 |
| PSM | R | 0.551502 | 0.734425 | 0.552609 | 0.721470 |

AP is average precision, not trapezoidal AUC-PR. VUS uses the previously pinned official LEFT RangeAUC_volume_opt, windowSize100/thre250. New six states were evaluated; accepted noU VUS was reused. VUS and point AP/ROC need not move in the same direction.

The same method table's operating panel follows. All percentages are rates, not counts; delay/miss units are original sample points. Strict and causal past-three confirmation are separate alarm streams, with no retroactive backfill. Point recall uses all labeled QUERY positives; coverage uses qualified complete-event points. One-to-one chronological first-unmatched overlap matching is not any-hit or maximum matching. Delay is hit-only; missed events are separately censored in events.csv. Unknown/ineligible events retain NA coverage/miss values, including observed_alarm_points=0 without implying a full miss.

| Task | Method | Stream | FP points | FPR % | Recall % | Event P/R/F1 % | Matched / eligible | Any-hit | Normal fragments | Coverage % | Long macro / pooled % | Max miss | Hit-only delay median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWAT | noU | strict | 3759 | 1.303 | 69.169 | 7.812/45.455/13.333 | 10/22 | 10 | 122 | 69.169 | 11.725/82.587 | 3939 | 135.0 |
| SWAT | noU | confirmed3 | 3529 | 1.223 | 69.123 | 8.065/45.455/13.699 | 10/22 | 10 | 118 | 69.123 | 11.460/82.566 | 3941 | 137.0 |
| SWAT | TT | strict | 3945 | 1.367 | 66.743 | 9.009/45.455/15.038 | 10/22 | 10 | 108 | 66.743 | 11.455/79.685 | 5067 | 131.5 |
| SWAT | TT | confirmed3 | 3740 | 1.296 | 66.710 | 9.091/45.455/15.152 | 10/22 | 10 | 107 | 66.710 | 11.310/79.669 | 5069 | 133.5 |
| SWAT | SS | strict | 3901 | 1.352 | 66.875 | 9.174/45.455/15.267 | 10/22 | 10 | 106 | 66.875 | 11.458/79.809 | 5018 | 126.0 |
| SWAT | SS | confirmed3 | 3699 | 1.282 | 66.842 | 9.174/45.455/15.267 | 10/22 | 10 | 106 | 66.842 | 11.314/79.793 | 5020 | 128.0 |
| SWAT | R | strict | 4054 | 1.405 | 67.072 | 8.547/45.455/14.388 | 10/22 | 10 | 114 | 67.072 | 11.492/79.989 | 4949 | 122.0 |
| SWAT | R | confirmed3 | 3838 | 1.330 | 67.039 | 8.696/45.455/14.599 | 10/22 | 10 | 112 | 67.039 | 11.347/79.974 | 4951 | 124.0 |
| PSM | noU | strict | 14036 | 29.669 | 50.870 | 8.955/37.500/14.458 | 18/48 | 28 | 185 | 73.659 | 86.955/92.407 | 486 | 0.0 |
| PSM | noU | confirmed3 | 13765 | 29.096 | 48.878 | 7.059/12.500/9.023 | 6/48 | 17 | 88 | 73.195 | 86.453/92.114 | 694 | 0.0 |
| PSM | TT | strict | 1509 | 3.190 | 18.331 | 5.037/56.250/9.247 | 27/48 | 30 | 350 | 21.300 | 29.341/25.073 | 708 | 1.0 |
| PSM | TT | confirmed3 | 1022 | 2.160 | 14.249 | 5.618/20.833/8.850 | 10/48 | 11 | 102 | 18.258 | 24.441/22.016 | 1228 | 7.0 |
| PSM | SS | strict | 9089 | 19.212 | 47.442 | 6.720/52.083/11.905 | 25/48 | 31 | 336 | 68.107 | 68.984/85.545 | 534 | 0.0 |
| PSM | SS | confirmed3 | 8604 | 18.187 | 44.469 | 6.250/18.750/9.375 | 9/48 | 16 | 128 | 67.197 | 66.713/84.927 | 1228 | 0.0 |
| PSM | R | strict | 2380 | 5.031 | 32.080 | 4.396/58.333/8.175 | 28/48 | 28 | 462 | 43.917 | 46.044/57.285 | 534 | 0.0 |
| PSM | R | confirmed3 | 1698 | 3.589 | 28.073 | 4.933/22.917/8.118 | 11/48 | 11 | 153 | 41.008 | 42.016/54.195 | 1228 | 0.0 |

## Learning-method contrasts

Each difference is first method minus second, n=1; no sample SD is estimable. Rate differences below are percentage points; counts and longest-miss changes retain their units. Full exact fields, including strict point F1 and event macro coverage, remain in methods.csv/learning_pairs.csv.

| Task | Contrast | Stream | AP pp | ROC pp | VUS-PR pp | VUS-ROC pp | FPR pp | Recall pp | Matched delta | Any-hit delta | Long macro pp | Long pooled pp | Max miss delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWAT | R−TT | strict | -0.590 | -0.209 | 0.214 | -0.607 | 0.038 | 0.329 | 0 | 0 | 0.037 | 0.304 | -118 |
| SWAT | R−noU | strict | -3.425 | -1.966 | 6.145 | 8.068 | 0.102 | -2.097 | 0 | 0 | -0.233 | -2.597 | 1010 |
| SWAT | R−SS | strict | 0.184 | -0.201 | 0.811 | -0.348 | 0.053 | 0.197 | 0 | 0 | 0.034 | 0.181 | -69 |
| SWAT | TT−noU | strict | -2.835 | -1.757 | 5.932 | 8.675 | 0.064 | -2.426 | 0 | 0 | -0.270 | -2.902 | 1128 |
| SWAT | R−TT | confirmed3 | -0.590 | -0.209 | 0.214 | -0.607 | 0.034 | 0.329 | 0 | 0 | 0.037 | 0.304 | -118 |
| SWAT | R−noU | confirmed3 | -3.425 | -1.966 | 6.145 | 8.068 | 0.107 | -2.085 | 0 | 0 | -0.113 | -2.592 | 1010 |
| SWAT | R−SS | confirmed3 | 0.184 | -0.201 | 0.811 | -0.348 | 0.048 | 0.197 | 0 | 0 | 0.034 | 0.181 | -69 |
| SWAT | TT−noU | confirmed3 | -2.835 | -1.757 | 5.932 | 8.675 | 0.073 | -2.413 | 0 | 0 | -0.150 | -2.897 | 1128 |
| PSM | R−TT | strict | 5.426 | 2.151 | 3.186 | 0.727 | 1.841 | 13.749 | 1 | -2 | 16.703 | 32.211 | -174 |
| PSM | R−noU | strict | 0.055 | -0.062 | 6.803 | 7.331 | -24.638 | -18.790 | 10 | 0 | -40.911 | -35.122 | 48 |
| PSM | R−SS | strict | 1.673 | 0.243 | 3.253 | 2.362 | -14.181 | -15.363 | 3 | -3 | -22.940 | -28.260 | 0 |
| PSM | TT−noU | strict | -5.371 | -2.213 | 3.617 | 6.604 | -26.479 | -32.539 | 9 | 2 | -57.614 | -67.333 | 222 |
| PSM | R−TT | confirmed3 | 5.426 | 2.151 | 3.186 | 0.727 | 1.429 | 13.824 | 1 | 0 | 17.575 | 32.179 | 0 |
| PSM | R−noU | confirmed3 | 0.055 | -0.062 | 6.803 | 7.331 | -25.507 | -20.805 | 5 | -6 | -44.436 | -37.919 | 534 |
| PSM | R−SS | confirmed3 | 1.673 | 0.243 | 3.253 | 2.362 | -14.598 | -16.396 | 2 | -5 | -24.696 | -30.732 | 0 |
| PSM | TT−noU | confirmed3 | -5.371 | -2.213 | 3.617 | 6.604 | -26.936 | -34.629 | 4 | -6 | -62.012 | -70.098 | 534 |

## Technical decision

- Ordinary TT reduces AP/ROC against noU on both tasks: SWaT AP -2.835pp, PSM -5.371pp. The VUS gains are retained; this is not a claim that every detection measure deteriorated. Natural mixed updating does not identify an anomaly-contamination cause.
- PSM R versus TT improves AP by5.426pp and ROC by2.151pp. C3 long-event macro/pooled coverage rises24.441/22.016% to42.016/54.195%, but normal FPR rises2.160% to3.589%. Matched events rise10 to11; any-hit stays11, which does not imply identical event identities. This is a bounded useful replay positive.
- PSM R restores noU AP approximately (.551502 versus .550952), not its sustained coverage. Against noU, C3 FPR falls29.096% to3.589% while recall falls48.878% to28.073%, long macro86.453% to42.016%, pooled92.114% to54.195%, and longest miss increases694 to1228. Against SS, R improves AP/ROC and lowers FPR but loses coverage. Thus its ranking benefit exceeds source-only continuation without operational dominance.
- SWaT R versus TT loses AP0.590pp/ROC0.209pp; small coverage improvements and higher false alarms accompany unchanged10 matched events. There is no comparable clear useful replay increment. All adapted AP/ROC remain below noU; improved VUS is separately retained.
- These are two exposed development tasks and one training seed, with mature official TimeRadar and replay attribution. They establish neither an original Hybrid mechanism, heterogeneous pretraining, contamination robustness nor general cross-domain transfer. No R for stopped B/C, extra seeds, losses, scores or successor is automatically scheduled.

## Permissions and immutable definitions

All native windows use B=1, own fractional order, SOURCE_FIT scaler, norm0, ten fixed complementary copies under isolated84551 RNG, correction1 variance, channel mean and terminal99. Only the original decoder793610/1636043 parameters update; backbone/order network/buffers remain frozen/eval. Original full-copy/full-window reconstruction MSE is divided by64 before gradient accumulation. TT=[T,Tprime], SS=[Sprime,S], R=[T,S],32+32 slots, matched T/S and independent additional streams. Every arm restores official weights and fresh AdamW,2000 endpoint updates/128000 actual window exposures; source/target pools and optimization are fixed in IMPLEMENTATION.md/config, not tuned on QUERY.

SWaT source[0,396000), target[495000,585000), QUERY[608535,944856)=336321; PSM train source[0,105984), test target[0,17568), QUERY[23016,87744)=64728. Record identities distinguish PSM train from test. Same original512 CAL points per task, own q99/higher/strict> threshold. SWaT CAL contains34 attack points, PSM512/all attack; neither defines a certified normal1% FPR budget. Long events have duration>512 with steady portion after start+512; SWaT9/PSM5 long complete events,22/48 total eligible. Every score is available at its endpoint, no future window inputs; these remain retrospective development evaluations, not online deployment certification.

The prior CAL prose38 was corrected to34 using archived methods.csv; no historical scores, thresholds, CSV or scientific conclusions were overwritten.

## Completion, cost and review

38 owned GPU jobs (two engineering gates, six trainings, thirty readout jobs), all OS0/cleanup, no failures or duplicate scientific work: **8.08072435820514 A40 GPU hours**, including startup/cleanup. Decoder-training occupancy4.207758699656179 GPUh; peaks202.07MiB SWaT/172.64MiB PSM allocated. Main metric CPU wall69.270933s; independent metadata/NumPy plus six shared-official VUS recomputations67.5104s, with separately recorded brief preflight/CLI overhead, not an exact all-command CPU total.

One bounded same-family role-separated reviewer checked16458 fields,16 method-stream rows,16 contrast rows and856 raw event rows; maximum discrepancy3.3306690738754696e-16. It checked actual endpoint tensors, paired plans, roles, masks, bank identities/formulas, calibration, AP/ROC, strict/C3 events, delay, long macro/pooled coverage and VUS. Shared VUS code is not an independent mathematical rewrite. No review neural forward or GPU job occurred. Public export rounding receives a final limited check in the same review.

PROVENANCE.json binds source/PT, all checkpoints/banks, original global seal, metrics and review by SHA. Full checkpoints include upstream frozen pretrained tensors and stay outside public Git, along with raw data, scalers and full banks. Only hashes, code/config and nonprivate aggregate/event metrics are published.
