# Complete official pretrained TimesFM3 frozen forecast baseline

Public SWaT31/PSM25 only; no training or adaptation. Every origin independently predicts one current point from strictly past512 joint multivariate observations. Project residual MSE is not an official anomaly head and not a Hybrid contribution.

| Domain | Method | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- | --- |
| SWAT | TimesFM3_frozen | 0.093973 | 0.288375 | 0.096544 | 0.298381 |
| PSM | TimesFM3_frozen | 0.344512 | 0.566754 | 0.376176 | 0.608554 |
| SWAT | TimeRadar_V_A | 0.795725 | 0.857852 | 0.576600 | 0.708990 |
| SWAT | TimesFM_RAW_D | 0.762338 | 0.838454 | 0.546680 | 0.696022 |
| PSM | TimeRadar_V_A | 0.550952 | 0.735048 | 0.484578 | 0.648161 |
| PSM | TimesFM_RAW_D | 0.567902 | 0.743845 | 0.575549 | 0.764875 |
| SWAT | MLP_noU | 0.783949 | 0.856207 | 0.543552 | 0.696882 |
| SWAT | CANDI_A | 0.783885 | 0.856961 | 0.541710 | 0.695700 |
| SWAT | MLP_R | 0.782007 | 0.863262 | 0.541789 | 0.702566 |
| PSM | MLP_noU | 0.606531 | 0.815061 | 0.627826 | 0.823338 |
| PSM | CANDI_A | 0.606531 | 0.815059 | 0.627825 | 0.823337 |
| PSM | MLP_R | 0.635775 | 0.828137 | 0.685514 | 0.852250 |
| SWAT | LEFT_noU | 0.607560 | 0.842267 | 0.606345 | 0.829634 |
| SWAT | LEFT_R | 0.629347 | 0.885512 | 0.639265 | 0.884546 |
| PSM | LEFT_noU | 0.586600 | 0.776988 | 0.577678 | 0.780826 |
| PSM | LEFT_R | 0.593208 | 0.759509 | 0.590745 | 0.771875 |
| PSM | HistoryMean512 | 0.303673 | 0.525589 | 0.332386 | 0.567946 |
| SWAT | HistoryMean512 | 0.087661 | 0.227761 | 0.090893 | 0.232695 |

## Original CAL operation

| Domain | Method | Stream | Normal FP | FPR % | Recall % | Matched/events | Any-hit | Event F1 % | Normal fragments | Long macro/pooled % | Longest miss | Hit-only median delay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWAT | TimesFM3_frozen | strict | 3638 | 1.260813 | 0.439542 | 21/22 | 21 | 1.275820 | 3117 | 1.242028/0.211509 | 31863 | 37.0 |
| SWAT | TimesFM3_frozen | confirmed3 | 90 | 0.031191 | 0.064885 | 6/22 | 6 | 15.000000 | 50 | 0.148020/0.043849 | 31867 | 364.0 |
| PSM | TimesFM3_frozen | strict | 1058 | 2.236361 | 6.412538 | 41/48 | 41 | 6.312548 | 905 | 4.038802/2.731707 | 485 | 2.0 |
| PSM | TimesFM3_frozen | confirmed3 | 34 | 0.071868 | 0.826684 | 14/48 | 14 | 29.166667 | 21 | 0.238479/0.113821 | 3699 | 9.5 |
| SWAT | TimeRadar_V_A | strict | 3759 | 1.302748 | 69.169266 | 10/22 | 10 | 13.333333 | 122 | 11.725089/82.586603 | 3939 | 135.0 |
| SWAT | TimeRadar_V_A | confirmed3 | 3529 | 1.223037 | 69.123218 | 10/22 | 10 | 13.698630 | 118 | 11.460402/82.565968 | 3941 | 137.0 |
| SWAT | TimesFM_RAW_D | strict | 274019 | 94.966106 | 97.982293 | 6/22 | 22 | 7.843137 | 142 | 88.942052/99.463489 | 701 | 0.0 |
| SWAT | TimesFM_RAW_D | confirmed3 | 273858 | 94.910308 | 97.959269 | 5/22 | 21 | 18.867925 | 47 | 88.888889/99.460909 | 721 | 0.0 |
| PSM | TimeRadar_V_A | strict | 14036 | 29.668773 | 50.869740 | 18/48 | 28 | 14.457831 | 185 | 86.955307/92.406504 | 486 | 0.0 |
| PSM | TimeRadar_V_A | confirmed3 | 13765 | 29.095944 | 48.877662 | 6/48 | 17 | 9.022556 | 88 | 86.452514/92.113821 | 694 | 0.0 |
| PSM | TimesFM_RAW_D | strict | 12067 | 25.506775 | 58.206556 | 36/48 | 45 | 4.058625 | 1429 | 90.614525/94.536585 | 246 | 0.0 |
| PSM | TimesFM_RAW_D | confirmed3 | 10412 | 22.008497 | 48.803031 | 22/48 | 30 | 20.560748 | 84 | 85.167598/91.365854 | 399 | 3.5 |
| SWAT | MLP_noU | strict | 280425 | 97.186218 | 98.024154 | 4/22 | 21 | 11.267606 | 63 | 88.888889/99.460909 | 721 | 0.0 |
| SWAT | MLP_noU | confirmed3 | 280339 | 97.156413 | 98.003223 | 4/22 | 21 | 11.764706 | 60 | 88.888889/99.460909 | 721 | 0.0 |
| SWAT | CANDI_A | strict | 281335 | 97.501594 | 98.013689 | 4/22 | 21 | 11.594203 | 61 | 88.888889/99.460909 | 721 | 0.0 |
| SWAT | CANDI_A | confirmed3 | 281256 | 97.474215 | 97.992758 | 4/22 | 21 | 13.114754 | 54 | 88.888889/99.460909 | 721 | 0.0 |
| SWAT | MLP_R | strict | 278767 | 96.611609 | 98.026247 | 5/22 | 21 | 11.494253 | 80 | 88.888889/99.460909 | 721 | 0.0 |
| SWAT | MLP_R | confirmed3 | 278647 | 96.570021 | 98.009502 | 5/22 | 21 | 12.820513 | 71 | 88.888889/99.460909 | 721 | 0.0 |
| PSM | MLP_noU | strict | 2047 | 4.326872 | 32.567886 | 29/48 | 31 | 25.327511 | 137 | 25.528286/73.008130 | 1000 | 1.0 |
| PSM | MLP_noU | confirmed3 | 1837 | 3.882982 | 31.356565 | 20/48 | 21 | 22.598870 | 99 | 23.817355/72.390244 | 1000 | 4.0 |
| PSM | CANDI_A | strict | 2047 | 4.326872 | 32.567886 | 29/48 | 31 | 25.327511 | 137 | 25.528286/73.008130 | 1000 | 1.0 |
| PSM | CANDI_A | confirmed3 | 1837 | 3.882982 | 31.356565 | 20/48 | 21 | 22.598870 | 99 | 23.817355/72.390244 | 1000 | 4.0 |
| PSM | MLP_R | strict | 471 | 0.995582 | 5.281589 | 29/48 | 31 | 37.662338 | 60 | 3.549037/3.235772 | 1482 | 2.0 |
| PSM | MLP_R | confirmed3 | 389 | 0.822254 | 4.104713 | 20/48 | 21 | 30.303030 | 52 | 2.697166/2.439024 | 1621 | 7.0 |
| SWAT | LEFT_noU | strict | 2852 | 0.988411 | 1.513280 | 13/22 | 13 | 9.885932 | 223 | 1.041518/0.325002 | 31840 | NA |
| SWAT | LEFT_noU | confirmed3 | 2429 | 0.841813 | 1.431651 | 12/22 | 12 | 10.300429 | 194 | 0.855017/0.291470 | 31840 | NA |
| SWAT | LEFT_R | strict | 2705 | 0.937465 | 1.320719 | 13/22 | 13 | 9.811321 | 226 | 1.022679/0.170239 | 31862 | NA |
| SWAT | LEFT_R | confirmed3 | 2276 | 0.788788 | 1.243276 | 12/22 | 12 | 10.256410 | 197 | 0.836806/0.141866 | 31862 | NA |
| PSM | LEFT_noU | strict | 8843 | 18.692004 | 49.595270 | 34/48 | 38 | 10.000000 | 489 | 80.002244/89.739837 | 399 | NA |
| PSM | LEFT_noU | confirmed3 | 8055 | 17.026359 | 45.467593 | 20/48 | 24 | 10.526316 | 251 | 76.406722/88.130081 | 399 | NA |
| PSM | LEFT_R | strict | 8065 | 17.047496 | 53.969803 | 35/48 | 36 | 9.055627 | 557 | 84.226025/92.032520 | 399 | NA |
| PSM | LEFT_R | confirmed3 | 7168 | 15.151451 | 50.014352 | 21/48 | 22 | 9.677419 | 281 | 79.975084/89.967480 | 399 | NA |
| PSM | HistoryMean512 | strict | 1428 | 3.018453 | 5.367702 | 41/48 | 41 | 8.836207 | 636 | 3.313360/4.113821 | 615 | 1.0 |
| PSM | HistoryMean512 | confirmed3 | 527 | 1.113953 | 2.462828 | 20/48 | 20 | 15.384615 | 155 | 1.935276/2.373984 | 1324 | 13.0 |

## TimesFM versus native TimeRadar: descriptive independent operations

| Domain | Stream | Any-hit both / TimesFM-only / TimeRadar-only / neither | Independently matched both / TimesFM-only / TimeRadar-only / neither | Normal FP both / TimesFM-only / TimeRadar-only |
| --- | --- | --- | --- | --- |
| SWAT | strict | 10 / 11 / 0 / 1 | 10 / 11 / 0 / 1 | 527 / 3111 / 3232 |
| SWAT | confirmed3 | 4 / 2 / 6 / 10 | 4 / 2 / 6 / 10 | 51 / 39 / 3478 |
| PSM | strict | 25 / 16 / 3 / 4 | 18 / 23 / 0 / 7 | 364 / 694 / 13672 |
| PSM | confirmed3 | 6 / 8 / 11 / 23 | 2 / 12 / 4 / 30 | 20 / 14 / 13745 |

Full event IDs are in EVENT_OVERLAP.csv; events.csv retains original boundaries, qualification, delay, long coverage and missingness. Independently matched sets are not the matching of a new union detector. No OR score/threshold is introduced.

## Fixed permissions and limitations

SWaT QUERY[608535,944856):336321 points; PSM[23016,87744):64728. TimesFM and TimeRadar use the exact same source-FIT scaler,512 legal CAL indexes and current observation arrays. Each computes its own q99/higher threshold and alarms strictly above. SWaT CAL contains34 attack points; PSM512/512 are attacks: neither implies a certified normal1%FPR budget. C3 uses three consecutive original alarms without backfill; original indices and availability are retained. Long means complete event duration>512, steady after start+512; macro and pooled denominators differ; unqualified results remainNA. Delay and miss units are original sample points.

LEFT remains native192-point retrospective full-window output with source-prefix statistics including SELECT; MLP/CANDI use10-point endpoints and their own existing source training/scalers/adaptation rules. These are common QUERY-axis complete-method comparisons, not equal-architecture, pretraining or normal-exposure ablations. Historical lightweight TimesFM_RAW_D had no official pretrained weights; its comparison does not isolate weights alone. SWaT HistoryMean512 lacks a compatible mixed-CAL bank and isNA for operation; no new inference or threshold substitution is performed. Official pretraining provenance is not independently certified free of benchmark exposure. Both tasks are exposed development tasks.

AP denotes Average Precision, distinct from trapezoidal AUC-PR. VUS uses the same pinned official RangeAUC_volume_opt(window100,thre250); old VUS rows are reused. Raw-score AP/ROC repeated in strict/C3 rows do not imply the confirmation rule changes raw ranking. LOW_FPR_SUPPLEMENT preserves empirical polyline pAUC0-1%/0-5%, area/bound (not McClish), discrete tie-group recall at caps1/3/5%, and separate C3 minima curves. QUERY curve points are retrospective descriptions only, never substitute deployment thresholds.

## Actual computation

23 actual owned GPU jobs, all OS0/cleared; 5.487239025 A40 GPU hours including startup/cleanup. Training/adaptation0. Metric CPU wall 47.726551s; old model inference/metrics not repeated. Model has330710976 frozen parameters; complete weightSHA `a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da`. Full tensors and forecasts stay outside Git. CodeApache2; weights retain separate TimesFM noncommercial terms.

## Scientific decision

The fixed complete pretrained512-history median-residual baseline does not establish a ranking or sustained-coverage advantage over native TimeRadar V. AP is0.093973/0.344512 and ROC0.288375/0.566754 on SWaT/PSM, below native V in both tasks. Even the descriptive1%FPR cap has strict recall only0.374657%/3.840634%, versus68.861586%/16.619783% for native V. Confirmation-specific cap1 recall is0.458380%/4.477869%, also below V68.936936%/17.113497%. Thus the negative ranking result cannot be explained solely by choosing the original CAL operating threshold. Score direction, median choice and thresholds remain fixed; no posthoc inversion is performed.

There is localized complementary triggering, which is retained rather than erased by the aggregate negative. SWaT strict operations share10 events, TimesFM alone hits11, V alone0; however TimesFM uses3638 normal false-positive points in3117 normal fragments, detects only210 anomaly points overall, and covers just0.211509% of pooled long steady states. The eleven extra strict events receive only1-22 alarm points each. After C3, overlap is4, TimesFM-only2 and V-only6. The two TimesFM-only events are E000026[842679,843280), one confirmed alarm point among601 and zero steady-after512 coverage, and E000031[931541,932010), six of469 points. This is additional short-lived evidence, not an established persistent-fault gain.

PSM strict any-hit sets are25 shared/16 TimesFM-only/3 V-only. C3 sets are6 shared/8 TimesFM-only/11 V-only; independently matched totals are14 versus6, while any-hit totals are14 versus17. These must not be interchanged. The eight TimesFM-only C3 events all last at most397 points, so they do not establish long-event steady coverage. TimesFM C3 normal FP34 versus V13765 is retained alongside recall0.826684% versus48.877662% and long pooled steady coverage0.113821% versus92.113821%. PSM CAL is entirely attack-labeled, so this is a fixed empirical operating tradeoff, not a matched normal-FPR certification.

The complete model modestly exceeds the cached HistoryMean512 ranking on both tasks, but remains far below native V, source-trained lightweight D and the compatible LEFT/MLP/CANDI references on AP/ROC. These complete methods differ in architecture, pretraining and source/target information; this does not identify a sole failure cause or disprove all forecasting/long-history mechanisms. The observation is restricted to the predeclared512-history, horizon1, median-MSE protocol and exposed public tasks. It establishes a full official pretrained external baseline, not a Hybrid contribution, lag mechanism, mixed-unlabeled adaptation or contamination robustness.

Retain the official frozen baseline and exact short-event evidence. There is currently no sufficient combined ranking/controlled-false-alarm/sustained benefit to justify automatically launching TimesFM adaptation or fusion. No new score, threshold, model or training follows this benchmark; next method decisions remain separate.

Public arrays are summary tables/figures only. available_index=t denotes the observation information footprint, not an assertion that offline batched execution was a measured live deployment. Runtime latency and GPU occupancy are reported separately. The first two public-export attempts failed on ranking-only NA handling and exclusive CSV output creation; originals are retained outside Git and a fresh successful export used unchanged metrics. They consumed no GPU or neural forward. Successful table/figure CPU wall24.614234s is separate from the main metric wall47.726551s and the bounded review.
