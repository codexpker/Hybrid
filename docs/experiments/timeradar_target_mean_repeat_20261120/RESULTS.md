# Fixed R-MeanT adaptation repetition

Two exposed public tasks; seeds2025/2026/2027 share one official pretrained initial state. Only adaptation randomness changes. noU is one fixed reference; TT/SS retain only their accepted seed2025 results. No repeated pretraining, independent-domain or blind-test claim is made.

## Per-seed ranking

| Task | Method | Seed | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- | --- | --- |
| SWAT | R | 2026 | 0.763614 | 0.838149 | 0.637617 | 0.793179 |
| SWAT | R-MeanT | 2026 | 0.759753 | 0.841749 | 0.612397 | 0.786580 |
| SWAT | R | 2027 | 0.765930 | 0.838772 | 0.640486 | 0.793614 |
| SWAT | R-MeanT | 2027 | 0.759074 | 0.841895 | 0.618142 | 0.787497 |
| PSM | R | 2026 | 0.558546 | 0.738881 | 0.564370 | 0.729383 |
| PSM | R-MeanT | 2026 | 0.562547 | 0.745142 | 0.552239 | 0.717619 |
| PSM | R | 2027 | 0.546953 | 0.734635 | 0.550390 | 0.722463 |
| PSM | R-MeanT | 2027 | 0.561034 | 0.743381 | 0.552240 | 0.716801 |
| SWAT | R-MeanT | 2025 | 0.759388 | 0.841620 | 0.610312 | 0.783951 |
| PSM | R-MeanT | 2025 | 0.560142 | 0.742163 | 0.546871 | 0.714094 |
| SWAT | TT | 2025 | 0.767373 | 0.840286 | 0.635916 | 0.795740 |
| SWAT | SS | 2025 | 0.759642 | 0.840201 | 0.629941 | 0.793154 |
| SWAT | R | 2025 | 0.761478 | 0.838195 | 0.638054 | 0.789673 |
| PSM | TT | 2025 | 0.497243 | 0.712915 | 0.520744 | 0.714204 |
| PSM | SS | 2025 | 0.534776 | 0.731992 | 0.520077 | 0.697853 |
| PSM | R | 2025 | 0.551502 | 0.734425 | 0.552609 | 0.721470 |
| SWAT | noU | fixed | 0.795725 | 0.857852 | 0.576600 | 0.708990 |
| PSM | noU | fixed | 0.550952 | 0.735048 | 0.484578 | 0.648161 |

AP is Average Precision. VUS uses the previously pinned implementation/window100/threshold250. Original seed2025 metrics and external baselines are reused. Complete exact rows and permissions remain in methods.csv.

## Original CAL operation

| Task | Method | Seed | Stream | Normal FP | FPR % | Recall % | Matched/events | Any-hit | Event F1 % | Normal fragments | Complete-event coverage % | Long macro/pooled % | Longest miss | Hit-only median delay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWAT | R | 2026 | strict | 4118 | 1.427 | 67.401 | 10/22 | 10 | 10.256 | 161 | 67.401 | 11.508/80.118 | 4899 | 63.5 |
| SWAT | R | 2026 | confirmed3 | 3850 | 1.334 | 67.340 | 10/22 | 10 | 14.815 | 107 | 67.340 | 11.363/80.103 | 4901 | 73.0 |
| SWAT | R-MeanT | 2026 | strict | 3789 | 1.313 | 66.365 | 9/22 | 9 | 13.740 | 106 | 66.365 | 11.398/79.311 | 5211 | 70.0 |
| SWAT | R-MeanT | 2026 | confirmed3 | 3585 | 1.242 | 66.335 | 9/22 | 9 | 13.953 | 104 | 66.335 | 11.253/79.295 | 5213 | 72.0 |
| SWAT | R | 2027 | strict | 4228 | 1.465 | 68.707 | 10/22 | 10 | 9.615 | 171 | 68.707 | 20.244/80.621 | 4850 | 63.5 |
| SWAT | R | 2027 | confirmed3 | 3960 | 1.372 | 68.636 | 10/22 | 10 | 15.038 | 103 | 68.636 | 20.099/80.606 | 4852 | 65.5 |
| SWAT | R-MeanT | 2027 | strict | 3877 | 1.344 | 66.423 | 10/22 | 10 | 15.625 | 103 | 66.423 | 11.414/79.347 | 5198 | 136.5 |
| SWAT | R-MeanT | 2027 | confirmed3 | 3681 | 1.276 | 66.390 | 10/22 | 10 | 15.625 | 103 | 66.390 | 11.269/79.331 | 5200 | 138.5 |
| PSM | R | 2026 | strict | 2723 | 5.756 | 34.405 | 30/48 | 30 | 7.864 | 509 | 46.172 | 47.880/60.130 | 534 | 0.0 |
| PSM | R | 2026 | confirmed3 | 1962 | 4.147 | 29.789 | 12/48 | 12 | 7.692 | 179 | 42.752 | 43.320/56.439 | 1228 | 0.0 |
| PSM | R-MeanT | 2026 | strict | 3130 | 6.616 | 36.983 | 29/48 | 30 | 8.443 | 478 | 50.891 | 50.264/68.276 | 569 | 0.5 |
| PSM | R-MeanT | 2026 | confirmed3 | 2416 | 5.107 | 32.740 | 10/48 | 11 | 6.515 | 174 | 47.982 | 45.908/65.593 | 1228 | 0.0 |
| PSM | R | 2027 | strict | 3028 | 6.400 | 33.285 | 30/48 | 30 | 7.853 | 519 | 45.063 | 48.457/58.081 | 534 | 0.0 |
| PSM | R | 2027 | confirmed3 | 2253 | 4.762 | 28.899 | 12/48 | 12 | 7.477 | 183 | 41.738 | 44.285/54.520 | 1228 | 0.0 |
| PSM | R-MeanT | 2027 | strict | 3243 | 6.855 | 37.740 | 28/48 | 29 | 8.034 | 514 | 51.592 | 49.954/69.577 | 569 | 0.0 |
| PSM | R-MeanT | 2027 | confirmed3 | 2477 | 5.236 | 33.653 | 11/48 | 12 | 7.432 | 184 | 49.214 | 46.023/67.854 | 1228 | 0.0 |
| SWAT | R-MeanT | 2025 | strict | 3815 | 1.322 | 66.499 | 10/22 | 10 | 14.286 | 114 | 66.499 | 11.411/79.417 | 5170 | 134.5 |
| SWAT | R-MeanT | 2025 | confirmed3 | 3597 | 1.247 | 66.461 | 10/22 | 10 | 15.748 | 102 | 66.461 | 11.266/79.401 | 5172 | 136.5 |
| PSM | R-MeanT | 2025 | strict | 4987 | 10.541 | 40.795 | 30/48 | 32 | 8.345 | 510 | 57.021 | 57.394/73.870 | 400 | 0.0 |
| PSM | R-MeanT | 2025 | confirmed3 | 4222 | 8.924 | 36.546 | 12/48 | 14 | 7.643 | 189 | 54.150 | 53.380/72.179 | 879 | 0.0 |
| SWAT | TT | 2025 | strict | 3945 | 1.367 | 66.743 | 10/22 | 10 | 15.038 | 108 | 66.743 | 11.455/79.685 | 5067 | 131.5 |
| SWAT | TT | 2025 | confirmed3 | 3740 | 1.296 | 66.710 | 10/22 | 10 | 15.152 | 107 | 66.710 | 11.310/79.669 | 5069 | 133.5 |
| SWAT | SS | 2025 | strict | 3901 | 1.352 | 66.875 | 10/22 | 10 | 15.267 | 106 | 66.875 | 11.458/79.809 | 5018 | 126.0 |
| SWAT | SS | 2025 | confirmed3 | 3699 | 1.282 | 66.842 | 10/22 | 10 | 15.267 | 106 | 66.842 | 11.314/79.793 | 5020 | 128.0 |
| SWAT | R | 2025 | strict | 4054 | 1.405 | 67.072 | 10/22 | 10 | 14.388 | 114 | 67.072 | 11.492/79.989 | 4949 | 122.0 |
| SWAT | R | 2025 | confirmed3 | 3838 | 1.330 | 67.039 | 10/22 | 10 | 14.599 | 112 | 67.039 | 11.347/79.974 | 4951 | 124.0 |
| PSM | TT | 2025 | strict | 1509 | 3.190 | 18.331 | 27/48 | 30 | 9.247 | 350 | 21.300 | 29.341/25.073 | 708 | 1.0 |
| PSM | TT | 2025 | confirmed3 | 1022 | 2.160 | 14.249 | 10/48 | 11 | 8.850 | 102 | 18.258 | 24.441/22.016 | 1228 | 7.0 |
| PSM | SS | 2025 | strict | 9089 | 19.212 | 47.442 | 25/48 | 31 | 11.905 | 336 | 68.107 | 68.984/85.545 | 534 | 0.0 |
| PSM | SS | 2025 | confirmed3 | 8604 | 18.187 | 44.469 | 9/48 | 16 | 9.375 | 128 | 67.197 | 66.713/84.927 | 1228 | 0.0 |
| PSM | R | 2025 | strict | 2380 | 5.031 | 32.080 | 28/48 | 28 | 8.175 | 462 | 43.917 | 46.044/57.285 | 534 | 0.0 |
| PSM | R | 2025 | confirmed3 | 1698 | 3.589 | 28.073 | 11/48 | 11 | 8.118 | 153 | 41.008 | 42.016/54.195 | 1228 | 0.0 |
| SWAT | noU | fixed | strict | 3759 | 1.303 | 69.169 | 10/22 | 10 | 13.333 | 122 | 69.169 | 11.725/82.587 | 3939 | 135.0 |
| SWAT | noU | fixed | confirmed3 | 3529 | 1.223 | 69.123 | 10/22 | 10 | 13.699 | 118 | 69.123 | 11.460/82.566 | 3941 | 137.0 |
| PSM | noU | fixed | strict | 14036 | 29.669 | 50.870 | 18/48 | 28 | 14.458 | 185 | 73.659 | 86.955/92.407 | 486 | 0.0 |
| PSM | noU | fixed | confirmed3 | 13765 | 29.096 | 48.878 | 6/48 | 17 | 9.023 | 88 | 73.195 | 86.453/92.114 | 694 | 0.0 |

CAL is the original512 mixed points:34 SWaT attack points and512 PSM attack points. Each model uses its own q99/higher threshold with strict >; this is not a certified normal-FPR budget. Strict and past-three/no-backfill confirmation remain separate. Complete-event coverage and all-QUERY point recall use different denominators; any-hit is not one-to-one matching. Long events exceed512 points, with steady coverage from event start+512; macro and pooled are both retained. Nonhits are right-censored and appear in events.csv.

## Paired changes

| Task | Contrast | Stream | Metric | n | Mean difference | Sample SD |
| --- | --- | --- | --- | --- | --- | --- |
| SWAT | R-MeanT−R | strict | AP (pp) | 3 | -0.426882 | 0.240936 |
| SWAT | R-MeanT−R | strict | ROC_AUC (pp) | 3 | 0.338252 | 0.024181 |
| SWAT | R-MeanT−R | strict | VUS_PR (pp) | 3 | -2.510224 | 0.270103 |
| SWAT | R-MeanT−R | strict | VUS_ROC (pp) | 3 | -0.614601 | 0.043913 |
| SWAT | R-MeanT−R | strict | FP | 3 | -306.333333 | 59.340823 |
| SWAT | R-MeanT−R | strict | TP | 3 | -620.000000 | 422.600284 |
| SWAT | R-MeanT−R | strict | normal_FPR (pp) | 3 | -0.106165 | 0.020566 |
| SWAT | R-MeanT−R | strict | Recall (pp) | 3 | -1.297696 | 0.884527 |
| SWAT | R-MeanT−R | strict | F1 (pp) | 3 | -0.632191 | 0.572736 |
| SWAT | R-MeanT−R | strict | M | 3 | -0.333333 | 0.577350 |
| SWAT | R-MeanT−R | strict | auxiliary_any_hit_events | 3 | -0.333333 | 0.577350 |
| SWAT | R-MeanT−R | strict | one_to_one_eventPrecision (pp) | 3 | 2.153907 | 2.083842 |
| SWAT | R-MeanT−R | strict | one_to_one_eventRecall (pp) | 3 | -1.515152 | 2.624319 |
| SWAT | R-MeanT−R | strict | one_to_one_eventF1 (pp) | 3 | 3.130296 | 3.071512 |
| SWAT | R-MeanT−R | strict | event_macro_coverage (pp) | 3 | -2.637979 | 2.490990 |
| SWAT | R-MeanT−R | strict | event_point_coverage (pp) | 3 | -1.297696 | 0.884527 |
| SWAT | R-MeanT−R | strict | long_steady_macro_coverage (pp) | 3 | -3.007017 | 5.042770 |
| SWAT | R-MeanT−R | strict | long_steady_point_coverage (pp) | 3 | -0.884727 | 0.357139 |
| SWAT | R-MeanT−R | strict | max_event_miss_points | 3 | 293.666667 | 65.454819 |
| SWAT | R-MeanT−R | strict | normal_alarm_fragments | 3 | -41.000000 | 36.097091 |
| SWAT | R-MeanT−R | strict | delay_hit_only_median | 3 | 30.666667 | 36.784281 |
| SWAT | R-MeanT−noU | strict | AP (pp) | 3 | -3.632057 | 0.033996 |
| SWAT | R-MeanT−noU | strict | ROC_AUC (pp) | 3 | -1.609725 | 0.013752 |
| SWAT | R-MeanT−noU | strict | VUS_PR (pp) | 3 | 3.701694 | 0.405501 |
| SWAT | R-MeanT−noU | strict | VUS_ROC (pp) | 3 | 7.701941 | 0.184060 |
| SWAT | R-MeanT−noU | strict | FP | 3 | 68.000000 | 45.210618 |
| SWAT | R-MeanT−noU | strict | TP | 3 | -1309.333333 | 32.083225 |
| SWAT | R-MeanT−noU | strict | normal_FPR (pp) | 3 | 0.023567 | 0.015669 |
| SWAT | R-MeanT−noU | strict | Recall (pp) | 3 | -2.740510 | 0.067152 |
| SWAT | R-MeanT−noU | strict | F1 (pp) | 3 | -1.978213 | 0.055850 |
| SWAT | R-MeanT−noU | strict | M | 3 | -0.333333 | 0.577350 |
| SWAT | R-MeanT−noU | strict | auxiliary_any_hit_events | 3 | -0.333333 | 0.577350 |
| SWAT | R-MeanT−noU | strict | one_to_one_eventPrecision (pp) | 3 | 0.909306 | 0.626277 |
| SWAT | R-MeanT−noU | strict | one_to_one_eventRecall (pp) | 3 | -1.515152 | 2.624319 |
| SWAT | R-MeanT−noU | strict | one_to_one_eventF1 (pp) | 3 | 1.217057 | 0.969750 |
| SWAT | R-MeanT−noU | strict | event_macro_coverage (pp) | 3 | -1.035766 | 0.157464 |
| SWAT | R-MeanT−noU | strict | event_point_coverage (pp) | 3 | -2.740510 | 0.067152 |
| SWAT | R-MeanT−noU | strict | long_steady_macro_coverage (pp) | 3 | -0.317792 | 0.008477 |
| SWAT | R-MeanT−noU | strict | long_steady_point_coverage (pp) | 3 | -3.228525 | 0.053756 |
| SWAT | R-MeanT−noU | strict | max_event_miss_points | 3 | 1254.000000 | 20.952327 |
| SWAT | R-MeanT−noU | strict | normal_alarm_fragments | 3 | -14.333333 | 5.686241 |
| SWAT | R-MeanT−noU | strict | delay_hit_only_median | 3 | -21.333333 | 37.829662 |
| SWAT | R-MeanT−R | confirmed3 | AP (pp) | 3 | -0.426882 | 0.240936 |
| SWAT | R-MeanT−R | confirmed3 | ROC_AUC (pp) | 3 | 0.338252 | 0.024181 |
| SWAT | R-MeanT−R | confirmed3 | VUS_PR (pp) | 3 | -2.510224 | 0.270103 |
| SWAT | R-MeanT−R | confirmed3 | VUS_ROC (pp) | 3 | -0.614601 | 0.043913 |
| SWAT | R-MeanT−R | confirmed3 | FP | 3 | -261.666667 | 19.218047 |
| SWAT | R-MeanT−R | confirmed3 | TP | 3 | -609.666667 | 414.019726 |
| SWAT | R-MeanT−R | confirmed3 | normal_FPR (pp) | 3 | -0.090685 | 0.006660 |
| SWAT | R-MeanT−R | confirmed3 | Recall (pp) | 3 | -1.276067 | 0.866567 |
| SWAT | R-MeanT−R | confirmed3 | F1 (pp) | 3 | -0.657910 | 0.589224 |
| SWAT | R-MeanT−R | confirmed3 | M | 3 | -0.333333 | 0.577350 |
| SWAT | R-MeanT−R | confirmed3 | auxiliary_any_hit_events | 3 | -0.333333 | 0.577350 |
| SWAT | R-MeanT−R | confirmed3 | one_to_one_eventPrecision (pp) | 3 | 0.271589 | 0.647028 |
| SWAT | R-MeanT−R | confirmed3 | one_to_one_eventRecall (pp) | 3 | -1.515152 | 2.624319 |
| SWAT | R-MeanT−R | confirmed3 | one_to_one_eventF1 (pp) | 3 | 0.291857 | 1.037477 |
| SWAT | R-MeanT−R | confirmed3 | event_macro_coverage (pp) | 3 | -2.554473 | 2.431414 |
| SWAT | R-MeanT−R | confirmed3 | event_point_coverage (pp) | 3 | -1.276067 | 0.866567 |
| SWAT | R-MeanT−R | confirmed3 | long_steady_macro_coverage (pp) | 3 | -3.007017 | 5.042770 |
| SWAT | R-MeanT−R | confirmed3 | long_steady_point_coverage (pp) | 3 | -0.884727 | 0.357139 |
| SWAT | R-MeanT−R | confirmed3 | max_event_miss_points | 3 | 293.666667 | 65.454819 |
| SWAT | R-MeanT−R | confirmed3 | normal_alarm_fragments | 3 | -4.333333 | 5.131601 |
| SWAT | R-MeanT−R | confirmed3 | delay_hit_only_median | 3 | 28.166667 | 39.409178 |
| SWAT | R-MeanT−noU | confirmed3 | AP (pp) | 3 | -3.632057 | 0.033996 |
| SWAT | R-MeanT−noU | confirmed3 | ROC_AUC (pp) | 3 | -1.609725 | 0.013752 |
| SWAT | R-MeanT−noU | confirmed3 | VUS_PR (pp) | 3 | 3.701694 | 0.405501 |
| SWAT | R-MeanT−noU | confirmed3 | VUS_ROC (pp) | 3 | 7.701941 | 0.184060 |
| SWAT | R-MeanT−noU | confirmed3 | FP | 3 | 92.000000 | 52.306787 |
| SWAT | R-MeanT−noU | confirmed3 | TP | 3 | -1303.333333 | 30.088758 |
| SWAT | R-MeanT−noU | confirmed3 | normal_FPR (pp) | 3 | 0.031884 | 0.018128 |
| SWAT | R-MeanT−noU | confirmed3 | Recall (pp) | 3 | -2.727951 | 0.062977 |
| SWAT | R-MeanT−noU | confirmed3 | F1 (pp) | 3 | -1.994612 | 0.064379 |
| SWAT | R-MeanT−noU | confirmed3 | M | 3 | -0.333333 | 0.577350 |
| SWAT | R-MeanT−noU | confirmed3 | auxiliary_any_hit_events | 3 | -0.333333 | 0.577350 |
| SWAT | R-MeanT−noU | confirmed3 | one_to_one_eventPrecision (pp) | 3 | 1.058479 | 0.618055 |
| SWAT | R-MeanT−noU | confirmed3 | one_to_one_eventRecall (pp) | 3 | -1.515152 | 2.624319 |
| SWAT | R-MeanT−noU | confirmed3 | one_to_one_eventF1 (pp) | 3 | 1.410210 | 1.002453 |
| SWAT | R-MeanT−noU | confirmed3 | event_macro_coverage (pp) | 3 | -0.951893 | 0.131757 |
| SWAT | R-MeanT−noU | confirmed3 | event_point_coverage (pp) | 3 | -2.727951 | 0.062977 |
| SWAT | R-MeanT−noU | confirmed3 | long_steady_macro_coverage (pp) | 3 | -0.197672 | 0.008477 |
| SWAT | R-MeanT−noU | confirmed3 | long_steady_point_coverage (pp) | 3 | -3.223366 | 0.053756 |
| SWAT | R-MeanT−noU | confirmed3 | max_event_miss_points | 3 | 1254.000000 | 20.952327 |
| SWAT | R-MeanT−noU | confirmed3 | normal_alarm_fragments | 3 | -15.000000 | 1.000000 |
| SWAT | R-MeanT−noU | confirmed3 | delay_hit_only_median | 3 | -21.333333 | 37.829662 |
| PSM | R-MeanT−R | strict | AP (pp) | 3 | 0.890795 | 0.504555 |
| PSM | R-MeanT−R | strict | ROC_AUC (pp) | 3 | 0.758157 | 0.125017 |
| PSM | R-MeanT−R | strict | VUS_PR (pp) | 3 | -0.533964 | 0.699929 |
| PSM | R-MeanT−R | strict | VUS_ROC (pp) | 3 | -0.826700 | 0.314703 |
| PSM | R-MeanT−R | strict | FP | 3 | 1076.333333 | 1329.067844 |
| PSM | R-MeanT−R | strict | TP | 3 | 914.333333 | 547.761201 |
| PSM | R-MeanT−R | strict | normal_FPR (pp) | 3 | 2.275113 | 2.809334 |
| PSM | R-MeanT−R | strict | Recall (pp) | 3 | 5.249058 | 3.144619 |
| PSM | R-MeanT−R | strict | F1 (pp) | 3 | 3.365800 | 1.293367 |
| PSM | R-MeanT−R | strict | M | 3 | -0.333333 | 2.081666 |
| PSM | R-MeanT−R | strict | auxiliary_any_hit_events | 3 | 1.000000 | 2.645751 |
| PSM | R-MeanT−R | strict | one_to_one_eventPrecision (pp) | 3 | 0.180752 | 0.142240 |
| PSM | R-MeanT−R | strict | one_to_one_eventRecall (pp) | 3 | -0.694444 | 4.336804 |
| PSM | R-MeanT−R | strict | one_to_one_eventF1 (pp) | 3 | 0.309860 | 0.232984 |
| PSM | R-MeanT−R | strict | event_macro_coverage (pp) | 3 | 1.257551 | 3.802726 |
| PSM | R-MeanT−R | strict | event_point_coverage (pp) | 3 | 8.116986 | 4.412686 |
| PSM | R-MeanT−R | strict | long_steady_macro_coverage (pp) | 3 | 5.076521 | 5.450829 |
| PSM | R-MeanT−R | strict | long_steady_point_coverage (pp) | 3 | 12.075881 | 4.249298 |
| PSM | R-MeanT−R | strict | max_event_miss_points | 3 | -21.333333 | 97.572195 |
| PSM | R-MeanT−R | strict | normal_alarm_fragments | 3 | 4.000000 | 40.261644 |
| PSM | R-MeanT−R | strict | delay_hit_only_median | 3 | 0.166667 | 0.288675 |
| PSM | R-MeanT−noU | strict | AP (pp) | 3 | 1.028886 | 0.121548 |
| PSM | R-MeanT−noU | strict | ROC_AUC (pp) | 3 | 0.851387 | 0.149741 |
| PSM | R-MeanT−noU | strict | VUS_PR (pp) | 3 | 6.587213 | 0.309984 |
| PSM | R-MeanT−noU | strict | VUS_ROC (pp) | 3 | 6.801094 | 0.184518 |
| PSM | R-MeanT−noU | strict | FP | 3 | -10249.333333 | 1041.053473 |
| PSM | R-MeanT−noU | strict | TP | 3 | -2153.666667 | 351.507231 |
| PSM | R-MeanT−noU | strict | normal_FPR (pp) | 3 | -21.664659 | 2.200540 |
| PSM | R-MeanT−noU | strict | Recall (pp) | 3 | -12.363894 | 2.017953 |
| PSM | R-MeanT−noU | strict | F1 (pp) | 3 | 4.097418 | 0.283915 |
| PSM | R-MeanT−noU | strict | M | 3 | 11.000000 | 1.000000 |
| PSM | R-MeanT−noU | strict | auxiliary_any_hit_events | 3 | 2.333333 | 1.527525 |
| PSM | R-MeanT−noU | strict | one_to_one_eventPrecision (pp) | 3 | -4.514021 | 0.114928 |
| PSM | R-MeanT−noU | strict | one_to_one_eventRecall (pp) | 3 | 22.916667 | 2.083333 |
| PSM | R-MeanT−noU | strict | one_to_one_eventF1 (pp) | 3 | -6.183878 | 0.213091 |
| PSM | R-MeanT−noU | strict | event_macro_coverage (pp) | 3 | -6.926852 | 2.080114 |
| PSM | R-MeanT−noU | strict | event_point_coverage (pp) | 3 | -20.491441 | 3.355334 |
| PSM | R-MeanT−noU | strict | long_steady_macro_coverage (pp) | 3 | -34.418234 | 4.208780 |
| PSM | R-MeanT−noU | strict | long_steady_point_coverage (pp) | 3 | -21.831978 | 2.927070 |
| PSM | R-MeanT−noU | strict | max_event_miss_points | 3 | 26.666667 | 97.572195 |
| PSM | R-MeanT−noU | strict | normal_alarm_fragments | 3 | 315.666667 | 19.731531 |
| PSM | R-MeanT−noU | strict | delay_hit_only_median | 3 | 0.166667 | 0.288675 |
| PSM | R-MeanT−R | confirmed3 | AP (pp) | 3 | 0.890795 | 0.504555 |
| PSM | R-MeanT−R | confirmed3 | ROC_AUC (pp) | 3 | 0.758157 | 0.125017 |
| PSM | R-MeanT−R | confirmed3 | VUS_PR (pp) | 3 | -0.533964 | 0.699929 |
| PSM | R-MeanT−R | confirmed3 | VUS_ROC (pp) | 3 | -0.826700 | 0.314703 |
| PSM | R-MeanT−R | confirmed3 | FP | 3 | 1067.333333 | 1266.741226 |
| PSM | R-MeanT−R | confirmed3 | TP | 3 | 939.333333 | 490.568378 |
| PSM | R-MeanT−R | confirmed3 | normal_FPR (pp) | 3 | 2.256089 | 2.677590 |
| PSM | R-MeanT−R | confirmed3 | Recall (pp) | 3 | 5.392579 | 2.816283 |
| PSM | R-MeanT−R | confirmed3 | F1 (pp) | 3 | 3.970340 | 1.339035 |
| PSM | R-MeanT−R | confirmed3 | M | 3 | -0.666667 | 1.527525 |
| PSM | R-MeanT−R | confirmed3 | auxiliary_any_hit_events | 3 | 0.666667 | 2.081666 |
| PSM | R-MeanT−R | confirmed3 | one_to_one_eventPrecision (pp) | 3 | -0.355343 | 0.366663 |
| PSM | R-MeanT−R | confirmed3 | one_to_one_eventRecall (pp) | 3 | -1.388889 | 3.182344 |
| PSM | R-MeanT−R | confirmed3 | one_to_one_eventF1 (pp) | 3 | -0.565541 | 0.572149 |
| PSM | R-MeanT−R | confirmed3 | event_macro_coverage (pp) | 3 | 0.699957 | 1.935775 |
| PSM | R-MeanT−R | confirmed3 | event_point_coverage (pp) | 3 | 8.616007 | 4.077218 |
| PSM | R-MeanT−R | confirmed3 | long_steady_macro_coverage (pp) | 3 | 5.229590 | 5.328910 |
| PSM | R-MeanT−R | confirmed3 | long_steady_point_coverage (pp) | 3 | 13.490515 | 4.416732 |
| PSM | R-MeanT−R | confirmed3 | max_event_miss_points | 3 | -116.333333 | 201.495244 |
| PSM | R-MeanT−R | confirmed3 | normal_alarm_fragments | 3 | 10.666667 | 22.143472 |
| PSM | R-MeanT−R | confirmed3 | delay_hit_only_median | 3 | 0.000000 | 0.000000 |
| PSM | R-MeanT−noU | confirmed3 | AP (pp) | 3 | 1.028886 | 0.121548 |
| PSM | R-MeanT−noU | confirmed3 | ROC_AUC (pp) | 3 | 0.851387 | 0.149741 |
| PSM | R-MeanT−noU | confirmed3 | VUS_PR (pp) | 3 | 6.587213 | 0.309984 |
| PSM | R-MeanT−noU | confirmed3 | VUS_ROC (pp) | 3 | 6.801094 | 0.184518 |
| PSM | R-MeanT−noU | confirmed3 | FP | 3 | -10726.666667 | 1025.539045 |
| PSM | R-MeanT−noU | confirmed3 | TP | 3 | -2537.000000 | 346.137256 |
| PSM | R-MeanT−noU | confirmed3 | normal_FPR (pp) | 3 | -22.673628 | 2.167746 |
| PSM | R-MeanT−noU | confirmed3 | Recall (pp) | 3 | -14.564556 | 1.987125 |
| PSM | R-MeanT−noU | confirmed3 | F1 (pp) | 3 | 2.319110 | 0.477222 |
| PSM | R-MeanT−noU | confirmed3 | M | 3 | 5.000000 | 1.000000 |
| PSM | R-MeanT−noU | confirmed3 | auxiliary_any_hit_events | 3 | -4.666667 | 1.527525 |
| PSM | R-MeanT−noU | confirmed3 | one_to_one_eventPrecision (pp) | 3 | -2.789568 | 0.355581 |
| PSM | R-MeanT−noU | confirmed3 | one_to_one_eventRecall (pp) | 3 | 10.416667 | 2.083333 |
| PSM | R-MeanT−noU | confirmed3 | one_to_one_eventF1 (pp) | 3 | -1.825756 | 0.600089 |
| PSM | R-MeanT−noU | confirmed3 | event_macro_coverage (pp) | 3 | -12.503205 | 1.559273 |
| PSM | R-MeanT−noU | confirmed3 | event_point_coverage (pp) | 3 | -22.746510 | 3.264304 |
| PSM | R-MeanT−noU | confirmed3 | long_steady_macro_coverage (pp) | 3 | -38.015803 | 4.281022 |
| PSM | R-MeanT−noU | confirmed3 | long_steady_point_coverage (pp) | 3 | -23.571816 | 3.346211 |
| PSM | R-MeanT−noU | confirmed3 | max_event_miss_points | 3 | 417.666667 | 201.495244 |
| PSM | R-MeanT−noU | confirmed3 | normal_alarm_fragments | 3 | 94.333333 | 7.637626 |
| PSM | R-MeanT−noU | confirmed3 | delay_hit_only_median | 3 | 0.000000 | 0.000000 |

Differences are within-domain and within-seed against R; noU is the same frozen reference for all three contrasts. Sample SD uses ddof=1, n=3. Seeds are not datasets or independent event samples. Low-FPR areas/cap coverage are descriptive QUERY calculations in the single LOW_FPR_SUPPLEMENT table; their cutoffs never replace CAL thresholds.


## Technical decision

PSM ranking gains repeat in all three pairs: R-MeanT minus R AP is+0.864,+0.400,+1.408pp, mean+0.891+/-0.505pp; ROC-AUC mean+0.758+/-0.125pp. Against the one frozen noU reference AP is+1.029+/-0.122pp. This is a real but task-limited ranking effect, not broad superiority: VUS-PR changes-0.534+/-0.700pp and VUS-ROC-0.827+/-0.315pp versus R. At original CAL, C3 normal FPR increases+2.256+/-2.678pp with recall+5.393+/-2.816pp, long steady macro+5.230+/-5.329pp and pooled+13.491+/-4.417pp. Matched events12/10/11 versus R11/12/12 and any-hit14/11/12 versus11/12/12 do not establish a repeated extra-event advantage. Longest miss improves only in2025 (1228 to879); in2026/2027 both arms remain1228.

On SWaT, AP decreases in all seeds (-0.427+/-0.241pp), ROC-AUC improves+0.338+/-0.024pp, and VUS-PR/ROC decline relative to R. C3 FPR falls0.091+/-0.007pp alongside recall-1.276+/-0.867pp and long macro-3.007+/-5.043pp. This does not demonstrate the corresponding sustained-detection advantage. The apparent precision of pooled coverage does not replace the variable per-event macro coverage.

Low-FPR evidence qualifies the PSM positive: strict normalized partial areas improve over R, but the C3 effect is supported more clearly through5% than through1% FPR. Whole-tie attainable recall, continuous-area gain, event matching and long macro/pooled coverage can move differently; the supplement retains every case. These are retrospective comparisons, not threshold recommendations. Against noU, reduced PSM false alarms coexist with substantially lower coverage, so the learned candidate is not an unconditional replacement for noU.

Retain this fixed candidate as a reproducible, limited PSM learning effect with SWaT negatives and operating costs. It is not established cross-task evidence preservation or contamination robustness, and mature TimeRadar, replay and the error/disagreement decomposition keep their original attribution. This matrix is complete; no loss, threshold, seed, structure or data extension is queued.

## Computation and retained evidence

Eight new adaptations and forty new readout parts plus four normal engineering checks consumed10.828305763A40GPUh, including startup and cleanup. All52 actual GPU jobs completed and cleared. A separate pre-start guard rejection used zero GPU seconds and2.233744947CPU wall seconds. The original thirty successful readouts were retained; only ten never-started parts were recovered after guarded resources became available. The original controller exception, exact-status limitation, CPU pidfd fix, resource receipts and original runtime are preserved outside Git. No successful training/inference was repeated.

The final16-bank seal and metric hashes are in PROVENANCE.json; actual53 dispatch records are in COMPUTE.csv. Checkpoints/optimizers embed official pretrained weights and remain archived rather than republished. MODEL_AND_CACHE_IDENTITIES.json carries their hashes, plans, scalers and all bank identities. Full per-seed observations, n=3 method summaries, paired differences, event details and low-FPR tables are provided, with old results and source snapshots preserved. The key review is role-separated within the same model family, not external double-blind peer review.
