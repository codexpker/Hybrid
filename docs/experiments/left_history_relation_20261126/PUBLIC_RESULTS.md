# Public detection with historical lag references

Scores cover every eligible current W192 window on its original grid. The first current window begins 384 points after each record origin, with history [128,384). A window's 192 scores become available at its end. Lag scores share one value across each of its six 32-point blocks. SMD averages entity metrics with equal weight over all 28 machines. New source training uses seed 2025.

## External comparison

The intersection begins at PSM 384, SWaT 495384, and SMD relative 384. Exclusive ends are PSM 87744, SWaT 944856, and SMD min(192 floor(N/192),100 floor(N/100)). [Per-entity axes and label counts](public_axes.csv)

### AP

| Detector | PSM | SWaT | SMD28 |
|---|---:|---:|---:|
| Frozen multisource LEFT | 0.49699 | 0.72392 | 0.49810 |
| Matched source continuation LEFT | 0.51477 | 0.72276 | 0.50223 |
| Last historical block, lag persistence | 0.33002 | 0.15994 | 0.09115 |
| Matched LEFT with lag persistence | 0.42968 | 0.71174 | 0.45995 |
| LEFT with learned historical lag references | 0.42415 | 0.71614 | 0.46340 |
| Target-normal LEFT from scratch | 0.57720 | 0.54388 | 0.49874 |
| Multisource LEFT with target-normal training | 0.55485 | 0.54251 | 0.51883 |
| Level deviation (SmoothL1) | 0.43868 | 0.74480 | 0.42859 |
| TimeRadar, archived official block readout | 0.43410 | 0.63931 | NA |
| TimeRadar, official endpoint variance | 0.53580 | NA | 0.44527 |
| DADA, official frozen readout | 0.39619 | NA | 0.41515 |
| TimeRCD, official weights, FIT-only adapter | 0.30278 | NA | 0.47365 |
| PCA residual | 0.53451 | NA | NA |
| CATCH | 0.43711 ± 0.00842 | NA | NA |

### ROC-AUC

| Detector | PSM | SWaT | SMD28 |
|---|---:|---:|---:|
| Frozen multisource LEFT | 0.70069 | 0.81137 | 0.86755 |
| Matched source continuation LEFT | 0.71084 | 0.81158 | 0.87603 |
| Last historical block, lag persistence | 0.53610 | 0.51749 | 0.61914 |
| Matched LEFT with lag persistence | 0.64949 | 0.81339 | 0.84436 |
| LEFT with learned historical lag references | 0.64553 | 0.81428 | 0.82960 |
| Target-normal LEFT from scratch | 0.77578 | 0.82074 | 0.87814 |
| Multisource LEFT with target-normal training | 0.75626 | 0.85550 | 0.88259 |
| Level deviation (SmoothL1) | 0.65349 | 0.83168 | 0.80302 |
| TimeRadar, archived official block readout | 0.67077 | 0.82186 | NA |
| TimeRadar, official endpoint variance | 0.73091 | NA | 0.81270 |
| DADA, official frozen readout | 0.60443 | NA | 0.85098 |
| TimeRCD, official weights, FIT-only adapter | 0.47672 | NA | 0.85949 |
| PCA residual | 0.79239 | NA | NA |
| CATCH | 0.64676 ± 0.00599 | NA | NA |

### VUS-PR

| Detector | PSM | SWaT | SMD28 |
|---|---:|---:|---:|
| Frozen multisource LEFT | 0.50261 | 0.49778 | 0.54243 |
| Matched source continuation LEFT | 0.51876 | 0.49775 | 0.54620 |
| Last historical block, lag persistence | 0.34094 | 0.15269 | 0.10770 |
| Matched LEFT with lag persistence | 0.44567 | 0.48642 | 0.48550 |
| LEFT with learned historical lag references | 0.43137 | 0.49023 | 0.47254 |
| Target-normal LEFT from scratch | 0.57443 | 0.53469 | 0.54269 |
| Multisource LEFT with target-normal training | 0.55351 | 0.53087 | 0.56492 |
| Level deviation (SmoothL1) | 0.44755 | 0.46411 | 0.42802 |
| TimeRadar, archived official block readout | 0.44251 | 0.51411 | NA |
| TimeRadar, official endpoint variance | 0.48438 | NA | 0.46819 |
| DADA, official frozen readout | 0.44004 | NA | 0.47532 |
| TimeRCD, official weights, FIT-only adapter | 0.34942 | NA | 0.51670 |
| PCA residual | 0.52887 | NA | NA |
| CATCH | 0.52746 ± 0.00866 | NA | NA |

### VUS-ROC

| Detector | PSM | SWaT | SMD28 |
|---|---:|---:|---:|
| Frozen multisource LEFT | 0.70672 | 0.66096 | 0.91216 |
| Matched source continuation LEFT | 0.71798 | 0.66483 | 0.91719 |
| Last historical block, lag persistence | 0.51054 | 0.51800 | 0.66645 |
| Matched LEFT with lag persistence | 0.66899 | 0.68838 | 0.89797 |
| LEFT with learned historical lag references | 0.65901 | 0.67604 | 0.88470 |
| Target-normal LEFT from scratch | 0.78534 | 0.80915 | 0.91780 |
| Multisource LEFT with target-normal training | 0.76629 | 0.84299 | 0.91995 |
| Level deviation (SmoothL1) | 0.65524 | 0.64387 | 0.86573 |
| TimeRadar, archived official block readout | 0.67889 | 0.76674 | NA |
| TimeRadar, official endpoint variance | 0.66498 | NA | 0.86965 |
| DADA, official frozen readout | 0.65150 | NA | 0.89919 |
| TimeRCD, official weights, FIT-only adapter | 0.53255 | NA | 0.90971 |
| PCA residual | 0.76594 | NA | NA |
| CATCH | 0.69619 ± 0.00657 | NA | NA |

## Full eligible window grid

| Method | Domain | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---:|---:|---:|---:|
| Frozen multisource LEFT | PSM | 0.49699 | 0.70069 | 0.50261 | 0.70672 |
| Frozen multisource LEFT | SWAT | 0.72392 | 0.81137 | 0.49778 | 0.66096 |
| Frozen multisource LEFT | SMD | 0.49809 | 0.86754 | 0.54243 | 0.91216 |
| Matched source continuation LEFT | PSM | 0.51477 | 0.71084 | 0.51876 | 0.71798 |
| Matched source continuation LEFT | SWAT | 0.72276 | 0.81158 | 0.49775 | 0.66483 |
| Matched source continuation LEFT | SMD | 0.50221 | 0.87601 | 0.54645 | 0.91718 |
| Last historical block, lag persistence | PSM | 0.33002 | 0.53610 | 0.34094 | 0.51054 |
| Last historical block, lag persistence | SWAT | 0.15994 | 0.51749 | 0.15269 | 0.51800 |
| Last historical block, lag persistence | SMD | 0.09114 | 0.61914 | 0.10767 | 0.66645 |
| Matched LEFT with lag persistence | PSM | 0.42968 | 0.64949 | 0.44567 | 0.66899 |
| Matched LEFT with lag persistence | SWAT | 0.71174 | 0.81339 | 0.48642 | 0.68838 |
| Matched LEFT with lag persistence | SMD | 0.45994 | 0.84436 | 0.48552 | 0.89798 |
| LEFT with learned historical lag references | PSM | 0.42415 | 0.64553 | 0.43137 | 0.65901 |
| LEFT with learned historical lag references | SWAT | 0.71614 | 0.81428 | 0.49023 | 0.67604 |
| LEFT with learned historical lag references | SMD | 0.46339 | 0.82960 | 0.47250 | 0.88472 |

PSM/SWaT source checkpoints use normal TEP and SMD; the SMD fold uses TEP and HAI. Source-transfer rows share target-normal prefix statistics. Target-normal LEFT also uses these records for normal training and selection. Level deviation uses normal statistics and the current observation. Native LEFT readouts use full W192. The persistence statistic compares fixed lag coordinates with the last observed history block.

TimeRadar endpoint uses past/current W100 and official frozen weights. The archived block0 readout retains its own grouped-window context. DADA uses frozen official weights and complete W100. TimeRCD uses FIT-only statistics and full windows up to 5000. CATCH uses W192 and target-normal training. Its three archived seeds give the mean and sample SD. PCA is a target-normal pointwise residual. NA indicates unavailable compatible coverage.

### Literature panel

| Method and reported metric | PSM | SWaT | SMD |
|---|---:|---:|---:|
| CrossAD VUS-PR | .5596 | .4767 | .2344 |
| CrossAD VUS-ROC | .7302 | .7865 | .8580 |
| ModernTCN VUS-PR | .4668 | .0941 | .1596 |
| ModernTCN VUS-ROC | .6480 | .2735 | .7707 |

Author-reported values retain their published data, metrics and coverage. [CrossAD Tables 2–3](https://arxiv.org/html/2510.12489) · [ModernTCN in LEFT Table 2](https://arxiv.org/html/2602.08638v2#S4.T2)

## Source update and readout comparisons

Differences below use every eligible W192 point through 192 floor(N/192) on each record. The learned method and continued LEFT share source windows and optimizer budget. Same-checkpoint readout contrasts evaluate self-lag, cross-lag, native, learned and persistence components without target fitting.

| Signed difference | Domain | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---:|---:|---:|---:|
| LEFT with learned historical lag references minus Frozen multisource LEFT | PSM | -0.07284 | -0.05516 | -0.07124 | -0.04771 |
| LEFT with learned historical lag references minus Frozen multisource LEFT | SWAT | -0.00778 | +0.00291 | -0.00755 | +0.01508 |
| LEFT with learned historical lag references minus Frozen multisource LEFT | SMD | -0.03469 | -0.03793 | -0.06993 | -0.02744 |
| LEFT with learned historical lag references minus Matched source continuation LEFT | PSM | -0.09062 | -0.06531 | -0.08739 | -0.05897 |
| LEFT with learned historical lag references minus Matched source continuation LEFT | SWAT | -0.00662 | +0.00270 | -0.00752 | +0.01121 |
| LEFT with learned historical lag references minus Matched source continuation LEFT | SMD | -0.03882 | -0.04641 | -0.07395 | -0.03246 |
| LEFT with learned historical lag references minus Matched LEFT with lag persistence | PSM | -0.00553 | -0.00396 | -0.01430 | -0.00997 |
| LEFT with learned historical lag references minus Matched LEFT with lag persistence | SWAT | +0.00441 | +0.00089 | +0.00381 | -0.01234 |
| LEFT with learned historical lag references minus Matched LEFT with lag persistence | SMD | +0.00345 | -0.01475 | -0.01302 | -0.01326 |
| LEFT with learned historical lag references minus Last historical block, lag persistence | PSM | +0.09413 | +0.10944 | +0.09043 | +0.14848 |
| LEFT with learned historical lag references minus Last historical block, lag persistence | SWAT | +0.55620 | +0.29680 | +0.33754 | +0.15804 |
| LEFT with learned historical lag references minus Last historical block, lag persistence | SMD | +0.37225 | +0.21047 | +0.36483 | +0.21827 |
| LEFT with learned historical lag references minus Joint checkpoint, native score only | PSM | -0.09116 | -0.06518 | -0.08765 | -0.05869 |
| LEFT with learned historical lag references minus Joint checkpoint, native score only | SWAT | -0.00705 | +0.00363 | -0.00900 | +0.01405 |
| LEFT with learned historical lag references minus Joint checkpoint, native score only | SMD | -0.03975 | -0.04758 | -0.07556 | -0.03385 |
| LEFT with learned historical lag references minus Joint checkpoint, native + self-lag | PSM | -0.02112 | -0.01770 | -0.02521 | -0.01969 |
| LEFT with learned historical lag references minus Joint checkpoint, native + self-lag | SWAT | -0.00588 | -0.00368 | -0.01132 | -0.00464 |
| LEFT with learned historical lag references minus Joint checkpoint, native + self-lag | SMD | +0.01479 | +0.01395 | +0.03208 | +0.02041 |
| LEFT with learned historical lag references minus Joint checkpoint, native + cross-lag | PSM | +0.03187 | +0.02842 | +0.03488 | +0.03118 |
| LEFT with learned historical lag references minus Joint checkpoint, native + cross-lag | SWAT | +0.00484 | +0.00409 | -0.00011 | -0.01161 |
| LEFT with learned historical lag references minus Joint checkpoint, native + cross-lag | SMD | +0.01214 | -0.00469 | +0.00995 | -0.00341 |
| LEFT with learned historical lag references minus Target-normal LEFT from scratch | PSM | -0.15305 | -0.13024 | -0.14306 | -0.12633 |
| LEFT with learned historical lag references minus Target-normal LEFT from scratch | SWAT | +0.17227 | -0.00646 | -0.04446 | -0.13311 |
| LEFT with learned historical lag references minus Target-normal LEFT from scratch | SMD | -0.03533 | -0.04852 | -0.07014 | -0.03306 |
| LEFT with learned historical lag references minus Multisource LEFT with target-normal training | PSM | -0.13070 | -0.11073 | -0.12214 | -0.10727 |
| LEFT with learned historical lag references minus Multisource LEFT with target-normal training | SWAT | +0.17363 | -0.04121 | -0.04064 | -0.16695 |
| LEFT with learned historical lag references minus Multisource LEFT with target-normal training | SMD | -0.05541 | -0.05296 | -0.09238 | -0.03520 |
| LEFT with learned historical lag references minus Level deviation (SmoothL1) | PSM | -0.01453 | -0.00795 | -0.01618 | +0.00378 |
| LEFT with learned historical lag references minus Level deviation (SmoothL1) | SWAT | -0.02866 | -0.01740 | +0.02612 | +0.03217 |
| LEFT with learned historical lag references minus Level deviation (SmoothL1) | SMD | +0.03482 | +0.02659 | +0.04459 | +0.01902 |
| Joint checkpoint, native score only minus Matched source continuation LEFT | PSM | +0.00054 | -0.00013 | +0.00026 | -0.00028 |
| Joint checkpoint, native score only minus Matched source continuation LEFT | SWAT | +0.00043 | -0.00093 | +0.00148 | -0.00284 |
| Joint checkpoint, native score only minus Matched source continuation LEFT | SMD | +0.00093 | +0.00118 | +0.00161 | +0.00139 |
| Learned relation error only minus Last historical block, lag persistence | PSM | -0.01062 | +0.00630 | -0.00940 | -0.00435 |
| Learned relation error only minus Last historical block, lag persistence | SWAT | +0.08852 | +0.12639 | +0.07142 | +0.11795 |
| Learned relation error only minus Last historical block, lag persistence | SMD | +0.00359 | -0.02892 | -0.01248 | -0.04731 |
| Joint checkpoint, native + persistence minus Matched LEFT with lag persistence | PSM | +0.00037 | +0.00023 | +0.00040 | +0.00016 |
| Joint checkpoint, native + persistence minus Matched LEFT with lag persistence | SWAT | +0.00057 | +0.00009 | -0.00130 | -0.00290 |
| Joint checkpoint, native + persistence minus Matched LEFT with lag persistence | SMD | +0.00153 | +0.00091 | +0.00281 | +0.00123 |
| Joint checkpoint, native + self-lag minus Joint checkpoint, native score only | PSM | -0.07004 | -0.04748 | -0.06244 | -0.03900 |
| Joint checkpoint, native + self-lag minus Joint checkpoint, native score only | SWAT | -0.00117 | +0.00731 | +0.00232 | +0.01869 |
| Joint checkpoint, native + self-lag minus Joint checkpoint, native score only | SMD | -0.05454 | -0.06153 | -0.10764 | -0.05426 |
| Joint checkpoint, native + cross-lag minus Joint checkpoint, native score only | PSM | -0.12303 | -0.09359 | -0.12253 | -0.08987 |
| Joint checkpoint, native + cross-lag minus Joint checkpoint, native score only | SWAT | -0.01188 | -0.00046 | -0.00889 | +0.02567 |
| Joint checkpoint, native + cross-lag minus Joint checkpoint, native score only | SMD | -0.05189 | -0.04289 | -0.08551 | -0.03044 |

[All absolute readouts](public_macro.csv) · [Raw-score bindings and entity metrics](public_entity_metrics.csv) · [Individual paired differences](public_entity_differences.csv) · [Medians, wins, ties and losses](public_paired_summary.csv)

## History Mean512

The shared axis starts at relative 512 for this comparison. The archived SWaT history cache covers only a suffix, so the comparisons below cover PSM and SMD.

| Method | Domain | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---:|---:|---:|---:|
| History Mean512 | PSM | 0.32104 | 0.53004 | 0.34987 | 0.57301 |
| Matched source continuation LEFT | PSM | 0.51491 | 0.71064 | 0.51894 | 0.71779 |
| LEFT with learned historical lag references | PSM | 0.42428 | 0.64526 | 0.43144 | 0.65870 |
| History Mean512 | SMD | 0.11135 | 0.66183 | 0.14012 | 0.75353 |
| Matched source continuation LEFT | SMD | 0.50213 | 0.87548 | 0.54576 | 0.91667 |
| LEFT with learned historical lag references | SMD | 0.46332 | 0.82874 | 0.47187 | 0.88376 |

AP is average precision. VUS uses the archived LEFT software, windowSize 100 and 250 thresholds. An independent tie-aware ranking calculation checked every AP/ROC against the archived score arrays. VUS uses one implementation with independently checked aggregates. [Device score times and CPU evaluator worker-seconds](public_costs.csv)
