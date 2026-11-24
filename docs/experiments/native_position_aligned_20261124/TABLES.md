# Complete-window source-training comparison, seed2025

All three decoder arms use the same official initialization, normal-window plan, 240-update cap and seven full-position SOURCE SELECT checkpoints. The reconstruction arm uses source-event labels for selection but not its training gradient. The supervised arms share the exact positive/normal windows and weights. Point-count weights represent window-sampling strata, not uniform abnormal time-point supervision.

| Domain | Method | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| PSM | Normal reconstruction control | 0.45528 | 0.69075 | 0.44872 | 0.67387 |
| PSM | Endpoint-supervised control | 0.44899 | 0.69144 | 0.45661 | 0.68777 |
| PSM | All-position supervised candidate | 0.46225 | 0.71037 | 0.45351 | 0.67682 |
| PSM | Official TimeRadar V | 0.44824 | 0.67842 | 0.45090 | 0.68182 |
| PSM | Official E | 0.43590 | 0.60779 | 0.43846 | 0.61526 |
| PSM | Official fixed V+E | 0.44836 | 0.61891 | 0.44821 | 0.62409 |
| PSM | Complete LEFT | 0.57690 | 0.77618 | 0.57413 | 0.78579 |
| SMD | Normal reconstruction control | 0.45069 | 0.84241 | 0.47706 | 0.88994 |
| SMD | Endpoint-supervised control | 0.43390 | 0.84721 | 0.46462 | 0.89466 |
| SMD | All-position supervised candidate | 0.45628 | 0.84825 | 0.47608 | 0.88921 |
| SMD | Official TimeRadar V | 0.42716 | 0.82717 | 0.45194 | 0.88232 |
| SMD | Official E | 0.42720 | 0.81995 | 0.45557 | 0.86840 |
| SMD | Official fixed V+E | 0.43725 | 0.82679 | 0.46495 | 0.87375 |
| SMD | Complete LEFT | 0.49809 | 0.87855 | 0.54184 | 0.91853 |
| MSL | Normal reconstruction control | 0.37066 | 0.67779 | 0.40455 | 0.72458 |
| MSL | Endpoint-supervised control | 0.35521 | 0.68378 | 0.39270 | 0.73002 |
| MSL | All-position supervised candidate | 0.36696 | 0.67751 | 0.41192 | 0.72658 |
| MSL | Official TimeRadar V | 0.34850 | 0.66935 | 0.38184 | 0.71798 |
| MSL | Official E | 0.39988 | 0.71093 | 0.43881 | 0.76245 |
| MSL | Official fixed V+E | 0.40407 | 0.71697 | 0.44381 | 0.76558 |
| MSL | Complete LEFT | 0.50834 | 0.76334 | 0.55308 | 0.79416 |
| SMAP | Normal reconstruction control | 0.33734 | 0.67429 | 0.33244 | 0.71321 |
| SMAP | Endpoint-supervised control | 0.33698 | 0.67140 | 0.33367 | 0.71295 |
| SMAP | All-position supervised candidate | 0.33681 | 0.66996 | 0.34702 | 0.71341 |
| SMAP | Official TimeRadar V | 0.30165 | 0.66240 | 0.29931 | 0.70426 |
| SMAP | Official E | 0.38124 | 0.71881 | 0.38102 | 0.76553 |
| SMAP | Official fixed V+E | 0.38662 | 0.72699 | 0.38210 | 0.77097 |
| SMAP | Complete LEFT | 0.49315 | 0.76287 | 0.52172 | 0.78494 |

All entries use the latest legal per-entity intersection of W100/stride100 outputs, archived causal coverage and LEFT scored positions. Every new method is an offline complete-window detector. No causal bank or binary threshold was generated. NASA uses sensor0 and entity-first macro averaging; SMD normal records participated in source training. LEFT is target-normal trained, with W96 MSL/W192 other datasets and native offline context. Each new arm has one project seed; no training SD is estimated.

## Candidate minus controls

| Domain | Reference | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| PSM | Endpoint-supervised control | 0.01326 | 0.01894 | -0.00310 | -0.01094 |
| PSM | Normal reconstruction control | 0.00697 | 0.01962 | 0.00478 | 0.00295 |
| PSM | Official TimeRadar V | 0.01402 | 0.03195 | 0.00260 | -0.00500 |
| PSM | Complete LEFT | -0.11464 | -0.06581 | -0.12062 | -0.10896 |
| SMD | Endpoint-supervised control | 0.02238 | 0.00104 | 0.01146 | -0.00545 |
| SMD | Normal reconstruction control | 0.00559 | 0.00584 | -0.00098 | -0.00073 |
| SMD | Official TimeRadar V | 0.02912 | 0.02108 | 0.02414 | 0.00689 |
| SMD | Complete LEFT | -0.04180 | -0.03030 | -0.06576 | -0.02932 |
| MSL | Endpoint-supervised control | 0.01176 | -0.00627 | 0.01923 | -0.00344 |
| MSL | Normal reconstruction control | -0.00370 | -0.00028 | 0.00737 | 0.00199 |
| MSL | Official TimeRadar V | 0.01846 | 0.00816 | 0.03009 | 0.00860 |
| MSL | Complete LEFT | -0.14138 | -0.08583 | -0.14116 | -0.06758 |
| SMAP | Endpoint-supervised control | -0.00018 | -0.00144 | 0.01335 | 0.00046 |
| SMAP | Normal reconstruction control | -0.00053 | -0.00433 | 0.01458 | 0.00021 |
| SMAP | Official TimeRadar V | 0.03516 | 0.00756 | 0.04771 | 0.00915 |
| SMAP | Complete LEFT | -0.15635 | -0.09292 | -0.17470 | -0.07153 |

[Full entity metrics](entity_metrics.csv) · [entity differences](paired_entity.csv) · [wins/ties/losses and median differences](paired_distribution.csv).
