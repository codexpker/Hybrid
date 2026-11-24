# Repeated complete-window source supervision

| Domain | Method | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| PSM | Normal reconstruction control | 0.45620 ± 0.00129 | 0.69217 ± 0.00127 | 0.44822 ± 0.00069 | 0.67093 ± 0.00260 |
| PSM | Endpoint-supervised control | 0.44911 ± 0.00151 | 0.69214 ± 0.00097 | 0.45660 ± 0.00067 | 0.68739 ± 0.00056 |
| PSM | All-position supervised candidate | 0.46376 ± 0.00159 | 0.71062 ± 0.00030 | 0.45373 ± 0.00048 | 0.67598 ± 0.00091 |
| PSM | Official TimeRadar V | 0.44824 | 0.67842 | 0.45090 | 0.68182 |
| PSM | Official E | 0.43590 | 0.60779 | 0.43846 | 0.61526 |
| PSM | Official fixed V+E | 0.44836 | 0.61891 | 0.44821 | 0.62409 |
| PSM | Complete LEFT | 0.57690 | 0.77618 | 0.57413 | 0.78579 |
| SMD | Normal reconstruction control | 0.45364 ± 0.00290 | 0.84469 ± 0.00203 | 0.48029 ± 0.00322 | 0.89108 ± 0.00104 |
| SMD | Endpoint-supervised control | 0.43535 ± 0.00187 | 0.84781 ± 0.00074 | 0.46581 ± 0.00104 | 0.89471 ± 0.00020 |
| SMD | All-position supervised candidate | 0.45637 ± 0.00020 | 0.84738 ± 0.00076 | 0.47604 ± 0.00031 | 0.88843 ± 0.00074 |
| SMD | Official TimeRadar V | 0.42716 | 0.82717 | 0.45194 | 0.88232 |
| SMD | Official E | 0.42720 | 0.81995 | 0.45557 | 0.86840 |
| SMD | Official fixed V+E | 0.43725 | 0.82679 | 0.46495 | 0.87375 |
| SMD | Complete LEFT | 0.49809 | 0.87855 | 0.54184 | 0.91853 |
| MSL | Normal reconstruction control | 0.36796 ± 0.00282 | 0.67966 ± 0.00202 | 0.40338 ± 0.00115 | 0.72586 ± 0.00126 |
| MSL | Endpoint-supervised control | 0.35530 ± 0.00153 | 0.68530 ± 0.00151 | 0.39288 ± 0.00063 | 0.73091 ± 0.00079 |
| MSL | All-position supervised candidate | 0.36780 ± 0.00079 | 0.67710 ± 0.00114 | 0.41239 ± 0.00142 | 0.72611 ± 0.00062 |
| MSL | Official TimeRadar V | 0.34850 | 0.66935 | 0.38184 | 0.71798 |
| MSL | Official E | 0.39988 | 0.71093 | 0.43881 | 0.76245 |
| MSL | Official fixed V+E | 0.40407 | 0.71697 | 0.44381 | 0.76558 |
| MSL | Complete LEFT | 0.50834 | 0.76334 | 0.55308 | 0.79416 |
| SMAP | Normal reconstruction control | 0.33696 ± 0.00403 | 0.67129 ± 0.00299 | 0.33645 ± 0.00495 | 0.71018 ± 0.00283 |
| SMAP | Endpoint-supervised control | 0.33553 ± 0.00199 | 0.67119 ± 0.00019 | 0.33378 ± 0.00280 | 0.71234 ± 0.00060 |
| SMAP | All-position supervised candidate | 0.33759 ± 0.00068 | 0.66927 ± 0.00066 | 0.34671 ± 0.00043 | 0.71283 ± 0.00074 |
| SMAP | Official TimeRadar V | 0.30165 | 0.66240 | 0.29931 | 0.70426 |
| SMAP | Official E | 0.38124 | 0.71881 | 0.38102 | 0.76553 |
| SMAP | Official fixed V+E | 0.38662 | 0.72699 | 0.38210 | 0.77097 |
| SMAP | Complete LEFT | 0.49315 | 0.76287 | 0.52172 | 0.78494 |

Common axes are unchanged intersections of archived causal coverage, W100 full-window scores and native LEFT scored positions. SMD and NASA are entity-first equal macro averages. Three project seeds share a single official pretrained initialization; n=3, sample SD ddof=1. The reconstruction control uses source labels only for validation checkpoint selection. LEFT has target-normal training and native offline W96 (MSL) / W192 context. No binary calibration or causal bank was added.

## All-position candidate minus references, paired across project seeds

| Domain | Reference | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| PSM | Endpoint-supervised control | 0.01465 ± 0.00134 | 0.01848 ± 0.00068 | -0.00287 ± 0.00031 | -0.01141 ± 0.00041 |
| PSM | Normal reconstruction control | 0.00756 ± 0.00052 | 0.01845 ± 0.00102 | 0.00551 ± 0.00064 | 0.00505 ± 0.00182 |
| PSM | Official TimeRadar V | 0.01552 ± 0.00159 | 0.03220 ± 0.00030 | 0.00283 ± 0.00048 | -0.00584 ± 0.00091 |
| PSM | Complete LEFT | -0.11313 ± 0.00159 | -0.06556 ± 0.00030 | -0.12040 ± 0.00048 | -0.10980 ± 0.00091 |
| SMD | Endpoint-supervised control | 0.02101 ± 0.00169 | -0.00043 ± 0.00134 | 0.01023 ± 0.00110 | -0.00628 ± 0.00072 |
| SMD | Normal reconstruction control | 0.00273 ± 0.00293 | 0.00269 ± 0.00275 | -0.00425 ± 0.00311 | -0.00264 ± 0.00166 |
| SMD | Official TimeRadar V | 0.02920 ± 0.00020 | 0.02021 ± 0.00076 | 0.02411 ± 0.00031 | 0.00611 ± 0.00074 |
| SMD | Complete LEFT | -0.04172 ± 0.00020 | -0.03116 ± 0.00076 | -0.06579 ± 0.00031 | -0.03009 ± 0.00074 |
| MSL | Endpoint-supervised control | 0.01250 ± 0.00137 | -0.00820 ± 0.00171 | 0.01951 ± 0.00079 | -0.00480 ± 0.00134 |
| MSL | Normal reconstruction control | -0.00016 ± 0.00331 | -0.00257 ± 0.00198 | 0.00901 ± 0.00238 | 0.00025 ± 0.00152 |
| MSL | Official TimeRadar V | 0.01930 ± 0.00079 | 0.00774 ± 0.00114 | 0.03055 ± 0.00142 | 0.00813 ± 0.00062 |
| MSL | Complete LEFT | -0.14054 ± 0.00079 | -0.08625 ± 0.00114 | -0.14070 ± 0.00142 | -0.06805 ± 0.00062 |
| SMAP | Endpoint-supervised control | 0.00205 ± 0.00241 | -0.00191 ± 0.00048 | 0.01293 ± 0.00249 | 0.00049 ± 0.00083 |
| SMAP | Normal reconstruction control | 0.00063 ± 0.00405 | -0.00202 ± 0.00233 | 0.01026 ± 0.00491 | 0.00265 ± 0.00266 |
| SMAP | Official TimeRadar V | 0.03594 ± 0.00068 | 0.00688 ± 0.00066 | 0.04741 ± 0.00043 | 0.00857 ± 0.00074 |
| SMAP | Complete LEFT | -0.15557 ± 0.00068 | -0.09360 ± 0.00066 | -0.17500 ± 0.00043 | -0.07211 ± 0.00074 |

## Per-seed common-axis model values

| Domain | Method | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| SMD | 2026_copy | 0.45649 | 0.84540 | 0.48350 | 0.89132 |
| SMD | 2026_terminal | 0.43470 | 0.84759 | 0.46643 | 0.89454 |
| SMD | 2026_all_positions | 0.45622 | 0.84686 | 0.47633 | 0.88773 |
| SMD | 2027_copy | 0.45373 | 0.84627 | 0.48031 | 0.89197 |
| SMD | 2027_terminal | 0.43747 | 0.84863 | 0.46639 | 0.89492 |
| SMD | 2027_all_positions | 0.45659 | 0.84705 | 0.47572 | 0.88836 |
| SMD | 2025_copy | 0.45069 | 0.84241 | 0.47706 | 0.88994 |
| SMD | 2025_terminal | 0.43390 | 0.84721 | 0.46462 | 0.89466 |
| SMD | 2025_all_positions | 0.45628 | 0.84825 | 0.47608 | 0.88921 |
| SMAP | 2026_copy | 0.33275 | 0.66830 | 0.33494 | 0.70760 |
| SMAP | 2026_terminal | 0.33327 | 0.67104 | 0.33104 | 0.71175 |
| SMAP | 2026_all_positions | 0.33788 | 0.66863 | 0.34622 | 0.71308 |
| SMAP | 2027_copy | 0.34079 | 0.67129 | 0.34198 | 0.70974 |
| SMAP | 2027_terminal | 0.33635 | 0.67112 | 0.33664 | 0.71232 |
| SMAP | 2027_all_positions | 0.33807 | 0.66923 | 0.34690 | 0.71200 |
| SMAP | 2025_copy | 0.33734 | 0.67429 | 0.33244 | 0.71321 |
| SMAP | 2025_terminal | 0.33698 | 0.67140 | 0.33367 | 0.71295 |
| SMAP | 2025_all_positions | 0.33681 | 0.66996 | 0.34702 | 0.71341 |
| MSL | 2026_copy | 0.36819 | 0.68181 | 0.40225 | 0.72710 |
| MSL | 2026_terminal | 0.35688 | 0.68680 | 0.39358 | 0.73119 |
| MSL | 2026_all_positions | 0.36854 | 0.67797 | 0.41398 | 0.72634 |
| MSL | 2027_copy | 0.36503 | 0.67939 | 0.40334 | 0.72588 |
| MSL | 2027_terminal | 0.35383 | 0.68532 | 0.39236 | 0.73152 |
| MSL | 2027_all_positions | 0.36790 | 0.67581 | 0.41126 | 0.72541 |
| MSL | 2025_copy | 0.37066 | 0.67779 | 0.40455 | 0.72458 |
| MSL | 2025_terminal | 0.35521 | 0.68378 | 0.39270 | 0.73002 |
| MSL | 2025_all_positions | 0.36696 | 0.67751 | 0.41192 | 0.72658 |
| PSM | 2026_copy | 0.45565 | 0.69256 | 0.44743 | 0.66891 |
| PSM | 2026_terminal | 0.44767 | 0.69173 | 0.45593 | 0.68674 |
| PSM | 2026_all_positions | 0.46361 | 0.71054 | 0.45341 | 0.67501 |
| PSM | 2027_copy | 0.45768 | 0.69321 | 0.44850 | 0.67001 |
| PSM | 2027_terminal | 0.45067 | 0.69325 | 0.45727 | 0.68766 |
| PSM | 2027_all_positions | 0.46542 | 0.71095 | 0.45428 | 0.67611 |
| PSM | 2025_copy | 0.45528 | 0.69075 | 0.44872 | 0.67387 |
| PSM | 2025_terminal | 0.44899 | 0.69144 | 0.45661 | 0.68777 |
| PSM | 2025_all_positions | 0.46225 | 0.71037 | 0.45351 | 0.67682 |

## Per-seed paired differences

| Domain | Seed | Reference | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|---|
| PSM | 2025 | terminal | 0.01326 | 0.01894 | -0.00310 | -0.01094 |
| PSM | 2025 | copy | 0.00697 | 0.01962 | 0.00478 | 0.00295 |
| PSM | 2025 | official_V | 0.01402 | 0.03195 | 0.00260 | -0.00500 |
| PSM | 2025 | LEFT | -0.11464 | -0.06581 | -0.12062 | -0.10896 |
| PSM | 2026 | terminal | 0.01595 | 0.01881 | -0.00252 | -0.01173 |
| PSM | 2026 | copy | 0.00797 | 0.01799 | 0.00598 | 0.00610 |
| PSM | 2026 | official_V | 0.01537 | 0.03213 | 0.00251 | -0.00681 |
| PSM | 2026 | LEFT | -0.11328 | -0.06564 | -0.12072 | -0.11077 |
| PSM | 2027 | terminal | 0.01475 | 0.01770 | -0.00299 | -0.01155 |
| PSM | 2027 | copy | 0.00773 | 0.01774 | 0.00578 | 0.00610 |
| PSM | 2027 | official_V | 0.01718 | 0.03253 | 0.00338 | -0.00571 |
| PSM | 2027 | LEFT | -0.11148 | -0.06524 | -0.11985 | -0.10967 |
| SMD | 2025 | terminal | 0.02238 | 0.00104 | 0.01146 | -0.00545 |
| SMD | 2025 | copy | 0.00559 | 0.00584 | -0.00098 | -0.00073 |
| SMD | 2025 | official_V | 0.02912 | 0.02108 | 0.02414 | 0.00689 |
| SMD | 2025 | LEFT | -0.04180 | -0.03030 | -0.06576 | -0.02932 |
| SMD | 2026 | terminal | 0.02152 | -0.00073 | 0.00990 | -0.00681 |
| SMD | 2026 | copy | -0.00026 | 0.00145 | -0.00717 | -0.00358 |
| SMD | 2026 | official_V | 0.02906 | 0.01969 | 0.02440 | 0.00541 |
| SMD | 2026 | LEFT | -0.04186 | -0.03169 | -0.06550 | -0.03079 |
| SMD | 2027 | terminal | 0.01912 | -0.00159 | 0.00933 | -0.00656 |
| SMD | 2027 | copy | 0.00286 | 0.00078 | -0.00459 | -0.00362 |
| SMD | 2027 | official_V | 0.02943 | 0.01987 | 0.02378 | 0.00604 |
| SMD | 2027 | LEFT | -0.04149 | -0.03150 | -0.06612 | -0.03017 |
| MSL | 2025 | terminal | 0.01176 | -0.00627 | 0.01923 | -0.00344 |
| MSL | 2025 | copy | -0.00370 | -0.00028 | 0.00737 | 0.00199 |
| MSL | 2025 | official_V | 0.01846 | 0.00816 | 0.03009 | 0.00860 |
| MSL | 2025 | LEFT | -0.14138 | -0.08583 | -0.14116 | -0.06758 |
| MSL | 2026 | terminal | 0.01166 | -0.00883 | 0.02040 | -0.00486 |
| MSL | 2026 | copy | 0.00035 | -0.00384 | 0.01173 | -0.00077 |
| MSL | 2026 | official_V | 0.02004 | 0.00862 | 0.03215 | 0.00836 |
| MSL | 2026 | LEFT | -0.13980 | -0.08538 | -0.13910 | -0.06783 |
| MSL | 2027 | terminal | 0.01407 | -0.00951 | 0.01890 | -0.00611 |
| MSL | 2027 | copy | 0.00288 | -0.00358 | 0.00792 | -0.00047 |
| MSL | 2027 | official_V | 0.01940 | 0.00646 | 0.02942 | 0.00743 |
| MSL | 2027 | LEFT | -0.14044 | -0.08753 | -0.14183 | -0.06875 |
| SMAP | 2025 | terminal | -0.00018 | -0.00144 | 0.01335 | 0.00046 |
| SMAP | 2025 | copy | -0.00053 | -0.00433 | 0.01458 | 0.00021 |
| SMAP | 2025 | official_V | 0.03516 | 0.00756 | 0.04771 | 0.00915 |
| SMAP | 2025 | LEFT | -0.15635 | -0.09292 | -0.17470 | -0.07153 |
| SMAP | 2026 | terminal | 0.00461 | -0.00241 | 0.01518 | 0.00133 |
| SMAP | 2026 | copy | 0.00513 | 0.00033 | 0.01129 | 0.00548 |
| SMAP | 2026 | official_V | 0.03623 | 0.00624 | 0.04692 | 0.00882 |
| SMAP | 2026 | LEFT | -0.15527 | -0.09424 | -0.17549 | -0.07186 |
| SMAP | 2027 | terminal | 0.00172 | -0.00190 | 0.01026 | -0.00032 |
| SMAP | 2027 | copy | -0.00271 | -0.00206 | 0.00492 | 0.00226 |
| SMAP | 2027 | official_V | 0.03642 | 0.00683 | 0.04759 | 0.00774 |
| SMAP | 2027 | LEFT | -0.15508 | -0.09365 | -0.17482 | -0.07294 |

Per-entity differences and wins/ties/losses/median differences are in paired_entity.csv and paired_distribution.csv. Entities are not independent training repeats. Full native axes remain separately available.
