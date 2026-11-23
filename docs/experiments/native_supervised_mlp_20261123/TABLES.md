# Complete supervised-readout comparison, seed 2025

PSM uses its complete causal test ranking axis; its operating panel uses the original 64,728-point suffix. Spacecraft and SMD ranking averages per-entity metrics equally. All operating counts preserve entity boundaries. The full and ≤240 MLP rows are source-selected across two fixed rates. The same heads serve all four targets.

| Dataset | Method | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| PSM | Official V | 0.53521 | 0.73065 | 0.48391 | 0.66457 |
| PSM | Official E | 0.38983 | 0.57110 | 0.40119 | 0.58264 |
| PSM | Official V+E | 0.42167 | 0.60470 | 0.42610 | 0.60846 |
| PSM | Reconstruction continuation / V | 0.53903 | 0.73152 | 0.49180 | 0.66665 |
| PSM | Source V supervision / V | 0.57717 | 0.76580 | 0.51378 | 0.67986 |
| PSM | Source E supervision / E | 0.52350 | 0.67552 | 0.50290 | 0.67104 |
| PSM | Supervised MLP, full source selection | 0.50494 | 0.72168 | 0.45263 | 0.60520 |
| PSM | Supervised MLP, ≤240 selection | 0.50494 | 0.72168 | 0.45263 | 0.60520 |
| MSL | Official V | 0.33559 | 0.62874 | 0.40570 | 0.70148 |
| MSL | Official E | 0.37804 | 0.69173 | 0.45967 | 0.76679 |
| MSL | Official V+E | 0.38658 | 0.70140 | 0.46949 | 0.77263 |
| MSL | Reconstruction continuation / V | 0.34152 | 0.63225 | 0.41388 | 0.70468 |
| MSL | Source V supervision / V | 0.35412 | 0.66573 | 0.42298 | 0.73565 |
| MSL | Source E supervision / E | 0.36975 | 0.68323 | 0.44146 | 0.74956 |
| MSL | Supervised MLP, full source selection | 0.28168 | 0.55090 | 0.31698 | 0.62785 |
| MSL | Supervised MLP, ≤240 selection | 0.28168 | 0.55090 | 0.31698 | 0.62785 |
| SMAP | Official V | 0.36560 | 0.64193 | 0.40267 | 0.68870 |
| SMAP | Official E | 0.42500 | 0.67890 | 0.46787 | 0.73709 |
| SMAP | Official V+E | 0.41425 | 0.70570 | 0.45512 | 0.75657 |
| SMAP | Reconstruction continuation / V | 0.36830 | 0.63893 | 0.39807 | 0.67898 |
| SMAP | Source V supervision / V | 0.38272 | 0.62680 | 0.42949 | 0.67618 |
| SMAP | Source E supervision / E | 0.39527 | 0.69797 | 0.43453 | 0.75068 |
| SMAP | Supervised MLP, full source selection | 0.28896 | 0.53892 | 0.31272 | 0.59110 |
| SMAP | Supervised MLP, ≤240 selection | 0.28896 | 0.53892 | 0.31272 | 0.59110 |
| SMD | Official V | 0.44423 | 0.81295 | 0.46636 | 0.86971 |
| SMD | Official E | 0.42871 | 0.80506 | 0.43255 | 0.84846 |
| SMD | Official V+E | 0.43842 | 0.81181 | 0.44154 | 0.85487 |
| SMD | Reconstruction continuation / V | 0.45021 | 0.82025 | 0.47609 | 0.87510 |
| SMD | Source V supervision / V | 0.44206 | 0.82705 | 0.43352 | 0.86098 |
| SMD | Source E supervision / E | 0.42858 | 0.81369 | 0.44149 | 0.86057 |
| SMD | Supervised MLP, full source selection | 0.37652 | 0.75892 | 0.31839 | 0.76763 |
| SMD | Supervised MLP, ≤240 selection | 0.37652 | 0.75892 | 0.31839 | 0.76763 |

## Source-selected MLP minus fixed controls

| Dataset | Method | Reference | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|---|
| PSM | Supervised MLP, full source selection | Source V supervision / V | -0.07223 | -0.04411 | -0.06115 | -0.07466 |
| PSM | Supervised MLP, full source selection | Source E supervision / E | -0.01856 | 0.04616 | -0.05027 | -0.06584 |
| PSM | Supervised MLP, full source selection | Official V | -0.03027 | -0.00897 | -0.03128 | -0.05936 |
| PSM | Supervised MLP, ≤240 selection | Source V supervision / V | -0.07223 | -0.04411 | -0.06115 | -0.07466 |
| PSM | Supervised MLP, ≤240 selection | Source E supervision / E | -0.01856 | 0.04616 | -0.05027 | -0.06584 |
| PSM | Supervised MLP, ≤240 selection | Official V | -0.03027 | -0.00897 | -0.03128 | -0.05936 |
| MSL | Supervised MLP, full source selection | Source V supervision / V | -0.07244 | -0.11483 | -0.10599 | -0.10780 |
| MSL | Supervised MLP, full source selection | Source E supervision / E | -0.08807 | -0.13233 | -0.12448 | -0.12172 |
| MSL | Supervised MLP, full source selection | Official V | -0.05391 | -0.07784 | -0.08871 | -0.07363 |
| MSL | Supervised MLP, ≤240 selection | Source V supervision / V | -0.07244 | -0.11483 | -0.10599 | -0.10780 |
| MSL | Supervised MLP, ≤240 selection | Source E supervision / E | -0.08807 | -0.13233 | -0.12448 | -0.12172 |
| MSL | Supervised MLP, ≤240 selection | Official V | -0.05391 | -0.07784 | -0.08871 | -0.07363 |
| SMAP | Supervised MLP, full source selection | Source V supervision / V | -0.09376 | -0.08788 | -0.11677 | -0.08508 |
| SMAP | Supervised MLP, full source selection | Source E supervision / E | -0.10632 | -0.15905 | -0.12181 | -0.15958 |
| SMAP | Supervised MLP, full source selection | Official V | -0.07664 | -0.10301 | -0.08995 | -0.09760 |
| SMAP | Supervised MLP, ≤240 selection | Source V supervision / V | -0.09376 | -0.08788 | -0.11677 | -0.08508 |
| SMAP | Supervised MLP, ≤240 selection | Source E supervision / E | -0.10632 | -0.15905 | -0.12181 | -0.15958 |
| SMAP | Supervised MLP, ≤240 selection | Official V | -0.07664 | -0.10301 | -0.08995 | -0.09760 |
| SMD | Supervised MLP, full source selection | Source V supervision / V | -0.06554 | -0.06813 | -0.11514 | -0.09335 |
| SMD | Supervised MLP, full source selection | Source E supervision / E | -0.05206 | -0.05477 | -0.12310 | -0.09294 |
| SMD | Supervised MLP, full source selection | Official V | -0.06771 | -0.05403 | -0.14797 | -0.10208 |
| SMD | Supervised MLP, ≤240 selection | Source V supervision / V | -0.06554 | -0.06813 | -0.11514 | -0.09335 |
| SMD | Supervised MLP, ≤240 selection | Source E supervision / E | -0.05206 | -0.05477 | -0.12310 | -0.09294 |
| SMD | Supervised MLP, ≤240 selection | Official V | -0.06771 | -0.05403 | -0.14797 | -0.10208 |

## Fixed CAL operating points

| Dataset | Method | Stream | Normal FPR | Recall | False fragments | Matched/eligible | Any-hit | Event F1 | Long macro | Long pooled | Longest miss | Hit-only delay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PSM | Official V | strict | 0.29669 | 0.50870 | 185 | 18/48 | 28 | 0.14458 | 0.86955 | 0.92407 | 486.00000 | 0.00000 |
| PSM | Official V | confirmed3 | 0.29096 | 0.48878 | 88 | 6/48 | 17 | 0.09023 | 0.86453 | 0.92114 | 694.00000 | 0.00000 |
| PSM | Official E | strict | 0.49790 | 0.64854 | 1057 | 27/48 | 42 | 0.04779 | 1.00000 | 1.00000 | 364.00000 | 0.00000 |
| PSM | Official E | confirmed3 | 0.46456 | 0.63172 | 432 | 15/48 | 30 | 0.06356 | 1.00000 | 1.00000 | 397.00000 | 0.00000 |
| PSM | Official V+E | strict | 0.52003 | 0.65210 | 934 | 30/48 | 43 | 0.06024 | 1.00000 | 1.00000 | 364.00000 | 0.00000 |
| PSM | Official V+E | confirmed3 | 0.49098 | 0.63505 | 354 | 17/48 | 30 | 0.08608 | 1.00000 | 1.00000 | 397.00000 | 0.00000 |
| PSM | Reconstruction continuation / V | strict | 0.29381 | 0.50416 | 176 | 18/48 | 28 | 0.14938 | 0.85028 | 0.91285 | 693.00000 | 0.00000 |
| PSM | Reconstruction continuation / V | confirmed3 | 0.28855 | 0.48436 | 77 | 6/48 | 17 | 0.10000 | 0.84525 | 0.90992 | 718.00000 | 0.00000 |
| PSM | Source V supervision / V | strict | 0.06724 | 0.40060 | 251 | 19/48 | 20 | 0.09291 | 0.55067 | 0.74797 | 874.00000 | 0.50000 |
| PSM | Source V supervision / V | confirmed3 | 0.05919 | 0.37333 | 93 | 10/48 | 11 | 0.10638 | 0.53069 | 0.73561 | 874.00000 | 3.00000 |
| PSM | Source E supervision / E | strict | 0.30371 | 0.57432 | 510 | 29/48 | 36 | 0.08788 | 0.90084 | 0.94228 | 399.00000 | 0.00000 |
| PSM | Source E supervision / E | confirmed3 | 0.28847 | 0.53999 | 175 | 12/48 | 20 | 0.10000 | 0.88212 | 0.93138 | 399.00000 | 0.00000 |
| PSM | Supervised MLP, full source selection | strict | 0.02701 | 0.25030 | 83 | 8/48 | 9 | 0.08377 | 0.36440 | 0.54081 | 1228.00000 | 1.00000 |
| PSM | Supervised MLP, full source selection | confirmed3 | 0.02424 | 0.24272 | 48 | 7/48 | 8 | 0.11667 | 0.35564 | 0.52927 | 1228.00000 | 5.50000 |
| PSM | Supervised MLP, ≤240 selection | strict | 0.02701 | 0.25030 | 83 | 8/48 | 9 | 0.08377 | 0.36440 | 0.54081 | 1228.00000 | 1.00000 |
| PSM | Supervised MLP, ≤240 selection | confirmed3 | 0.02424 | 0.24272 | 48 | 7/48 | 8 | 0.11667 | 0.35564 | 0.52927 | 1228.00000 | 5.50000 |
| MSL | Official V | strict | 0.28858 | 0.28908 | 881 | 26/35 | 27 | 0.04610 | 0.27345 | 0.18152 | 1141.00000 | 6.00000 |
| MSL | Official V | confirmed3 | 0.26958 | 0.24079 | 162 | 19/35 | 20 | 0.13428 | 0.14084 | 0.10747 | 1141.00000 | 6.50000 |
| MSL | Official E | strict | 0.26102 | 0.29951 | 557 | 29/35 | 29 | 0.07117 | 0.22312 | 0.26671 | 1114.00000 | 7.00000 |
| MSL | Official E | confirmed3 | 0.24778 | 0.25045 | 141 | 27/35 | 27 | 0.22594 | 0.12917 | 0.21298 | 1114.00000 | 9.00000 |
| MSL | Official V+E | strict | 0.27097 | 0.30659 | 520 | 27/35 | 29 | 0.07077 | 0.22551 | 0.27064 | 1114.00000 | 5.00000 |
| MSL | Official V+E | confirmed3 | 0.25870 | 0.25972 | 116 | 25/35 | 27 | 0.24876 | 0.13275 | 0.21887 | 1114.00000 | 9.00000 |
| MSL | Reconstruction continuation / V | strict | 0.26100 | 0.29900 | 826 | 26/35 | 27 | 0.05098 | 0.31547 | 0.19397 | 1141.00000 | 6.00000 |
| MSL | Reconstruction continuation / V | confirmed3 | 0.24304 | 0.26243 | 163 | 19/35 | 20 | 0.14786 | 0.24116 | 0.14351 | 1141.00000 | 8.00000 |
| MSL | Source V supervision / V | strict | 0.29366 | 0.31483 | 684 | 28/35 | 30 | 0.06174 | 0.27795 | 0.16514 | 1136.00000 | 5.00000 |
| MSL | Source V supervision / V | confirmed3 | 0.27859 | 0.27015 | 165 | 24/35 | 26 | 0.17082 | 0.21404 | 0.12189 | 1141.00000 | 13.50000 |
| MSL | Source E supervision / E | strict | 0.27110 | 0.29938 | 572 | 29/35 | 30 | 0.07099 | 0.20017 | 0.23001 | 1114.00000 | 6.50000 |
| MSL | Source E supervision / E | confirmed3 | 0.25772 | 0.25122 | 145 | 26/35 | 27 | 0.20968 | 0.10453 | 0.17235 | 1114.00000 | 9.00000 |
| MSL | Supervised MLP, full source selection | strict | 0.27480 | 0.30003 | 689 | 22/35 | 22 | 0.05439 | 0.19735 | 0.32307 | 1141.00000 | 12.50000 |
| MSL | Supervised MLP, full source selection | confirmed3 | 0.25595 | 0.27723 | 401 | 21/35 | 21 | 0.08537 | 0.17766 | 0.29292 | 1141.00000 | 18.00000 |
| MSL | Supervised MLP, ≤240 selection | strict | 0.27480 | 0.30003 | 689 | 22/35 | 22 | 0.05439 | 0.19735 | 0.32307 | 1141.00000 | 12.50000 |
| MSL | Supervised MLP, ≤240 selection | confirmed3 | 0.25595 | 0.27723 | 401 | 21/35 | 21 | 0.08537 | 0.17766 | 0.29292 | 1141.00000 | 18.00000 |
| SMAP | Official V | strict | 0.12064 | 0.40392 | 2277 | 46/57 | 46 | 0.03489 | 0.22878 | 0.45949 | 2701.00000 | 29.50000 |
| SMAP | Official V | confirmed3 | 0.11211 | 0.39328 | 607 | 28/57 | 28 | 0.07735 | 0.20459 | 0.44493 | 2812.00000 | 45.00000 |
| SMAP | Official E | strict | 0.09112 | 0.49020 | 3620 | 45/57 | 45 | 0.02247 | 0.23396 | 0.46647 | 2401.00000 | 17.00000 |
| SMAP | Official E | confirmed3 | 0.07940 | 0.47737 | 497 | 30/57 | 30 | 0.09202 | 0.21367 | 0.45416 | 2401.00000 | 35.50000 |
| SMAP | Official V+E | strict | 0.08224 | 0.45345 | 3646 | 47/57 | 47 | 0.02333 | 0.24050 | 0.47016 | 2401.00000 | 18.00000 |
| SMAP | Official V+E | confirmed3 | 0.07006 | 0.44085 | 472 | 30/57 | 30 | 0.09449 | 0.22166 | 0.45867 | 2401.00000 | 35.50000 |
| SMAP | Reconstruction continuation / V | strict | 0.12231 | 0.40288 | 2252 | 44/57 | 44 | 0.03429 | 0.22548 | 0.45730 | 2701.00000 | 32.00000 |
| SMAP | Reconstruction continuation / V | confirmed3 | 0.11386 | 0.39376 | 592 | 29/57 | 29 | 0.08274 | 0.20435 | 0.44479 | 2812.00000 | 45.00000 |
| SMAP | Source V supervision / V | strict | 0.13486 | 0.43837 | 3095 | 45/57 | 45 | 0.02664 | 0.23031 | 0.45491 | 2368.00000 | 28.00000 |
| SMAP | Source V supervision / V | confirmed3 | 0.12222 | 0.42464 | 808 | 36/57 | 36 | 0.07895 | 0.20322 | 0.44445 | 2401.00000 | 34.50000 |
| SMAP | Source E supervision / E | strict | 0.08318 | 0.38908 | 2890 | 45/57 | 45 | 0.02753 | 0.24554 | 0.45580 | 2401.00000 | 22.00000 |
| SMAP | Source E supervision / E | confirmed3 | 0.07243 | 0.37855 | 805 | 34/57 | 34 | 0.07211 | 0.21338 | 0.44473 | 2401.00000 | 35.50000 |
| SMAP | Supervised MLP, full source selection | strict | 0.13956 | 0.48852 | 2633 | 36/57 | 36 | 0.02609 | 0.30494 | 0.57852 | 1651.00000 | 25.50000 |
| SMAP | Supervised MLP, full source selection | confirmed3 | 0.12758 | 0.48159 | 1388 | 30/57 | 30 | 0.04024 | 0.30394 | 0.57695 | 1651.00000 | 41.00000 |
| SMAP | Supervised MLP, ≤240 selection | strict | 0.13956 | 0.48852 | 2633 | 36/57 | 36 | 0.02609 | 0.30494 | 0.57852 | 1651.00000 | 25.50000 |
| SMAP | Supervised MLP, ≤240 selection | confirmed3 | 0.12758 | 0.48159 | 1388 | 30/57 | 30 | 0.04024 | 0.30394 | 0.57695 | 1651.00000 | 41.00000 |
| SMD | Official V | strict | 0.19334 | 0.52975 | 4534 | 250/327 | 278 | 0.09376 | 0.58222 | 0.69921 | 631.00000 | 0.00000 |
| SMD | Official V | confirmed3 | 0.18411 | 0.49772 | 1189 | 149/327 | 180 | 0.17696 | 0.57438 | 0.69519 | 837.00000 | 2.00000 |
| SMD | Official E | strict | 0.16652 | 0.49015 | 5751 | 248/327 | 283 | 0.07155 | 0.48396 | 0.58023 | 371.00000 | 0.00000 |
| SMD | Official E | confirmed3 | 0.15543 | 0.43975 | 1221 | 136/327 | 174 | 0.15385 | 0.46413 | 0.57008 | 841.00000 | 2.00000 |
| SMD | Official V+E | strict | 0.17767 | 0.49793 | 6062 | 247/327 | 282 | 0.06930 | 0.48384 | 0.58023 | 371.00000 | 0.00000 |
| SMD | Official V+E | confirmed3 | 0.16577 | 0.45357 | 1368 | 144/327 | 181 | 0.15467 | 0.46437 | 0.57043 | 841.00000 | 2.00000 |
| SMD | Reconstruction continuation / V | strict | 0.19467 | 0.53335 | 4417 | 248/327 | 277 | 0.09467 | 0.58637 | 0.70096 | 342.00000 | 0.00000 |
| SMD | Reconstruction continuation / V | confirmed3 | 0.18560 | 0.49935 | 1222 | 152/327 | 183 | 0.17441 | 0.57819 | 0.69659 | 837.00000 | 2.00000 |
| SMD | Source V supervision / V | strict | 0.19204 | 0.54286 | 3349 | 175/327 | 208 | 0.08774 | 0.57985 | 0.70061 | 837.00000 | 0.00000 |
| SMD | Source V supervision / V | confirmed3 | 0.18468 | 0.51885 | 1305 | 123/327 | 158 | 0.14236 | 0.57300 | 0.69816 | 837.00000 | 2.00000 |
| SMD | Source E supervision / E | strict | 0.17775 | 0.46380 | 8111 | 266/327 | 282 | 0.05637 | 0.48711 | 0.55031 | 387.00000 | 0.00000 |
| SMD | Source E supervision / E | confirmed3 | 0.16093 | 0.40857 | 2190 | 136/327 | 155 | 0.09663 | 0.46456 | 0.51654 | 1204.00000 | 2.00000 |
| SMD | Supervised MLP, full source selection | strict | 0.19517 | 0.49355 | 1892 | 123/327 | 153 | 0.10686 | 0.52890 | 0.61365 | 861.00000 | 0.00000 |
| SMD | Supervised MLP, full source selection | confirmed3 | 0.19079 | 0.48084 | 1069 | 112/327 | 142 | 0.15819 | 0.52634 | 0.61032 | 863.00000 | 2.00000 |
| SMD | Supervised MLP, ≤240 selection | strict | 0.19517 | 0.49355 | 1892 | 123/327 | 153 | 0.10686 | 0.52890 | 0.61365 | 861.00000 | 0.00000 |
| SMD | Supervised MLP, ≤240 selection | confirmed3 | 0.19079 | 0.48084 | 1069 | 112/327 | 142 | 0.15819 | 0.52634 | 0.61032 | 863.00000 | 2.00000 |
