# Table 2. Same-checkpoint endpoint and complete-window comparison

Every difference uses the identical per-entity common axis defined in Table1. Full-window minus endpoint measures a readout/context change at fixed parameters. It is separate from differences between training objectives. No result is selected by target performance.

## Same weights, different readout

| Domain | Weights / score | Endpoint AP | Complete-window AP | Δ AP | Δ ROC_AUC | Δ VUS_PR | Δ VUS_ROC |
|---|---|---|---|---|---|---|---|
| PSM | Reconstruction continuation | 0.53884 ± 0.00209 | 0.45283 ± 0.00291 | -0.08601 ± 0.00171 | -0.04444 ± 0.00245 | -0.04059 ± 0.00536 | 0.01032 ± 0.00324 |
| PSM | Source-supervised native V | 0.57306 ± 0.00469 | 0.44922 ± 0.00150 | -0.12384 ± 0.00380 | -0.07172 ± 0.00253 | -0.05177 ± 0.00460 | 0.01130 ± 0.00406 |
| PSM | Official V | 0.53528 | 0.44824 | -0.08704 | -0.05257 | -0.03269 | 0.01705 |
| PSM | Official E | 0.38734 | 0.43590 | 0.04856 | 0.03812 | 0.03999 | 0.03391 |
| PSM | Official V+E | 0.41947 | 0.44836 | 0.02889 | 0.01549 | 0.02454 | 0.01673 |
| PSM | E-supervised E | 0.52232 | 0.38952 | -0.13281 | -0.09783 | -0.09392 | -0.06940 |
| SMD | Reconstruction continuation | 0.44968 ± 0.00260 | 0.44041 ± 0.00487 | -0.00927 ± 0.00236 | 0.01859 ± 0.00043 | -0.00983 ± 0.00215 | 0.01287 ± 0.00077 |
| SMD | Source-supervised native V | 0.44023 ± 0.00255 | 0.43614 ± 0.00139 | -0.00409 ± 0.00344 | 0.02228 ± 0.00077 | 0.03680 ± 0.00485 | 0.03516 ± 0.00124 |
| SMD | Official V | 0.44480 | 0.42716 | -0.01764 | 0.01396 | -0.01526 | 0.01236 |
| SMD | Official E | 0.42914 | 0.42720 | -0.00194 | 0.01463 | 0.02194 | 0.01966 |
| SMD | Official V+E | 0.43885 | 0.43725 | -0.00159 | 0.01473 | 0.02219 | 0.01853 |
| SMD | E-supervised E | 0.42899 | 0.42158 | -0.00741 | 0.01989 | 0.03219 | 0.02015 |
| MSL | Reconstruction continuation | 0.34945 ± 0.00189 | 0.36811 ± 0.00294 | 0.01866 ± 0.00180 | 0.03923 ± 0.00149 | -0.01920 ± 0.00167 | 0.01454 ± 0.00086 |
| MSL | Source-supervised native V | 0.36592 ± 0.00188 | 0.35562 ± 0.00159 | -0.01030 ± 0.00260 | 0.01827 ± 0.00298 | -0.04205 ± 0.00390 | -0.00418 ± 0.00496 |
| MSL | Official V | 0.34256 | 0.34850 | 0.00594 | 0.03664 | -0.03285 | 0.01317 |
| MSL | Official E | 0.39044 | 0.39988 | 0.00943 | 0.01777 | -0.03125 | -0.00465 |
| MSL | Official V+E | 0.39986 | 0.40407 | 0.00421 | 0.01322 | -0.03671 | -0.00828 |
| MSL | E-supervised E | 0.38119 | 0.39125 | 0.01005 | 0.03478 | -0.01586 | 0.02332 |
| SMAP | Reconstruction continuation | 0.36633 ± 0.00216 | 0.32903 ± 0.01050 | -0.03730 ± 0.00878 | 0.03928 ± 0.00178 | -0.07356 ± 0.01182 | 0.03569 ± 0.00021 |
| SMAP | Source-supervised native V | 0.38265 ± 0.00218 | 0.33422 ± 0.00185 | -0.04843 ± 0.00109 | 0.03891 ± 0.00409 | -0.09670 ± 0.00138 | 0.02978 ± 0.00391 |
| SMAP | Official V | 0.36555 | 0.30165 | -0.06390 | 0.01971 | -0.10338 | 0.01401 |
| SMAP | Official E | 0.42536 | 0.38124 | -0.04412 | 0.03832 | -0.08762 | 0.02690 |
| SMAP | Official V+E | 0.41481 | 0.38662 | -0.02820 | 0.01900 | -0.07365 | 0.01155 |
| SMAP | E-supervised E | 0.39631 | 0.38984 | -0.00647 | 0.01737 | -0.04439 | 0.01217 |

## Source supervision within each fixed readout

Differences are candidate minus official V or same-seed reconstruction continuation. Each value is the mean ± sample SD of three paired project-seed differences. Entity macro averaging precedes seed aggregation.
| Domain | Candidate readout | Reference | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|---|
| PSM | full_window | official_V | 0.00098 ± 0.00150 | 0.01386 ± 0.00085 | 0.00569 ± 0.00067 | 0.00519 ± 0.00057 |
| PSM | full_window | copy | -0.00361 ± 0.00182 | 0.00499 ± 0.00265 | 0.00704 ± 0.00122 | 0.01026 ± 0.00158 |
| PSM | full_window | LEFT | -0.12768 ± 0.00150 | -0.08391 ± 0.00085 | -0.11754 ± 0.00067 | -0.09877 ± 0.00057 |
| PSM | causal_endpoint | official_V | 0.03779 ± 0.00469 | 0.03301 ± 0.00277 | 0.02477 ± 0.00488 | 0.01093 ± 0.00388 |
| PSM | causal_endpoint | copy | 0.03422 ± 0.00345 | 0.03227 ± 0.00201 | 0.01822 ± 0.00407 | 0.00928 ± 0.00332 |
| PSM | causal_endpoint | LEFT | -0.00383 ± 0.00469 | -0.01219 ± 0.00277 | -0.06576 ± 0.00488 | -0.11008 ± 0.00388 |
| SMD | full_window | official_V | 0.00898 ± 0.00139 | 0.02119 ± 0.00068 | 0.01488 ± 0.00071 | 0.01267 ± 0.00049 |
| SMD | full_window | copy | -0.00426 ± 0.00349 | 0.01102 ± 0.00306 | 0.00177 ± 0.00613 | 0.00803 ± 0.00173 |
| SMD | full_window | LEFT | -0.06194 ± 0.00139 | -0.03019 ± 0.00068 | -0.07502 ± 0.00071 | -0.02354 ± 0.00049 |
| SMD | causal_endpoint | official_V | -0.00457 ± 0.00255 | 0.01287 ± 0.00109 | -0.03718 ± 0.00539 | -0.01014 ± 0.00147 |
| SMD | causal_endpoint | copy | -0.00945 ± 0.00399 | 0.00734 ± 0.00323 | -0.04486 ± 0.00777 | -0.01426 ± 0.00323 |
| SMD | causal_endpoint | LEFT | -0.05785 ± 0.00255 | -0.05246 ± 0.00109 | -0.11182 ± 0.00539 | -0.05870 ± 0.00147 |
| MSL | full_window | official_V | 0.00712 ± 0.00159 | 0.01671 ± 0.00074 | 0.01058 ± 0.00114 | 0.01346 ± 0.00022 |
| MSL | full_window | copy | -0.01249 ± 0.00393 | 0.01057 ± 0.00232 | -0.01128 ± 0.00408 | 0.00935 ± 0.00141 |
| MSL | full_window | LEFT | -0.15272 ± 0.00159 | -0.07728 ± 0.00074 | -0.16067 ± 0.00114 | -0.06272 ± 0.00022 |
| MSL | causal_endpoint | official_V | 0.02336 ± 0.00188 | 0.03507 ± 0.00299 | 0.01979 ± 0.00284 | 0.03082 ± 0.00502 |
| MSL | causal_endpoint | copy | 0.01647 ± 0.00178 | 0.03152 ± 0.00373 | 0.01158 ± 0.00273 | 0.02807 ± 0.00562 |
| MSL | causal_endpoint | LEFT | -0.14242 ± 0.00188 | -0.09556 ± 0.00299 | -0.11861 ± 0.00284 | -0.05854 ± 0.00502 |
| SMAP | full_window | official_V | 0.03257 ± 0.00185 | 0.00822 ± 0.00081 | 0.03402 ± 0.00294 | 0.00741 ± 0.00071 |
| SMAP | full_window | copy | 0.00518 ± 0.00965 | -0.00456 ± 0.00307 | 0.00920 ± 0.00783 | -0.00304 ± 0.00231 |
| SMAP | full_window | LEFT | -0.15894 ± 0.00185 | -0.09226 ± 0.00081 | -0.18839 ± 0.00294 | -0.07328 ± 0.00071 |
| SMAP | causal_endpoint | official_V | 0.01709 ± 0.00218 | -0.01099 ± 0.00473 | 0.02734 ± 0.00331 | -0.00836 ± 0.00456 |
| SMAP | causal_endpoint | copy | 0.01632 ± 0.00221 | -0.00418 ± 0.00715 | 0.03234 ± 0.00568 | 0.00286 ± 0.00509 |
| SMAP | causal_endpoint | LEFT | -0.11051 ± 0.00218 | -0.13117 ± 0.00473 | -0.09170 ± 0.00331 | -0.10306 ± 0.00456 |

The LEFT comparison always retains LEFT’s native offline readout. Seed-wise and per-entity differences, counts of wins/ties/losses and medians are in [paired_distribution.csv](paired_distribution.csv). The underlying differences are in [paired_entity.csv](paired_entity.csv).

## Entity distribution of the full-window candidate-minus-LEFT AP

Counts compare each fixed training seed with LEFT on the exact per-entity joint axis; entities are not extra training repetitions.
| Domain | Candidate seed | Macro AP delta | Median AP delta | Wins/ties/losses |
|---|---|---|---|---|
| PSM | 2025_point_weighted | -0.12757 | -0.12757 | 0/0/1 |
| PSM | 2026_point_weighted | -0.12923 | -0.12923 | 0/0/1 |
| PSM | 2027_point_weighted | -0.12623 | -0.12623 | 0/0/1 |
| SMD | 2025_point_weighted | -0.06182 | -0.04550 | 7/0/21 |
| SMD | 2026_point_weighted | -0.06339 | -0.04796 | 6/0/22 |
| SMD | 2027_point_weighted | -0.06062 | -0.04471 | 6/0/22 |
| MSL | 2025_point_weighted | -0.15218 | -0.07096 | 6/0/21 |
| MSL | 2026_point_weighted | -0.15147 | -0.07325 | 6/0/21 |
| MSL | 2027_point_weighted | -0.15451 | -0.07181 | 6/0/21 |
| SMAP | 2025_point_weighted | -0.16012 | -0.02454 | 19/0/34 |
| SMAP | 2026_point_weighted | -0.15989 | -0.02665 | 19/0/34 |
| SMAP | 2027_point_weighted | -0.15680 | -0.01103 | 20/0/33 |
