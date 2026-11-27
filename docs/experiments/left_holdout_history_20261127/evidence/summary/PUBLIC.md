# Holdout LEFT public evaluation

All scores are available at the end of the current W192 window. The Z and N stages use the same legal TEST timestamps within each panel. External axes can trim tails. SMD, when eligible, averages entity metrics equally. The project uses one training seed. Only CATCH uses three archived seeds.

Operation points are NA. This full-score comparison does not replace the old CAL panel.

## Frozen source models (Z), complete internal axis

### AP

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.507302 | 0.724501 | 0.449466 |
| Holdout LEFT + frozen pretrained TimesFM | 0.506734 | 0.722993 | 0.456272 |
| Holdout LEFT + frozen random TimesFM | 0.506900 | 0.723429 | 0.450543 |

### ROC_AUC

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.700661 | 0.811404 | 0.849665 |
| Holdout LEFT + frozen pretrained TimesFM | 0.701277 | 0.811133 | 0.855641 |
| Holdout LEFT + frozen random TimesFM | 0.700201 | 0.811207 | 0.849601 |

### VUS_PR

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.518731 | 0.500210 | 0.469733 |
| Holdout LEFT + frozen pretrained TimesFM | 0.517999 | 0.497547 | 0.484279 |
| Holdout LEFT + frozen random TimesFM | 0.517642 | 0.497253 | 0.472347 |

### VUS_ROC

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.716707 | 0.668224 | 0.904495 |
| Holdout LEFT + frozen pretrained TimesFM | 0.716922 | 0.665291 | 0.909122 |
| Holdout LEFT + frozen random TimesFM | 0.715972 | 0.665247 | 0.904741 |

## Frozen source models (Z), external common axis

### AP

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.507302 | 0.724501 | 0.449481 |
| Holdout LEFT + frozen pretrained TimesFM | 0.506734 | 0.722993 | 0.456291 |
| Holdout LEFT + frozen random TimesFM | 0.506900 | 0.723429 | 0.450558 |
| Native LEFT matched source continuation | 0.526270 | 0.722702 | 0.500441 |
| Unmasked LEFT + frozen pretrained TimesFM | 0.526647 | 0.722609 | 0.494131 |
| Earlier multisource LEFT | 0.497206 | 0.723927 | 0.498182 |
| Native LEFT trained on target normal data | 0.577398 | 0.543881 | 0.499187 |
| Earlier multisource LEFT + target normal training | 0.555001 | 0.542517 | 0.519018 |
| Normal-level deviation | 0.439188 | 0.744804 | 0.428476 |
| CATCH (three archived seeds) | 0.437276 ± 0.008436 | NA | NA |
| PCA residual | 0.534679 | NA | NA |
| TimeRadar (archived block score) | 0.434810 | 0.639315 | NA |
| TimeRadar endpoint V | 0.536488 | NA | 0.444987 |
| DADA | 0.396595 | NA | 0.415141 |
| TimeRCD (FIT-only statistics) | 0.303144 | NA | 0.473921 |

### ROC_AUC

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.700661 | 0.811404 | 0.849675 |
| Holdout LEFT + frozen pretrained TimesFM | 0.701277 | 0.811133 | 0.855653 |
| Holdout LEFT + frozen random TimesFM | 0.700201 | 0.811207 | 0.849610 |
| Native LEFT matched source continuation | 0.711257 | 0.811901 | 0.875045 |
| Unmasked LEFT + frozen pretrained TimesFM | 0.711475 | 0.812082 | 0.868094 |
| Earlier multisource LEFT | 0.700389 | 0.811299 | 0.866936 |
| Native LEFT trained on target normal data | 0.775512 | 0.820681 | 0.877656 |
| Earlier multisource LEFT + target normal training | 0.755856 | 0.855454 | 0.882029 |
| Normal-level deviation | 0.653725 | 0.831611 | 0.802073 |
| CATCH (three archived seeds) | 0.646309 ± 0.006074 | NA | NA |
| PCA residual | 0.792130 | NA | NA |
| TimeRadar (archived block score) | 0.671221 | 0.821814 | NA |
| TimeRadar endpoint V | 0.731327 | NA | 0.811991 |
| DADA | 0.604198 | NA | 0.850385 |
| TimeRCD (FIT-only statistics) | 0.476508 | NA | 0.859168 |

### VUS_PR

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.518731 | 0.500210 | 0.469824 |
| Holdout LEFT + frozen pretrained TimesFM | 0.517999 | 0.497547 | 0.484409 |
| Holdout LEFT + frozen random TimesFM | 0.517642 | 0.497253 | 0.472302 |
| Native LEFT matched source continuation | 0.527736 | 0.497658 | 0.543549 |
| Unmasked LEFT + frozen pretrained TimesFM | 0.528103 | 0.497345 | 0.535253 |
| Earlier multisource LEFT | 0.502864 | 0.497560 | 0.541924 |
| Native LEFT trained on target normal data | 0.574585 | 0.534715 | 0.542440 |
| Earlier multisource LEFT + target normal training | 0.553671 | 0.530836 | 0.565223 |
| Normal-level deviation | 0.448158 | 0.464219 | 0.427435 |
| CATCH (three archived seeds) | 0.527603 ± 0.008662 | NA | NA |
| PCA residual | 0.529230 | NA | NA |
| TimeRadar (archived block score) | 0.443332 | 0.514125 | NA |
| TimeRadar endpoint V | 0.485394 | NA | 0.467207 |
| DADA | 0.440465 | NA | 0.475523 |
| TimeRCD (FIT-only statistics) | 0.349831 | NA | 0.516771 |

### VUS_ROC

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.716707 | 0.668224 | 0.904497 |
| Holdout LEFT + frozen pretrained TimesFM | 0.716922 | 0.665291 | 0.909143 |
| Holdout LEFT + frozen random TimesFM | 0.715972 | 0.665247 | 0.904755 |
| Native LEFT matched source continuation | 0.718670 | 0.665546 | 0.915820 |
| Unmasked LEFT + frozen pretrained TimesFM | 0.718901 | 0.666311 | 0.912140 |
| Earlier multisource LEFT | 0.706424 | 0.660764 | 0.911587 |
| Native LEFT trained on target normal data | 0.785010 | 0.809105 | 0.917348 |
| Earlier multisource LEFT + target normal training | 0.765883 | 0.842850 | 0.919536 |
| Normal-level deviation | 0.655534 | 0.643787 | 0.864645 |
| CATCH (three archived seeds) | 0.695732 ± 0.006690 | NA | NA |
| PCA residual | 0.765810 | NA | NA |
| TimeRadar (archived block score) | 0.679425 | 0.766598 | NA |
| TimeRadar endpoint V | 0.665784 | NA | 0.868872 |
| DADA | 0.651296 | NA | 0.898696 |
| TimeRCD (FIT-only statistics) | 0.532382 | NA | 0.909346 |

## Target normal training (N), complete internal axis

### AP

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.533509 | 0.719360 | 0.452147 |
| Holdout LEFT + frozen pretrained TimesFM | 0.541018 | 0.719539 | 0.455440 |
| Holdout LEFT + frozen random TimesFM | 0.532497 | 0.719307 | 0.453062 |
| Matched-budget native LEFT normal training | 0.567882 | 0.720594 | 0.499478 |

### ROC_AUC

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.712808 | 0.813110 | 0.847882 |
| Holdout LEFT + frozen pretrained TimesFM | 0.717313 | 0.812415 | 0.849297 |
| Holdout LEFT + frozen random TimesFM | 0.712598 | 0.812522 | 0.848925 |
| Matched-budget native LEFT normal training | 0.732309 | 0.813057 | 0.870980 |

### VUS_PR

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.543270 | 0.501293 | 0.472982 |
| Holdout LEFT + frozen pretrained TimesFM | 0.550638 | 0.503943 | 0.481181 |
| Holdout LEFT + frozen random TimesFM | 0.541490 | 0.503409 | 0.474385 |
| Matched-budget native LEFT normal training | 0.567789 | 0.506516 | 0.543579 |

### VUS_ROC

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.730335 | 0.679903 | 0.901478 |
| Holdout LEFT + frozen pretrained TimesFM | 0.735943 | 0.680104 | 0.903261 |
| Holdout LEFT + frozen random TimesFM | 0.729892 | 0.679566 | 0.902173 |
| Matched-budget native LEFT normal training | 0.743930 | 0.683174 | 0.914086 |

## Target normal training (N), external common axis

### AP

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.533509 | 0.719360 | 0.452164 |
| Holdout LEFT + frozen pretrained TimesFM | 0.541018 | 0.719539 | 0.455457 |
| Holdout LEFT + frozen random TimesFM | 0.532497 | 0.719307 | 0.453078 |
| Matched-budget native LEFT normal training | 0.567882 | 0.720594 | 0.499495 |
| Native LEFT matched source continuation | 0.526270 | 0.722702 | 0.500441 |
| Unmasked LEFT + frozen pretrained TimesFM | 0.526647 | 0.722609 | 0.494131 |
| Earlier multisource LEFT | 0.497206 | 0.723927 | 0.498182 |
| Native LEFT trained on target normal data | 0.577398 | 0.543881 | 0.499187 |
| Earlier multisource LEFT + target normal training | 0.555001 | 0.542517 | 0.519018 |
| Normal-level deviation | 0.439188 | 0.744804 | 0.428476 |
| CATCH (three archived seeds) | 0.437276 ± 0.008436 | NA | NA |
| PCA residual | 0.534679 | NA | NA |
| TimeRadar (archived block score) | 0.434810 | 0.639315 | NA |
| TimeRadar endpoint V | 0.536488 | NA | 0.444987 |
| DADA | 0.396595 | NA | 0.415141 |
| TimeRCD (FIT-only statistics) | 0.303144 | NA | 0.473921 |

### ROC_AUC

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.712808 | 0.813110 | 0.847893 |
| Holdout LEFT + frozen pretrained TimesFM | 0.717313 | 0.812415 | 0.849307 |
| Holdout LEFT + frozen random TimesFM | 0.712598 | 0.812522 | 0.848935 |
| Matched-budget native LEFT normal training | 0.732309 | 0.813057 | 0.870999 |
| Native LEFT matched source continuation | 0.711257 | 0.811901 | 0.875045 |
| Unmasked LEFT + frozen pretrained TimesFM | 0.711475 | 0.812082 | 0.868094 |
| Earlier multisource LEFT | 0.700389 | 0.811299 | 0.866936 |
| Native LEFT trained on target normal data | 0.775512 | 0.820681 | 0.877656 |
| Earlier multisource LEFT + target normal training | 0.755856 | 0.855454 | 0.882029 |
| Normal-level deviation | 0.653725 | 0.831611 | 0.802073 |
| CATCH (three archived seeds) | 0.646309 ± 0.006074 | NA | NA |
| PCA residual | 0.792130 | NA | NA |
| TimeRadar (archived block score) | 0.671221 | 0.821814 | NA |
| TimeRadar endpoint V | 0.731327 | NA | 0.811991 |
| DADA | 0.604198 | NA | 0.850385 |
| TimeRCD (FIT-only statistics) | 0.476508 | NA | 0.859168 |

### VUS_PR

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.543270 | 0.501293 | 0.473115 |
| Holdout LEFT + frozen pretrained TimesFM | 0.550638 | 0.503943 | 0.481266 |
| Holdout LEFT + frozen random TimesFM | 0.541490 | 0.503409 | 0.474499 |
| Matched-budget native LEFT normal training | 0.567789 | 0.506516 | 0.543441 |
| Native LEFT matched source continuation | 0.527736 | 0.497658 | 0.543549 |
| Unmasked LEFT + frozen pretrained TimesFM | 0.528103 | 0.497345 | 0.535253 |
| Earlier multisource LEFT | 0.502864 | 0.497560 | 0.541924 |
| Native LEFT trained on target normal data | 0.574585 | 0.534715 | 0.542440 |
| Earlier multisource LEFT + target normal training | 0.553671 | 0.530836 | 0.565223 |
| Normal-level deviation | 0.448158 | 0.464219 | 0.427435 |
| CATCH (three archived seeds) | 0.527603 ± 0.008662 | NA | NA |
| PCA residual | 0.529230 | NA | NA |
| TimeRadar (archived block score) | 0.443332 | 0.514125 | NA |
| TimeRadar endpoint V | 0.485394 | NA | 0.467207 |
| DADA | 0.440465 | NA | 0.475523 |
| TimeRCD (FIT-only statistics) | 0.349831 | NA | 0.516771 |

### VUS_ROC

| Method | PSM | SWAT | SMD |
|---|---:|---:|---:|
| Holdout LEFT | 0.730335 | 0.679903 | 0.901491 |
| Holdout LEFT + frozen pretrained TimesFM | 0.735943 | 0.680104 | 0.903284 |
| Holdout LEFT + frozen random TimesFM | 0.729892 | 0.679566 | 0.902194 |
| Matched-budget native LEFT normal training | 0.743930 | 0.683174 | 0.914098 |
| Native LEFT matched source continuation | 0.718670 | 0.665546 | 0.915820 |
| Unmasked LEFT + frozen pretrained TimesFM | 0.718901 | 0.666311 | 0.912140 |
| Earlier multisource LEFT | 0.706424 | 0.660764 | 0.911587 |
| Native LEFT trained on target normal data | 0.785010 | 0.809105 | 0.917348 |
| Earlier multisource LEFT + target normal training | 0.765883 | 0.842850 | 0.919536 |
| Normal-level deviation | 0.655534 | 0.643787 | 0.864645 |
| CATCH (three archived seeds) | 0.695732 ± 0.006690 | NA | NA |
| PCA residual | 0.765810 | NA | NA |
| TimeRadar (archived block score) | 0.679425 | 0.766598 | NA |
| TimeRadar endpoint V | 0.665784 | NA | 0.868872 |
| DADA | 0.651296 | NA | 0.898696 |
| TimeRCD (FIT-only statistics) | 0.532382 | NA | 0.909346 |

## Matched differences

| Domain | Stage | Axis | Contrast | AP | ROC-AUC | VUS-PR | VUS-ROC |
|---|---|---|---|---:|---:|---:|---:|
| PSM | Z | complete | B-A | -0.000568 | +0.000615 | -0.000732 | +0.000215 |
| PSM | Z | complete | B-C | -0.000166 | +0.001076 | +0.000356 | +0.000950 |
| PSM | Z | nonoverlap | B-B_nonoverlap | -0.000215 | +0.001062 | -0.000259 | +0.001112 |
| PSM | Z | external | A-e59_native_Z | -0.018968 | -0.010596 | -0.009005 | -0.001963 |
| PSM | Z | external | B-e59_native_Z | -0.019536 | -0.009981 | -0.009737 | -0.001748 |
| PSM | Z | external | B-e59_history_Z | -0.019912 | -0.010198 | -0.010104 | -0.001979 |
| PSM | N | complete | B-A | +0.007509 | +0.004505 | +0.007369 | +0.005608 |
| PSM | N | complete | B-C | +0.008521 | +0.004715 | +0.009149 | +0.006050 |
| PSM | N | complete | B-native | -0.026864 | -0.014997 | -0.017150 | -0.007988 |
| PSM | N | nonoverlap | B-B_nonoverlap | +0.000071 | +0.000429 | -0.000140 | +0.000210 |
| PSM | N | external | A-e59_native_Z | +0.007239 | +0.001550 | +0.015534 | +0.011665 |
| PSM | N | external | B-e59_native_Z | +0.014748 | +0.006055 | +0.022902 | +0.017272 |
| PSM | N | external | B-e59_history_Z | +0.014372 | +0.005837 | +0.022535 | +0.017042 |
| SWAT | Z | complete | B-A | -0.001508 | -0.000270 | -0.002663 | -0.002933 |
| SWAT | Z | complete | B-C | -0.000437 | -0.000074 | +0.000294 | +0.000044 |
| SWAT | Z | nonoverlap | B-B_nonoverlap | -0.002022 | -0.003156 | -0.005436 | -0.004652 |
| SWAT | Z | external | A-e59_native_Z | +0.001799 | -0.000497 | +0.002552 | +0.002678 |
| SWAT | Z | external | B-e59_native_Z | +0.000291 | -0.000768 | -0.000111 | -0.000255 |
| SWAT | Z | external | B-e59_history_Z | +0.000383 | -0.000949 | +0.000202 | -0.001020 |
| SWAT | N | complete | B-A | +0.000180 | -0.000694 | +0.002650 | +0.000202 |
| SWAT | N | complete | B-C | +0.000232 | -0.000107 | +0.000534 | +0.000539 |
| SWAT | N | complete | B-native | -0.001054 | -0.000642 | -0.002572 | -0.003070 |
| SWAT | N | nonoverlap | B-B_nonoverlap | -0.000049 | +0.000206 | +0.000378 | +0.000187 |
| SWAT | N | external | A-e59_native_Z | -0.003343 | +0.001209 | +0.003635 | +0.014357 |
| SWAT | N | external | B-e59_native_Z | -0.003163 | +0.000514 | +0.006285 | +0.014558 |
| SWAT | N | external | B-e59_history_Z | -0.003070 | +0.000333 | +0.006599 | +0.013793 |
| SMD | Z | complete | B-A | +0.006805 | +0.005976 | +0.014546 | +0.004628 |
| SMD | Z | complete | B-C | +0.005729 | +0.006040 | +0.011932 | +0.004381 |
| SMD | Z | nonoverlap | B-B_nonoverlap | +0.000119 | -0.000168 | -0.000216 | -0.000454 |
| SMD | Z | external | A-e59_native_Z | -0.050959 | -0.025370 | -0.073725 | -0.011323 |
| SMD | Z | external | B-e59_native_Z | -0.044150 | -0.019391 | -0.059140 | -0.006678 |
| SMD | Z | external | B-e59_history_Z | -0.037840 | -0.012440 | -0.050844 | -0.002998 |
| SMD | N | complete | B-A | +0.003293 | +0.001415 | +0.008200 | +0.001783 |
| SMD | N | complete | B-C | +0.002378 | +0.000372 | +0.006796 | +0.001088 |
| SMD | N | complete | B-native | -0.044038 | -0.021684 | -0.062397 | -0.010825 |
| SMD | N | nonoverlap | B-B_nonoverlap | -0.000469 | -0.000244 | -0.000710 | -0.000332 |
| SMD | N | external | A-e59_native_Z | -0.048277 | -0.027152 | -0.070434 | -0.014329 |
| SMD | N | external | B-e59_native_Z | -0.044984 | -0.025737 | -0.062283 | -0.012536 |
| SMD | N | external | B-e59_history_Z | -0.038674 | -0.018787 | -0.053987 | -0.008856 |

## Controlled core ROC-AUC

| Domain | Background | Stage | Contrast | Holdout | Pretrained history | Random history |
|---|---|---|---|---:|---:|---:|
| PSM | coupled_AR | Z | weak / weak_control | 0.509766 | 0.505127 | 0.510010 |
| PSM | coupled_AR | Z | lag / lag_amplitude | 0.073486 | 0.050537 | 0.063477 |
| PSM | coupled_AR | Z | lag / synchronous | 0.485840 | 0.488281 | 0.489014 |
| PSM | normal_SELECT | Z | weak / weak_control | 0.503906 | 0.503906 | 0.511719 |
| PSM | normal_SELECT | Z | lag / lag_amplitude | 0.480469 | 0.484375 | 0.468750 |
| PSM | normal_SELECT | Z | lag / synchronous | 0.515625 | 0.511719 | 0.515625 |
| PSM | coupled_AR | N | weak / weak_control | 0.507080 | 0.507080 | 0.509033 |
| PSM | coupled_AR | N | lag / lag_amplitude | 0.068848 | 0.073975 | 0.068359 |
| PSM | coupled_AR | N | lag / synchronous | 0.485596 | 0.477539 | 0.481445 |
| PSM | normal_SELECT | N | weak / weak_control | 0.503906 | 0.511719 | 0.503906 |
| PSM | normal_SELECT | N | lag / lag_amplitude | 0.460938 | 0.464844 | 0.457031 |
| PSM | normal_SELECT | N | lag / synchronous | 0.503906 | 0.496094 | 0.503906 |
| SWAT | coupled_AR | Z | weak / weak_control | 0.507324 | 0.515869 | 0.508545 |
| SWAT | coupled_AR | Z | lag / lag_amplitude | 0.257568 | 0.159668 | 0.216797 |
| SWAT | coupled_AR | Z | lag / synchronous | 0.509033 | 0.506348 | 0.501709 |
| SWAT | normal_SELECT | Z | weak / weak_control | 0.199219 | 0.394531 | 0.136719 |
| SWAT | normal_SELECT | Z | lag / lag_amplitude | 0.507812 | 0.503906 | 0.496094 |
| SWAT | normal_SELECT | Z | lag / synchronous | 0.480469 | 0.488281 | 0.492188 |
| SWAT | coupled_AR | N | weak / weak_control | 0.495361 | 0.489746 | 0.491455 |
| SWAT | coupled_AR | N | lag / lag_amplitude | 0.469238 | 0.464111 | 0.423584 |
| SWAT | coupled_AR | N | lag / synchronous | 0.542969 | 0.557617 | 0.560059 |
| SWAT | normal_SELECT | N | weak / weak_control | 0.058594 | 0.070312 | 0.066406 |
| SWAT | normal_SELECT | N | lag / lag_amplitude | 0.492188 | 0.500000 | 0.500000 |
| SWAT | normal_SELECT | N | lag / synchronous | 0.500000 | 0.492188 | 0.511719 |

## Measured stage cost

| Stage | Summed wall seconds | Native calls | History calls |
|---|---:|---:|---:|
| joint source train and SELECT | 10396.098 | 19152 | 12198 |
| isolation and archived C bypass | 11.535 | 165 | 0 |
| CPU ranking metric computation | 776.081 | 0 | 0 |
| joint normal train and SELECT | 2770.891 | 26820 | 496 |
| TEST historical feature extraction | 0.000 | 0 | 0 |
| TEST frozen scoring | 225.095 | 6930 | 0 |
| controlled scoring | 14.827 | 864 | 0 |
