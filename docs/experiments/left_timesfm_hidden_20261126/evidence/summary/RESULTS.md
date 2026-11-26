# Frozen TimesFM hidden-conditioned LEFT: sealed results

Newly trained rows use seed2025. CATCH retains archived seeds 0, 1 and 2. SMD reports all 28 entities with equal weights; entities are not training repetitions. All current-window scores are native LEFT, available only at the 192-point current-window end. Preceding 512 observations are restricted to the same record and role. The frozen TimesFM residual forecast baseline appears only on the compatible mixed-role intersection; it has no complete-test cache here.

Source continuation uses a fixed 2000-update endpoint in each fold; normal SOURCE SELECT is diagnostic only and never selects a checkpoint. D assigns an earlier legal same-record history with isolated randomness; it is not a frequency-preserving permutation. Short TEP runs can make the reassigned histories overlap substantially. The saved SOURCE SELECT fixture quantifies feature differences and overlap without another forward pass.

actual_costs.csv reports measured stage wall-clock times with a timing_scope field. Encoding timers exclude model initialization and state hashes, and score timers exclude encoding. These phases are not exact GPU kernel times or a complete end-to-end billing ledger. Waiting for source checkpoints is not charged as model computation. CPU metric reevaluation is listed separately.

## Complete native common axis

#### AP

| Method | PSM | SWaT | SMD28 macro |
|---|---:|---:|---:|
| Matched source continuation LEFT | 0.526270 | 0.722702 | 0.500424 |
| LEFT with history mean/variance | 0.526313 | 0.722822 | 0.505100 |
| LEFT with pretrained TimesFM hidden history | 0.526647 | 0.722609 | 0.494111 |
| LEFT with causally reassigned TimesFM history | 0.526648 | 0.722544 | 0.502862 |
| LEFT with frozen random TimesFM history | 0.526143 | 0.722673 | 0.498045 |
| Earlier multi-source LEFT | 0.497206 | 0.723927 | 0.498165 |
| Multi-source LEFT + target-normal training | 0.555001 | 0.542517 | 0.518995 |
| LEFT target-normal training from scratch | 0.577398 | 0.543881 | 0.499170 |
| Normal-level deviation | 0.439188 | 0.744804 | 0.428455 |
| PCA residual | 0.534679 | NA | NA |
| Official TimeRadar fixed block readout | 0.434810 | 0.639315 | NA |
| CATCH (archived seeds, mean ± sample SD) | 0.437276 ± 0.008436 | NA | NA |

#### ROC_AUC

| Method | PSM | SWaT | SMD28 macro |
|---|---:|---:|---:|
| Matched source continuation LEFT | 0.711257 | 0.811901 | 0.875029 |
| LEFT with history mean/variance | 0.711203 | 0.811829 | 0.871561 |
| LEFT with pretrained TimesFM hidden history | 0.711475 | 0.812082 | 0.868077 |
| LEFT with causally reassigned TimesFM history | 0.711509 | 0.811961 | 0.876073 |
| LEFT with frozen random TimesFM history | 0.711137 | 0.811835 | 0.873171 |
| Earlier multi-source LEFT | 0.700389 | 0.811299 | 0.866925 |
| Multi-source LEFT + target-normal training | 0.755856 | 0.855454 | 0.882005 |
| LEFT target-normal training from scratch | 0.775512 | 0.820681 | 0.877635 |
| Normal-level deviation | 0.653725 | 0.831611 | 0.802068 |
| PCA residual | 0.792130 | NA | NA |
| Official TimeRadar fixed block readout | 0.671221 | 0.821814 | NA |
| CATCH (archived seeds, mean ± sample SD) | 0.646309 ± 0.006074 | NA | NA |

#### VUS_PR

| Method | PSM | SWaT | SMD28 macro |
|---|---:|---:|---:|
| Matched source continuation LEFT | 0.527736 | 0.497658 | 0.543501 |
| LEFT with history mean/variance | 0.527711 | 0.497651 | 0.554570 |
| LEFT with pretrained TimesFM hidden history | 0.528103 | 0.497345 | 0.535364 |
| LEFT with causally reassigned TimesFM history | 0.528168 | 0.496963 | 0.545122 |
| LEFT with frozen random TimesFM history | 0.527609 | 0.497712 | 0.542814 |
| Earlier multi-source LEFT | 0.502864 | 0.497560 | 0.541909 |
| Multi-source LEFT + target-normal training | 0.553671 | 0.530836 | 0.565102 |
| LEFT target-normal training from scratch | 0.574585 | 0.534715 | 0.542521 |
| Normal-level deviation | 0.448158 | 0.464219 | 0.427299 |
| PCA residual | 0.529230 | NA | NA |
| Official TimeRadar fixed block readout | 0.443332 | 0.514125 | NA |
| CATCH (archived seeds, mean ± sample SD) | 0.527603 ± 0.008662 | NA | NA |

#### VUS_ROC

| Method | PSM | SWaT | SMD28 macro |
|---|---:|---:|---:|
| Matched source continuation LEFT | 0.718670 | 0.665546 | 0.915799 |
| LEFT with history mean/variance | 0.718581 | 0.665231 | 0.915476 |
| LEFT with pretrained TimesFM hidden history | 0.718901 | 0.666311 | 0.912121 |
| LEFT with causally reassigned TimesFM history | 0.719028 | 0.665765 | 0.915991 |
| LEFT with frozen random TimesFM history | 0.718624 | 0.665607 | 0.915547 |
| Earlier multi-source LEFT | 0.706424 | 0.660764 | 0.911569 |
| Multi-source LEFT + target-normal training | 0.765883 | 0.842850 | 0.919506 |
| LEFT target-normal training from scratch | 0.785010 | 0.809105 | 0.917336 |
| Normal-level deviation | 0.655534 | 0.643787 | 0.864631 |
| PCA residual | 0.765810 | NA | NA |
| Official TimeRadar fixed block readout | 0.679425 | 0.766598 | NA |
| CATCH (archived seeds, mean ± sample SD) | 0.695732 ± 0.006690 | NA | NA |
## External intersection axis

#### AP

| Method | PSM | SWaT | SMD28 macro |
|---|---:|---:|---:|
| Matched source continuation LEFT | 0.526270 | 0.722702 | 0.500441 |
| LEFT with history mean/variance | 0.526313 | 0.722822 | 0.505111 |
| LEFT with pretrained TimesFM hidden history | 0.526647 | 0.722609 | 0.494131 |
| LEFT with causally reassigned TimesFM history | 0.526648 | 0.722544 | 0.502881 |
| LEFT with frozen random TimesFM history | 0.526143 | 0.722673 | 0.498063 |
| Earlier multi-source LEFT | 0.497206 | 0.723927 | 0.498182 |
| Multi-source LEFT + target-normal training | 0.555001 | 0.542517 | 0.519018 |
| LEFT target-normal training from scratch | 0.577398 | 0.543881 | 0.499187 |
| Normal-level deviation | 0.439188 | 0.744804 | 0.428476 |
| PCA residual | 0.534679 | NA | NA |
| Official TimeRadar fixed block readout | 0.434810 | 0.639315 | NA |
| DADA official pretrained | 0.396595 | NA | 0.415141 |
| TimeRCD official, FIT-only statistics | 0.303144 | NA | 0.473921 |
| Official TimeRadar endpoint V | 0.536488 | NA | 0.444987 |
| CATCH (archived seeds, mean ± sample SD) | 0.437276 ± 0.008436 | NA | NA |

#### ROC_AUC

| Method | PSM | SWaT | SMD28 macro |
|---|---:|---:|---:|
| Matched source continuation LEFT | 0.711257 | 0.811901 | 0.875045 |
| LEFT with history mean/variance | 0.711203 | 0.811829 | 0.871569 |
| LEFT with pretrained TimesFM hidden history | 0.711475 | 0.812082 | 0.868094 |
| LEFT with causally reassigned TimesFM history | 0.711509 | 0.811961 | 0.876089 |
| LEFT with frozen random TimesFM history | 0.711137 | 0.811835 | 0.873186 |
| Earlier multi-source LEFT | 0.700389 | 0.811299 | 0.866936 |
| Multi-source LEFT + target-normal training | 0.755856 | 0.855454 | 0.882029 |
| LEFT target-normal training from scratch | 0.775512 | 0.820681 | 0.877656 |
| Normal-level deviation | 0.653725 | 0.831611 | 0.802073 |
| PCA residual | 0.792130 | NA | NA |
| Official TimeRadar fixed block readout | 0.671221 | 0.821814 | NA |
| DADA official pretrained | 0.604198 | NA | 0.850385 |
| TimeRCD official, FIT-only statistics | 0.476508 | NA | 0.859168 |
| Official TimeRadar endpoint V | 0.731327 | NA | 0.811991 |
| CATCH (archived seeds, mean ± sample SD) | 0.646309 ± 0.006074 | NA | NA |

#### VUS_PR

| Method | PSM | SWaT | SMD28 macro |
|---|---:|---:|---:|
| Matched source continuation LEFT | 0.527736 | 0.497658 | 0.543549 |
| LEFT with history mean/variance | 0.527711 | 0.497651 | 0.554666 |
| LEFT with pretrained TimesFM hidden history | 0.528103 | 0.497345 | 0.535253 |
| LEFT with causally reassigned TimesFM history | 0.528168 | 0.496963 | 0.544934 |
| LEFT with frozen random TimesFM history | 0.527609 | 0.497712 | 0.542944 |
| Earlier multi-source LEFT | 0.502864 | 0.497560 | 0.541924 |
| Multi-source LEFT + target-normal training | 0.553671 | 0.530836 | 0.565223 |
| LEFT target-normal training from scratch | 0.574585 | 0.534715 | 0.542440 |
| Normal-level deviation | 0.448158 | 0.464219 | 0.427435 |
| PCA residual | 0.529230 | NA | NA |
| Official TimeRadar fixed block readout | 0.443332 | 0.514125 | NA |
| DADA official pretrained | 0.440465 | NA | 0.475523 |
| TimeRCD official, FIT-only statistics | 0.349831 | NA | 0.516771 |
| Official TimeRadar endpoint V | 0.485394 | NA | 0.467207 |
| CATCH (archived seeds, mean ± sample SD) | 0.527603 ± 0.008662 | NA | NA |

#### VUS_ROC

| Method | PSM | SWaT | SMD28 macro |
|---|---:|---:|---:|
| Matched source continuation LEFT | 0.718670 | 0.665546 | 0.915820 |
| LEFT with history mean/variance | 0.718581 | 0.665231 | 0.915483 |
| LEFT with pretrained TimesFM hidden history | 0.718901 | 0.666311 | 0.912140 |
| LEFT with causally reassigned TimesFM history | 0.719028 | 0.665765 | 0.916004 |
| LEFT with frozen random TimesFM history | 0.718624 | 0.665607 | 0.915569 |
| Earlier multi-source LEFT | 0.706424 | 0.660764 | 0.911587 |
| Multi-source LEFT + target-normal training | 0.765883 | 0.842850 | 0.919536 |
| LEFT target-normal training from scratch | 0.785010 | 0.809105 | 0.917348 |
| Normal-level deviation | 0.655534 | 0.643787 | 0.864645 |
| PCA residual | 0.765810 | NA | NA |
| Official TimeRadar fixed block readout | 0.679425 | 0.766598 | NA |
| DADA official pretrained | 0.651296 | NA | 0.898696 |
| TimeRCD official, FIT-only statistics | 0.532382 | NA | 0.909346 |
| Official TimeRadar endpoint V | 0.665784 | NA | 0.868872 |
| CATCH (archived seeds, mean ± sample SD) | 0.695732 ± 0.006690 | NA | NA |
The previous full LEFT source, scratch and target-normal rows retain their separate learning permissions. TimeRadar endpoint, DADA and TimeRCD rows retain their original preprocessing and availability; common points do not make information permissions identical. CATCH summarizes archived seeds with sample standard deviation; individual values remain in entity_metrics.csv. No external method is retrained.

## Independent mixed-role axis

#### AP

| Method | PSM | SWaT | SMD28 macro |
|---|---:|---:|---:|
| Matched source continuation LEFT | 0.547434 | 0.789938 | 0.561068 |
| Matched LEFT + ordinary source replay U | 0.581121 | 0.785986 | 0.572650 |
| LEFT with pretrained TimesFM hidden history | 0.547884 | 0.789851 | 0.552371 |
| TimesFM hidden LEFT + ordinary source replay U | 0.579317 | 0.785674 | 0.568230 |
| Earlier multi-source LEFT frozen | 0.505103 | 0.791359 | 0.559092 |
| Random-initialized LEFT mixed U | 0.586439 | 0.788191 | 0.562302 |
| Earlier LEFT ordinary mixed U | 0.581909 | 0.779990 | 0.572168 |
| Earlier LEFT ordinary source replay U | 0.597384 | 0.786150 | 0.576585 |
| Official pretrained TimesFM median forecast residual | 0.344382 | 0.094007 | NA |
| Earlier LEFT window-reliability replay U | NA | NA | 0.575462 |
| Earlier LEFT source-only continuation | NA | NA | 0.559107 |

#### ROC_AUC

| Method | PSM | SWaT | SMD28 macro |
|---|---:|---:|---:|
| Matched source continuation LEFT | 0.735898 | 0.843021 | 0.882945 |
| Matched LEFT + ordinary source replay U | 0.751238 | 0.842110 | 0.893220 |
| LEFT with pretrained TimesFM hidden history | 0.736094 | 0.843377 | 0.870633 |
| TimesFM hidden LEFT + ordinary source replay U | 0.749907 | 0.842347 | 0.886882 |
| Earlier multi-source LEFT frozen | 0.724563 | 0.842253 | 0.873979 |
| Random-initialized LEFT mixed U | 0.751179 | 0.842148 | 0.889660 |
| Earlier LEFT ordinary mixed U | 0.768667 | 0.841598 | 0.889359 |
| Earlier LEFT ordinary source replay U | 0.766836 | 0.841902 | 0.896745 |
| Official pretrained TimesFM median forecast residual | 0.566867 | 0.288345 | NA |
| Earlier LEFT window-reliability replay U | NA | NA | 0.896428 |
| Earlier LEFT source-only continuation | NA | NA | 0.877629 |

#### VUS_PR

| Method | PSM | SWaT | SMD28 macro |
|---|---:|---:|---:|
| Matched source continuation LEFT | 0.550867 | 0.629790 | 0.600836 |
| Matched LEFT + ordinary source replay U | 0.577741 | 0.634617 | 0.614971 |
| LEFT with pretrained TimesFM hidden history | 0.551252 | 0.629088 | 0.588657 |
| TimesFM hidden LEFT + ordinary source replay U | 0.575707 | 0.634598 | 0.608311 |
| Earlier multi-source LEFT frozen | 0.516533 | 0.630820 | 0.602790 |
| Random-initialized LEFT mixed U | 0.579141 | 0.633919 | 0.608295 |
| Earlier LEFT ordinary mixed U | 0.589567 | 0.632938 | 0.618206 |
| Earlier LEFT ordinary source replay U | 0.596942 | 0.633984 | 0.622716 |
| Official pretrained TimesFM median forecast residual | 0.376067 | 0.096578 | NA |
| Earlier LEFT window-reliability replay U | NA | NA | 0.621058 |
| Earlier LEFT source-only continuation | NA | NA | 0.599184 |

#### VUS_ROC

| Method | PSM | SWaT | SMD28 macro |
|---|---:|---:|---:|
| Matched source continuation LEFT | 0.748275 | 0.737104 | 0.918942 |
| Matched LEFT + ordinary source replay U | 0.764475 | 0.741355 | 0.925004 |
| LEFT with pretrained TimesFM hidden history | 0.748423 | 0.736855 | 0.912694 |
| TimesFM hidden LEFT + ordinary source replay U | 0.763008 | 0.742051 | 0.917787 |
| Earlier multi-source LEFT frozen | 0.735727 | 0.734043 | 0.916195 |
| Random-initialized LEFT mixed U | 0.761167 | 0.741338 | 0.922942 |
| Earlier LEFT ordinary mixed U | 0.781413 | 0.744598 | 0.925027 |
| Earlier LEFT ordinary source replay U | 0.782078 | 0.740229 | 0.928316 |
| Official pretrained TimesFM median forecast residual | 0.608701 | 0.298351 | NA |
| Earlier LEFT window-reliability replay U | NA | NA | 0.928834 |
| Earlier LEFT source-only continuation | NA | NA | 0.916230 |
The original 512 CAL points remain fixed. When any lacks a full legal 512-point history, threshold operations are NA; no replacement CAL, zero padding across roles, or threshold selection is performed. Historical PSM all-attack CAL and prior SMD nominal-normal CAL results retain their old protocol.

## Paired evidence

All four metrics and per-entity signs/medians are in paired_summary.csv and paired_entities.csv. Differences are computed only after confirming identical current start/stop axes. The controlled panel reuses the archived edits with 256 actual historical observations and left padding to 512. Synthetic records each offer only one legal history, so D retains identity there; D cannot test history reassignment in that controlled group.
