# Frozen TimesFM history conditioning: complete Z comparison

New source learning uses seed2025 and the same fixed 2,000-update budget. SMD28 is the equal-weight mean of all 28 per-machine metrics, not 28 training repetitions. CATCH retains archived seeds 0, 1 and 2; no other row has an invented standard deviation. These are local cache-based results.

## External common timestamps

The following panels use the separately frozen external intersection, including PSM [576,87744) and SWaT [495576,944856). SMD endpoints are entity-specific and may be shortened by the external methods' existing coverage. Complete-axis learning contrasts are reported separately below. Equal timestamps do not imply equal context or training access.

### AP

| Training group | Complete method | PSM | SWaT | SMD28 macro |
|---|---|---:|---:|---:|
| Project source, Z | Earlier complete multi-source LEFT | 0.497206 | 0.723927 | 0.498182 |
| Matched source continuation, Z | Native LEFT continuation | 0.526270 | 0.722702 | 0.500441 |
| Matched source continuation, Z | LEFT + historical mean/variance | 0.526313 | 0.722822 | 0.505111 |
| Matched source continuation, Z | LEFT + pretrained TimesFM hidden history | 0.526647 | 0.722609 | 0.494131 |
| Matched source continuation, Z | LEFT + earlier reassigned hidden history | 0.526648 | 0.722544 | 0.502881 |
| Matched source continuation, Z | LEFT + frozen random TimesFM history | 0.526143 | 0.722673 | 0.498063 |
| Target normal training | Native LEFT from scratch | 0.577398 | 0.543881 | 0.499187 |
| Source then target normal | Multi-source LEFT + target normal training | 0.555001 | 0.542517 | 0.519018 |
| Target normal training | CATCH, archived three-seed mean ± sample SD | 0.437276 ± 0.008436 | NA | NA |
| Target normal fit | PCA residual | 0.534679 | NA | NA |
| Target normal statistics | Normal-level deviation | 0.439188 | 0.744804 | 0.428476 |
| Official pretrained | TimeRadar, archived block readout | 0.434810 | 0.639315 | NA |
| Official pretrained | TimeRadar, endpoint V | 0.536488 | NA | 0.444987 |
| Official pretrained | DADA | 0.396595 | NA | 0.415141 |
| Official pretrained | TimeRCD, FIT-only statistics | 0.303144 | NA | 0.473921 |

### ROC_AUC

| Training group | Complete method | PSM | SWaT | SMD28 macro |
|---|---|---:|---:|---:|
| Project source, Z | Earlier complete multi-source LEFT | 0.700389 | 0.811299 | 0.866936 |
| Matched source continuation, Z | Native LEFT continuation | 0.711257 | 0.811901 | 0.875045 |
| Matched source continuation, Z | LEFT + historical mean/variance | 0.711203 | 0.811829 | 0.871569 |
| Matched source continuation, Z | LEFT + pretrained TimesFM hidden history | 0.711475 | 0.812082 | 0.868094 |
| Matched source continuation, Z | LEFT + earlier reassigned hidden history | 0.711509 | 0.811961 | 0.876089 |
| Matched source continuation, Z | LEFT + frozen random TimesFM history | 0.711137 | 0.811835 | 0.873186 |
| Target normal training | Native LEFT from scratch | 0.775512 | 0.820681 | 0.877656 |
| Source then target normal | Multi-source LEFT + target normal training | 0.755856 | 0.855454 | 0.882029 |
| Target normal training | CATCH, archived three-seed mean ± sample SD | 0.646309 ± 0.006074 | NA | NA |
| Target normal fit | PCA residual | 0.792130 | NA | NA |
| Target normal statistics | Normal-level deviation | 0.653725 | 0.831611 | 0.802073 |
| Official pretrained | TimeRadar, archived block readout | 0.671221 | 0.821814 | NA |
| Official pretrained | TimeRadar, endpoint V | 0.731327 | NA | 0.811991 |
| Official pretrained | DADA | 0.604198 | NA | 0.850385 |
| Official pretrained | TimeRCD, FIT-only statistics | 0.476508 | NA | 0.859168 |

### VUS_PR

| Training group | Complete method | PSM | SWaT | SMD28 macro |
|---|---|---:|---:|---:|
| Project source, Z | Earlier complete multi-source LEFT | 0.502864 | 0.497560 | 0.541924 |
| Matched source continuation, Z | Native LEFT continuation | 0.527736 | 0.497658 | 0.543549 |
| Matched source continuation, Z | LEFT + historical mean/variance | 0.527711 | 0.497651 | 0.554666 |
| Matched source continuation, Z | LEFT + pretrained TimesFM hidden history | 0.528103 | 0.497345 | 0.535253 |
| Matched source continuation, Z | LEFT + earlier reassigned hidden history | 0.528168 | 0.496963 | 0.544934 |
| Matched source continuation, Z | LEFT + frozen random TimesFM history | 0.527609 | 0.497712 | 0.542944 |
| Target normal training | Native LEFT from scratch | 0.574585 | 0.534715 | 0.542440 |
| Source then target normal | Multi-source LEFT + target normal training | 0.553671 | 0.530836 | 0.565223 |
| Target normal training | CATCH, archived three-seed mean ± sample SD | 0.527603 ± 0.008662 | NA | NA |
| Target normal fit | PCA residual | 0.529230 | NA | NA |
| Target normal statistics | Normal-level deviation | 0.448158 | 0.464219 | 0.427435 |
| Official pretrained | TimeRadar, archived block readout | 0.443332 | 0.514125 | NA |
| Official pretrained | TimeRadar, endpoint V | 0.485394 | NA | 0.467207 |
| Official pretrained | DADA | 0.440465 | NA | 0.475523 |
| Official pretrained | TimeRCD, FIT-only statistics | 0.349831 | NA | 0.516771 |

### VUS_ROC

| Training group | Complete method | PSM | SWaT | SMD28 macro |
|---|---|---:|---:|---:|
| Project source, Z | Earlier complete multi-source LEFT | 0.706424 | 0.660764 | 0.911587 |
| Matched source continuation, Z | Native LEFT continuation | 0.718670 | 0.665546 | 0.915820 |
| Matched source continuation, Z | LEFT + historical mean/variance | 0.718581 | 0.665231 | 0.915483 |
| Matched source continuation, Z | LEFT + pretrained TimesFM hidden history | 0.718901 | 0.666311 | 0.912140 |
| Matched source continuation, Z | LEFT + earlier reassigned hidden history | 0.719028 | 0.665765 | 0.916004 |
| Matched source continuation, Z | LEFT + frozen random TimesFM history | 0.718624 | 0.665607 | 0.915569 |
| Target normal training | Native LEFT from scratch | 0.785010 | 0.809105 | 0.917348 |
| Source then target normal | Multi-source LEFT + target normal training | 0.765883 | 0.842850 | 0.919536 |
| Target normal training | CATCH, archived three-seed mean ± sample SD | 0.695732 ± 0.006690 | NA | NA |
| Target normal fit | PCA residual | 0.765810 | NA | NA |
| Target normal statistics | Normal-level deviation | 0.655534 | 0.643787 | 0.864645 |
| Official pretrained | TimeRadar, archived block readout | 0.679425 | 0.766598 | NA |
| Official pretrained | TimeRadar, endpoint V | 0.665784 | NA | 0.868872 |
| Official pretrained | DADA | 0.651296 | NA | 0.898696 |
| Official pretrained | TimeRCD, FIT-only statistics | 0.532382 | NA | 0.909346 |

Project Z uses the existing target normal-prefix statistics, with no target parameter update. The PSM/SWaT source folds use TEP+SMD; the SMD fold uses TEP+HAI. The candidate retains variable-specific 512-observation historical states; TEP source histories contain 256 valid points in a masked 512-slot input. Native LEFT scores become available at the end of the current 192-point window. Target-normal LEFT uses additional target normal optimization and validation. CATCH and PCA retain target FIT-only scaling; level deviation uses the LEFT normal-prefix scaler. Archived TimeRadar block processing, causal endpoint V, DADA and long-context TimeRCD retain their original inference and normalization permissions. Upstream TimesFM target-dataset membership is not established. CrossAD/ModernTCN paper-reported rows remain in the existing literature panel, outside this local common-axis comparison.

### Context and statistics permissions

| Method group | Target parameter learning / statistics | Input and score availability |
|---|---|---|
| Project source LEFT and A–E | No target parameter update; public panels use the bound complete normal-prefix scaler, including its normal SELECT portion | Current W192; B–E additionally use preceding history. Scores available at current-window end |
| Target-normal LEFT, with or without source initialization | Target normal FIT optimization and normal SELECT; normal-prefix statistics | Native W192 full-window score, available at its end |
| CATCH | Archived target FIT training and FIT-only scaling | W192, full-window offline readout |
| PCA | FIT-only scaling and PCA fitted on at most 50,000 FIT observations | Current observation only; pointwise residual |
| Normal-level deviation | Same normal-prefix statistics as LEFT; no learned encoder | Current observation only; pointwise SmoothL1 to zero |
| TimeRadar archived block readout | Released pretrained weights; original archived per-dataset preprocessing | W100 block readout; window-end availability |
| TimeRadar endpoint V | Released pretrained weights; archived target FIT statistics | W100 ending at the scored point; pointwise causal endpoint |
| DADA | Released pretrained weights; full published normal-training statistics, plus the retained author internal-normalization flags | W100 nonoverlapping blocks, ten copies and native variance; block-end availability |
| TimeRCD FIT-only adapter | Published pretrained weights; archived target FIT statistics, no test-statistic fitting | Native context capped at 5,000 observations; each score waits for its actual window end |

The [archived external-method contract](../native_endpoint_external_20261125/RESULTS.md) retains the original model identities, normalization flags and footprints. DADA's unreleased pretraining procedure and upstream TimesFM training membership are not reconstructed or assumed to match project source access. The separate historical TimesFM forecast-residual cache covers only the mixed suffix here. It is not silently treated as a complete-test result, and its score is never fused with the new detector.

## Matched learning contrasts on the complete history-valid axis

| Domain | Contrast | Δ AP | Δ ROC-AUC | Δ VUS-PR | Δ VUS-ROC |
|---|---|---:|---:|---:|---:|
| PSM | C−A | +0.000377 | +0.000218 | +0.000367 | +0.000230 |
| PSM | C−B | +0.000333 | +0.000273 | +0.000392 | +0.000320 |
| PSM | C−D | -0.000001 | -0.000033 | -0.000065 | -0.000128 |
| PSM | C−E | +0.000504 | +0.000338 | +0.000494 | +0.000277 |
| SWAT | C−A | -0.000093 | +0.000181 | -0.000313 | +0.000765 |
| SWAT | C−B | -0.000213 | +0.000253 | -0.000306 | +0.001081 |
| SWAT | C−D | +0.000066 | +0.000121 | +0.000382 | +0.000547 |
| SWAT | C−E | -0.000063 | +0.000247 | -0.000368 | +0.000704 |
| SMD | C−A | -0.006313 | -0.006952 | -0.008137 | -0.003678 |
| SMD | C−B | -0.010990 | -0.003484 | -0.019207 | -0.003355 |
| SMD | C−D | -0.008751 | -0.007995 | -0.009759 | -0.003870 |
| SMD | C−E | -0.003934 | -0.005094 | -0.007451 | -0.003426 |

| SMD28 contrast | Metric | Mean difference | Median difference | Win / tie / loss |
|---|---|---:|---:|---:|
| C−A | AP | -0.006313 | -0.004177 | 12 / 0 / 16 |
| C−A | ROC_AUC | -0.006952 | -0.002311 | 9 / 0 / 19 |
| C−A | VUS_PR | -0.008137 | -0.001322 | 9 / 1 / 18 |
| C−A | VUS_ROC | -0.003678 | -0.001673 | 8 / 1 / 19 |
| C−B | AP | -0.010990 | -0.006057 | 10 / 0 / 18 |
| C−B | ROC_AUC | -0.003484 | -0.004229 | 8 / 0 / 20 |
| C−B | VUS_PR | -0.019207 | -0.004881 | 10 / 1 / 17 |
| C−B | VUS_ROC | -0.003355 | -0.000361 | 11 / 1 / 16 |
| C−D | AP | -0.008751 | -0.004133 | 6 / 0 / 22 |
| C−D | ROC_AUC | -0.007995 | -0.001371 | 9 / 0 / 19 |
| C−D | VUS_PR | -0.009759 | -0.003868 | 9 / 1 / 18 |
| C−D | VUS_ROC | -0.003870 | -0.000221 | 10 / 1 / 17 |
| C−E | AP | -0.003934 | -0.003290 | 10 / 0 / 18 |
| C−E | ROC_AUC | -0.005094 | -0.001360 | 11 / 0 / 17 |
| C−E | VUS_PR | -0.007451 | -0.004124 | 9 / 1 / 18 |
| C−E | VUS_ROC | -0.003426 | -0.001152 | 8 / 1 / 19 |

All differences above use the same within-entity complete axis, not the external-intersection values. A/B/C/D/E denote native continuation, historical statistics, pretrained hidden states, earlier reassigned hidden states and frozen random hidden states respectively. The D assignment is causal and same-record, but is not a frequency-preserving permutation. Short TEP histories overlap strongly; this limits how strongly a small C−D contrast can be interpreted.
