# Paired source and full-U development results

Single project training seed 2025. Values below retain separate public and QUERY panels. All ranking differences are computed on identical ticks within a panel. LEFT retains its offline context permission. No calibration threshold is introduced.

## Z: public

| Method | PSM AP | PSM ROC-AUC | PSM VUS-PR | PSM VUS-ROC | SWaT AP | SWaT ROC-AUC | SWaT VUS-PR | SWaT VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TimeRadar + trainable history context | 0.495524 | 0.719034 | 0.468786 | 0.658618 | 0.728971 | 0.838661 | 0.469702 | 0.672024 |
| Matched native decoder continuation | 0.492870 | 0.717044 | 0.469319 | 0.661607 | 0.722847 | 0.834580 | 0.461426 | 0.664574 |
| Official TimeRadar, matched project readout | 0.494071 | 0.714661 | 0.462378 | 0.654056 | 0.734178 | 0.839639 | 0.475467 | 0.671378 |
| Frozen multi-source LEFT (offline W192) | 0.497257 | 0.700362 | 0.502917 | 0.706403 | 0.723928 | 0.811287 | 0.497585 | 0.660756 |
| Native LEFT, target-normal training (offline W192) | 0.577412 | 0.775419 | 0.574606 | 0.784912 | 0.543882 | 0.820670 | 0.534722 | 0.809097 |

Axes: PSM [611,87744); SWaT [495611,944856).
The target-normal LEFT row is an N background reference, not source-frozen Z. Both LEFT rows are offline W192; TimeRadar rows are causal endpoints. Target normal-prefix statistics include normal SELECT for every row.

## Z: QUERY

| Method | PSM AP | PSM ROC-AUC | PSM VUS-PR | PSM VUS-ROC | SWaT AP | SWaT ROC-AUC | SWaT VUS-PR | SWaT VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TimeRadar + trainable history context | 0.494804 | 0.711088 | 0.456147 | 0.633278 | 0.787593 | 0.855493 | 0.563266 | 0.702569 |
| Matched native decoder continuation | 0.491406 | 0.709107 | 0.457245 | 0.635653 | 0.784892 | 0.851313 | 0.555679 | 0.696760 |
| Official TimeRadar, matched project readout | 0.494384 | 0.709309 | 0.449681 | 0.628119 | 0.792434 | 0.857547 | 0.575100 | 0.707909 |
| Frozen multi-source LEFT (offline W192) | 0.504839 | 0.724752 | 0.516162 | 0.735922 | 0.791359 | 0.842269 | 0.630793 | 0.734052 |
| Native LEFT, target-normal training (offline W192) | 0.587050 | 0.778099 | 0.577899 | 0.781838 | 0.608253 | 0.841497 | 0.607626 | 0.829731 |

Axes: PSM [23115,87744); SWaT [608634,944856).
The target-normal LEFT row is an N background reference, not source-frozen Z. Both LEFT rows are offline W192; TimeRadar rows are causal endpoints. Target normal-prefix statistics include normal SELECT for every row.

## U: QUERY

| Method | PSM AP | PSM ROC-AUC | PSM VUS-PR | PSM VUS-ROC | SWaT AP | SWaT ROC-AUC | SWaT VUS-PR | SWaT VUS-ROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TimeRadar + trainable history context | 0.445443 | 0.693170 | 0.450203 | 0.671977 | 0.756319 | 0.849155 | 0.627997 | 0.794651 |
| Matched native decoder continuation | 0.478992 | 0.704771 | 0.470963 | 0.661983 | 0.752374 | 0.846517 | 0.621046 | 0.797047 |

Axes: PSM [23115,87744); SWaT [608634,944856).

## Paired differences

| Comparison | Dataset | Stage/panel | AP | ROC-AUC | VUS-PR | VUS-ROC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| history_minus_native | PSM | Z/public | +0.002653 | +0.001990 | -0.000533 | -0.002989 |
| history_minus_native | PSM | Z/QUERY | +0.003399 | +0.001981 | -0.001099 | -0.002375 |
| history_minus_native | PSM | U/QUERY | -0.033549 | -0.011602 | -0.020760 | +0.009994 |
| native_U_minus_Z | PSM | U-Z/QUERY | -0.012414 | -0.004336 | +0.013718 | +0.026330 |
| history_U_minus_Z | PSM | U-Z/QUERY | -0.049361 | -0.017918 | -0.005944 | +0.038700 |
| history_minus_native | SWaT | Z/public | +0.006124 | +0.004081 | +0.008276 | +0.007449 |
| history_minus_native | SWaT | Z/QUERY | +0.002701 | +0.004180 | +0.007586 | +0.005809 |
| history_minus_native | SWaT | U/QUERY | +0.003945 | +0.002639 | +0.006951 | -0.002396 |
| native_U_minus_Z | SWaT | U-Z/QUERY | -0.032517 | -0.004796 | +0.065367 | +0.100287 |
| history_U_minus_Z | SWaT | U-Z/QUERY | -0.031273 | -0.006338 | +0.064731 | +0.092082 |
