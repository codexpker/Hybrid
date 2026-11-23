# Objective and readout comparison on complete causal axes

V/V and E/E change the supervised decoder objective and deployed evidence; cross-readouts are predetermined descriptive controls. The MLP changes objective, updated component and inference together. Source labels and original FIT/SELECT windows are fixed, but the selected exposure and two-rate development budget differ. All rows retain their original full causal axes. This panel is separate from the LEFT intersections.

## AP

| Training / readout | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| copy_V | 0.53903 | 0.45021 | 0.34152 | 0.36830 |
| Vtrain_V | 0.57717 | 0.44206 | 0.35412 | 0.38272 |
| Etrain_E | 0.52350 | 0.42858 | 0.36975 | 0.39527 |
| Vtrain_E | 0.55301 | 0.39016 | 0.37085 | 0.42851 |
| Etrain_V | 0.45800 | 0.44134 | 0.32687 | 0.35378 |
| MLP_full | 0.50494 | 0.37652 | 0.28168 | 0.28896 |

## ROC_AUC

| Training / readout | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| copy_V | 0.73152 | 0.82025 | 0.63225 | 0.63893 |
| Vtrain_V | 0.76580 | 0.82705 | 0.66573 | 0.62680 |
| Etrain_E | 0.67552 | 0.81369 | 0.68323 | 0.69797 |
| Vtrain_E | 0.68619 | 0.81022 | 0.67817 | 0.70164 |
| Etrain_V | 0.63872 | 0.83302 | 0.61482 | 0.60904 |
| MLP_full | 0.72168 | 0.75892 | 0.55090 | 0.53892 |

## VUS_PR

| Training / readout | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| copy_V | 0.49180 | 0.47609 | 0.41388 | 0.39807 |
| Vtrain_V | 0.51378 | 0.43352 | 0.42298 | 0.42949 |
| Etrain_E | 0.50290 | 0.44149 | 0.44146 | 0.43453 |
| Vtrain_E | 0.51334 | 0.39734 | 0.43768 | 0.46907 |
| Etrain_V | 0.44784 | 0.49259 | 0.39115 | 0.39606 |
| MLP_full | 0.45263 | 0.31839 | 0.31698 | 0.31272 |

## VUS_ROC

| Training / readout | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| copy_V | 0.66665 | 0.87510 | 0.70468 | 0.67898 |
| Vtrain_V | 0.67986 | 0.86098 | 0.73565 | 0.67618 |
| Etrain_E | 0.67104 | 0.86057 | 0.74956 | 0.75068 |
| Vtrain_E | 0.66157 | 0.85187 | 0.74420 | 0.75565 |
| Etrain_V | 0.62058 | 0.88260 | 0.68144 | 0.65864 |
| MLP_full | 0.60520 | 0.76763 | 0.62785 | 0.59110 |
