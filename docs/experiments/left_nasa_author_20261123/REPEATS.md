# Project-training repetition and same-axis NASA uncertainty

Each SD is the sample SD across three project-training seeds sharing the official checkpoint (n=3, ddof=1). NASA entities and SMD machines are not extra training repetitions. LEFT has one seed and no SD.

## Original complete causal axes

| Domain | Training | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| PSM | copy | 0.53874 ± 0.00208 | 0.73136 ± 0.00094 | 0.49047 ± 0.00519 | 0.66623 ± 0.00197 |
| PSM | point_weighted | 0.57307 ± 0.00467 | 0.76382 ± 0.00275 | 0.50889 ± 0.00487 | 0.67571 ± 0.00386 |
| MSL | copy | 0.34139 ± 0.00138 | 0.63255 ± 0.00164 | 0.41301 ± 0.00162 | 0.70444 ± 0.00166 |
| MSL | point_weighted | 0.35267 ± 0.00162 | 0.66251 ± 0.00281 | 0.42019 ± 0.00249 | 0.73057 ± 0.00468 |
| SMAP | copy | 0.36623 ± 0.00213 | 0.63499 ± 0.00417 | 0.39735 ± 0.00220 | 0.67737 ± 0.00170 |
| SMAP | point_weighted | 0.38287 ± 0.00213 | 0.63104 ± 0.00462 | 0.42961 ± 0.00357 | 0.68028 ± 0.00497 |
| SMD | copy | 0.44909 ± 0.00258 | 0.81849 ± 0.00365 | 0.47381 ± 0.00443 | 0.87381 ± 0.00271 |
| SMD | point_weighted | 0.43963 ± 0.00254 | 0.82581 ± 0.00108 | 0.42868 ± 0.00545 | 0.85949 ± 0.00150 |

SMD complete-axis repetition is retained in [the original macro table](../native_event_smd28_20261123/native/ranking_macro.csv) and [paired-repeat summary](../native_event_smd28_20261123/native/pairs_summary.csv). These full axes differ from LEFT common axes.

## Recomputed NASA LEFT common axes

| Domain | Training | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| MSL | copy | 0.34757 ± 0.00183 | 0.63540 ± 0.00160 | 0.41767 ± 0.00181 | 0.70638 ± 0.00161 |
| MSL | point_weighted | 0.36176 ± 0.00174 | 0.66650 ± 0.00289 | 0.42622 ± 0.00258 | 0.73367 ± 0.00472 |
| SMAP | copy | 0.36578 ± 0.00215 | 0.63572 ± 0.00417 | 0.39726 ± 0.00244 | 0.67886 ± 0.00167 |
| SMAP | point_weighted | 0.38202 ± 0.00219 | 0.63151 ± 0.00474 | 0.42889 ± 0.00359 | 0.68131 ± 0.00490 |
