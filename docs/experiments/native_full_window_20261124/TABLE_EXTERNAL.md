# Compatible external cache panels

PSM uses [512,87744), jointly covered by W100, causal endpoints, LEFT/CATCH and HistoryMean512. SMD uses the main per-machine intersection [99,min(100 floor(N/100),192 floor(N/192))). These panels preserve the external models’ original training, normalization and context permissions. TimeRCD uses its already frozen raw class-logit difference, with A/B both retained. A normalizes its whole passed role, including query observations; B uses archived FIT-only statistics. Neither is a causal alarm detector. There is no model inference or new threshold.

## PSM

| Method | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|
| CATCH W192 seed0 | 0.44684 | 0.65261 | 0.53644 | 0.70178 |
| CATCH W192 seed1 | 0.43112 | 0.64050 | 0.51913 | 0.68863 |
| CATCH W192 seed2 | 0.43365 | 0.64614 | 0.52704 | 0.69708 |
| HistoryMean512 | 0.32104 | 0.53004 | 0.34987 | 0.57301 |
| LEFT native W192 | 0.57733 | 0.77558 | 0.57451 | 0.78509 |
| PCA residual | 0.53463 | 0.79223 | 0.52910 | 0.76589 |
| TimeRCD A_official_role logit | 0.35995 | 0.55216 | 0.41212 | 0.60495 |
| TimeRCD B_fit_only logit | 0.30301 | 0.47656 | 0.34969 | 0.53241 |
| causal_endpoint 2025_copy | 0.54007 | 0.73203 | 0.49301 | 0.66751 |
| causal_endpoint 2025_point_weighted | 0.57763 | 0.76547 | 0.51410 | 0.67959 |
| causal_endpoint 2026_copy | 0.53759 | 0.73091 | 0.48588 | 0.66483 |
| causal_endpoint 2026_point_weighted | 0.56852 | 0.76051 | 0.50436 | 0.67201 |
| causal_endpoint 2027_copy | 0.54168 | 0.73269 | 0.49598 | 0.66869 |
| causal_endpoint 2027_point_weighted | 0.57452 | 0.76469 | 0.50908 | 0.67461 |
| causal_endpoint Etrain_E | 0.52266 | 0.67362 | 0.50163 | 0.66896 |
| causal_endpoint official_E | 0.38751 | 0.56764 | 0.39864 | 0.57913 |
| causal_endpoint official_V | 0.53627 | 0.73120 | 0.48507 | 0.66546 |
| causal_endpoint official_V_plus_E | 0.41957 | 0.60120 | 0.42372 | 0.60494 |
| full_window 2025_copy | 0.45249 | 0.68783 | 0.44976 | 0.67574 |
| full_window 2025_point_weighted | 0.45059 | 0.69237 | 0.45806 | 0.68730 |
| full_window 2026_copy | 0.45232 | 0.68438 | 0.45157 | 0.67895 |
| full_window 2026_point_weighted | 0.44894 | 0.69227 | 0.45744 | 0.68743 |
| full_window 2027_copy | 0.45743 | 0.69115 | 0.45208 | 0.67768 |
| full_window 2027_point_weighted | 0.45190 | 0.69374 | 0.45876 | 0.68830 |
| full_window Etrain_E | 0.38973 | 0.57496 | 0.40759 | 0.59890 |
| full_window official_E | 0.43615 | 0.60622 | 0.43865 | 0.61344 |
| full_window official_V | 0.44940 | 0.67880 | 0.45232 | 0.68237 |
| full_window official_V_plus_E | 0.44863 | 0.61742 | 0.44847 | 0.62246 |

## SMD

| Method | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|
| TimeRCD A_official_role logit | 0.48696 | 0.83814 | 0.51960 | 0.88987 |
| TimeRCD B_fit_only logit | 0.47295 | 0.85942 | 0.51514 | 0.90975 |

For SMD, the main common-axis native/LEFT rows from Table1 are on these exact same points. The underlying [external entity rows](external_entity.csv) retain source score SHA and permission. Missing NASA TimeRCD/CATCH/PCA caches are NA; no inference was added to complete them.
