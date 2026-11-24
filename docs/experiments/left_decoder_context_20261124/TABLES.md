### Complete public benchmark on common scored points

| Task | Complete method | AP | ROC_AUC | VUS_PR | VUS_ROC | Training and context permissions | Train / readout seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PSM | Native LEFT | 0.576639 | 0.776095 | 0.572611 | 0.785067 | target normal FIT / normal SELECT; prefix scaler; W192 offline | 977.81 / 0.79 |
| PSM | Observation-conditioned LEFT | 0.576648 | 0.776028 | 0.573074 | 0.784999 | target normal FIT / normal SELECT; prefix scaler; W192 offline | 941.15 / 0.91 |
| PSM | Prototype-conditioned LEFT | 0.577939 | 0.776990 | 0.574991 | 0.786316 | target normal FIT / normal SELECT; prefix scaler; W192 offline | 906.31 / 0.79 |
| PSM | CATCH W192 seed0 | 0.445990 | 0.653418 | 0.536671 | 0.703657 | target FIT-only scaler; FIT trains, SELECT selects; full W192 | NA |
| PSM | CATCH W192 seed1 | 0.430264 | 0.641399 | 0.519273 | 0.690691 | target FIT-only scaler; FIT trains, SELECT selects; full W192 | NA |
| PSM | CATCH W192 seed2 | 0.433044 | 0.647248 | 0.526447 | 0.698733 | target FIT-only scaler; FIT trains, SELECT selects; full W192 | NA |
| PSM | PCA residual | 0.534404 | 0.793321 | 0.528991 | 0.766686 | target FIT-only scaler and PCA fitted on at most 50000 FIT points; score only uses x_t | NA |
| PSM | TimeRadar official block0 seed2024 | 0.433265 | 0.670669 | 0.441259 | 0.678365 | native norm=0, B64, L1; prefix scaler includes SELECT; complete W100 and batch-shared order | NA |
| PSM | TimeRadar common FIT scaler seed2024 | 0.455371 | 0.683823 | 0.458071 | 0.689163 | FIT-only scaler, B64, norm=0, L1; complete W100 and batch-shared order | NA |
| SWAT | Native LEFT | 0.543866 | 0.820853 | 0.534601 | 0.809255 | target normal FIT / normal SELECT; prefix scaler; W192 offline | 3241.90 / 3.87 |
| SWAT | Observation-conditioned LEFT | 0.490931 | 0.791439 | 0.457082 | 0.730007 | target normal FIT / normal SELECT; prefix scaler; W192 offline | 3804.55 / 3.96 |
| SWAT | Prototype-conditioned LEFT | 0.140230 | 0.529687 | 0.142682 | 0.501934 | target normal FIT / normal SELECT; prefix scaler; W192 offline | 3968.55 / 3.93 |
| SWAT | TimeRadar official block0 | 0.639299 | 0.821970 | 0.514061 | 0.766831 | native norm=0, B128, L1; prefix scaler includes SELECT; complete W100 and batch-shared order | NA |
| SWAT | CATCH W192 | NA | NA | NA | NA | No compatible full-axis cache | NA |
| SWAT | PCA residual | NA | NA | NA | NA | No compatible full-axis cache | NA |

### Matched structural ablation

| Task | Contrast | AP | ROC_AUC | VUS_PR | VUS_ROC |
| --- | --- | --- | --- | --- | --- |
| PSM | Prototype-conditioned minus Native LEFT | +0.001301 | +0.000895 | +0.002381 | +0.001249 |
| PSM | Prototype-conditioned minus Observation-conditioned LEFT | +0.001291 | +0.000962 | +0.001917 | +0.001317 |
| SWAT | Prototype-conditioned minus Native LEFT | -0.403636 | -0.291166 | -0.391919 | -0.307321 |
| SWAT | Prototype-conditioned minus Observation-conditioned LEFT | -0.350700 | -0.261753 | -0.314399 | -0.228073 |

### Parameters and observed cost

| Task | Method | Registered parameters | Adam state parameters | Active modulation parameters | Selected epoch | Updates | Train seconds | Readout seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PSM | Native LEFT | 4519985 | 4486705 | 0 | 20 | 16540 | 977.814 | 0.794 |
| PSM | Observation-conditioned LEFT | 4519985 | 4519985 | 33280 | 15 | 14886 | 941.146 | 0.911 |
| PSM | Prototype-conditioned LEFT | 4519985 | 4519985 | 33280 | 15 | 14886 | 906.308 | 0.795 |
| SWAT | Native LEFT | 3742202 | 3708922 | 0 | 13 | 49488 | 3241.902 | 3.868 |
| SWAT | Observation-conditioned LEFT | 3742202 | 3742202 | 33280 | 15 | 55674 | 3804.551 | 3.957 |
| SWAT | Prototype-conditioned LEFT | 3742202 | 3742202 | 33280 | 16 | 58767 | 3968.552 | 3.926 |

PSM uses the test-relative axis [0, 87744). SWaT uses the absolute axis [495000, 944856). Both are offline whole-window comparisons. The matched LEFT arms share seed2025 and normal-data permissions. TimeRadar, CATCH and PCA retain the permissions recorded in their cached metadata. Historical costs are shown as NA when this table lacks a comparable measurement. Operation metrics are NA because a compatible independent calibration split has not been established on these full axes.
