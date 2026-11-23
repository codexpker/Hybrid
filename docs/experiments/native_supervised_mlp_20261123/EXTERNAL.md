# External comparisons on their own common axes

Each panel fixes its own common points. Native causal rows and offline/target-trained external rows retain their original information permissions. These panels must not be subtracted from full-axis main results.

## PSM [512,87744), 87,232 points

| Method | AP | ROC_AUC | VUS_PR | VUS_ROC | Training and readout permission |
|---|---|---|---|---|---|
| MLP_full | 0.50667 | 0.72365 | 0.45480 | 0.60747 | frozen official encoder; real public source-label MLP; W100 endpoint; target FIT scaler |
| MLP_240 | 0.50667 | 0.72365 | 0.45480 | 0.60747 | frozen official encoder; real public source-label MLP; W100 endpoint; target FIT scaler |
| official_V | 0.53627 | 0.73120 | 0.48507 | 0.66546 | native causal W100; archived source permissions |
| official_E | 0.38751 | 0.56764 | 0.39864 | 0.57913 | native causal W100; archived source permissions |
| copy_V | 0.54007 | 0.73203 | 0.49301 | 0.66751 | native causal W100; archived source permissions |
| Vtrain_V | 0.57763 | 0.76547 | 0.51410 | 0.67959 | native causal W100; archived source permissions |
| Etrain_E | 0.52266 | 0.67362 | 0.50163 | 0.66896 | native causal W100; archived source permissions |
| official_V_plus_E | 0.41957 | 0.60120 | 0.42372 | 0.60494 | native causal W100; archived source permissions |
| Vtrain_E | 0.55097 | 0.68326 | 0.51129 | 0.65859 | native causal W100; archived source permissions |
| Etrain_V | 0.45952 | 0.64025 | 0.44948 | 0.62223 | native causal W100; archived source permissions |
| LEFT native W192 | 0.57733 | 0.77558 | 0.57451 | 0.78509 | target [0,105984) FIT; SELECT [105984,132481); scaler [0,132481); score reads full W192 |
| CATCH W192 seed0 | 0.44684 | 0.65261 | 0.53644 | 0.70178 | target FIT-only scaler; FIT trains, SELECT selects; full W192 |
| CATCH W192 seed1 | 0.43112 | 0.64050 | 0.51913 | 0.68863 | target FIT-only scaler; FIT trains, SELECT selects; full W192 |
| CATCH W192 seed2 | 0.43365 | 0.64614 | 0.52704 | 0.69708 | target FIT-only scaler; FIT trains, SELECT selects; full W192 |
| PCA residual | 0.53463 | 0.79223 | 0.52910 | 0.76589 | target FIT-only scaler and PCA fitted on at most 50000 FIT points; score only uses x_t |

## PSM [512,87841), 87,329 points

| Method | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|
| official_V | 0.53620 | 0.73087 | 0.48545 | 0.66534 |
| official_E | 0.39001 | 0.56908 | 0.40123 | 0.58039 |
| copy_V | 0.53997 | 0.73166 | 0.49327 | 0.66726 |
| Vtrain_V | 0.57762 | 0.76528 | 0.51461 | 0.67955 |
| Etrain_E | 0.52384 | 0.67449 | 0.50319 | 0.66973 |
| official_V_plus_E | 0.42177 | 0.60249 | 0.42615 | 0.60610 |
| LEFT W192 endpoint | 0.56992 | 0.76206 | 0.57900 | 0.78039 |
| HistoryMean512 | 0.32262 | 0.53043 | 0.35175 | 0.57346 |
| MLP_full | 0.50621 | 0.72237 | 0.45470 | 0.60667 |
| MLP_240 | 0.50621 | 0.72237 | 0.45470 | 0.60667 |

## SMAP [512,N) per sensor-zero entity, 400,481 points

| Method | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|
| history_mean512 | 0.31813 | 0.63386 | 0.33667 | 0.68625 |
| persistence | 0.17663 | 0.49406 | 0.17391 | 0.51372 |
| RoCA_FULL_T | 0.20348 | 0.44316 | 0.19488 | 0.49970 |
| official_V | 0.37369 | 0.64316 | 0.41078 | 0.69159 |
| copy_V | 0.37577 | 0.63879 | 0.40770 | 0.68183 |
| Vtrain_V | 0.38709 | 0.62611 | 0.43472 | 0.67795 |
| official_E | 0.42785 | 0.67846 | 0.47061 | 0.73671 |
| official_V_plus_E | 0.41737 | 0.70516 | 0.45823 | 0.75771 |
| Etrain_E | 0.39938 | 0.69729 | 0.43917 | 0.75230 |
| MLP_full | 0.29254 | 0.53734 | 0.31727 | 0.58834 |
| MLP_240 | 0.29254 | 0.53734 | 0.31727 | 0.58834 |

## SMD [99,192 floor(N/192)) per machine, 703,212 points

| Method | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|
| Etrain_E | 0.42896 | 0.81391 | 0.44245 | 0.86086 |
| LEFT | 0.49807 | 0.87853 | 0.54180 | 0.91851 |
| Vtrain_V | 0.44266 | 0.82731 | 0.43472 | 0.86130 |
| copy_V | 0.45079 | 0.82049 | 0.47729 | 0.87535 |
| official_E | 0.42912 | 0.80530 | 0.43357 | 0.84872 |
| official_V | 0.44479 | 0.81320 | 0.46718 | 0.86997 |
| official_V_plus_E | 0.43882 | 0.81205 | 0.44273 | 0.85519 |
| MLP_full | 0.37710 | 0.75929 | 0.31949 | 0.76793 |
| MLP_240 | 0.37710 | 0.75929 | 0.31949 | 0.76793 |

LEFT is target-normal trained. PSM native W192 LEFT and CATCH use the full offline window. The PSM endpoint LEFT row uses its project causal endpoint connection. SMD LEFT is offline whole-window; its public normal training permissions do not provide the project independent CAL. SMAP RoCA and simple baselines retain their archived permissions. MSL compatible entity-resolved external scores are missing.
