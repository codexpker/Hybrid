# Complete public method comparison

All entries are local measurements. PSM uses [512,87744), 87,232 points; SMD uses [99,192 floor(N/192)) within each machine; NASA uses [99,W floor(N/W)) within each entity, W96 for MSL and W192 for SMAP. Entity means are unweighted within NASA and SMD. These common points align outcomes while preserving distinct source supervision, target-normal training and offline context permissions. Each row below uses seed2025; separate repetition estimates follow.

## AP

| Method | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| Official TimeRadar V | 0.53627 | 0.44479 | 0.34078 | 0.36502 |
| Official TimeRadar E | 0.38751 | 0.42912 | 0.38602 | 0.42477 |
| Official fixed V+E | 0.41957 | 0.43882 | 0.39507 | 0.41422 |
| Reconstruction continuation / V | 0.54007 | 0.45079 | 0.34792 | 0.36788 |
| Source-supervised native V (candidate) | 0.57763 | 0.44266 | 0.36355 | 0.38185 |
| Source-supervised E / E | 0.52266 | 0.42896 | 0.37895 | 0.39566 |
| Supervised MLP | 0.50667 | 0.37710 | 0.29162 | 0.28740 |
| LEFT | 0.57733 | 0.49807 | 0.50800 | 0.49312 |

## ROC_AUC

| Method | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| Official TimeRadar V | 0.73120 | 0.81320 | 0.63182 | 0.64252 |
| Official TimeRadar E | 0.56764 | 0.80530 | 0.69213 | 0.68028 |
| Official fixed V+E | 0.60120 | 0.81205 | 0.70234 | 0.70776 |
| Reconstruction continuation / V | 0.73203 | 0.82049 | 0.63514 | 0.63963 |
| Source-supervised native V (candidate) | 0.76547 | 0.82731 | 0.66980 | 0.62720 |
| Source-supervised E / E | 0.67362 | 0.81391 | 0.68451 | 0.70006 |
| Supervised MLP | 0.72365 | 0.75929 | 0.55546 | 0.53829 |
| LEFT | 0.77558 | 0.87853 | 0.76252 | 0.76284 |

## VUS_PR

| Method | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| Official TimeRadar V | 0.48507 | 0.46718 | 0.40974 | 0.40221 |
| Official TimeRadar E | 0.39864 | 0.43357 | 0.46363 | 0.46807 |
| Official fixed V+E | 0.42372 | 0.44273 | 0.47360 | 0.45518 |
| Reconstruction continuation / V | 0.49301 | 0.47729 | 0.41830 | 0.39790 |
| Source-supervised native V (candidate) | 0.51410 | 0.43472 | 0.42916 | 0.42903 |
| Source-supervised E / E | 0.50163 | 0.44245 | 0.44674 | 0.43497 |
| Supervised MLP | 0.45480 | 0.31949 | 0.32752 | 0.31115 |
| LEFT | 0.57451 | 0.54180 | 0.55285 | 0.52170 |

## VUS_ROC

| Method | PSM | SMD28 | MSL27 | SMAP53 |
|---|---|---|---|---|
| Official TimeRadar V | 0.66546 | 0.86997 | 0.70357 | 0.69006 |
| Official TimeRadar E | 0.57913 | 0.84872 | 0.76590 | 0.73844 |
| Official fixed V+E | 0.60494 | 0.85519 | 0.77255 | 0.75922 |
| Reconstruction continuation / V | 0.66751 | 0.87535 | 0.70670 | 0.68044 |
| Source-supervised native V (candidate) | 0.67959 | 0.86130 | 0.73865 | 0.67737 |
| Source-supervised E / E | 0.66896 | 0.86086 | 0.75027 | 0.75341 |
| Supervised MLP | 0.60747 | 0.76793 | 0.63193 | 0.58941 |
| LEFT | 0.78509 | 0.91851 | 0.79337 | 0.78490 |

The official, reconstruction-continuation and source-supervised candidates use W100 endpoint observations. The project-trained heads and decoders use real SWaT source labels for selection; the supervised arms also use them in training. Official scores have no project updates. The MLP freezes the representation; the V/E arms update its native decoder. LEFT uses domain/entity normal target training and full-window offline scoring. The project candidate has no target-gradient updates in this panel. NASA has sensor zero only. LEFT independent-CAL operation is NA.
