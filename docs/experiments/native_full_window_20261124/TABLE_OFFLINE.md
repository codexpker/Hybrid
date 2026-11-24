# Table 1. Fixed-weight complete-window offline comparison

All methods in this table use identical scored points within each entity: the intersection of W100 complete-window outputs, archived causal endpoints and native LEFT coverage. PSM begins at99; SMD and NASA also begin at99 in each independent record. The last point is min(100 floor(N/100), W_LEFT floor(N/W_LEFT))−1. LEFT W192 applies to PSM, SMD and SMAP; MSL uses W96. Every TimeRadar full-window score is available at the containing W100 endpoint. Complete-window output is a project readout of fixed checkpoints. The three-seed mean and sample SD concern project source training at a common official initialization (n=3, ddof=1). Single frozen/E/LEFT models have no invented SD.

| Domain | Method | Project seeds | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|---|
| PSM | Official TimeRadar V | 1 | 0.44824 | 0.67842 | 0.45090 | 0.68182 |
| PSM | Official E | 1 | 0.43590 | 0.60779 | 0.43846 | 0.61526 |
| PSM | Official fixed V+E | 1 | 0.44836 | 0.61891 | 0.44821 | 0.62409 |
| PSM | Reconstruction continuation / V | 3 | 0.45283 ± 0.00291 | 0.68729 ± 0.00337 | 0.44955 ± 0.00120 | 0.67675 ± 0.00157 |
| PSM | Source-supervised native V | 3 | 0.44922 ± 0.00150 | 0.69227 ± 0.00085 | 0.45659 ± 0.00067 | 0.68701 ± 0.00057 |
| PSM | Source-supervised E / E | 1 | 0.38952 | 0.57682 | 0.40739 | 0.60085 |
| PSM | LEFT | 1 | 0.57690 | 0.77618 | 0.57413 | 0.78579 |
| SMD | Official TimeRadar V | 1 | 0.42716 | 0.82717 | 0.45194 | 0.88232 |
| SMD | Official E | 1 | 0.42720 | 0.81995 | 0.45557 | 0.86840 |
| SMD | Official fixed V+E | 1 | 0.43725 | 0.82679 | 0.46495 | 0.87375 |
| SMD | Reconstruction continuation / V | 3 | 0.44041 ± 0.00487 | 0.83733 ± 0.00368 | 0.46505 ± 0.00622 | 0.88696 ± 0.00200 |
| SMD | Source-supervised native V | 3 | 0.43614 ± 0.00139 | 0.84836 ± 0.00068 | 0.46682 ± 0.00071 | 0.89499 ± 0.00049 |
| SMD | Source-supervised E / E | 1 | 0.42158 | 0.83382 | 0.47466 | 0.88102 |
| SMD | LEFT | 1 | 0.49809 | 0.87855 | 0.54184 | 0.91853 |
| MSL | Official TimeRadar V | 1 | 0.34850 | 0.66935 | 0.38184 | 0.71798 |
| MSL | Official E | 1 | 0.39988 | 0.71093 | 0.43881 | 0.76245 |
| MSL | Official fixed V+E | 1 | 0.40407 | 0.71697 | 0.44381 | 0.76558 |
| MSL | Reconstruction continuation / V | 3 | 0.36811 ± 0.00294 | 0.67549 ± 0.00174 | 0.40369 ± 0.00295 | 0.72209 ± 0.00163 |
| MSL | Source-supervised native V | 3 | 0.35562 ± 0.00159 | 0.68606 ± 0.00074 | 0.39242 ± 0.00114 | 0.73144 ± 0.00022 |
| MSL | Source-supervised E / E | 1 | 0.39125 | 0.72008 | 0.43678 | 0.77492 |
| MSL | LEFT | 1 | 0.50834 | 0.76334 | 0.55308 | 0.79416 |
| SMAP | Official TimeRadar V | 1 | 0.30165 | 0.66240 | 0.29931 | 0.70426 |
| SMAP | Official E | 1 | 0.38124 | 0.71881 | 0.38102 | 0.76553 |
| SMAP | Official fixed V+E | 1 | 0.38662 | 0.72699 | 0.38210 | 0.77097 |
| SMAP | Reconstruction continuation / V | 3 | 0.32903 ± 0.01050 | 0.67517 ± 0.00245 | 0.32412 ± 0.00975 | 0.71470 ± 0.00187 |
| SMAP | Source-supervised native V | 3 | 0.33422 ± 0.00185 | 0.67061 ± 0.00081 | 0.33332 ± 0.00294 | 0.71166 ± 0.00071 |
| SMAP | Source-supervised E / E | 1 | 0.38984 | 0.71764 | 0.39124 | 0.76577 |
| SMAP | LEFT | 1 | 0.49315 | 0.76287 | 0.52172 | 0.78494 |

Official TimeRadar uses no project parameter updates. Reconstruction continuation is selected by real source-event validation but receives only nominal reconstruction gradients. The native-V/E supervised decoders use nominal source reconstruction plus real SWaT source pairs. LEFT uses the full target-normal records with its archived scaler/validation permissions and offline context. Current TimeRadar input scalers and mask identities remain fixed. NASA uses sensor zero only. SMD normal training records were among the project sources. MLP remains in its prior causal panel; no window logit was broadcast. No CAL, thresholds or online latency are evaluated in this readout experiment.

[Seed-specific macro values](macro.csv) · [complete per-entity rankings](entity_metrics.csv) · [same-axis repetitions](repeats_common.csv).
