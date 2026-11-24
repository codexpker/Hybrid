# Public common-axis comparison

| Task | Method | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| PSM | Official initialization, no target updates | 0.426134 | 0.663406 | 0.434457 | 0.670174 |
| PSM | Source reconstruction initialization, no target updates | 0.434304 | 0.680116 | 0.436703 | 0.669819 |
| PSM | All-position source-supervised initialization, no target updates | 0.438880 | 0.698536 | 0.440585 | 0.672816 |
| PSM | Official initialization + decoder-only target-normal training | 0.423674 | 0.668193 | 0.428627 | 0.651129 |
| PSM | Source reconstruction initialization + decoder-only target-normal training | 0.419972 | 0.665718 | 0.427917 | 0.652606 |
| PSM | All-position source-supervised initialization + decoder-only target-normal training | 0.421540 | 0.668633 | 0.429409 | 0.654356 |
| PSM | Complete LEFT | 0.576895 | 0.776184 | 0.574128 | 0.785786 |
| PSM | CATCH W192 seed0 | 0.446458 | 0.653765 | 0.536065 | 0.702828 |
| PSM | CATCH W192 seed1 | 0.430683 | 0.641721 | 0.518706 | 0.689799 |
| PSM | CATCH W192 seed2 | 0.433369 | 0.647474 | 0.526733 | 0.698310 |
| PSM | PCA residual | 0.534437 | 0.793117 | 0.528731 | 0.766991 |
| PSM | Official initialization + full-parameter target-normal training | 0.426145 | 0.670999 | 0.447756 | 0.674564 |
| PSM | Source reconstruction initialization + full-parameter target-normal training | 0.425982 | 0.671106 | 0.447632 | 0.674083 |
| PSM | All-position source-supervised initialization + full-parameter target-normal training | 0.425088 | 0.669928 | 0.445920 | 0.671730 |
| MSL | Official initialization, no target updates | 0.255976 | 0.576677 | 0.287096 | 0.653113 |
| MSL | Source reconstruction initialization, no target updates | 0.307240 | 0.604195 | 0.352391 | 0.671524 |
| MSL | All-position source-supervised initialization, no target updates | 0.282944 | 0.603217 | 0.336841 | 0.671066 |
| MSL | Official initialization + decoder-only target-normal training | 0.361896 | 0.679476 | 0.395182 | 0.725680 |
| MSL | Source reconstruction initialization + decoder-only target-normal training | 0.367920 | 0.684501 | 0.403348 | 0.727514 |
| MSL | All-position source-supervised initialization + decoder-only target-normal training | 0.371977 | 0.688752 | 0.407930 | 0.730448 |
| MSL | Complete LEFT | 0.508343 | 0.763344 | 0.553084 | 0.794160 |
| MSL | Official initialization + full-parameter target-normal training | 0.464447 | 0.714421 | 0.517487 | 0.750163 |
| MSL | Source reconstruction initialization + full-parameter target-normal training | 0.443015 | 0.715395 | 0.488308 | 0.748806 |
| MSL | All-position source-supervised initialization + full-parameter target-normal training | 0.431968 | 0.717039 | 0.477313 | 0.750918 |

# Training scope and initialization contrasts

| Task | Method | Reference | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|---|
| PSM | Official initialization + full-parameter target-normal training | Official initialization + decoder-only target-normal training | +0.002471 | +0.002806 | +0.019129 | +0.023435 |
| PSM | Source reconstruction initialization + full-parameter target-normal training | Source reconstruction initialization + decoder-only target-normal training | +0.006010 | +0.005388 | +0.019715 | +0.021477 |
| PSM | All-position source-supervised initialization + full-parameter target-normal training | All-position source-supervised initialization + decoder-only target-normal training | +0.003547 | +0.001295 | +0.016511 | +0.017373 |
| PSM | All-position source-supervised initialization + full-parameter target-normal training | Source reconstruction initialization + full-parameter target-normal training | -0.000894 | -0.001178 | -0.001712 | -0.002353 |
| PSM | All-position source-supervised initialization + full-parameter target-normal training | Official initialization + full-parameter target-normal training | -0.001057 | -0.001071 | -0.001836 | -0.002834 |
| PSM | Official initialization + full-parameter target-normal training | Complete LEFT | -0.150750 | -0.105185 | -0.126372 | -0.111222 |
| PSM | Official initialization + full-parameter target-normal training | PCA residual | -0.108292 | -0.122119 | -0.080975 | -0.092427 |
| PSM | Source reconstruction initialization + full-parameter target-normal training | Complete LEFT | -0.150913 | -0.105078 | -0.126497 | -0.111703 |
| PSM | Source reconstruction initialization + full-parameter target-normal training | PCA residual | -0.108456 | -0.122011 | -0.081100 | -0.092909 |
| PSM | All-position source-supervised initialization + full-parameter target-normal training | Complete LEFT | -0.151807 | -0.106256 | -0.128208 | -0.114056 |
| PSM | All-position source-supervised initialization + full-parameter target-normal training | PCA residual | -0.109350 | -0.123189 | -0.082811 | -0.095262 |
| PSM | Official initialization + decoder-only target-normal training | Complete LEFT | -0.153222 | -0.107991 | -0.145501 | -0.134657 |
| PSM | Official initialization + decoder-only target-normal training | PCA residual | -0.110764 | -0.124924 | -0.100104 | -0.115862 |
| PSM | Source reconstruction initialization + decoder-only target-normal training | Complete LEFT | -0.156923 | -0.110466 | -0.146211 | -0.133180 |
| PSM | Source reconstruction initialization + decoder-only target-normal training | PCA residual | -0.114465 | -0.127399 | -0.100814 | -0.114386 |
| PSM | All-position source-supervised initialization + decoder-only target-normal training | Complete LEFT | -0.155355 | -0.107551 | -0.144719 | -0.131429 |
| PSM | All-position source-supervised initialization + decoder-only target-normal training | PCA residual | -0.112897 | -0.124484 | -0.099322 | -0.112635 |
| MSL | Official initialization + full-parameter target-normal training | Official initialization + decoder-only target-normal training | +0.102551 | +0.034945 | +0.122305 | +0.024482 |
| MSL | Source reconstruction initialization + full-parameter target-normal training | Source reconstruction initialization + decoder-only target-normal training | +0.075095 | +0.030894 | +0.084961 | +0.021293 |
| MSL | All-position source-supervised initialization + full-parameter target-normal training | All-position source-supervised initialization + decoder-only target-normal training | +0.059991 | +0.028288 | +0.069384 | +0.020470 |
| MSL | All-position source-supervised initialization + full-parameter target-normal training | Source reconstruction initialization + full-parameter target-normal training | -0.011047 | +0.001645 | -0.010995 | +0.002112 |
| MSL | All-position source-supervised initialization + full-parameter target-normal training | Official initialization + full-parameter target-normal training | -0.032479 | +0.002619 | -0.040174 | +0.000755 |
| MSL | Official initialization + full-parameter target-normal training | Complete LEFT | -0.043896 | -0.048924 | -0.035597 | -0.043998 |
| MSL | Source reconstruction initialization + full-parameter target-normal training | Complete LEFT | -0.065328 | -0.047950 | -0.064775 | -0.045354 |
| MSL | All-position source-supervised initialization + full-parameter target-normal training | Complete LEFT | -0.076375 | -0.046305 | -0.075771 | -0.043243 |
| MSL | Official initialization + decoder-only target-normal training | Complete LEFT | -0.146447 | -0.083868 | -0.157902 | -0.068480 |
| MSL | Source reconstruction initialization + decoder-only target-normal training | Complete LEFT | -0.140423 | -0.078843 | -0.149736 | -0.066647 |
| MSL | All-position source-supervised initialization + decoder-only target-normal training | Complete LEFT | -0.136366 | -0.074593 | -0.145154 | -0.063712 |

Both tasks use project seed2025 and the previously archived common offline axes (87,645 PSM points and 69,495 MSL points). MSL averages entity metrics with equal weight over all 27 sensor-0 records. The three initializations and target-normal data, scaler, permutation plans and update counts are shared across the two trainable scopes. Zero-update rows inherit the same scaler. The source reconstruction initialization historically used public source-event labels for checkpoint selection. The all-position initialization additionally received a source pair gradient. Neither target training scope uses target anomaly labels. The frozen external scores preserve their original permissions: LEFT uses target-normal training with W192 on PSM and W96 on MSL, CATCH uses target FIT-only statistics and a full W192 context, and PCA uses target FIT-only statistics and the observation at the scored time. CATCH seeds remain separate. No compatible independent CAL operation metrics are asserted. The historical PSM CAL consists of attack labels, and the MSL normal scaler uses the original validation/CAL part of the normal arrays.
