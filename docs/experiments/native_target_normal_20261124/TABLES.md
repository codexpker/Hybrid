# Target-normal training, seed2025

| Domain | Complete method | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| PSM | Official initialization, no target updates | 0.426134 | 0.663406 | 0.434457 | 0.670174 |
| PSM | Source reconstruction initialization, no target updates | 0.434304 | 0.680116 | 0.436703 | 0.669819 |
| PSM | All-position source-supervised initialization, no target updates | 0.438880 | 0.698536 | 0.440585 | 0.672816 |
| PSM | Official initialization + target-normal training | 0.423674 | 0.668193 | 0.428627 | 0.651129 |
| PSM | Source reconstruction initialization + target-normal training | 0.419972 | 0.665718 | 0.427917 | 0.652606 |
| PSM | All-position source supervision + target-normal training | 0.421540 | 0.668633 | 0.429409 | 0.654356 |
| PSM | Complete LEFT | 0.576895 | 0.776184 | 0.574128 | 0.785786 |
| PSM | CATCH W192 seed0 | 0.446458 | 0.653765 | 0.536065 | 0.702828 |
| PSM | CATCH W192 seed1 | 0.430683 | 0.641721 | 0.518706 | 0.689799 |
| PSM | CATCH W192 seed2 | 0.433369 | 0.647474 | 0.526733 | 0.698310 |
| PSM | PCA residual | 0.534437 | 0.793117 | 0.528731 | 0.766991 |
| MSL | Official initialization, no target updates | 0.255976 | 0.576677 | 0.287096 | 0.653113 |
| MSL | Source reconstruction initialization, no target updates | 0.307240 | 0.604195 | 0.352391 | 0.671524 |
| MSL | All-position source-supervised initialization, no target updates | 0.282944 | 0.603217 | 0.336841 | 0.671066 |
| MSL | Official initialization + target-normal training | 0.361896 | 0.679476 | 0.395182 | 0.725680 |
| MSL | Source reconstruction initialization + target-normal training | 0.367920 | 0.684501 | 0.403348 | 0.727514 |
| MSL | All-position source supervision + target-normal training | 0.371977 | 0.688752 | 0.407930 | 0.730448 |
| MSL | Complete LEFT | 0.508343 | 0.763344 | 0.553084 | 0.794160 |

All six rows per task use the new shared target-normal preprocessing. Each target-trained model uses only normal target windows and normal reconstruction-MSE checkpoint selection, including epoch0. The three initializations retain their different official/source-label histories. This is a single project-training seed, not a three-seed result. LEFT and CATCH retain native offline windows and their own training/scaler permissions. Equal timestamps do not imply equal context or training permissions. No independent-CAL operating metrics are asserted.

## Initialization and training effects

| Domain | Contrast | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| PSM | source-supervised minus reconstruction initialization after target training | +0.001568 | +0.002915 | +0.001492 | +0.001751 |
| PSM | source-supervised minus official initialization after target training | -0.002133 | +0.000440 | +0.000782 | +0.003228 |
| PSM | source-supervised minus reconstruction initialization before target training | +0.004576 | +0.018420 | +0.003882 | +0.002997 |
| PSM | source-supervised minus official initialization before target training | +0.012747 | +0.035130 | +0.006127 | +0.002642 |
| PSM | target training effect, official initialization | -0.002460 | +0.004787 | -0.005830 | -0.019045 |
| PSM | target training effect, source reconstruction initialization | -0.014332 | -0.014398 | -0.008786 | -0.017214 |
| PSM | target training effect, source-supervised initialization | -0.017340 | -0.029903 | -0.011176 | -0.018459 |
| PSM | complete source-supervised target-trained detector minus LEFT | -0.155355 | -0.107551 | -0.144719 | -0.131429 |
| MSL | source-supervised minus reconstruction initialization after target training | +0.004057 | +0.004250 | +0.004582 | +0.002935 |
| MSL | source-supervised minus official initialization after target training | +0.010081 | +0.009275 | +0.012748 | +0.004768 |
| MSL | source-supervised minus reconstruction initialization before target training | -0.024296 | -0.000978 | -0.015550 | -0.000458 |
| MSL | source-supervised minus official initialization before target training | +0.026968 | +0.026539 | +0.049745 | +0.017953 |
| MSL | target training effect, official initialization | +0.105920 | +0.102799 | +0.108086 | +0.072567 |
| MSL | target training effect, source reconstruction initialization | +0.060680 | +0.080307 | +0.050957 | +0.055989 |
| MSL | target training effect, source-supervised initialization | +0.089033 | +0.085535 | +0.071089 | +0.059382 |
| MSL | complete source-supervised target-trained detector minus LEFT | -0.136366 | -0.074593 | -0.145154 | -0.063712 |

Complete native axes and common axes remain separate in entity_metrics.csv and macro.csv. All27 MSL entities are evaluated separately before equal macro aggregation. Entity wins/ties/losses and median differences are in paired_distribution.csv; they are not independent training repetitions.
