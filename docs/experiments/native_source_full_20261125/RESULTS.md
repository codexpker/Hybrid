# Native parameter scope and source-supervised deployment statistics

All cells give AP / ROC-AUC / VUS-PR / VUS-ROC. SMD and NASA are equal-entity means. The new models use seed2025. Different methods retain their respective training and context permissions.

## Complete method comparison on the external common axis

| Method | PSM | SMD28 | MSL27 | SMAP53 |
|---|---:|---:|---:|---:|
| official_V | 0.5353 / 0.7310 / 0.4836 / 0.6648 | 0.4448 / 0.8132 / 0.4672 / 0.8700 | 0.3426 / 0.6327 / 0.4147 / 0.7048 | 0.3656 / 0.6427 / 0.4027 / 0.6902 |
| official_E | 0.3873 / 0.5697 / 0.3985 / 0.5814 | 0.4291 / 0.8053 / 0.4336 / 0.8487 | 0.3904 / 0.6932 / 0.4701 / 0.7671 | 0.4254 / 0.6805 / 0.4686 / 0.7386 |
| official_V_plus_E | 0.4195 / 0.6034 / 0.4237 / 0.6074 | 0.4388 / 0.8121 / 0.4428 / 0.8552 | 0.3999 / 0.7037 / 0.4805 / 0.7739 | 0.4148 / 0.7080 / 0.4558 / 0.7594 |
| Source reconstruction continuation (seed 2025) | 0.5391 / 0.7319 / 0.4915 / 0.6669 | 0.4508 / 0.8205 / 0.4773 / 0.8754 | 0.3498 / 0.6360 / 0.4235 / 0.7078 | 0.3684 / 0.6398 / 0.3982 / 0.6806 |
| Source-supervised endpoint V (seed 2025) | 0.5772 / 0.7660 / 0.5133 / 0.6799 | 0.4427 / 0.8273 / 0.4348 / 0.8613 | 0.3679 / 0.6712 / 0.4377 / 0.7410 | 0.3825 / 0.6274 / 0.4305 / 0.6782 |
| Source reconstruction continuation (seed 2026) | 0.5366 / 0.7307 / 0.4844 / 0.6642 | 0.4467 / 0.8145 / 0.4696 / 0.8710 | 0.3474 / 0.6348 / 0.4207 / 0.7058 | 0.3641 / 0.6315 / 0.3998 / 0.6773 |
| Source-supervised endpoint V (seed 2026) | 0.5680 / 0.7608 / 0.5035 / 0.6722 | 0.4404 / 0.8253 / 0.4311 / 0.8598 | 0.3642 / 0.6664 / 0.4325 / 0.7349 | 0.3805 / 0.6310 / 0.4265 / 0.6805 |
| Source reconstruction continuation (seed 2027) | 0.5408 / 0.7326 / 0.4945 / 0.6682 | 0.4515 / 0.8212 / 0.4777 / 0.8760 | 0.3511 / 0.6380 / 0.4244 / 0.7090 | 0.3664 / 0.6364 / 0.3950 / 0.6792 |
| Source-supervised endpoint V (seed 2027) | 0.5741 / 0.7652 / 0.5083 / 0.6750 | 0.4376 / 0.8256 / 0.4242 / 0.8584 | 0.3657 / 0.6657 / 0.4332 / 0.7310 | 0.3849 / 0.6368 / 0.4331 / 0.6870 |
| Complete LEFT | 0.5769 / 0.7762 / 0.5741 / 0.7858 | 0.4981 / 0.8785 / 0.5418 / 0.9185 | 0.5083 / 0.7633 / 0.5531 / 0.7942 | 0.4932 / 0.7629 / 0.5217 / 0.7849 |
| DADA official V | 0.3957 / 0.6050 / 0.4395 / 0.6520 | 0.4146 / 0.8514 / 0.4747 / 0.8997 | 0.3957 / 0.7336 / 0.4298 / 0.7747 | 0.4288 / 0.6459 / 0.4459 / 0.6826 |
| TimeRCD official weights, FIT-only adapter | 0.3019 / 0.4762 / 0.3485 / 0.5320 | 0.4729 / 0.8594 / 0.5151 / 0.9098 | 0.4827 / 0.7631 / 0.5298 / 0.8065 | 0.4427 / 0.6749 / 0.4280 / 0.7038 |
| CATCH W192 seed0 | 0.4465 / 0.6538 / 0.5361 / 0.7028 | NA | NA | NA |
| CATCH W192 seed1 | 0.4307 / 0.6417 / 0.5187 / 0.6898 | NA | NA | NA |
| CATCH W192 seed2 | 0.4334 / 0.6475 / 0.5267 / 0.6983 | NA | NA | NA |
| PCA residual | 0.5344 / 0.7931 / 0.5287 / 0.7670 | NA | NA | NA |
| Full normal reconstruction / V (seed 2025) | 0.5373 / 0.7343 / 0.5075 / 0.6809 | 0.4602 / 0.8339 / 0.4866 / 0.8843 | 0.3449 / 0.6379 / 0.4094 / 0.7031 | 0.3533 / 0.6150 / 0.3821 / 0.6597 |
| Full V supervision / V (seed 2025) | 0.5419 / 0.7878 / 0.5063 / 0.6974 | 0.3317 / 0.7682 / 0.3743 / 0.8302 | 0.3595 / 0.6902 / 0.4320 / 0.7522 | 0.3611 / 0.6257 / 0.3987 / 0.6788 |
| Full E supervision / E (seed 2025) | 0.5176 / 0.6721 / 0.4972 / 0.6685 | 0.4205 / 0.8101 / 0.4312 / 0.8566 | 0.3858 / 0.6939 / 0.4663 / 0.7668 | 0.4099 / 0.7094 / 0.4453 / 0.7578 |
| Decoder E supervision / E (seed 2025) | 0.5223 / 0.6747 / 0.5013 / 0.6702 | 0.4290 / 0.8139 / 0.4425 / 0.8609 | 0.3812 / 0.6853 / 0.4526 / 0.7516 | 0.3963 / 0.7003 / 0.4356 / 0.7536 |

## Original source contract: scope and criterion on the complete causal axis

| Method | PSM | SMD28 | MSL27 | SMAP53 |
|---|---:|---:|---:|---:|
| Source reconstruction continuation (seed 2025) | 0.5390 / 0.7315 / 0.4918 / 0.6667 | 0.4502 / 0.8202 / 0.4761 / 0.8751 | 0.3415 / 0.6322 / 0.4139 / 0.7047 | 0.3683 / 0.6389 / 0.3981 / 0.6790 |
| Full normal reconstruction / V (seed 2025) | 0.5368 / 0.7333 / 0.5073 / 0.6802 | 0.4596 / 0.8337 / 0.4854 / 0.8840 | 0.3360 / 0.6336 / 0.4001 / 0.6994 | 0.3533 / 0.6145 / 0.3821 / 0.6584 |
| Source-supervised endpoint V (seed 2025) | 0.5772 / 0.7658 / 0.5138 / 0.6799 | 0.4421 / 0.8271 / 0.4335 / 0.8610 | 0.3541 / 0.6657 / 0.4230 / 0.7356 | 0.3827 / 0.6268 / 0.4295 / 0.6762 |
| Full V supervision / V (seed 2025) | 0.5438 / 0.7884 / 0.5078 / 0.6976 | 0.3310 / 0.7679 / 0.3732 / 0.8298 | 0.3499 / 0.6848 / 0.4207 / 0.7483 | 0.3620 / 0.6257 / 0.3999 / 0.6784 |
| Decoder E supervision / E (seed 2025) | 0.5235 / 0.6755 / 0.5029 / 0.6710 | 0.4286 / 0.8137 / 0.4415 / 0.8606 | 0.3697 / 0.6832 / 0.4415 / 0.7496 | 0.3953 / 0.6980 / 0.4345 / 0.7507 |
| Full E supervision / E (seed 2025) | 0.5186 / 0.6729 / 0.4987 / 0.6693 | 0.4201 / 0.8098 / 0.4301 / 0.8563 | 0.3768 / 0.6917 / 0.4553 / 0.7647 | 0.4093 / 0.7072 / 0.4448 / 0.7553 |

## Paired differences on the complete causal axis

| Contrast | PSM | SMD28 | MSL27 | SMAP53 |
|---|---:|---:|---:|---:|
| Full normal reconstruction / V minus Source reconstruction continuation | -0.0022 / +0.0018 / +0.0155 / +0.0136 | +0.0094 / +0.0134 / +0.0093 / +0.0089 | -0.0055 / +0.0013 / -0.0138 / -0.0052 | -0.0150 / -0.0244 / -0.0160 / -0.0206 |
| Full V supervision / V minus Source-supervised endpoint V | -0.0333 / +0.0226 / -0.0060 / +0.0177 | -0.1110 / -0.0592 / -0.0604 / -0.0312 | -0.0043 / +0.0191 / -0.0022 / +0.0126 | -0.0207 / -0.0011 / -0.0296 / +0.0022 |
| Full E supervision / E minus Decoder E supervision / E | -0.0049 / -0.0026 / -0.0042 / -0.0018 | -0.0085 / -0.0039 / -0.0114 / -0.0043 | +0.0071 / +0.0084 / +0.0139 / +0.0152 | +0.0140 / +0.0093 / +0.0103 / +0.0047 |
| Full V supervision / V minus Full normal reconstruction / V | +0.0070 / +0.0551 / +0.0005 / +0.0174 | -0.1285 / -0.0658 / -0.1123 / -0.0542 | +0.0139 / +0.0512 / +0.0206 / +0.0488 | +0.0087 / +0.0112 / +0.0178 / +0.0200 |
| Full V supervision / V minus Full E supervision / E | +0.0252 / +0.1155 / +0.0091 / +0.0283 | -0.0890 / -0.0419 / -0.0570 / -0.0265 | -0.0269 / -0.0069 / -0.0346 / -0.0164 | -0.0473 / -0.0815 / -0.0449 / -0.0770 |

All new checkpoints are selected by their original score-specific source-event risk, including step0. The reconstruction control also uses source labels for SELECT. PSM is an exposed development task. SMD normal training records were source-visible. NASA uses only sensor zero. SWaT's source-labeled intervals are excluded from target benchmarks.

Operating metrics use the preserved separate PSM suffix, with all512 CAL points attack-labeled. NASA uses small CAL panels and q99/higher equals their sample maxima. SMD calibration is nominally normal. Strict and confirmed-three streams, one-to-one matching, any-hit and long-event coverage are exported without changing these permissions.

[Full-precision methods](methods.csv) · [Entity metrics](entity_metrics.csv) · [Pairs](entity_pairs.csv) · [Median and wins/ties/losses](paired_distribution.csv) · [Operations](operating.csv) · [Costs](EVALUATION.json).

## Interpretation and actual cost

Full-model V supervision lowered AP relative to decoder-only V in all four domains. PSM and MSL improved ROC-AUC and VUS-ROC while losing AP and VUS-PR. SMD declined on all four metrics, with AP wins in five and losses in twenty-three machines. SMAP's AP mean fell despite27 positive and26 negative entity differences, with a near-zero median.

With full updates, V supervision exceeded reconstruction in mean AP on PSM, MSL and SMAP but lost12.85 AP percentage points on SMD. Its AP exceeded full E on PSM only. The full reconstruction control improved all four SMD metrics over its decoder counterpart. Full E improved all four measures on MSL and SMAP. These are task-dependent effects of the complete training and source-selection procedures.

The selected update numbers are40,200,120 for reconstruction,V,E. Nominal exposures at those checkpoints are1,280,6,400,3,840; labeled pair exposures are0,3,000,1,800. All trajectories executed240 updates. Normal reconstruction also used source-event SELECT. Independent initial pretraining, seed repetitions and target adaptation are outside this comparison.

Formal training and source-validation time was 0.338 A40-hours. Causal score generation, input checks and output writing totaled 4.828 A40-hours summed over GPUs. The three models each executed their own encoder and adaptive order. All109 objects retained both scores and timestamp footprints. CPU metric workers accumulated 1763.1 seconds. Focused preliminary gradient checks and source/target bookkeeping are outside the formal GPU totals.

PSM three-confirmation false-positive rate and recall for full V were25.61% and60.76%, versus22.71% and44.24% for full reconstruction. One-to-one matches were14 versus7 out of48, while both had17 any-hit events. On SMD these rates were20.14%/38.28% for full V and18.92%/47.56% for reconstruction; matched events were148 versus159 out of327. NASA long-event statistics include four MSL and ten SMAP long events. The full operating exports retain these differences and all ineligible events.

The fixed full-parameter extension does not strengthen the existing V candidate's cross-task AP competitiveness. The archived decoder-only three-seed candidate remains a separate result. No new partial-unfreezing, loss, learning-rate or adaptation search follows this comparison.
