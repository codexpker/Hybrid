# Fixed source-support comparison

Each cell gives AP / ROC-AUC / VUS-PR / VUS-ROC. Results for multi-entity datasets are equal-entity means. The three new methods use project seed 2025.

## Complete detectors on the common external axis

| Method | PSM | SMD28 | MSL27 | SMAP53 |
|---|---:|---:|---:|---:|
| Official TimeRadar V | 0.5353 / 0.7310 / 0.4836 / 0.6648 | 0.4448 / 0.8132 / 0.4672 / 0.8700 | 0.3426 / 0.6327 / 0.4147 / 0.7048 | 0.3656 / 0.6427 / 0.4027 / 0.6902 |
| Official TimeRadar E | 0.3873 / 0.5697 / 0.3985 / 0.5814 | 0.4291 / 0.8053 / 0.4336 / 0.8487 | 0.3904 / 0.6932 / 0.4701 / 0.7671 | 0.4254 / 0.6805 / 0.4686 / 0.7386 |
| Official weights, fixed V + E | 0.4195 / 0.6034 / 0.4237 / 0.6074 | 0.4388 / 0.8121 / 0.4428 / 0.8552 | 0.3999 / 0.7037 / 0.4805 / 0.7739 | 0.4148 / 0.7080 / 0.4558 / 0.7594 |
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
| SWaT event supervision, matched V | 0.5812 / 0.7683 / 0.5152 / 0.6799 | 0.4430 / 0.8286 / 0.4314 / 0.8608 | 0.3704 / 0.6713 / 0.4400 / 0.7392 | 0.3794 / 0.6172 / 0.4218 / 0.6673 |
| SWaT + TEP event supervision, V | 0.5637 / 0.7557 / 0.4841 / 0.6628 | 0.4370 / 0.8245 / 0.4197 / 0.8560 | 0.3687 / 0.6557 / 0.4349 / 0.7212 | 0.3811 / 0.6299 / 0.4273 / 0.6790 |
| SWaT + TEP event supervision, E | 0.5212 / 0.6712 / 0.4960 / 0.6641 | 0.4237 / 0.8040 / 0.4341 / 0.8509 | 0.3692 / 0.6819 / 0.4368 / 0.7463 | 0.4097 / 0.6971 / 0.4520 / 0.7501 |

All new scores become available at the window endpoint. Complete LEFT and DADA use full windows. TimeRCD FIT-only uses the whole input block for context. Normalization permissions, input hashes and timestamp intersections are fixed by the preceding external benchmark. PSM PCA uses the current observation. Its CATCH comparisons have target FIT training. The auxiliary historical-context and whole-QUERY normalization panels are preserved separately in methods.csv.

## Support and deployment statistic on the full causal axis

| Paired difference | PSM | SMD28 | MSL27 | SMAP53 |
|---|---:|---:|---:|---:|
| Expanded V minus SWaT event supervision, matched V | -0.0174 / -0.0125 / -0.0310 / -0.0171 | -0.0062 / -0.0042 / -0.0121 / -0.0048 | +0.0001 / -0.0141 / -0.0041 / -0.0164 | +0.0021 / +0.0124 / +0.0059 / +0.0111 |
| Expanded V minus SWaT + TEP event supervision, E | +0.0416 / +0.0837 / -0.0125 / -0.0020 | +0.0130 / +0.0204 / -0.0147 / +0.0050 | +0.0008 / -0.0282 / -0.0038 / -0.0272 | -0.0275 / -0.0659 / -0.0223 / -0.0698 |

Each difference was computed within entity on identical points before aggregation. The two V models differ in the source of positive and paired-normal FIT windows. Both use the same combined source-validation panel. The V/E contrast compares training, score-specific selection and deployment together.

[All panels and full precision](methods.csv) · [Entity scores](entity_metrics.csv) · [Within-entity differences](entity_pairs.csv) · [Median differences and win/tie/loss counts](paired_distribution.csv) · [Operating costs](operating.csv).

The development domain is PSM. SMD supplied nominal source records. NASA uses sensor zero per entity. SWaT supplied labeled source events. PSM operating metrics use the separate 64,728-point suffix and its 512 attack-labeled CAL points. SMD uses its nominal CAL. The small NASA CAL sets have q99/higher equal to the sample maximum.

For expanded V relative to matched V, full-axis AP win/tie/loss counts were 14/0/14 on SMD, 11/0/16 on MSL and 29/3/21 on SMAP. SMAP showed the clearest ranking gain, with a higher three-confirmation false-positive rate. Expanded E achieved higher SMAP AP than expanded V, while frozen official E still had higher SMAP AP than both. TEP-specific improvement of E cannot be estimated from these three arms.

Source training and validation used 0.325 A40-hours. Per-entity causal readout used 3.585 A40-hours summed across lanes, including normal CAL, input checks and score writing. Sharing the frozen encoder and reusing byte-identical windows required 1,164,454 distinct window computations across entities for the three decoders. There were 1,286,816 QUERY endpoints and 17,060 CAL endpoints. CPU metric workers accumulated 1278.3 seconds; final aggregation elapsed 51.4 seconds including waits for the last entity metrics. Data transfer and one-off engineering checks are outside the formal GPU times.

[Source file identities](INPUT.json) · [Sampling permissions and plan digest](PLAN.json) · [Realized exposures](EXPOSURES.json) · [Timing](EVALUATION.json) · [Numerical verification](RECHECK.json).
