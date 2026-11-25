# Fixed endpoint V and complete external TSAD methods

This round compares the original endpoint-supervised native V detector on the admitted PSM, SMD28, MSL27 and SMAP53 records. Project source SELECT fixed the three checkpoint states before the target comparison. Decoder-only source supervision used real public SWaT events and archived nominal SWaT and SMD windows. The source reconstruction controls use the same nominal pool and target inference with source-event SELECT. PSM is development-exposed, SMD is a held-out test of seen normal source machines, and both NASA sets use one continuous sensor per entity. SWaT's source-supervised run remains on the source side of this fold. No training or target threshold selection is added here.

DADA is imported from the released implementation and checkpoint at ea172389acb466a11a7caac2d18b5ead754aaa43. The weight SHA-256 is `66db3cc568d3df550bdb0aab7cf40c191acc1acd4bfee62b2b63af843369d850`. Each entity uses non-overlapping W100 inputs, ten complementary copies, the author's variance score and per-entity standardization on the full published normal training array. The author flags for internal normalization are PSM 1, SMD 0, MSL 0 and SMAP 1. Batch sizes are 32, 32, 128 and 32. The label-free window adapter keeps entity boundaries and starts the published random seed 2024 at each entity's QUERY. This defines the run's mask stream. The upstream release does not include its protected pretraining code. The source and checkpoint bytes stay outside this repository. [ICLR paper](https://openreview.net/forum?id=aKcd7ImG5e) · [Released implementation](https://github.com/iambowen/DADA/tree/ea172389acb466a11a7caac2d18b5ead754aaa43).

TimeRCD uses b031773e323bc0e6c9dcd8c9119b800ec4d70f8c and the published checkpoint revision 0880420070a524efe14d7e163b34dcca90d2be1b. Original sealed multi-weight scores for PSM and SMD28 are reused after raw input, axis, logit and availability checks. The multi SHA-256 is `a497ac910f24f8fb347d690a21b60d74bc04785ebc777880553c158dea4434d0`. The official API requires the uni weight for single-channel MSL and SMAP, with SHA-256 `be8906c6a62f16cedd8fe0fe48ef11c4363099633ed6ba8780af819337f7ffaa`. The model and head retain W5000 capped by series length, batch1, valid padding and the original abnormal-minus-normal class logit. B normalizes once with the archived target FIT. For MSL that is the first 60% of each normal array, and for SMAP it is the archived raw entity FIT. Zero standard deviations use the archived TimeRCD guard. A computes statistics from the whole QUERY role. The two weights are independent published variants. [ICML paper](https://proceedings.mlr.press/v306/lan26b.html) · [Checkpoint revision](https://huggingface.co/thu-sail-lab/Time-RCD/tree/0880420070a524efe14d7e163b34dcca90d2be1b/best_model).


Every row reports AP / ROC-AUC / VUS-PR / VUS-ROC. Metrics use labels only for evaluation. VUS uses the pinned LEFT implementation with windowSize 100 and thre 250. Each entity is evaluated independently before equal-weight domain aggregation. SD describes the three project seeds. The frozen external detectors are single fixed references. Candidate and continuation rows in the offline-common table are causal scores restricted to the same timestamps as the offline references.

The causal full test range is [99, N) per entity. The offline-common range starts at 99 and ends before the smaller of the complete DADA W100 and LEFT W96/W192 block endpoints. It is the same for every method listed within that domain panel. History Mean uses [512, 87744) on PSM and [512, N) per SMAP entity. The PSM PCA and History Mean rows use only their covered points. CATCH full W192 and LEFT retain future context within a window. TimeRCD B waits for the actual last input in its window. TimeRCD A waits for the end of QUERY to determine normalization. The historical operating panels retain the 64,728-point PSM suffix and its 512 all-attack CAL points, the NASA nominal normal small-sample CAL with a maximum-value q99/higher cutoff, and SMD's nominal-normal CAL. These thresholds are empirical and their false-alarm interpretation follows the archived protocols. The existing strict and three-confirmation scores, one-to-one event matches, any-hit and eligibility stay separate. New external operating values are NA.

| Dataset | Entities | Channels | Full causal points | Offline-common points | History-common points |
| --- | --- | --- | --- | --- | --- |
| PSM | 1 | 25 | 87,742 | 87,645 | 87,232 |
| SMD28 | 28 | 38 | 705,648 | 702,996 | NA |
| MSL27 | 27 | 1 | 71,056 | 69,495 | NA |
| SMAP53 | 53 | 1 | 422,370 | 417,265 | 400,481 |

## Causal full-axis methods

| Method | PSM | SMD28 | MSL27 | SMAP53 | Training and statistics | Score availability |
| --- | --- | --- | --- | --- | --- | --- |
| Frozen TimeRadar V (release L=1) | 0.5352 / 0.7306 / 0.4839 / 0.6646 | 0.4442 / 0.8130 / 0.4664 / 0.8697 | 0.3356 / 0.6287 / 0.4057 / 0.7015 | 0.3656 / 0.6419 / 0.4027 / 0.6887 | Official pretrained TimeRadar, no project updates, archived target FIT | causal W100 endpoint |
| Frozen TimeRadar E | 0.3898 / 0.5711 / 0.4012 / 0.5826 | 0.4287 / 0.8051 / 0.4326 / 0.8485 | 0.3780 / 0.6917 / 0.4597 / 0.7668 | 0.4250 / 0.6789 / 0.4679 / 0.7371 | Official pretrained TimeRadar, no project updates, archived target FIT | causal W100 endpoint |
| Frozen TimeRadar fixed V+E | 0.4217 / 0.6047 / 0.4261 / 0.6085 | 0.4384 / 0.8118 / 0.4415 / 0.8549 | 0.3866 / 0.7014 / 0.4695 / 0.7726 | 0.4143 / 0.7057 / 0.4551 / 0.7566 | Official pretrained TimeRadar, no project updates, archived target FIT | causal W100 endpoint |
| Source reconstruction continuation (2025) | 0.5390 / 0.7315 / 0.4918 / 0.6667 | 0.4502 / 0.8202 / 0.4761 / 0.8751 | 0.3415 / 0.6322 / 0.4139 / 0.7047 | 0.3683 / 0.6389 / 0.3981 / 0.6790 | SWaT real-event SELECT, nominal source reconstruction gradients, no target update, archived FIT | causal W100 endpoint |
| Source-supervised endpoint V (2025) | 0.5772 / 0.7658 / 0.5138 / 0.6799 | 0.4421 / 0.8271 / 0.4335 / 0.8610 | 0.3541 / 0.6657 / 0.4230 / 0.7356 | 0.3827 / 0.6268 / 0.4295 / 0.6762 | SWaT real-event pairs and SELECT, nominal SWaT/SMD reconstruction, no target update, archived FIT | causal W100 endpoint |
| Source reconstruction continuation (2026) | 0.5365 / 0.7304 / 0.4847 / 0.6641 | 0.4461 / 0.8143 / 0.4687 / 0.8707 | 0.3399 / 0.6311 / 0.4111 / 0.7027 | 0.3641 / 0.6306 / 0.3991 / 0.6756 | SWaT real-event SELECT, nominal source reconstruction gradients, no target update, archived FIT | causal W100 endpoint |
| Source-supervised endpoint V (2026) | 0.5680 / 0.7607 / 0.5040 / 0.6722 | 0.4398 / 0.8251 / 0.4297 / 0.8595 | 0.3509 / 0.6612 / 0.4182 / 0.7296 | 0.3808 / 0.6304 / 0.4261 / 0.6788 | SWaT real-event pairs and SELECT, nominal SWaT/SMD reconstruction, no target update, archived FIT | causal W100 endpoint |
| Source reconstruction continuation (2027) | 0.5407 / 0.7322 / 0.4949 / 0.6680 | 0.4509 / 0.8209 / 0.4766 / 0.8756 | 0.3427 / 0.6343 / 0.4140 / 0.7060 | 0.3663 / 0.6354 / 0.3949 / 0.6775 | SWaT real-event SELECT, nominal source reconstruction gradients, no target update, archived FIT | causal W100 endpoint |
| Source-supervised endpoint V (2027) | 0.5741 / 0.7650 / 0.5088 / 0.6750 | 0.4370 / 0.8253 / 0.4228 / 0.8580 | 0.3530 / 0.6606 / 0.4194 / 0.7264 | 0.3851 / 0.6360 / 0.4332 / 0.6858 | SWaT real-event pairs and SELECT, nominal SWaT/SMD reconstruction, no target update, archived FIT | causal W100 endpoint |
| Source-supervised endpoint V (three-seed mean ± sample SD) | 0.5731 ± 0.0047 / 0.7638 ± 0.0027 / 0.5089 ± 0.0049 / 0.6757 ± 0.0039 | 0.4396 ± 0.0025 / 0.8258 ± 0.0011 / 0.4287 ± 0.0054 / 0.8595 ± 0.0015 | 0.3527 ± 0.0016 / 0.6625 ± 0.0028 / 0.4202 ± 0.0025 / 0.7306 ± 0.0047 | 0.3829 ± 0.0021 / 0.6310 ± 0.0046 / 0.4296 ± 0.0036 / 0.6803 ± 0.0050 | SWaT real-event pairs and SELECT, nominal SWaT/SMD reconstruction, no target update, archived FIT | causal W100 endpoint |

## Offline-common comparison

| Method | PSM | SMD28 | MSL27 | SMAP53 | Training and statistics | Score availability |
| --- | --- | --- | --- | --- | --- | --- |
| Frozen TimeRadar V (release L=1) | 0.5353 / 0.7310 / 0.4836 / 0.6648 | 0.4448 / 0.8132 / 0.4672 / 0.8700 | 0.3426 / 0.6327 / 0.4147 / 0.7048 | 0.3656 / 0.6427 / 0.4027 / 0.6902 | Official pretrained TimeRadar, no project updates, archived target FIT | causal W100 endpoint |
| Frozen TimeRadar E | 0.3873 / 0.5697 / 0.3985 / 0.5814 | 0.4291 / 0.8053 / 0.4336 / 0.8487 | 0.3904 / 0.6932 / 0.4701 / 0.7671 | 0.4254 / 0.6805 / 0.4686 / 0.7386 | Official pretrained TimeRadar, no project updates, archived target FIT | causal W100 endpoint |
| Frozen TimeRadar fixed V+E | 0.4195 / 0.6034 / 0.4237 / 0.6074 | 0.4388 / 0.8121 / 0.4428 / 0.8552 | 0.3999 / 0.7037 / 0.4805 / 0.7739 | 0.4148 / 0.7080 / 0.4558 / 0.7594 | Official pretrained TimeRadar, no project updates, archived target FIT | causal W100 endpoint |
| Source reconstruction continuation (2025) | 0.5391 / 0.7319 / 0.4915 / 0.6669 | 0.4508 / 0.8205 / 0.4773 / 0.8754 | 0.3498 / 0.6360 / 0.4235 / 0.7078 | 0.3684 / 0.6398 / 0.3982 / 0.6806 | SWaT real-event SELECT, nominal source reconstruction gradients, no target update, archived FIT | causal W100 endpoint |
| Source-supervised endpoint V (2025) | 0.5772 / 0.7660 / 0.5133 / 0.6799 | 0.4427 / 0.8273 / 0.4348 / 0.8613 | 0.3679 / 0.6712 / 0.4377 / 0.7410 | 0.3825 / 0.6274 / 0.4305 / 0.6782 | SWaT real-event pairs and SELECT, nominal SWaT/SMD reconstruction, no target update, archived FIT | causal W100 endpoint |
| Source reconstruction continuation (2026) | 0.5366 / 0.7307 / 0.4844 / 0.6642 | 0.4467 / 0.8145 / 0.4696 / 0.8710 | 0.3474 / 0.6348 / 0.4207 / 0.7058 | 0.3641 / 0.6315 / 0.3998 / 0.6773 | SWaT real-event SELECT, nominal source reconstruction gradients, no target update, archived FIT | causal W100 endpoint |
| Source-supervised endpoint V (2026) | 0.5680 / 0.7608 / 0.5035 / 0.6722 | 0.4404 / 0.8253 / 0.4311 / 0.8598 | 0.3642 / 0.6664 / 0.4325 / 0.7349 | 0.3805 / 0.6310 / 0.4265 / 0.6805 | SWaT real-event pairs and SELECT, nominal SWaT/SMD reconstruction, no target update, archived FIT | causal W100 endpoint |
| Source reconstruction continuation (2027) | 0.5408 / 0.7326 / 0.4945 / 0.6682 | 0.4515 / 0.8212 / 0.4777 / 0.8760 | 0.3511 / 0.6380 / 0.4244 / 0.7090 | 0.3664 / 0.6364 / 0.3950 / 0.6792 | SWaT real-event SELECT, nominal source reconstruction gradients, no target update, archived FIT | causal W100 endpoint |
| Source-supervised endpoint V (2027) | 0.5741 / 0.7652 / 0.5083 / 0.6750 | 0.4376 / 0.8256 / 0.4242 / 0.8584 | 0.3657 / 0.6657 / 0.4332 / 0.7310 | 0.3849 / 0.6368 / 0.4331 / 0.6870 | SWaT real-event pairs and SELECT, nominal SWaT/SMD reconstruction, no target update, archived FIT | causal W100 endpoint |
| Complete LEFT | 0.5769 / 0.7762 / 0.5741 / 0.7858 | 0.4981 / 0.8785 / 0.5418 / 0.9185 | 0.5083 / 0.7633 / 0.5531 / 0.7942 | 0.4932 / 0.7629 / 0.5217 / 0.7849 | Target normal FIT and SELECT, scaler uses the full normal prefix, full-window offline | full W96 MSL or W192 elsewhere, window-end |
| DADA official V | 0.3957 / 0.6050 / 0.4395 / 0.6520 | 0.4146 / 0.8514 / 0.4747 / 0.8997 | 0.3957 / 0.7336 / 0.4298 / 0.7747 | 0.4288 / 0.6459 / 0.4459 / 0.6826 | Official multidomain pretraining with synthetic anomalies, full published normal target array for scaling | full W100, window-end |
| TimeRCD official weights, FIT-only adapter | 0.3019 / 0.4762 / 0.3485 / 0.5320 | 0.4729 / 0.8594 / 0.5151 / 0.9098 | 0.4827 / 0.7631 / 0.5298 / 0.8065 | 0.4427 / 0.6749 / 0.4280 / 0.7038 | Official synthetic contextual supervision, archived target FIT statistics only | full W5000 capped by length, actual window-end |
| CATCH W192 seed0 | 0.4465 / 0.6538 / 0.5361 / 0.7028 | NA | NA | NA | Target FIT-only scaling, FIT training and SELECT, full W192 | full W192, window-end |
| CATCH W192 seed1 | 0.4307 / 0.6417 / 0.5187 / 0.6898 | NA | NA | NA | Target FIT-only scaling, FIT training and SELECT, full W192 | full W192, window-end |
| CATCH W192 seed2 | 0.4334 / 0.6475 / 0.5267 / 0.6983 | NA | NA | NA | Target FIT-only scaling, FIT training and SELECT, full W192 | full W192, window-end |
| PCA residual | 0.5344 / 0.7931 / 0.5287 / 0.7670 | NA | NA | NA | Target FIT-only scaler and PCA fitted on at most 50,000 FIT points, current observation | current value only, pointwise |
| Source-supervised endpoint V (three-seed mean ± sample SD) | 0.5731 ± 0.0047 / 0.7640 ± 0.0028 / 0.5084 ± 0.0049 / 0.6757 ± 0.0039 | 0.4402 ± 0.0025 / 0.8261 ± 0.0011 / 0.4300 ± 0.0054 / 0.8598 ± 0.0015 | 0.3659 ± 0.0019 / 0.6678 ± 0.0030 / 0.4345 ± 0.0028 / 0.7356 ± 0.0050 | 0.3826 ± 0.0022 / 0.6317 ± 0.0047 / 0.4300 ± 0.0033 / 0.6819 ± 0.0046 | SWaT real-event pairs and SELECT, nominal SWaT/SMD reconstruction, no target update, archived FIT | causal W100 endpoint |

## Whole-QUERY statistics reference

| Method | PSM | SMD28 | MSL27 | SMAP53 | Training and statistics | Score availability |
| --- | --- | --- | --- | --- | --- | --- |
| TimeRCD official whole-QUERY normalization | 0.3588 / 0.5517 / 0.4109 / 0.6044 | 0.4870 / 0.8381 / 0.5196 / 0.8899 | 0.4947 / 0.7487 / 0.5453 / 0.7785 | 0.4264 / 0.6046 / 0.4143 / 0.6251 | Same TimeRCD weights and score, entire QUERY used for statistics | full W5000 capped by length, end of QUERY for statistics |

## Causal History Mean intersection

| Method | PSM | SMD28 | MSL27 | SMAP53 | Training and statistics | Score availability |
| --- | --- | --- | --- | --- | --- | --- |
| Frozen TimeRadar V (release L=1) | 0.5363 / 0.7312 / 0.4851 / 0.6655 | NA | NA | 0.3737 / 0.6432 / 0.4108 / 0.6916 | Official pretrained TimeRadar, no project updates, archived target FIT | causal W100 endpoint |
| Frozen TimeRadar E | 0.3875 / 0.5676 / 0.3986 / 0.5791 | NA | NA | 0.4279 / 0.6785 / 0.4706 / 0.7367 | Official pretrained TimeRadar, no project updates, archived target FIT | causal W100 endpoint |
| Frozen TimeRadar fixed V+E | 0.4196 / 0.6012 / 0.4237 / 0.6049 | NA | NA | 0.4174 / 0.7052 / 0.4582 / 0.7577 | Official pretrained TimeRadar, no project updates, archived target FIT | causal W100 endpoint |
| Source reconstruction continuation (2025) | 0.5401 / 0.7320 / 0.4930 / 0.6675 | NA | NA | 0.3758 / 0.6388 / 0.4077 / 0.6818 | SWaT real-event SELECT, nominal source reconstruction gradients, no target update, archived FIT | causal W100 endpoint |
| Source-supervised endpoint V (2025) | 0.5776 / 0.7655 / 0.5141 / 0.6796 | NA | NA | 0.3871 / 0.6261 / 0.4347 / 0.6780 | SWaT real-event pairs and SELECT, nominal SWaT/SMD reconstruction, no target update, archived FIT | causal W100 endpoint |
| Source reconstruction continuation (2026) | 0.5376 / 0.7309 / 0.4859 / 0.6648 | NA | NA | 0.3716 / 0.6306 / 0.4070 / 0.6773 | SWaT real-event SELECT, nominal source reconstruction gradients, no target update, archived FIT | causal W100 endpoint |
| Source-supervised endpoint V (2026) | 0.5685 / 0.7605 / 0.5044 / 0.6720 | NA | NA | 0.3852 / 0.6296 / 0.4317 / 0.6804 | SWaT real-event pairs and SELECT, nominal SWaT/SMD reconstruction, no target update, archived FIT | causal W100 endpoint |
| Source reconstruction continuation (2027) | 0.5417 / 0.7327 / 0.4960 / 0.6687 | NA | NA | 0.3739 / 0.6353 / 0.4037 / 0.6804 | SWaT real-event SELECT, nominal source reconstruction gradients, no target update, archived FIT | causal W100 endpoint |
| Source-supervised endpoint V (2027) | 0.5745 / 0.7647 / 0.5091 / 0.6746 | NA | NA | 0.3898 / 0.6358 / 0.4380 / 0.6874 | SWaT real-event pairs and SELECT, nominal SWaT/SMD reconstruction, no target update, archived FIT | causal W100 endpoint |
| History Mean512 | 0.3210 / 0.5300 / 0.3499 / 0.5730 | NA | NA | 0.3181 / 0.6339 / 0.3367 / 0.6862 | No parameter training, historical scaler, 512 past observations and the current value | past W512 and current value, pointwise |
| PCA residual | 0.5346 / 0.7922 / 0.5291 / 0.7659 | NA | NA | NA | Target FIT-only scaler and PCA fitted on at most 50,000 FIT points, current observation | current value only, pointwise |
| Source-supervised endpoint V (three-seed mean ± sample SD) | 0.5736 ± 0.0046 / 0.7636 ± 0.0027 / 0.5092 ± 0.0049 / 0.6754 ± 0.0039 | NA | NA | 0.3874 ± 0.0023 / 0.6305 ± 0.0049 / 0.4348 ± 0.0032 / 0.6819 ± 0.0049 | SWaT real-event pairs and SELECT, nominal SWaT/SMD reconstruction, no target update, archived FIT | causal W100 endpoint |

## Paired AP distribution on the offline-common axes

Rows subtract the reference from the fixed candidate in the same entity. Exact ties are separate. Full four-metric per-entity differences, medians and win/tie/loss distributions are in the linked CSVs.

| Dataset | Seed | Reference | Mean ΔAP | Median ΔAP | W / T / L | n |
| --- | --- | --- | --- | --- | --- | --- |
| PSM | 2025 | DADA official V | +0.181528 | +0.181528 | 1 / 0 / 0 | 1 |
| PSM | 2025 | TimeRCD official weights, FIT-only adapter | +0.275248 | +0.275248 | 1 / 0 / 0 | 1 |
| PSM | 2025 | Complete LEFT | +0.000285 | +0.000285 | 1 / 0 / 0 | 1 |
| PSM | 2026 | DADA official V | +0.172303 | +0.172303 | 1 / 0 / 0 | 1 |
| PSM | 2026 | TimeRCD official weights, FIT-only adapter | +0.266023 | +0.266023 | 1 / 0 / 0 | 1 |
| PSM | 2026 | Complete LEFT | -0.008940 | -0.008940 | 0 / 0 / 1 | 1 |
| PSM | 2027 | DADA official V | +0.178401 | +0.178401 | 1 / 0 / 0 | 1 |
| PSM | 2027 | TimeRCD official weights, FIT-only adapter | +0.272121 | +0.272121 | 1 / 0 / 0 | 1 |
| PSM | 2027 | Complete LEFT | -0.002841 | -0.002841 | 0 / 0 / 1 | 1 |
| SMD28 | 2025 | DADA official V | +0.028067 | -0.013910 | 13 / 0 / 15 | 28 |
| SMD28 | 2025 | TimeRCD official weights, FIT-only adapter | -0.030269 | -0.040126 | 11 / 0 / 17 | 28 |
| SMD28 | 2025 | Complete LEFT | -0.055408 | -0.044432 | 8 / 0 / 20 | 28 |
| SMD28 | 2026 | DADA official V | +0.025809 | -0.019520 | 13 / 0 / 15 | 28 |
| SMD28 | 2026 | TimeRCD official weights, FIT-only adapter | -0.032528 | -0.040438 | 11 / 0 / 17 | 28 |
| SMD28 | 2026 | Complete LEFT | -0.057666 | -0.042132 | 8 / 0 / 20 | 28 |
| SMD28 | 2027 | DADA official V | +0.022986 | -0.018939 | 12 / 0 / 16 | 28 |
| SMD28 | 2027 | TimeRCD official weights, FIT-only adapter | -0.035350 | -0.041195 | 11 / 0 / 17 | 28 |
| SMD28 | 2027 | Complete LEFT | -0.060488 | -0.042650 | 7 / 0 / 21 | 28 |
| MSL27 | 2025 | DADA official V | -0.027744 | -0.009399 | 11 / 0 / 16 | 27 |
| MSL27 | 2025 | TimeRCD official weights, FIT-only adapter | -0.114743 | -0.066858 | 11 / 0 / 16 | 27 |
| MSL27 | 2025 | Complete LEFT | -0.140434 | -0.064325 | 8 / 0 / 19 | 27 |
| MSL27 | 2026 | DADA official V | -0.031477 | -0.011872 | 11 / 0 / 16 | 27 |
| MSL27 | 2026 | TimeRCD official weights, FIT-only adapter | -0.118476 | -0.059495 | 11 / 0 / 16 | 27 |
| MSL27 | 2026 | Complete LEFT | -0.144167 | -0.066409 | 8 / 0 / 19 | 27 |
| MSL27 | 2027 | DADA official V | -0.029974 | -0.012954 | 11 / 0 / 16 | 27 |
| MSL27 | 2027 | TimeRCD official weights, FIT-only adapter | -0.116973 | -0.063546 | 11 / 0 / 16 | 27 |
| MSL27 | 2027 | Complete LEFT | -0.142664 | -0.066922 | 8 / 0 / 19 | 27 |
| SMAP53 | 2025 | DADA official V | -0.046275 | -0.001383 | 24 / 0 / 29 | 53 |
| SMAP53 | 2025 | TimeRCD official weights, FIT-only adapter | -0.060170 | -0.023558 | 18 / 0 / 35 | 53 |
| SMAP53 | 2025 | Complete LEFT | -0.110670 | -0.075159 | 16 / 0 / 37 | 53 |
| SMAP53 | 2026 | DADA official V | -0.048211 | -0.001244 | 24 / 0 / 29 | 53 |
| SMAP53 | 2026 | TimeRCD official weights, FIT-only adapter | -0.062105 | -0.019941 | 18 / 0 / 35 | 53 |
| SMAP53 | 2026 | Complete LEFT | -0.112606 | -0.075159 | 16 / 0 / 37 | 53 |
| SMAP53 | 2027 | DADA official V | -0.043850 | -0.001411 | 24 / 0 / 29 | 53 |
| SMAP53 | 2027 | TimeRCD official weights, FIT-only adapter | -0.057744 | -0.021935 | 20 / 0 / 33 | 53 |
| SMAP53 | 2027 | Complete LEFT | -0.108245 | -0.075159 | 16 / 0 / 37 | 53 |

The original causal banks retain the three-seed AP gains over frozen V on PSM, MSL and SMAP, with a loss on the seen-normal-source SMD28 task. The frozen E and V+E comparisons also remain relevant. On the causal NASA axes, both exceed the source-supervised V candidate's AP. These scores use a single continuous sensor per NASA entity. Raw rank differences between the complete detectors reflect different pretraining, scaling and availability.

On PSM's offline-common timestamps all three candidate seeds exceed DADA and FIT-only TimeRCD on the four metrics. The candidate's mean AP is 0.5731, close to LEFT at 0.5769. The seed2025 AP edge over LEFT is 0.000286 and reverses in the other two seeds. LEFT's ROC-AUC and both VUS measures are higher for all three seeds. PCA also exceeds the candidate on those same three measures.

On SMD28 the candidate's mean AP is 0.4402 versus DADA at 0.4146. Every seed has a negative median AP difference and only 12–13 of 28 machines win. DADA has higher ROC-AUC and both VUS measures. FIT-only TimeRCD and complete LEFT exceed the candidate on all four domain macro metrics. The candidate continues to fall below the source reconstruction controls in AP and both VUS measures on the full causal SMD axis.

On MSL, FIT-only TimeRCD reaches AP 0.4827 and LEFT reaches 0.5083, compared with the candidate's 0.3659. The three candidate seeds each win AP on 11 of 27 entities against TimeRCD. On SMAP, DADA and FIT-only TimeRCD reach 0.4288 and 0.4427 AP, compared with 0.3826. The candidate's VUS-PR mean (0.4300) is close to FIT-only TimeRCD (0.4280), while its ROC-AUC and VUS-ROC are lower. TimeRCD's official whole-QUERY normalization changes AP in different directions across domains and is reported on a separate statistics track.

The competitive evidence is therefore strongest on the development PSM record. The seen-normal-source SMD task has a majority-machine AP deficit against all three principal offline comparators despite a positive mean contrast with DADA. NASA remains below the stronger complete methods in AP. The distinct contribution tested here is direct source-event supervision of the deployed endpoint variance. Evidence for that objective comes from matched source contrasts. The external table establishes where the resulting complete detector is competitive under its causal readout and where ranking gaps remain.

## Cost and recheck

DADA produces 109 new banks in 35.123 occupied A40-seconds. The two TimeRCD normalization passes on NASA80 use 5.985 occupied A40-seconds. Fifty-eight PSM/SMD TimeRCD banks are reused. Total occupied GPU time is 41.107 A40-seconds across four lanes including model loading and I/O, with 17 seconds of queue wall time. Data and axis preparation takes 8.850 CPU wall-seconds. Evaluation takes 715.426 CPU wall-seconds. The DADA variance identity on normal windows and the official TimeRCD uni API logit/probability parity are checked during initial loads. The independent tie-group AP and ROC-AUC recomputation covers every metric row (2,943) with maximum error 5.55e-16. The domain macro metrics and all pair summaries are independently reconstructed from their per-entity CSVs. VUS uses a single pinned integration pass plus independent aggregation.

[Axis and model identities](AXES_AND_INPUTS.json) · [Full-precision domain metrics](domain_metrics.csv) · [Per-entity results](entity_metrics.csv) · [Per-entity paired differences](entity_pairs.csv) · [All metric pair distributions](paired_distribution.csv) · [Inference costs](inference_cost.csv) · [Evaluation receipt](EVALUATION.json). Raw weights and score banks remain outside Git.

The existing [V/E/MLP internal comparisons](../native_supervised_mlp_20261123/RESULTS.md) and [target training-range control](../native_target_normal_full_20261124/RESULTS.md) remain on their archived protocols. [Original source repetitions and calibration](../native_event_rank_repeats_20261122/RESULTS.md) · [SMD event evaluation](../native_event_smd28_20261123/RESULTS.md).
