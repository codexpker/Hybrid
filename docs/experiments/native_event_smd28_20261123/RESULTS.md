# SMD28 multivariate held-out test

All 28 OmniAnomaly SMD machines and their 38 original variables belong to this evaluation. The seven causal detectors reuse one official TimeRadar checkpoint and six previously selected source-trained decoders. Nominal source FIT included SMD28 and SWaT. Labeled source-window pairs and labeled validation events came from SWaT. Each decoder saw the same source-event SELECT criterion, including the reconstruction-gradient control. Point weighting uses additional paired-window exposure. The project used no SMD test values or labels for training, selection or configuration.

## Common test-point ranking

Each machine contributes endpoints [99,192 floor(N/192)), 703,212 points in total. Ranking is computed separately per machine and then averaged with equal machine weights. For the two source-trained methods, ± gives the sample SD between three project-training seeds sharing the official initialization. LEFT uses one project seed and a separate normally trained model for each machine. Its score becomes available at the end of each 192-point window; native V becomes available at the scored endpoint. Common points align the outcomes; the models have different training and observation permissions.

| Method | Permission | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| Official TimeRadar V | Causal endpoint, shared across machines | 0.44479 | 0.81320 | 0.46718 | 0.86997 |
| Reconstruction continuation | Causal endpoint, shared across machines | 0.44967 ± 0.00260 | 0.81873 ± 0.00365 | 0.47487 ± 0.00454 | 0.87408 ± 0.00273 |
| Point-weighted source supervision | Causal endpoint, shared across machines | 0.44021 ± 0.00255 | 0.82606 ± 0.00109 | 0.42998 ± 0.00539 | 0.85982 ± 0.00148 |
| LEFT | Offline, trained per machine | 0.49807 | 0.87853 | 0.54180 | 0.91851 |

[Common-axis per-machine scores](common/common_entity.csv) · [checkpoint averages](common/common_macro_by_checkpoint.csv). AP denotes average precision. Both VUS metrics use the pinned LEFT implementation, windowSize=100 and thre=250.

## Methods on their complete scoring axes

| Method | Entity-specific axis | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|---|
| Official TimeRadar V | [99,N) | 0.44423 | 0.81295 | 0.46636 | 0.86971 |
| Reconstruction continuation | [99,N) | 0.44909 ± 0.00258 | 0.81849 ± 0.00365 | 0.47381 ± 0.00443 | 0.87381 ± 0.00271 |
| Point-weighted source supervision | [99,N) | 0.43963 ± 0.00254 | 0.82581 ± 0.00108 | 0.42868 ± 0.00545 | 0.85949 ± 0.00150 |
| LEFT | [0,192 floor(N/192)) | 0.49751 | 0.87868 | 0.54152 | 0.91862 |

The full causal axis has 705,648 scored points; LEFT has 705,984 full-window points, from 708,420 input test positions in total. Set membership differs at the beginning and end of each machine. [LEFT training and score identities](left/entity_native.csv) · [native V per-machine results](native/entity_methods.csv).

## Paired causal effects

All differences below are point weighting minus the named control on each machine's full causal axis. Counts and medians characterize the 28 paired entity differences. Training-seed SD uses n=3 and ddof=1.

| Control | Seed | AP | ROC_AUC | VUS_PR | VUS_ROC | AP wins/ties/losses | Median ΔAP |
|---|---|---|---|---|---|---|---|
| Official TimeRadar V | 2025 | -0.00217 | +0.01410 | -0.03284 | -0.00873 | 12/0/16 | -0.00653 |
| Reconstruction continuation | 2025 | -0.00816 | +0.00680 | -0.04257 | -0.01412 | 12/0/16 | -0.00773 |
| Official TimeRadar V | 2026 | -0.00439 | +0.01212 | -0.03661 | -0.01021 | 12/0/16 | -0.00826 |
| Reconstruction continuation | 2026 | -0.00629 | +0.01078 | -0.03895 | -0.01120 | 12/0/16 | -0.00722 |
| Official TimeRadar V | 2027 | -0.00723 | +0.01235 | -0.04358 | -0.01173 | 12/0/16 | -0.00858 |
| Reconstruction continuation | 2027 | -0.01392 | +0.00437 | -0.05386 | -0.01765 | 10/0/18 | -0.00774 |

| Control | AP | ROC_AUC | VUS_PR | VUS_ROC |
|---|---|---|---|---|
| Reconstruction continuation | -0.00946 ± 0.00397 | 0.00732 ± 0.00324 | -0.04513 ± 0.00778 | -0.01432 ± 0.00323 |
| Official TimeRadar V | -0.00460 ± 0.00254 | 0.01286 ± 0.00108 | -0.03767 ± 0.00545 | -0.01022 ± 0.00150 |

[All paired metrics and entity medians](native/pairs.csv).

## Operating outcomes

Each machine and checkpoint uses its own 512-point project CAL, q99/higher and strict >. A three-point confirmation starts on the third consecutive crossing; no backfill. Counts below sum within each seed across separate machines, then average across seeds for trained models. Point rates use total points. Event matching is one-to-one. Any-hit counts all qualifying events touched by an alarm. Long steady coverage excludes the first 512 points of each eligible long event; macro gives each such event equal weight and pooled gives each steady point equal weight. Delay is reported only for hit events. LEFT used this CAL in its public normal scaler and validation; independent CAL operations for that row are unavailable.

| Stream | Method | Normal FPR | False fragments | Recall | Matched / eligible | Any-hit | Event F1 | Long macro / pooled | Longest miss | Delay |
|---|---|---|---|---|---|---|---|---|---|---|
| strict | Official TimeRadar V | 0.1933 | 4534.0 | 0.5298 | 250.0/327 | 278.0 | 0.0938 | 0.5822/0.6992 | 631.0 | 0.0 |
| strict | Reconstruction continuation | 0.1951 | 4404.7 | 0.5346 | 246.0/327 | 277.3 | 0.0945 | 0.5856/0.7008 | 333.0 | 0.0 |
| strict | Point-weighted source supervision | 0.1911 | 3307.3 | 0.5420 | 172.3/327 | 205.3 | 0.0874 | 0.5792/0.7004 | 837.0 | 0.0 |
| confirmed3 | Official TimeRadar V | 0.1841 | 1189.0 | 0.4977 | 149.0/327 | 180.0 | 0.1770 | 0.5744/0.6952 | 837.0 | 2.0 |
| confirmed3 | Reconstruction continuation | 0.1861 | 1204.7 | 0.5019 | 148.3/327 | 182.3 | 0.1735 | 0.5772/0.6963 | 837.0 | 2.0 |
| confirmed3 | Point-weighted source supervision | 0.1838 | 1276.7 | 0.5181 | 123.0/327 | 158.0 | 0.1449 | 0.5730/0.6982 | 838.0 | 2.0 |

[Full operating table](native/operating.csv) · [event detail](native/events.csv).

## Evidence and scope

Point weighting increased ROC-AUC against both causal controls in every seed. On SMD it reduced AP, VUS-PR, and VUS-ROC against both in every seed. At the chosen operating points slightly more anomalous points were covered, yet three-confirmation matched fewer complete events and produced more normal alarm fragments. The long steady results cover 12 events and 5,715 steady points. On the shared timestamps LEFT had higher values for all four ranking metrics. Its machine-specific normal training and full-window lookahead limit attribution of that performance difference to its architecture.

The frozen recipe remains a source-supervised candidate with task-dependent evidence. PSM development, MSL and SMAP showed repeated AP and VUS-PR advantages. Their recorded costs include SMAP false-alarm fragmentation and reduced PSM suffix coverage under its attack-only CAL. NASA used one sensor per entity. The present multivariate test sets a boundary on the AP claim. Retain the same scoring rule for any separately authorized target-adaptation comparison. This batch tested no U, F or contamination intervention.

Seven-checkpoint scoring consumed 8.0982 A40-hours summed across entity receipts, with a separate approximately seven-second numerical check. The 28 independent LEFT fits and native readouts consumed 1.6488 A40-hours. LEFT reached epoch 20 on 15 machines and selected epoch 20 on ten. The three older LEFT runs had different normal FIT/SELECT/scaler permissions and remain archived separately. [LEFT configuration and permissions](left/README.md) · [bank identities](native/banks.csv) · [independent cache-based verification](review.json).
