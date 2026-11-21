# Reproducibility

This is the author-side record for the frozen SWaT and PSM results in *Source Replay for Mixed-Unlabeled Adaptation in Multivariate Time-Series Anomaly Detection*. The manuscript and archived numeric evidence are pinned at [submission snapshot `47232b29c2fcf0b399f5d157400bbacc1f3863e1`](https://github.com/codexpker/Hybrid/commit/47232b29c2fcf0b399f5d157400bbacc1f3863e1). The anonymous files contain methods and derived result tables without a link to the named author repository.

## Data versions and access

| Task | Provider and acquisition | Exact study version |
|:--|:--|:--|
| SWaT | Secure Water Treatment sensor and actuator benchmark. Request the sensor historian data and label documentation through [SUTD iTrust data access](https://www.sutd.edu.sg/itrust/itrust-labs/datasets) under its terms. Relevant public source family is [SWaT.A2_Dec 2015](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/). | The study uses the archived DADA-format long CSV with SHA-256 `a8bad2c5f770dbd04e7ad44f58f565caf1c4fe1a68ab62d2d8625adb4d4cdceb`, not a newly acquired raw release. The unstandardized wide input has 944,919 rows and 31 features and SHA-256 `d2304906b800ea47365b864886bd83fa79de9202190f75dccb6ef7f22d00cafa`. The benchmark export's upstream raw iTrust normal-file revision (v0 versus v1) and original-to-DADA conversion chain were not bound in the archive. Row counts or a fresh iTrust download alone do not establish equivalence. [Study input binding](../../configs/left-exposure-swat-20261118.json) |
| PSM | Public Pooled Server Metrics data. Obtain `train.csv`, `test.csv`, and `test_label.csv` from the [eBay RANSynCoders release](https://github.com/eBay/RANSynCoders/tree/948e61eb579815ff2a6bce721332d8220d59ca47/data). This pins the provider branch checked on 2026-11-21. The observation and label CSV headers were checked. | The study uses the DADA-format long CSV with SHA-256 `38e446209e82df94b82c11aa43d87e3671da86167ed794c1b7be5eb91fc0e8ac`, exported as raw arrays of shape `[132481,25]` (source) and `[87841,25]` (target). The public release corresponds in shape, split and channel naming. Byte or value equality between a fresh provider download and the frozen long CSV was not established. Exact source and target array, label, and scaler SHA-256 values are in the [PSM task identity](../../configs/left-author-psm-source-replay-20261118.json). |

## Channel map and preprocessing

SWaT starts with 51 feature blocks in the order of first appearance in the frozen long CSV, excluding `label`. The index below is zero-based. The name `" AIT201"` retains its original leading space. The removed original indices are `2,4,9,10,11,13,15,19,20,21,22,29,30,31,32,33,42,43,48,50`.

| Model index | Original index | Original header |
|--:|--:|:--|
| 0 | 0 | `FIT101` |
| 1 | 1 | `LIT101` |
| 2 | 3 | `P101` |
| 3 | 5 | `" AIT201"` |
| 4 | 6 | `AIT202` |
| 5 | 7 | `AIT203` |
| 6 | 8 | `FIT201` |
| 7 | 12 | `P203` |
| 8 | 14 | `P205` |
| 9 | 16 | `DPIT301` |
| 10 | 17 | `FIT301` |
| 11 | 18 | `LIT301` |
| 12 | 23 | `P301` |
| 13 | 24 | `P302` |
| 14 | 25 | `AIT401` |
| 15 | 26 | `AIT402` |
| 16 | 27 | `FIT401` |
| 17 | 28 | `LIT401` |
| 18 | 34 | `AIT501` |
| 19 | 35 | `AIT502` |
| 20 | 36 | `AIT503` |
| 21 | 37 | `AIT504` |
| 22 | 38 | `FIT501` |
| 23 | 39 | `FIT502` |
| 24 | 40 | `FIT503` |
| 25 | 41 | `FIT504` |
| 26 | 44 | `PIT501` |
| 27 | 45 | `PIT502` |
| 28 | 46 | `PIT503` |
| 29 | 47 | `FIT601` |
| 30 | 49 | `P602` |

PSM model column (i) is literal `feature_i` for (i=0,ldots,24), in increasing order. The provider's `timestamp_(min)` and `label` are excluded from model input. The frozen long export stores 25 consecutive feature blocks followed by a label block, with 220,322 rows per block. Stack blocks without reordering and split at 132,481. In SWaT the conversion writes each retained channel to row `date-1` and separates the label column. Both frozen inputs are finite before standardization. These study-level exports apply no new imputation. Any processing upstream of the frozen benchmark CSVs belongs to their respective provenance chains.

Use the archived per-feature `StandardScaler` for each task, fit on the full normal-source prefix under the LEFT native protocol, including source VALIDATION. Fit ranges are `[0,495000)` in SWaT and `[0,132481)` in PSM. The fitting inputs use the documented Float64 Fortran-layout path. Apply the fixed ((x-mu)/sigma) transform in the archived data pipeline and cast sampled network inputs to float32. No target observations enter scaler fit, and the scaler is fixed for all arms. Fitting the scaler on SOURCE_FIT alone defines a different experiment.

## Chronological roles

All ranges are zero-based and half-open. SWaT indexes the combined timeline. In PSM, SOURCE_FIT and VALIDATION use source coordinates. U-FIT, CAL, and QUERY use target-test-relative coordinates.

| Role or axis | SWaT | PSM |
|:--|:--|:--|
| SOURCE_FIT | `[0,396000)` | `[0,105984)` |
| VALIDATION | `[396000,495000)` | `[105984,132481)` |
| U-FIT | `[495000,585000)` | `[0,17568)` |
| Raw CAL role | `[585544,607479)` | `[18112,21960)` |
| Legal CAL score support | `[585624,607320)` | `[18240,21888)` |
| Raw QUERY role | `[608023,944919)` | `[22504,87841)` |
| Legal QUERY score support | `[608472,944856)` | `[22656,87744)` |
| Final QUERY crop | `[608535,944856)` | `[22656,87744)` |

SOURCE_FIT supplies native fitting and source replay. VALIDATION selects the source checkpoint and is included in source scaling. U-FIT supplies the unlabeled target draws. For the B/C intervention, a separate constructor uses labels within U-FIT only. The learner receives only numerical windows and a slot map. CAL supplies scores for a threshold without consulting labels. QUERY labels are used after scoring for evaluation. PSM has 512/512 selected CAL points with attack labels, and SWaT has 38/512. These are post hoc counts. An empirical threshold here does not give an equal or certified normal false-alarm budget.

## Model initialization and adaptation

Use the [official LEFT code at `3fadb4811797075e233076b9faceb0d8ec59b697`](https://github.com/DezhengWang/Left/tree/3fadb4811797075e233076b9faceb0d8ec59b697) and its task-specific `model_configs_Left.json`. Both source stages use window 192, batch 128, train stride 1, validation stride 192, native Adam and learning-rate schedule, and normal validation loss for selection. SWaT training stopped at epoch 17 under the native early stopping rule (cap 20, patience 3) with weights selected from epoch 14. The accepted noU scoring state uses archived runtime step 52,581. These are separate fields from the selected checkpoint's training step. PSM used the full 20 epochs, selected epoch 20, and started at runtime step 16,540. Restore the existing task-specific checkpoints, prototypes, scaler, and runtime step for all arms. The [SWaT native report](../../docs/experiments/left_native_swat_20261118/RESULTS.md) and [PSM report](../../docs/experiments/left_psm_replay_20261118/RESULTS.md) identify the source runs. Task-specific checkpoints were reused, without independent pretraining repetitions or transfer between SWaT and PSM.

| Arm | Batch and treatment |
|:--|:--|
| noU | Single frozen source state per task, shared by all seed comparisons. |
| TT | `[T, T_prime]` from the authorized target-fit role. |
| SS | `[S_prime, S]` from SOURCE_FIT only. |
| R | `[T, S]` with matched target and normal-source halves. |

The three update arms train all LEFT parameters with one native forward/state update and full native loss per 256-window batch (128 per half), fresh Adam with learning rate `1e-4`, and exactly 2,000 optimizer updates. The chosen endpoint is step 2,000 of adaptation. Seeds 2025, 2026, and 2027 seed Python, NumPy, Torch CPU and CUDA streams in the [adaptation implementation](../../src/hybrid_tsad/left_domain_replay.py). Sample legal width-192 start offsets uniformly with replacement. Use `default_rng(seed)` for T and `SeedSequence(seed).spawn(3)` for S, T_prime, S_prime in that order. In each task and seed, TT and R share T, and SS and R share S. No timing-based or query-based selection occurs.

For the [controlled replacement constructor](../../src/hybrid_tsad/left_exposure_bundle.py), select 25,600 of the 256,000 common T slots without replacement and share the selection across B/C. Draw donors with replacement independently from fully attack-labeled (B) or fully nominally normal (C) legal U-FIT windows. Both arms use identical donor windows within each condition. The seed is `SeedSequence([seed, 20261118, 10]).spawn(3)` in slot, B, and C order. Exact matches concern slots and within-condition cross-arm donor identity. The pools are not matched by donor time, operating state or similarity. One changed slot is one window presentation. The 5% of whole-batch presentations does not specify a unique-timepoint anomaly rate. The final manuscript cites absolute B comparisons separately from the interaction `(R-B − R-C) − (TT-B − TT-C)`.

## Readout and metrics

- **Axis and scores.** Use unmodified LEFT native `infer`, nonoverlapping 192-point windows aligned at 495000 (SWaT) and 0 (PSM), with complete window footprints inside the authorized CAL and QUERY roles. Average the native per-channel score in float32. All scores in a window first become available at its last observed index. These are retrospective offline scores. Include partially covered events in point metrics but exclude them from complete-event measures.
- **Point ranks.** Use scikit-learn `average_precision_score` and `roc_auc_score` on the finite aligned score and label arrays. AP uses the stepwise precision-recall average. No point adjustment is used. VUS uses LEFT `metricor.RangeAUC_volume_opt` with `windowSize=100` and `thre=250` under [the pinned metric entry](../../src/hybrid_tsad/left_native_swat_metrics.py). The approved metric source SHA-256 is `808a07460a84d31c87edce1d803e1df231111ec53ffa702ee32eb54a79897dfc`.
- **Alarm streams.** Select 512 fixed equally spaced integer CAL positions by `np.linspace(first,last,512,dtype=np.int64)`. Compute each model's empirical `q=.99` quantile with `method="higher"`. Strict alarms are scores greater than the threshold. C3 confirms upon the third consecutive above-threshold scored index, with no backfill and a reset at gaps. Calculate strict and C3 metrics independently. The two CAL roles and the all-attack PSM consequence are unchanged.
- **Events and coverage.** Recover contiguous binary label runs as original events. Complete events have full valid coverage on the fixed axis. Any-hit requires at least one aligned alarm. One-to-one matches by chronological first-unmatched overlap, with each alarm fragment and event used at most once. Event precision uses the number of eligible predicted fragments and recall uses the number of complete true events. F1 is their harmonic mean. Pool present alarm points over complete eligible event durations for total coverage. For events longer than 512 points, define steady coverage starting at original onset +512. Report both the mean of the within-event coverages and the sum over steady points divided by all eligible steady points. Longest continuous miss and time offsets are in original samples. Location and availability offset medians use hits, and censored cases are recorded separately.
- **Summaries.** For the three seeds, form contrasts within task and seed before the arithmetic mean and sample standard deviation (ddof=1). A fixed noU reference supplies one score per task. Repeated observations, overlapping windows and reused event donors do not add independent experimental units.

## Runtime and repeatability limits

The archived SWaT native run environment receipt gives Linux 6.8.0-147 x86_64, Python **3.12.13**, PyTorch **2.8.0+cu128**, CUDA runtime **12.8**, NumPy **1.26.4**, SciPy **1.15.3**, pandas **2.3.3**, and scikit-learn **1.7.2**. The archived PSM preflight records the same Python and core stack plus matplotlib **3.10.9**, einops **0.8.1**, tqdm **4.67.1**, and the separate THOP import receipt identifies distribution **0.1.1**, runtime **0.1.1-2209072238**. These are observations from the named historical environment, not a complete transitive lockfile from every job. Training and scoring records identify NVIDIA A40 devices. The frozen PR/ROC SVG separately embeds a Matplotlib **3.10.8** creator tag. Do not substitute a driver-reported CUDA ability for the Torch-linked runtime. The supplement lists recorded A40 job-wall costs by phase. A new run on other hardware or software is not asserted to be bit-identical.

## Access and verification

| Artifact | How to obtain it |
|:--|:--|
| Public dataset source | Obtain the files from iTrust and eBay using the provider links above. Verify the study export fingerprints before claiming exact replay. Public access to a source benchmark does not establish redistribution rights for its derived data. |
| Public implementation and configs | The named repository contains train, readout, metric, and construction source, and the [SWaT](../../configs/left-exposure-swat-20261118.json) and [PSM](../../configs/left-exposure-psm-20261118.json) exposure configs. The [archived exposure report](../../docs/experiments/left_replay_exposure_repeat_20261118/RESULTS.md) names the prepare, queue, and evaluation entry points. They require a workspace containing the bound inputs, selectable checkpoints, scaler and score state. |
| Original experiment inputs, weights, optimizer states, and full point score banks | Retained in the controlled experiment archive. They are not uploaded in this package. A named custodian, request mechanism and sharing conditions must be confirmed by the authors before publication. Where access is approved, compare file SHA-256 against the frozen task receipts. Public download of third-party data does not substitute for archive access. |
| Derived aggregate tables | The [anonymous table directory](anonymous/tables/) contains byte-for-byte copies of eleven public derived CSVs. The [supplement](anonymous/supplement.pdf) maps them to experiments and protocols. Reviewing displayed values requires only the derived tables. Recreating raw scores requires the additional assets above. |

The channel mapping was recovered from the frozen export records. Unresolved items for independent from-scratch reproduction are the raw SWaT provider-revision-to-DADA conversion chain, exact equivalence of fresh PSM provider files to the frozen long export, an all-job dependency lockfile, and an authorized archive access arrangement.
