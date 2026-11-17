# TimeRCD source reference-consistency training: fixed B-only comparison

Two matched source training arms use the same official architecture/initial weights, generated X/Y, source-instance roles, exposure and update budget. The source normalization is the intended factor: batch input statistics versus lawful historical reference statistics. Both are selected on the same reference-normalized source SELECT evaluation. No target QUERY or target labels enter training, source generation, model selection or threshold fitting. This is synthetic supervised anomaly learning with auxiliary masked reconstruction, not pure unsupervised learning and not a proven mixed-target robustness contribution.

## Common-axis SMD28

| Method | Valid n/28 | AP macro | AP median | ROC macro | Strict normal FPR (%) | C3 normal FPR (%) | C3 event recall (%) | C3 event F1 (%) | C3 anomaly coverage (%) | C3 long steady (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_fit_only | 28 | 0.473616 | 0.455082 | 0.859199 | 7.254556 | 6.424975 | 63.473894 | 28.469269 | 42.887990 | 29.107410 |
| zero | 28 | 0.418819 | 0.390820 | 0.854602 | 2.852814 | 1.741473 | 70.987042 | 26.376632 | 37.999980 | 20.663850 |
| decoder_head | 28 | 0.421510 | 0.411386 | 0.856739 | 3.123627 | 1.977645 | 73.646476 | 26.716859 | 39.250573 | 23.166605 |
| self | 28 | 0.421886 | 0.411146 | 0.856922 | 3.110444 | 1.969814 | 73.339045 | 26.586089 | 39.130174 | 22.932959 |
| cross | 28 | 0.422329 | 0.406186 | 0.856826 | 3.066436 | 1.941567 | 74.002604 | 27.080740 | 39.088128 | 22.748049 |
| history_mean512 | 28 | 0.111308 | 0.109026 | 0.661920 | 1.037069 | 0.287152 | 46.943001 | 29.822504 | 6.589165 | 1.144671 |
| LEFT_native_same_roles | 3 | 0.376827 | 0.388016 | 0.886443 | 4.479876 | 4.184639 | 48.863636 | 23.717728 | 20.825705 | 70.590130 |
| batch | 28 | 0.468639 | 0.466406 | 0.854407 | 7.606784 | 6.871171 | 63.583311 | 26.304407 | 42.203705 | 24.002682 |
| reference | 28 | 0.458957 | 0.490735 | 0.860279 | 7.234067 | 6.574864 | 64.824839 | 24.882210 | 40.419339 | 22.514915 |

LEFT covers only 3/28 compatible SMD entities; its subset macro is not directly comparable with all-28 results. Long-event statistics retain their own eligible denominator in the complete macro table. SMD28 is one historically exposed data domain.

## PSM common axis

| Method | AP | ROC | Strict normal FPR (%) | C3 normal FPR (%) | C3 any-hit events | C3 one-to-one matched events | C3 event F1 (%) | C3 anomaly coverage (%) | C3 long steady (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_fit_only | 0.303012 | 0.476555 | 2.205100 | 1.718961 | 28 | 24 | 28.070175 | 4.939317 | 0.465931 |
| zero | 0.432502 | 0.670965 | 0.000000 | 0.000000 | 2 | 2 | 5.479452 | 0.098786 | 0.000000 |
| decoder_head | 0.427837 | 0.678159 | 1.285249 | 0.411470 | 10 | 10 | 8.230453 | 1.891053 | 2.243277 |
| self | 0.427206 | 0.678538 | 1.309079 | 0.463897 | 10 | 10 | 8.658009 | 1.947502 | 2.263043 |
| cross | 0.426164 | 0.680347 | 2.662642 | 1.251887 | 11 | 11 | 5.459057 | 4.233700 | 5.956195 |
| history_mean512 | 0.316848 | 0.530450 | 0.546509 | 0.084200 | 12 | 12 | 20.168067 | 1.006680 | 0.157057 |
| LEFT_native_same_roles | 0.571098 | 0.764667 | 0.000000 | 0.000000 | 8 | 8 | 20.000000 | 1.618214 | 0.039610 |
| batch | 0.267263 | 0.390799 | 2.117722 | 1.836524 | 31 | 28 | 37.837838 | 4.760561 | 0.381607 |
| reference | 0.324151 | 0.509528 | 1.650647 | 1.404401 | 30 | 30 | 40.816327 | 5.165114 | 0.959428 |

## Fixed paired differences, SMD28

| Candidate minus reference | AP mean difference | AP median difference | Positive / tie / negative | Scope |
|---|---:|---:|---|---|
| reference − batch | -0.009683 | 0.000346 | 14 / 0 / 14 | MATCHED_SOURCE_NORMALIZATION |
| reference − B_fit_only | -0.014660 | 0.002034 | 15 / 0 / 13 | OFFICIAL_ZERO_UPDATE_REFERENCE |

## Scoring, permissions and preserved limits

- The sole target track is original B_fit_only: each raw CAL/QUERY role uses its original target FIT mean/std exactly once. CAL and QUERY are separate contexts. Official context window5000, batch1, anomaly head and abnormal-minus-normal logit expression remain fixed.
- Every method uses its own lawful original CAL q99/higher threshold and strict greater-than; neither old numeric thresholds nor QUERY labels are substituted. A and B statistics permissions are not mixed. All reported new methods are offline retrospective, not online detectors.
- common_29x9.csv is the matched scoring-axis table. native_29x7.csv preserves method-own full QUERY axes separately; no cross-axis subtraction. TimeRadar and compatible LEFT/history rows are reused and retain their differing model/training/context permissions.
- Prior source continuation is historical unmatched evidence. It is not substituted for the new batch arm or used to choose source scale, checkpoints or hyperparameters.
- Any-hit is distinct from one-to-one event matching. Full strict/C3 points, normal alarm fragments, event metrics, long-steady coverage and longest missed spans remain in the CSVs. First-hit offsets are offline positions, not causal delay claims.
- One training seed and exposed development datasets limit repeatability and independent confirmation claims. This comparison does not prove lag identification, fractional-feature benefit or mixed-unlabeled target adaptation robustness.

## Resource accounting

parameters_and_compute.csv retains selected steps, updates, source sample exposure and actual controller profile/training/scoring GPU wall time per arm. Raw arrays, checkpoint files, private server paths and receipts are withheld from the public repository. Their numerical identities and safe export hashes are retained in PROVENANCE_PUBLIC.json.

## Fixed scientific outcome

The result is **mixed development evidence**, not a broadly established detection gain and not a reason to erase the PSM positive result.

On PSM, historical-reference-consistent source training improves AP from 0.267263 to 0.324151 (+5.6887 percentage points) and ROC from 0.390799 to 0.509528 relative to the matching batch-normalized training arm. C3 normal FPR falls from 1.8365% to 1.4044%; one-to-one event recall improves from 40.0000% to 42.8571%, event F1 from 0.378378 to 0.408163, and anomaly point coverage from 4.7606% to 5.1651%. Long-steady coverage improves from 0.3816% to 0.9594% but remains very low. C3 any-hit events decrease from31 to30 despite one-to-one matches increasing from28 to30; longest missed span is3788 versus3787 points. Event discovery and sustained detection are therefore not uniformly improved. PSM AP also exceeds the original B checkpoint's 0.303012; it remains below TimeRadar 0.432502 and compatible LEFT 0.571098 under their disclosed differing permissions.

SMD28 does not replicate a stable AP or sustained-coverage gain: reference AP is 0.458957 versus matching batch 0.468639 (mean difference -0.009683, median +0.000346; 14 positive / 14 negative). ROC improves to 0.860279 from 0.854407. C3 normal FPR decreases from 6.8712% to 6.5749%, but anomaly coverage falls from 42.2037% to 40.4193%, long-steady coverage from 24.0027% to 22.5149%, and event F1 from 0.263044 to 0.248822. The original B checkpoint still has higher SMD macro AP 0.473616 and long-steady coverage 29.1074%. Single-entity gains are retained in the full tables, not selected into a favorable subset.

Under the identical reference-normalized synthetic SELECT task, the selected reference-trained model achieves joint loss 0.471745 (CE 0.037173, masked MSE 0.434571) versus batch-trained 1.924638 (CE 0.139374, MSE 1.785264), from the same step0 joint loss 4.130315. This supports better held-out synthetic task fit with matching reference normalization. It does **not** establish generalization of normal real-world response or controlled real-domain false alarms. The mixed QUERY detection evidence above must be judged separately.

Both native architectures have exactly 37,056,515 trainable parameters and complete 1536 updates / 4608 FIT sample exposures. Source SELECT chooses batch step1536 and reference step1024 without target labels. Total recorded controller GPU time is 0.344096626 GPUh (profile 0.004976226, training 0.089865976, scoring 0.249254425). A prelaunch GPU occupancy-check block after 50 successful score jobs remains archived; only eight never-launched jobs were later executed, with no repetition of the successful50.

The fixed task is complete. A broad cross-domain gain, persistent-detection solution or Hybrid innovation is not established; this does not negate the narrower PSM learning result or declare the overall research route NO-GO. No new normalization variant, model, seed or scoring search is started automatically.

Primary verification recomputed all116 bank hashes and all116 new native/common metric rows directly from archived head-logit differences, original CAL quantiles, AP/ROC and strict/C3 TP/FP. All agree within 1e-13. This is primary-agent verification, not external independent review. The actual executed evaluator source was preserved before evaluation and matches its raw RESULT source hash.
