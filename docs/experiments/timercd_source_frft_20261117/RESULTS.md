# TimeRCD-source local FRFT residual: fixed matched evaluation

Hybrid is the proposed candidate, not a renamed official model. This one-seed development comparison tests local patch representation only. Native/time/FRFT share the official initial weights, synthetic instances, source validation, exposure, backbone/head update policy and budget. The temporal and FRFT residuals have equal parameter counts. No target QUERY or its labels enter training or selection. This is synthetic supervision plus auxiliary masked self-supervision, not purely unsupervised training.

## Common-axis SMD28

| Method / permission | Valid n/28 | AP macro | AP median | ROC macro | Strict normal FPR (%) | C3 normal FPR (%) | C3 event recall (%) | C3 event F1 (%) | C3 anomaly coverage (%) | C3 long steady (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_official_role | 28 | 0.487280 | 0.500909 | 0.838074 | 0.575883 | 0.281396 | 56.812464 | 39.407260 | 30.229972 | 4.769747 |
| B_fit_only | 28 | 0.473616 | 0.455082 | 0.859199 | 7.254556 | 6.424975 | 63.473894 | 28.469269 | 42.887990 | 29.107410 |
| zero | 28 | 0.418819 | 0.390820 | 0.854602 | 2.852814 | 1.741473 | 70.987042 | 26.376632 | 37.999980 | 20.663850 |
| decoder_head | 28 | 0.421510 | 0.411386 | 0.856739 | 3.123627 | 1.977645 | 73.646476 | 26.716859 | 39.250573 | 23.166605 |
| self | 28 | 0.421886 | 0.411146 | 0.856922 | 3.110444 | 1.969814 | 73.339045 | 26.586089 | 39.130174 | 22.932959 |
| cross | 28 | 0.422329 | 0.406186 | 0.856826 | 3.066436 | 1.941567 | 74.002604 | 27.080740 | 39.088128 | 22.748049 |
| history_mean512 | 28 | 0.111308 | 0.109026 | 0.661920 | 1.037069 | 0.287152 | 46.943001 | 29.822504 | 6.589165 | 1.144671 |
| LEFT_native_same_roles | 3 | 0.376827 | 0.388016 | 0.886443 | 4.479876 | 4.184639 | 48.863636 | 23.717728 | 20.825705 | 70.590130 |
| native__A_official_role | 28 | 0.499294 | 0.502510 | 0.855502 | 0.528429 | 0.301711 | 58.599027 | 38.633250 | 30.563587 | 3.474968 |
| native__B_fit_only | 28 | 0.468240 | 0.466005 | 0.854074 | 7.152144 | 6.423489 | 66.466085 | 26.900859 | 40.575330 | 24.082655 |
| time__A_official_role | 28 | 0.497949 | 0.514657 | 0.853285 | 0.530135 | 0.299136 | 57.027277 | 36.641637 | 30.497752 | 3.414612 |
| time__B_fit_only | 28 | 0.460733 | 0.457616 | 0.848843 | 7.099486 | 6.335406 | 65.844028 | 25.939415 | 39.437611 | 24.276968 |
| frft__A_official_role | 28 | 0.495144 | 0.490226 | 0.853897 | 0.549341 | 0.307519 | 58.766625 | 37.976268 | 30.422329 | 3.426535 |
| frft__B_fit_only | 28 | 0.463244 | 0.466654 | 0.850688 | 7.014661 | 6.237123 | 65.797586 | 25.925576 | 39.420065 | 24.573650 |

LEFT has compatible results for only 3/28 SMD entities. Its subset summary is not an all-28 comparison. SMD28 represents one exposed domain, not 28 independent datasets. Long-steady metrics retain their own non-NA denominator (see macro CSV).

## PSM common axis

| Method / permission | AP | ROC | Strict normal FPR (%) | C3 normal FPR (%) | C3 any-hit events | C3 one-to-one matched events | C3 event F1 (%) | C3 anomaly coverage (%) | C3 long steady (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_official_role | 0.359946 | 0.552157 | 2.595917 | 2.151084 | 34 | 32 | 34.224599 | 5.588484 | 2.001377 |
| B_fit_only | 0.303012 | 0.476555 | 2.205100 | 1.718961 | 28 | 24 | 28.070175 | 4.939317 | 0.465931 |
| zero | 0.432502 | 0.670965 | 0.000000 | 0.000000 | 2 | 2 | 5.479452 | 0.098786 | 0.000000 |
| decoder_head | 0.427837 | 0.678159 | 1.285249 | 0.411470 | 10 | 10 | 8.230453 | 1.891053 | 2.243277 |
| self | 0.427206 | 0.678538 | 1.309079 | 0.463897 | 10 | 10 | 8.658009 | 1.947502 | 2.263043 |
| cross | 0.426164 | 0.680347 | 2.662642 | 1.251887 | 11 | 11 | 5.459057 | 4.233700 | 5.956195 |
| history_mean512 | 0.316848 | 0.530450 | 0.546509 | 0.084200 | 12 | 12 | 20.168067 | 1.006680 | 0.157057 |
| LEFT_native_same_roles | 0.571098 | 0.764667 | 0.000000 | 0.000000 | 8 | 8 | 20.000000 | 1.618214 | 0.039610 |
| native__A_official_role | 0.336883 | 0.496320 | 1.258241 | 0.845182 | 29 | 29 | 29.145729 | 4.440681 | 2.534527 |
| native__B_fit_only | 0.311615 | 0.471643 | 1.174041 | 0.829295 | 31 | 31 | 33.155080 | 3.678615 | 0.723674 |
| time__A_official_role | 0.329224 | 0.477774 | 1.334498 | 0.910319 | 28 | 28 | 28.000000 | 4.548876 | 2.394321 |
| time__B_fit_only | 0.309769 | 0.466401 | 1.180396 | 0.811820 | 30 | 30 | 32.432432 | 3.763289 | 1.009392 |
| frft__A_official_role | 0.327174 | 0.478879 | 1.224879 | 0.818175 | 28 | 28 | 28.426396 | 4.144322 | 2.234052 |
| frft__B_fit_only | 0.305800 | 0.459043 | 1.220113 | 0.835650 | 30 | 30 | 32.967033 | 3.716248 | 1.018737 |

## Matched source-arm increments (SMD28)

| Permission | Candidate − reference | AP mean difference | AP median difference | Positive / tie / negative |
|---|---|---:|---:|---|
| A_official_role | frft__A_official_role − native__A_official_role | -0.004151 | -0.003591 | 10 / 0 / 18 |
| A_official_role | frft__A_official_role − time__A_official_role | -0.002805 | -0.001188 | 11 / 0 / 17 |
| A_official_role | frft__A_official_role − A_official_role | 0.007863 | 0.013446 | 19 / 0 / 9 |
| B_fit_only | frft__B_fit_only − native__B_fit_only | -0.004996 | -0.003976 | 9 / 0 / 19 |
| B_fit_only | frft__B_fit_only − time__B_fit_only | 0.002511 | 0.000408 | 14 / 0 / 14 |
| B_fit_only | frft__B_fit_only − B_fit_only | -0.010373 | -0.007515 | 11 / 0 / 17 |

## Permissions, scoring and preserved limits

- B_fit_only is the main constrained track: the same original raw FIT statistics are used once. A_official_role separately normalizes each complete CAL/QUERY role using that role; it remains an offline native-statistics reference. Both use full noncausal context windows, never an online claim.
- Every new and original TimeRCD comparison row uses the archived abnormal-minus-normal head logit. Each score computes its own original CAL q99/higher threshold and uses strict greater-than. No inverse-probability logits, score fusion or QUERY-fitted threshold is used.
- Main differences use common_29x14.csv exclusively. Native full QUERY axes are separately preserved in native_29x12.csv; never subtract results across those axes. Official probability results remain archived in the prior release.
- TimeRadar zero/Head/Self/mean-Cross, History Mean512 and compatible LEFT rows are immutable prior results. Their different training/context permissions remain disclosed in table metadata. The old reconstruction-adaptation branch is closed and is not recovered here.
- Any-hit and one-to-one event matches are different quantities. Full strict/C3 metrics, false-alarm points/fragments, sustained coverage and longest missed spans remain in the CSVs. First-hit positions are offline index offsets, not causal detection delays.
- Single source seed2025 and exposed development data cannot establish broad repeatability, multi-source pretraining benefit, delay-mechanism identification or mixed-unlabeled robustness.

## Cost and source provenance

See parameters_and_compute.csv for parameter counts, selected step, sample exposure and per-arm profile/training/scoring GPU wall time. The three arms have identical source update budgets. PROVENANCE_PUBLIC.json contains only content identities; raw checkpoint files, role arrays, private machine paths and raw receipts are not redistributed.

This fixed candidate does not authorize a follow-up architecture, normalization, scoring, seed or threshold search. Scientific interpretation is based on the complete fixed tables, including losses.

## Fixed scientific outcome

This one bounded local-patch candidate does **not establish an independent FRFT detection gain** over both matched controls. It is not promoted as a Hybrid contribution and does not authorize follow-up variants.

On the main B track, SMD28 AP is 0.463244, below native continuation 0.468240 (mean paired difference -0.004996, 9 positive / 19 negative) and only 0.002511 above the equal-capacity temporal residual (14 / 14). ROC is 0.850688 versus native 0.854074. C3 normal FPR decreases from 6.4235% to 6.2371%, but anomaly coverage also decreases from 40.5753% to 39.4201% and event F1 from 0.269009 to 0.259256. Thus the false-alarm reduction is not a free sustained-detection gain.

PSM B AP is 0.305800 versus native 0.311615 and temporal 0.309769. C3 event recall is 42.8571%, coverage 3.7162%, and normal FPR 0.8357%; these do not show a superior detection/false-alarm tradeoff. All PSM negative results remain reported.

A is not substituted for the main B decision: its SMD28 FRFT AP 0.495144 exceeds the original checkpoint 0.487280, but native continuation achieves 0.499294 and temporal 0.497949. FRFT is lower than native on 18/28 and lower than temporal on 17/28. Its PSM AP 0.327174 is also below both matched source-trained controls. The gain over the original checkpoint on A is therefore not attributable specifically to the FRFT branch.

All three source training arms completed 1536 updates; joint source validation selected step1536 for all. FIT contains 1536 unique generated instances, SELECT 256 independent generated instances; each arm exposes 4608 FIT samples across three passes, not 4608 independent instances. Temporal and FRFT arms both add 17,953 parameters. Actual controller GPU accounting is 0.833360714 GPUh: profile 0.005828270, training 0.113279097, and scoring 0.714253347. A prelaunch GPU occupancy-check block (underlying cause unconfirmed) after 167 successful score jobs is retained; only the seven never-launched jobs were subsequently executed, with no repetition of successful jobs. Numerical interpretation does not support a new architecture, scoring, normalization or fusion search in this task.

Primary raw-cache recomputation independently of the producer checked all 348 bank hashes and all 348 new native/common metric rows: head-logit difference, original CAL threshold, AP/ROC, and strict/C3 TP/FP match within 1e-13. This is a primary-agent verification, not an external independent audit.

The decision is **limited NO-GO for this local FRFT residual candidate**, not a general NO-GO for source continuation, discriminative learning, fractional representations or the overall Hybrid research goal. No further variant is launched.

The exact executed evaluator source was recovered by reversing known publication-only edits, and its bytes match the execution-recorded SHA256 `6b4cb5033196315f15a901e65bde2486676d11373a836ddfc5a8042616aa6c2b`. It is preserved outside Git with an exact diff and recovery record. Later edits only guard publication output, distinguish execution/publication provenance hashes, and clarify an occupancy-check comment. Numerical evaluation instructions and raw RESULT identity are unchanged.
