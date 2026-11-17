# Historical-reference source training: fixed three-seed repetition

Source seeds2025/2026/2027 repeat exactly the accepted batch/reference training recipe. Seed2025 metrics are hash-reused without retraining, rescoring or reevaluation. Only the two prespecified new training seeds are evaluated. Source data and roles stay fixed; within each seed, batch/reference share draws, masks and official initialization. Source SELECT masks remain fixed. Target inference remains original B_fit_only with the fixed logit score, context and lawful CAL quantile. No QUERY feedback chooses a variant.

## PSM common-axis repetitions

| Method | AP mean ± sample SD | ROC mean ± sample SD | Strict normal FPR (%) | C3 normal FPR (%) | C3 one-to-one event recall (%) | C3 event F1 (%) | C3 anomaly coverage (%) | C3 long steady (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batch | 0.275184 ± 0.009215 | 0.405086 ± 0.014800 | 2.055763 ± 0.403142 | 1.702015 ± 0.319106 | 41.428571 ± 5.150788 | 36.073270 ± 2.277308 | 5.166682 ± 1.225241 | 0.634962 ± 0.384019 |
| reference | 0.320317 ± 0.004261 | 0.496779 ± 0.012620 | 1.644822 ± 0.005107 | 1.385866 ± 0.024320 | 42.380952 ± 2.182179 | 38.425136 ± 2.073431 | 5.257629 ± 0.148186 | 1.578184 ± 0.580709 |
| B_fit_only (reused fixed result) | 0.303012 | 0.476555 | 2.205100 | 1.718961 | 34.285714 | 28.070175 | 4.939317 | 0.465931 |
| zero (reused fixed result) | 0.432502 | 0.670965 | 0.000000 | 0.000000 | 2.857143 | 5.479452 | 0.098786 | 0.000000 |
| decoder_head (reused fixed result) | 0.427837 | 0.678159 | 1.285249 | 0.411470 | 14.285714 | 8.230453 | 1.891053 | 2.243277 |
| self (reused fixed result) | 0.427206 | 0.678538 | 1.309079 | 0.463897 | 14.285714 | 8.658009 | 1.947502 | 2.263043 |
| cross (reused fixed result) | 0.426164 | 0.680347 | 2.662642 | 1.251887 | 15.714286 | 5.459057 | 4.233700 | 5.956195 |
| history_mean512 (reused fixed result) | 0.316848 | 0.530450 | 0.546509 | 0.084200 | 17.142857 | 20.168067 | 1.006680 | 0.157057 |
| LEFT_native_same_roles (reused fixed result) | 0.571098 | 0.764667 | 0.000000 | 0.000000 | 11.428571 | 20.000000 | 1.618214 | 0.039610 |

## SMD28: per-seed entity macro, then mean ± sample SD across seeds

| Method | AP macro | AP median | ROC macro | Strict normal FPR (%) | C3 normal FPR (%) | C3 one-to-one event recall (%) | C3 event F1 (%) | C3 anomaly coverage (%) | C3 long steady (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batch | 0.466477 ± 0.002025 | 0.467518 ± 0.000971 | 0.855642 ± 0.004873 | 7.557921 ± 0.163539 | 6.817768 ± 0.164715 | 61.781341 ± 1.611753 | 26.259885 ± 0.058736 | 41.880185 ± 0.431477 | 24.381042 ± 0.528266 |
| reference | 0.459378 ± 0.009706 | 0.499777 ± 0.008128 | 0.857813 ± 0.002600 | 7.130079 ± 0.150746 | 6.445852 ± 0.212904 | 65.493217 ± 0.864972 | 25.186951 ± 0.540210 | 40.097039 ± 0.698182 | 22.080750 ± 0.480261 |

The table first averages eligible metrics over the same fixed SMD28 entities within each seed, then summarizes the three resulting seed values. It does not concatenate84 entity-seed sequences or treat them as independent domains. Entity medians are summarized separately from macros; noneligible long-event metrics stay NA.

## Per-seed PSM detection counts

| Seed | Method | AP | ROC | C3 any-hit | C3 one-to-one matches | C3 longest missed span |
|---|---|---:|---:|---:|---:|---:|
| 2025 | batch | 0.267263 | 0.390799 | 31 | 28 | 3787 |
| 2025 | reference | 0.324151 | 0.509528 | 30 | 30 | 3788 |
| 2026 | batch | 0.272991 | 0.404109 | 28 | 26 | 3787 |
| 2026 | reference | 0.321070 | 0.496516 | 31 | 28 | 3787 |
| 2027 | batch | 0.285298 | 0.420350 | 35 | 33 | 3665 |
| 2027 | reference | 0.315729 | 0.484292 | 31 | 31 | 3787 |

## Fixed paired AP differences

| Scope | Reference minus comparator | Seed2025 | Seed2026 | Seed2027 | Mean ± sample SD |
|---|---|---:|---:|---:|---:|
| PSM | reference − batch | 0.056887 | 0.048080 | 0.030431 | 0.045133 ± 0.013472 |
| SMD28 entity macro | reference − batch | -0.009683 | -0.014735 | 0.003120 | -0.007099 ± 0.009204 |
| PSM | reference − B_fit_only | 0.021139 | 0.018059 | 0.012717 | 0.017305 ± 0.004261 |
| SMD28 entity macro | reference − B_fit_only | -0.014660 | -0.023727 | -0.004328 | -0.014238 ± 0.009706 |

Per-machine, per-seed paired deltas and signs remain in the complete CSVs, including all unfavorable cases. Reference-minus-batch is the matched training comparison; reference minus official B is a zero-update reference, not a same-training-budget control.

## Scope, calibration and evidence limits

- Every seed retains its own selected checkpoint and CAL q99/higher threshold with strict greater-than. All new and official TimeRCD scores use archived abnormal-minus-normal head logits. No probability inversion, per-machine score selection, threshold refitting on QUERY or cross-axis subtraction occurs.
- Common-axis new-arm tables and method-own full-axis tables are separate. External baseline scores/metrics are reused once and do not receive an artificial three-seed variance. Compatible LEFT covers PSM and only3/28 SMD machines; missing LEFT results remain NA.
- Strict and C3 metrics, any-hit and one-to-one events, false-alarm points/fragments and sustained coverage remain distinct. First-hit positions are offline offsets, not verified online delays.
- Sample SD uses ddof1 across three training seeds. It describes training randomness on the same exposed PSM/SMD data and events, not unseen-domain confirmation. No pooled-point significance or84-independent-domain inference is made.
- Original2025 positive and negative findings are preserved. No model, normalization recipe, mask, selection rule or inference score is changed in this repetition task.

## Cost and provenance

parameters_and_compute.csv distinguishes reused2025 accounting from newly executed2026/2027 training/scoring. Source datasets and raw banks, weights, private host paths and raw receipts stay outside Git. Safe hashes identify accepted2025 metrics and all public exports. No follow-up candidate or automatic training is authorized by this publication.

## Fixed repetition outcome

**The PSM ranking learning increment reproduces across all three fixed seeds; a broadly stable detection or sustained-coverage mechanism does not.** Preserve the current local recipe as bounded development evidence, without automatically changing it or expanding the experiment matrix.

PSM reference AP is 0.324151 / 0.321070 / 0.315729, versus batch 0.267263 / 0.272991 / 0.285298. The paired AP increments are +0.056887 / +0.048080 / +0.030431, with mean +0.045133 and sample SD0.013472. Relative to the original official B result, all three AP differences remain positive (mean +0.017305, sample SD0.004261). This is a repeated source-learning effect on PSM under the fixed recipe, not an independent-data confirmation or a new architecture contribution. Compatible TimeRadar and LEFT still have higher PSM AP under their disclosed information permissions.

PSM C3 event F1 improves in all three matched seed comparisons. However, event discovery and false-alarm/coverage costs are not uniform: any-hit differences are -1 / +3 / -4, one-to-one event-match differences +2 / +2 / -2, seed2026 C3 FPR is slightly higher with reference (1.3949% vs1.3377%), and seed2027 anomaly coverage is lower (5.4285% vs6.5434%). Mean long-steady coverage remains low at1.5782% for reference versus0.6350% for batch. The longest missed event span is not consistently reduced. The repeated ranking and F1 evidence does not establish a sustained-detection solution.

SMD28 paired macro AP differences are -0.009683 / -0.014735 / +0.003120: mean -0.007099, sample SD0.009204. Per-seed AP positive/negative entity counts are14/14,10/18,14/14. Five machines have positive differences in all three seeds, eight negative in all three, and fifteen mixed signs. These are descriptions of the same28 exposed entities, not84 independent data domains or a basis for selecting favorable machines.

SMD reference macro AP is0.459378±0.009706 versus batch0.466477±0.002025. Reference is below the original official B AP0.473616 in all three seeds. C3 FPR decreases in each seed (three-seed mean6.4459% vs6.8178%), but C3 anomaly coverage (40.0970% vs41.8802%), long-steady coverage (22.0807% vs24.3810%) and event F1 (0.251870 vs0.262599) are lower in each paired seed. The false-alarm decrease must not be described as a free detection benefit.

The fixed repetition task is complete. PSM positive evidence is preserved, all SMD and unfavorable seed results remain visible, and no broad benefit, contamination robustness or persistent-anomaly resolution is claimed. No additional seed, mask, normalization variant, scoring rule or automatic new candidate is started.

## Completed computation and primary verification

New2026/2027 computation used0.739157647 GPUh: seed2026 training0.105676820 plus scoring0.265464823; seed2027 training0.102506732 plus scoring0.265509272. No new GPU profile was run; the accepted profile was reused. Seed2025's previously recorded0.344096626 GPUh is shown separately as historical accounting and is not charged again. Four new trainings each completed1536 updates with4608 sample exposures from the same immutable source instances. The four selected checkpoints remain source-selected; source SELECT masks are unchanged.

A GPU occupancy-check block after112 successful scoring jobs was retained. Only four never-launched seed2027 jobs were resumed once on available devices, without rerunning successful work. The two terminal seals each retain all58 score jobs /116 CAL-QUERY banks and their actual per-seed costs.

Primary verification checked all232 new raw-bank hashes and recomputed all232 new native/common metric rows directly from archived head-logit differences, original CAL q99/higher thresholds, AP/ROC and strict/C3 TP/FP. It also verified the three-seed paired means and sample SDs. All passed; seed2025 metrics were reused without reevaluation. This is primary-agent verification, not an external independent audit. The executed aggregator and evaluator source snapshots retain their execution identities.
