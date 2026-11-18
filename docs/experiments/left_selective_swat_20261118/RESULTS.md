# Fixed SWaT LEFT-native loss retention: B/C matched experiment

Four new arms retain the complete native model, per-window reconstruction objectives, initial checkpoint, original training plan and native endpoint scorer. The sole update factor is whether103 of128 window losses are selected by lowest loss or a fixed random subset. The original full128 still drive forward computation and native prototype updates. No training label, donor event ID or QUERY label enters the loss selector.

## Main own-CAL comparison

| Method | AP | ROC | Strict point recall (%) | Strict normal FPR (%) | C3 normal FPR (%) | C3 any-hit | C3 one-to-one | C3 event F1 (%) | C3 anomaly point coverage (%) | Long9 steady macro (%) | Long9 steady pooled (%) | C3 longest miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U_B | 0.234456 | 0.696179 | 0.856060 | 1.258805 | 1.100112 | 11 | 11 | 9.361702 | 0.780710 | 1.250793 | 0.260517 | 31838 |
| B_random | 0.208111 | 0.672568 | 0.874898 | 1.238016 | 1.084173 | 11 | 11 | 9.322034 | 0.797455 | 1.249537 | 0.250200 | 31840 |
| B_trim | 0.131428 | 0.476023 | 1.601189 | 1.687069 | 1.522486 | 11 | 11 | 8.270677 | 1.521653 | 1.463181 | 0.371431 | 31863 |
| U_C | 0.164798 | 0.515301 | 1.404441 | 1.347854 | 1.193318 | 11 | 11 | 9.205021 | 1.339557 | 1.254853 | 0.198612 | 31864 |
| C_random | 0.186609 | 0.558069 | 1.396069 | 1.329490 | 1.176340 | 11 | 11 | 9.322034 | 1.333278 | 1.231035 | 0.193454 | 31864 |
| C_trim | 0.130900 | 0.455771 | 1.789564 | 1.511051 | 1.348893 | 11 | 11 | 8.730159 | 1.705842 | 1.326979 | 0.410121 | 31828 |
| noU | 0.151321 | 0.546587 | 1.471419 | 1.380424 | 1.213761 | 11 | 11 | 8.730159 | 1.387697 | 1.167543 | 0.257938 | 31842 |
| U_A | 0.167294 | 0.521121 | 1.400255 | 1.335726 | 1.181191 | 11 | 11 | 9.243697 | 1.333278 | 1.242944 | 0.196033 | 31864 |
| HistoryMean512 | 0.087649 | 0.227811 | 0.251167 | 1.207871 | 1.125406 | 5 | 5 | 6.896552 | 0.230236 | 1.534654 | 0.123810 | 35900 |
| TimeRadar | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| TSAD-C | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

U_B/U_C are the corresponding ordinary full128-loss updates; B/C are two matched training conditions on one fixed QUERY, not independent repeats. U_A, noU and History Mean512 are historical context controls. Only the four new arms are evaluated here; accepted historical metrics are hash-reused without reevaluation.

## Matched update differences

| Contrast | AP difference | ROC difference | C3 FPR difference | C3 point coverage difference | Long9 macro difference |
|---|---:|---:|---:|---:|---:|
| B_trim-minus-ordinary | -0.103028 | -0.220156 | 0.004224 | 0.007409 | 0.002124 |
| B_trim-minus-random | -0.076684 | -0.196545 | 0.004383 | 0.007242 | 0.002136 |
| C_trim-minus-ordinary | -0.033898 | -0.059530 | 0.001556 | 0.003663 | 0.000721 |
| C_trim-minus-random | -0.055709 | -0.102298 | 0.001726 | 0.003726 | 0.000959 |
| ordinary_B-minus-C | 0.069658 | 0.180878 | -0.000932 | -0.005588 | -0.000041 |
| trim_B-minus-C | 0.000528 | 0.020252 | 0.001736 | -0.001842 | 0.001362 |
| random_B-minus-C | 0.021502 | 0.114499 | -0.000922 | -0.005358 | 0.000185 |
| B-C_trim_difference-minus-ordinary | -0.069130 | -0.160626 | 0.002668 | 0.003747 | 0.001403 |
| B-C_trim_difference-minus-random | -0.020975 | -0.094247 | 0.002658 | 0.003516 | 0.001177 |

Differences use raw metric fractions. A positive false-alarm difference is a cost, not a gain. B-minus-C and difference-in-differences are descriptive condition contrasts with the same source/checkpoint and QUERY, not independent-run causal or significance evidence.

## Post-training retention audit

| Method | Replacement category | Offered donor slots | Donor loss-retained (%) | High-loss nominal proxy slots | Proxy loss-retained (%) |
|---|---|---:|---:|---:|---:|
| U_B | replacement_attack_donor | 25600 | 100.000000 | 48817 | 100.000000 |
| B_random | replacement_attack_donor | 25600 | 80.183594 | 48817 | 80.471967 |
| B_trim | replacement_attack_donor | 25600 | 53.617188 | 48817 | 23.815474 |
| U_C | replacement_nominal_normal_donor | 25600 | 100.000000 | 50000 | 100.000000 |
| C_random | replacement_nominal_normal_donor | 25600 | 80.183594 | 50000 | 80.478000 |
| C_trim | replacement_nominal_normal_donor | 25600 | 95.148438 | 50000 | 3.644000 |

The high-loss nominal proxy is fixed before reading the new outputs: the top25 of128 slots per update in the accepted ordinary-U-C native loss archive; exact ties favor earlier slot identity. For B, attack-replacement slots are excluded from the nominal group; for C, the paired nominal-normal donor slots are included. The full CSV separates background and normal-donor subsets. This is relative training-loss difficulty, not authenticated plant mode, health or contamination truth. Labels are joined only after all four trainings and all eight score banks seal, never returned to training.

B attack donors contain13 independent original events, but their overlapping window presentations are repeated exposures, not extra events. C donors are label-normal gaps in the same earlier DONOR interval, not new FIT or certified healthy records. All nominal background descriptions retain this limitation.

The offered replacement fraction is10%. The new gradient subset retains103/128=80.46875% (206000 of256000 presentations), not exactly80%. supervision_exposure.csv reports offered window replacement, time-label target support, retained positive-time support and the unknown physical affected-channel fraction separately. All full256000 windows still participate in forward/native prototype state: dropping their explicit loss is not proof that the training state was cleaned. Retained loss mass is not gradient share.

## Frozen metric and external permissions

- Every method uses its own original512 CAL points linspace(446173,494999,512,int64), q99/higher and strict greater-than. Only own_CAL_q99 is published. No fixed-noU threshold extension, QUERY calibration or score search is introduced.
- QUERY stays608535:944919, with22 complete events and9 long events strictly>512; steady coverage starts at original event start+512. The original strict/C3, chronological one-to-one matching and normal-fragment definitions are reused. New native scores are actual causal endpoints; sample offsets are not converted into unverified physical time.
- H512 has extra CAL fitting for per-channel median residual scales as well as its threshold. The complete accepted seven frozen TimeRCD rows are retained separately in frozen_TimeRCD_external.csv: source-only training and offline5000 context differ from these LEFT-native target-normal training/endpoint permissions. No cross-permission mechanism attribution follows. Compatible TimeRadar banks remain unavailable; TSAD-C remains NA under its completed official compatibility finding.

## Completed resource use

Actual additional GPU time is 0.282923474 GPUh, including the finite CUDA equivalence gate and four training/four scoring lifecycles. Each new arm has2000 updates and uses only endpoint2000; SELECT is diagnostic. Source weights, old U training, old scores and historical metrics are not rerun or overwritten. Raw arrays, model weights and private host paths are withheld from public exports.

## Fixed scientific decision: limited NO-GO

The fixed lowest-loss truncation recipe does not improve the complete detector over matching ordinary U or random retention. B_trim AP0.131428 is below U_B0.234456 and B_random0.208111; C_trim AP0.130900 is below U_C0.164798 and C_random0.186609. ROC decreases in both conditions as well. The C condition has no injected attack-label windows, so its degradation cannot be attributed to anomalous donor pollution.

C3 normal FPR increases with truncation in both comparisons: B_trim1.5225% versus U_B1.1001% /B_random1.0842%, and C_trim1.3489% versus U_C1.1933% /C_random1.1763%. All six B/C ordinary/random/trim methods hit11 of22 events with the original one-to-one rule; no event-discovery increment is obtained. Truncation increases some anomaly-point and long-steady coverage, but this is accompanied by more false alarms and lower event F1, with longest misses still around31828–31863 points. It is not an independently useful sustained-detection gain.

Post-training donor annotation shows that B_trim reduces loss participation of attack-event windows to53.6172% versus random80.1836%. However, it also retains only23.8155% of high-loss nominal background slots under the fixed old-U-C audit proxy, versus random80.4720%. In C, the corresponding high-loss nominal group retains only3.6440% versus random80.4780%; meanwhile C's ordinary nominal-normal donor windows retain95.1484%. These statistics reveal strong suppression of difficult nominal training slots as well as partial attack-window suppression. They do not identify health or physical contamination, and they do not isolate a unique causal explanation for every score change. Lower attack-window retention alone did not become better detection.

B_trim's retained attack-time support is6.6631% of retained target windows/times, versus B_random9.9646% and ordinary U_B10%. The true affected-channel scalar fraction remains unknown. Full-batch native forward/prototype state still sees all offered windows, including those without explicit retained losses. This experiment therefore does not certify cleaned training or a general robust-learning mechanism.

The candidate is stopped at this fixed recipe. No truncation fraction, event subset, threshold, loss coefficient, retention schedule or replacement dose is changed, and no follow-up candidate is started. The broader mixed-unlabeled research goal and all historical results remain intact.

## Independent bounded verification

The designated independent auditor completed the raw recomputation and checked the eight new own-CAL stream rows (152 compared table values); all matched. No extra threshold track is published. This is a same-model-family, role-separated review, not external blinded validation. The audit receipt is retained locally and referenced by its safe hash. The actual evaluator snapshot matches the source hash recorded during evaluation; later source additions only implement safe publication and do not change the numerical run.
