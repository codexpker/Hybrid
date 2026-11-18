# Fixed additional-anomaly exposure under native LEFT source replay

## Scope and frozen implementation

This is a fixed seed2025 controlled comparison of ordinary target-only TT and source replay R. Both start from each domain's accepted native LEFT checkpoint, scaler, prototypes and actual running step. The native network, complete loss, Adam1e-4, batch256,2000 updates, fixed endpoint, whole-window192 inference and per-model mixed-CAL q99/higher/strict-greater rule are unchanged. No P/V preservation or teacher is used. P/V and earlier stopped candidates retain their original results.

A denotes the accepted natural-mixture runs and is reused, together with noU/SS. Only TT-B,TT-C,R-B,R-C were trained in each of SWaT and PSM, eight new jobs in total. B and C replace the same25,600 of256,000 shared-T exposure slots. This is10% of T and5% of512,000 total window presentations. TT/R receive identical T windows, slots and corresponding donor bank within a condition; T-prime and S retain the original plans. Donor random streams are isolated from the original sampling and model RNG. Construction uses SeedSequence([2025,20261118,10]).spawn(3) for slots,B,C, respectively. Donor draws are uniform over legal stride-one starts with replacement, with no score-based filter.

The constructor may read labels only within current U-FIT to identify complete192-point all-positive B and all-zero C windows. Training reads normalized numeric windows and slot-to-row maps, without donor starts, event IDs or labels. These identities and labels remain in separate audit-only artifacts. This is label-assisted stress-condition construction with label-free updates, not an algorithm discovering contamination labels. Event labels indicate annotated times, not proof that every sensor is physically corrupted. C denotes nominal normal annotation, not independently certified physical health.

## Donor eligibility and actual reuse

| Domain | Current U-FIT | Legal B windows | B event intervals | Legal C windows | Used unique B / max repeat | Used unique C / max repeat |
|---|---|---:|---:|---:|---:|---:|
| SWAT | [495000,585000) | 3468 | 9 | 82903 | 3466 / 19 | 21982 / 4 |
| PSM | [0,17568) | 649 | 5 | 12852 | 649 / 63 | 11080 / 10 |

Each condition presents25,600 donors. The rest are repeated presentations, not independent events. SWaT repeats22,134 B presentations and3,618 C presentations; PSM repeats24,951 B and14,520 C presentations. PSM U-FIT has22 positive label runs, including short and boundary-clipped runs; only five support complete all-positive192 windows. Old SWaT donor libraries were not reused, and no CAL/QUERY window was used to fill the donor pool.

## Effective exposure and interpretation

All rows contain512,000 whole windows,98,304,000 observed time-position presentations, no padding, and either31 or25 channels. The table counts repeated event-labeled time positions, not independent samples or physically affected-channel truth. S is counted as nominal normal under the original author source-FIT role; it is not newly certified from target labels.

| Domain | Arm | A event-time exposure | B event-time exposure | C event-time exposure | B-A, percentage points | B-C, percentage points |
|---|---|---:|---:|---:|---:|---:|
| SWAT | TT | 5694824/98304000 (5.7931%) | 10330422/98304000 (10.5086%) | 5415222/98304000 (5.5086%) | +4.7156 | +5.0000 |
| SWAT | R | 2877932/98304000 (2.9276%) | 7513530/98304000 (7.6432%) | 2598330/98304000 (2.6432%) | +4.7156 | +5.0000 |
| PSM | TT | 10096583/98304000 (10.2708%) | 14515839/98304000 (14.7663%) | 9600639/98304000 (9.7663%) | +4.4955 | +5.0000 |
| PSM | R | 5036405/98304000 (5.1233%) | 9455661/98304000 (9.6188%) | 4540461/98304000 (4.6188%) | +4.4955 | +5.0000 |

B-C adds exactly4,915,200 annotated time-position presentations in both arms. B-A is lower because replaced T slots already have some positive annotation. TT retains natural target exposure in T-prime while R uses source S; total background composition therefore differs even with identical controlled replacement.

The actual float32 values changed by B/C account for4.1064%/4.0723% of all presented scalar values in SWaT and4.9397%/4.9191% in PSM. This is numerical input change, not physical anomaly prevalence. Native time-frequency transforms and multiscale targets distribute information across positions. The reconstruction objective has no padding/excluded-window mask in these runs, but event-time fractions are not exact shares of anomalous gradients or of every loss term. Detailed counts and donor repetitions are in [SWAT_exposure.json](SWAT_exposure.json) and [PSM_exposure.json](PSM_exposure.json).

## Evaluation permissions

SWaT retains QUERY[608535,944856),336321 points,22 complete events and nine long events. PSM retains QUERY[22656,87744),65088 points,48 complete events and five long events. Boundary-truncated PSM events remain in observed point metrics but are excluded from complete-event metrics. Strict point recall and complete-event coverage thus have different denominators in PSM.

All models use the same512 legal CAL locations in their own score units, q99/higher/strict-greater. SWaT has38/512 attack-labeled CAL selections; PSM has512/512 and its complete CAL role is all attack-labeled. The original roles and thresholds are retained without label cleanup. They define empirical operating points, not a certified1% normal false-alarm budget. Whole-window scores become available at the192-point window end; even confirmed-three-point metrics are offline retrospective.

AP is average precision, distinct from trapezoidal PR area. VUS uses the frozen author metricor.RangeAUC_volume_opt implementation, windowSize100 and thre250. The common point axes, normal false-positive counts, chronological one-to-one event matching, any-hit counts, long steady-state definition (event length>512, after original start+512) and censored offsets retain the accepted evaluator definitions.

## Absolute results

All eight new runs and16 banks completed successfully, seed2025 only. A, noU and SS values below reuse accepted artifacts. R is established source replay, not a newly named Hybrid architecture.

### SWAT

| Arm | Condition | AP | ROC-AUC | VUS-PR | VUS-ROC | Strict FP | Strict FPR | Strict recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| noU | reference | 0.607560 | 0.842267 | 0.606345 | 0.829634 | 5073 | 1.758% | 60.960% |
| SS | reference | 0.601081 | 0.833884 | 0.595455 | 0.813485 | 5371 | 1.861% | 61.605% |
| TT | A | 0.610342 | 0.884582 | 0.618301 | 0.881642 | 5344 | 1.852% | 54.587% |
| TT | B | 0.577183 | 0.885519 | 0.582214 | 0.881045 | 5467 | 1.895% | 10.829% |
| TT | C | 0.599753 | 0.885894 | 0.603000 | 0.880903 | 5499 | 1.906% | 49.358% |
| R | A | 0.629347 | 0.885512 | 0.639265 | 0.884546 | 5143 | 1.782% | 58.327% |
| R | B | 0.624099 | 0.886953 | 0.631569 | 0.884275 | 5234 | 1.814% | 56.914% |
| R | C | 0.622314 | 0.880514 | 0.631736 | 0.878923 | 5095 | 1.766% | 58.614% |

| Arm | Condition | C3 FP | C3 any-hit | C3 1:1 matched | C3 1:1 P/R/F1 | Complete-event point coverage | Long steady pooled / macro | Longest miss | First hit / available median offset |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| noU | reference | 4324 | 15/22 | 15/22 | 0.0440/0.6818/0.0826 | 60.789% | 73.203% / 10.949% | 3074 | 40.0/188.0 |
| SS | reference | 4543 | 15/22 | 15/22 | 0.0404/0.6818/0.0763 | 61.419% | 74.028% / 11.160% | 3074 | 40.0/188.0 |
| TT | A | 4530 | 15/22 | 15/22 | 0.0393/0.6818/0.0743 | 54.369% | 65.333% / 10.102% | 5970 | 40.0/188.0 |
| TT | B | 4646 | 15/22 | 15/22 | 0.0235/0.6818/0.0455 | 8.835% | 9.216% / 3.450% | 14836 | 40.0/188.0 |
| TT | C | 4686 | 15/22 | 15/22 | 0.0260/0.6818/0.0502 | 48.186% | 57.716% / 9.174% | 6547 | 40.0/188.0 |
| R | A | 4362 | 15/22 | 15/22 | 0.0390/0.6818/0.0737 | 57.948% | 69.731% / 10.637% | 4434 | 40.0/188.0 |
| R | B | 4446 | 15/22 | 15/22 | 0.0402/0.6818/0.0759 | 56.697% | 68.181% / 10.448% | 5059 | 40.0/188.0 |
| R | C | 4322 | 15/22 | 15/22 | 0.0388/0.6818/0.0733 | 58.225% | 70.082% / 10.680% | 4431 | 40.0/188.0 |

### PSM

| Arm | Condition | AP | ROC-AUC | VUS-PR | VUS-ROC | Strict FP | Strict FPR | Strict recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| noU | reference | 0.583688 | 0.771345 | 0.575447 | 0.775647 | 7381 | 15.602% | 43.613% |
| SS | reference | 0.575764 | 0.762970 | 0.571271 | 0.770729 | 7033 | 14.866% | 43.495% |
| TT | A | 0.572727 | 0.733239 | 0.562617 | 0.740466 | 12006 | 25.378% | 63.665% |
| TT | B | 0.579550 | 0.740236 | 0.568306 | 0.746560 | 11019 | 23.292% | 60.341% |
| TT | C | 0.571787 | 0.731490 | 0.561991 | 0.739135 | 12061 | 25.494% | 63.918% |
| R | A | 0.587891 | 0.748194 | 0.586017 | 0.761687 | 5294 | 11.190% | 42.314% |
| R | B | 0.593225 | 0.756182 | 0.588131 | 0.767302 | 5559 | 11.750% | 42.753% |
| R | C | 0.588581 | 0.748001 | 0.586964 | 0.761680 | 5341 | 11.290% | 42.680% |

| Arm | Condition | C3 FP | C3 any-hit | C3 1:1 matched | C3 1:1 P/R/F1 | Complete-event point coverage | Long steady pooled / macro | Longest miss | First hit / available median offset |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| noU | reference | 6721 | 21/48 | 18/48 | 0.0604/0.3750/0.1040 | 65.710% | 83.902% / 65.355% | 399 | 2.0/82.0 |
| SS | reference | 6324 | 20/48 | 17/48 | 0.0525/0.3542/0.0914 | 65.293% | 83.789% / 65.152% | 399 | 2.0/86.0 |
| TT | A | 11388 | 26/48 | 21/48 | 0.0837/0.4375/0.1405 | 80.955% | 97.350% / 95.447% | 399 | 2.0/86.0 |
| TT | B | 10367 | 22/48 | 21/48 | 0.0808/0.4375/0.1364 | 78.596% | 95.057% / 91.339% | 399 | 2.0/94.0 |
| TT | C | 11462 | 27/48 | 21/48 | 0.0854/0.4375/0.1429 | 81.173% | 97.593% / 95.866% | 399 | 2.0/90.0 |
| R | A | 4578 | 17/48 | 17/48 | 0.0512/0.3542/0.0895 | 61.901% | 81.772% / 59.734% | 399 | 3.0/78.0 |
| R | B | 4832 | 18/48 | 18/48 | 0.0529/0.3750/0.0928 | 62.147% | 81.935% / 61.108% | 399 | 3.0/80.0 |
| R | C | 4615 | 17/48 | 17/48 | 0.0523/0.3542/0.0912 | 62.128% | 82.033% / 60.284% | 399 | 3.0/78.0 |

Offsets are in original sample points and summarize hit events only; misses are right-censored separately in the full CSV. They describe retrospective localization and when the full input became available, not real-time fault discovery. Longest misses and coverage include missed portions under the fixed complete-event rule. One-to-one metrics use chronological first-unmatched-overlap matching; any-hit is separately reported even when the displayed counts happen to coincide.

## Within-arm replacement response and paired interaction

Values below are absolute differences on the same domain and scoring axis. AP/ROC/coverage use percentage points, FP uses points. Higher is better for ranking/coverage, lower for FP/miss length; a positive I_FP is an adverse interaction.

| Domain | Arm | Contrast | AP, pp | ROC, pp | Strict FP | Strict recall, pp | C3 complete-event coverage, pp | C3 any-hit |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SWAT | TT | B-C | -2.2570 | -0.0374 | -32 | -38.5290 | -39.3516 | +0 |
| SWAT | TT | B-A | -3.3160 | +0.0937 | +123 | -43.7575 | -45.5345 | +0 |
| SWAT | TT | C-A | -1.0590 | +0.1311 | +155 | -5.2285 | -6.1829 | +0 |
| SWAT | R | B-C | +0.1785 | +0.6439 | +139 | -1.6996 | -1.5279 | +0 |
| SWAT | R | B-A | -0.5248 | +0.1441 | +91 | -1.4128 | -1.2516 | +0 |
| SWAT | R | C-A | -0.7033 | -0.4998 | -48 | +0.2867 | +0.2763 | +0 |
| PSM | TT | B-C | +0.7764 | +0.8746 | -1042 | -3.5773 | -2.5772 | -5 |
| PSM | TT | B-A | +0.6824 | +0.6997 | -987 | -3.3241 | -2.3593 | -4 |
| PSM | TT | C-A | -0.0940 | -0.1749 | +55 | +0.2531 | +0.2179 | +1 |
| PSM | R | B-C | +0.4643 | +0.8181 | +218 | +0.0731 | +0.0190 | +1 |
| PSM | R | B-A | +0.5333 | +0.7988 | +265 | +0.4387 | +0.2464 | +1 |
| PSM | R | C-A | +0.0690 | -0.0193 | +47 | +0.3656 | +0.2274 | +0 |

For each metric M, I_M=(M_RB-M_RC)-(M_TTB-M_TTC).

| Domain | I_AP, pp | I_ROC, pp | I_strict FP | I_strict FPR, pp | I_C3 FP | I_C3 coverage, pp | I_C3 long pooled / macro, pp | I_C3 any-hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SWAT | +2.4355 | +0.6814 | +171 | +0.0593 | +164 | +37.8236 | +46.5991 / +5.4931 | +0 |
| PSM | -0.3120 | -0.0565 | +1260 | +2.6633 | +1312 | +2.5962 | +2.4390 / +5.3511 | +6 |

All strict and C3 numerical differences, including VUS, event matching, false-alarm fragments, long macro/pooled coverage, offsets and misses, are preserved in [contrasts.csv](contrasts.csv), [interaction.csv](interaction.csv) and [direct R-TT comparisons](between_arm.csv). No across-axis or cross-domain subtraction is used.

## Technical decision

SWaT shows a bounded differential replacement response. TT B-C AP is-2.2570pp, C3 complete-event coverage-39.3516pp and long steady pooled coverage-48.5001pp. R B-C AP is+0.1785pp while C3 coverage is-1.5280pp and long pooled coverage-1.9010pp. The resulting positive AP/coverage interactions do not mean every R outcome improves: R-B increases strict false positives by139 versus R-C, C3 false positives by124, and both retain15/22 discovered events. Relative to R-A, R-B AP and coverage decrease. Its sustained coverage remains below noU/SS. Thus the fixed stress condition supplies task-specific evidence of retention relative to TT, not overall operating superiority. R-B long steady macro coverage is only10.448%, despite pooled long coverage68.181%; persistent detection is not solved.

PSM does not show an AP-specific contamination penalty: B-C improves AP for both R(+0.4643pp) and TT(+0.7764pp), yielding I_AP=-0.3120pp. PSM I_C3 coverage is+2.5962pp and I_any-hit is+6, but R-B still discovers18/48 versus TT-B22/48, with lower complete-event coverage(62.147% versus78.596%). Its positive any-hit interaction partly reflects TT's C-to-B any-hit loss, not more absolute event discovery by R. TT-B and TT-C both have21 one-to-one matches despite22 versus27 any-hit events; R-B/C have18/17 one-to-one matches. Thus I_any-hit=+6 while I_matched_count=+1. These event notions are not interchangeable. The strict-FP interaction is+1260 points and C3-FP interaction+1312, both adverse. Mixed all-attack CAL also limits interpretation of thresholded effects.

R-B ranks better than TT-B on both domains under this seed, but the absolute coverage and false-positive trade-offs are domain dependent. This is one seed with the same trained initial state per domain and repeated donors from few labeled event intervals. The previous A repetitions do not provide B/C replication or independent event-level uncertainty. Controls remove changes in nominal replacement count and ordinary normal replacement, but not every difference in donor state/content. Thresholded responses include each model's own fixed CAL response as well as score changes. They cannot by themselves identify a unique training failure or demonstrate general contamination robustness.

The established LEFT+source-replay control remains the current basis. The experiment adds evidence about its scope and cost; mature replay or these stress conditions are not renamed as an original Hybrid contribution. There is no automatic dose escalation, seed expansion, new loss, teacher, weighting or follow-up variant. P/V remains stopped.

## Reproduction, tests and resources

Source implementation was published before results at5d585333a6170f38995e9c48943160296bc4a639. Fixed task snapshots are [SWaT](../../../configs/left-exposure-swat-20261118.json) and [PSM](../../../configs/left-exposure-psm-20261118.json). The versioned preparation CLI reads the original pinned preservation-task files only as asset templates and explicitly removes preservation; the resulting exact task snapshots above contain no P/V loss. Training uses the native worker with numeric-only exposure bundles. The incremental test module checks full-batch identity without intervention, untouched second halves/unselected entries and Python/NumPy/Torch RNG, plus complete-window donor eligibility. Three tests passed; the unrelated anyio pytest deprecation warning is retained. No full-project re-audit was performed.

Example entry points, using a fresh output directory and original external workspace assets:

```text
python -m hybrid_tsad.left_exposure_run prepare --workspace WORKSPACE --output NEW_RAW --configs hybrid_ad/configs
python -m hybrid_tsad.left_exposure_run queue --workspace WORKSPACE --output NEW_RAW --gpus 0 1 2 3
python -m hybrid_tsad.left_exposure_evaluation --workspace WORKSPACE --root NEW_RAW --out NEW_METRICS --workers 4
```

These commands document the completed run; they do not authorize duplicate execution. Local raw identities, optimizer/endpoint/trace/score artifacts and original OS/cleanup receipts are retained. Eight formal jobs exited0 without retries or failed GPU jobs, using four A40 cards in two waves. Per-job device ordinals, UTC intervals, peaks and exit codes are in [job_costs.csv](job_costs.csv). Aggregate measured occupancy is0.5849222277 GPU hours including startup/readout/cleanup; peak allocated training memory is2.908GiB and readout0.466GiB. Main CPU evaluation took47.672 seconds; old accepted A/noU/SS VUS and metrics were reused. No new normal initialization or external model was trained.

The existing [paper draft](../../../manuscript/next_generation_20261115/draft.md) and [method definition](../../../manuscript/next_generation_20261115/method_definition.md) now distinguish current native LEFT/replay comparisons from the stopped density, TimeRadar adaptation and P/V historical designs. Original experiment snapshots and a pre-existing user edit remain preserved.

## Bounded independent acceptance

An implementation-independent agent from the same model family verified original U-FIT labels, exact replacement/donor random streams, all four normalized donor banks, source/scaler/initial-state identities, actual training exposure, eight endpoints and sixteen score banks. All550,502,400 numeric donor scalars match independent reconstruction from original inputs and frozen scalers exactly. New raw AP/ROC/VUS, strict/C3 measures, any-hit, chronological one-to-one matching, long coverage and first-hit/availability offsets were independently recomputed. Main JSON1624 numeric cells and CSV1592 numeric cells agree exactly, maximum absolute difference0. Independent CPU wall time was145.765 seconds. This is bounded same-family role independence, not external or blind peer review. No additional training, inference or repeated historical VUS was used for review. See [verification_summary.json](verification_summary.json).

Source/data roles remain label-assisted construction followed by unlabeled training. Historical results and main/review snapshots are retained. The current R base remains a mature learning control, with this limited SWaT pressure-response positive and the PSM limitations preserved. No new method follows automatically.
