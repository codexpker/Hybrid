# Contiguous holdout reconstruction with frozen history

The [protocol](PROTOCOL.md) fixes the information available to each prediction, all four masks, the source and normal-target training budgets, and the AP trigger for evaluating all SMD28. The earlier architecture retains its [complete evidence](../left_timesfm_hidden_20261126/RESULTS.md).

## Frozen bypass of the previous model

Use the same 32 archived normal SOURCE SELECT windows in each domain for history injection and exact bypass. The bypass removes the hook's residual addition and keeps all parameters, prototypes and the native step fixed. The normal objective is the complete native LEFT validation loss. Score differences aggregate the native per-channel output over the same samples, times and channels.

| Domain | Injected residual / native token RMS | Normal objective on | Normal objective bypassed | Score RMS change | Maximum absolute score change |
|---|---:|---:|---:|---:|---:|
| PSM | 0.082031 | 1.121314 | 1.123692 | 0.026496 | 0.119900 |
| SWaT | 0.092530 | 0.351536 | 0.356011 | 0.076454 | 0.403569 |

The learned history path changes both the normal objective and output scores in this fixed panel. The previously logged gate trajectories remain available for [PSM](../left_timesfm_hidden_20261126/source/PSM/CURVES.json) and [SWaT](../left_timesfm_hidden_20261126/source/SWAT/CURVES.json).

## Information separation

For each arm in each domain, every mask passed a real normal-input check. With a fixed visible input, history and model state, replacing the hidden truth shifted the loss while the temporal, inverse-frequency and full-rate multiscale prediction tensors remained bitwise equal. The conditioner in the two history arms had a nonzero gate for this engineering check and received finite, positive projection gradients. The scientific jobs initialize that gate at the prescribed zero. Each normal training forward advanced the native step once. Use the [PSM](PSM_CHECKS.json) and [SWaT](SWAT_CHECKS.json) receipts for model and fixture identities.

## PSM and SWaT development results

The paired AP differences use the full legal TEST axis separately in each domain. A is the holdout detector alone, B adds official pretrained history, and C adds random frozen history with the same architecture.

| Domain | Stage | B AP | B minus A (pp) | B minus C (pp) | B minus matching unmasked LEFT (pp) |
|---|---|---:|---:|---:|---:|
| PSM | Z | 0.506734 | −0.057 | −0.017 | −1.954 |
| PSM | N | 0.541018 | +0.751 | +0.852 | −2.686 |
| SWaT | Z | 0.722993 | −0.151 | −0.044 | +0.029 |
| SWaT | N | 0.719539 | +0.018 | +0.023 | −0.105 |

Z uses the archived native matched-source continuation as its unmasked reference. N uses the new native LEFT continuation on the same normal windows, update budget and normal validation procedure as the three holdout arms. PSM N exceeds both holdout controls on all four ranking metrics and triggers the fixed SMD28 extension. The matched native N detector leads it on all four. SWaT N produces very small AP differences and mixed ROC/VUS changes.

For PSM, B has the lowest normal SELECT objective among the three holdout arms in both Z and N. Z nevertheless loses AP to both controls. In the same-weight earlier-history test, replacing the aligned 512-point history with a disjoint earlier history changes PSM N AP from 0.541626 to 0.541555 on the common eligible subset. On SWaT it changes N AP from 0.719552 to 0.719601. In SWaT Z, the earlier history has higher AP than aligned history on the subset (0.725028 versus 0.723006). These comparisons give little evidence that the development improvement requires the correctly corresponding past.

## Fixed controlled responses

The controlled panel uses the original 64 coupled autoregressive records and 16 disjoint public normal SELECT blocks per width. The public PSM backgrounds are nominally normal. An identical frozen history is supplied to each intervention and its paired control. Core, transition and full-window results are all retained. These records lack a second legal comparable history, so the non-overlap replacement does not apply.

The largest relative weak-change shift occurs in SWaT Z normal SELECT blocks, where B obtains core ROC-AUC 0.394531 versus 0.199219 for A and 0.136719 for C. All three rank that change below the fixed smooth control more often than above it. Across the two simulated widths, weak-change discrimination stays close to 0.5. PSM N synthetic relative-delay versus amplitude ROC-AUC is 0.073975 for B. SWaT N's synthetic delay-versus-synchronous value is 0.557617 for B and 0.560059 for C. Thus isolated relative changes do not establish a consistent weak-change or delay-specific advantage.

## Complete SMD28 evaluation

The fixed PSM N outcome triggered extension to all 28 38-variable machines. The SMD source fold uses TEP and HAI normal records. The new source and target jobs retain a single project seed. The following are equal-weight means of within-machine metrics on the complete internal axis.

| Stage | Complete method | AP | ROC-AUC | VUS-PR | VUS-ROC |
|---|---|---:|---:|---:|---:|
| Z | Holdout LEFT | 0.449466 | 0.849665 | 0.469733 | 0.904495 |
| Z | Holdout LEFT + frozen pretrained history | 0.456272 | 0.855641 | 0.484279 | 0.909122 |
| Z | Holdout LEFT + frozen random history | 0.450543 | 0.849601 | 0.472347 | 0.904741 |
| N | Holdout LEFT | 0.452147 | 0.847882 | 0.472982 | 0.901478 |
| N | Holdout LEFT + frozen pretrained history | 0.455440 | 0.849297 | 0.481181 | 0.903261 |
| N | Holdout LEFT + frozen random history | 0.453062 | 0.848925 | 0.474385 | 0.902173 |
| N | Matched-budget native LEFT | 0.499478 | 0.870980 | 0.543579 | 0.914086 |

| Stage | AP comparison | Mean (pp) | Median (pp) | Machine wins / ties / losses |
|---|---|---:|---:|---:|
| Z | Pretrained history vs no history | +0.681 | +0.313 | 19 / 0 / 9 |
| Z | Pretrained history vs random history | +0.573 | +0.408 | 21 / 0 / 7 |
| N | Pretrained history vs no history | +0.329 | +0.052 | 16 / 0 / 12 |
| N | Pretrained history vs random history | +0.238 | +0.128 | 16 / 0 / 12 |
| N | Pretrained history vs native LEFT | −4.404 | −1.638 | 7 / 0 / 21 |
| Z | Corresponding vs non-overlapping history | +0.012 | +0.020 | 15 / 0 / 13 |
| N | Corresponding vs non-overlapping history | −0.047 | −0.009 | 12 / 0 / 16 |

The last two rows use the restricted same-weight replacement axis. Paired inference drops the first three starts in each machine for this contrast only. The official encoder gives a modest gain over the matched holdout controls in SMD and all four macro measures move in the same direction. On the independent external intersection, B's frozen AP still trails the archived unmasked native continuation by 4.415 pp. Under matching normal training, the unmasked native reference remains ahead on every macro ranking metric. The B arm's AP itself decreases by 0.083 pp from Z to N on their shared full axis. These results retain both the within-holdout gain and the absolute detection cost.

## Deliverables and measured cost

The [complete method tables](evidence/summary/PUBLIC.md) separate Z and N, the complete internal axis and the external intersection, and include AP, ROC-AUC, VUS-PR and VUS-ROC. The [entity metric rows](evidence/summary/metrics.csv), [paired machine distributions](evidence/summary/paired_entities.csv), [paired summaries](evidence/summary/paired_summary.csv), [normal SELECT outcomes](evidence/summary/learning.csv), [all controlled comparisons](evidence/controlled/comparisons.csv) and [stage ledger](evidence/summary/costs.csv) retain all comparisons. Native full-window scores and holdout scores become available at window end. External methods retain their archived scaler, target training and context permissions. The fixed CAL panels and historical operation NA remain in force.

| Stage | Summed stage-wall seconds | Current/LEFT batch forwards | Frozen history encoder calls |
|---|---:|---:|---:|
| Matched source learning and SELECT | 10,396.10 | 19,152 | 12,198 |
| Matched normal-target learning and SELECT | 2,770.89 | 26,820 | 496 |
| Sealed TEST score computation | 225.10 | 6,930 | 0 |
| Controlled score computation | 14.83 | 864 | 0 |
| CPU ranking metric computation | 776.08 | 0 | 0 |

These totals count stages jointly for matched arms. Successful GPU-work stage wall intervals sum to 3.724 h. Frozen history accounts for 8,871.78 seconds of that wall time, with simultaneous execution of the two encoders during SMD source training. The timings are stage-wall measurements. Device activity inside parallel stages and unattended elapsed time use different denominators. Archived TEST and controlled feature banks are checksum-verified and reused. The table counts batch calls, with four current predictions per holdout window at inference. Model loading, input hashing and rolling checkpoints fall inside the applicable enclosing stages. The successful source check and prior bypass total 165 additional current forwards over 11.53 seconds already enclosed by the source wall intervals.

The sealed matrix contains nine source and 120 normal-target endpoints. It records 18,000 source optimizer steps and 14,080 normal-target optimizer steps. The failed PSM launch reached no scientific score or optimizer update. In a separate interruption, the normal-target checkpoint receipt writer stopped after PSM's second epoch because it used a create-only helper for rolling files. Atomic replacement corrected persistence and execution resumed from the sealed first epoch. The 18 second-epoch updates per arm and 65 validation calls of the interrupted attempt were repeated. They are outside the sealed trajectory counts. Measured feature reloading after the interruption is included in the recorded N stage. Logs from the initial import and bypass-count preflight failures are also preserved.

The [final CPU verification](evidence/FINAL_CHECK.json) rechecks checkpoint identities and earliest normal SELECT selection, score bank hashes and window footprints, and AP/ROC from the original score arrays. It also confirms all 2,000 official and random source feature hashes per fold against the prior computation. VUS retains the fixed evaluator and is checked through its sealed rows and aggregations. Original tensor banks and selected states remain on the research server at `/home/a312/tsad_research/results/left_holdout_history_20261127_v1`.

The verification covers 129 selected checkpoints, 270 public score banks and 540 AP/ROC rows. Recalculated AP and ROC agree exactly with the sealed rows using the same library. The ranking pipeline also checks them against the project's independent rank calculation. The transferred summaries were checked against the raw rows for all 116 macro rows and 156 paired summaries, including VUS aggregation. All 645 files match the [evidence manifest](evidence/evidence_manifest.json). The archive SHA-256 is `ee38c886888da36010f9f5d05cb0121fe9da1443a19ac108c952cc39cbd22c02`.

## Scientific assessment

Holding out the scored values exposes a limited benefit from the official history representation within the new detector. PSM demonstrates it after normal target training, while SMD demonstrates it most clearly with frozen source weights. SWaT offers little corresponding AP increment. All three N comparisons retain a lead for unmasked native LEFT. The non-overlap interventions and fixed weak/delay controls give limited evidence for a contribution specifically from the correct historical correspondence. Thus the within-architecture ranking increment is worth preserving alongside the stronger native benchmark, and the evidence remains insufficient for a general improvement or a weak-anomaly and lag-mechanism claim. The present validation ends with Z, N and the fixed controls.
