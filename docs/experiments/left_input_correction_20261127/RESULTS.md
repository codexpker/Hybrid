# Source development of bounded LEFT training-input correction

**The prespecified source selection retains R0, ordinary replay.** All 16 endpoints completed 2,000 updates. No nonzero repair met the mean proxy AP gain of 0.005, and the best finite repair was slightly below whole-window weighting. The source experiment has no real target detection result for the repair. This is a group-level decision for these eight fixed procedures. [Frozen protocol](PROTOCOL.md) · [Selection, before rounding](selection.json)

## Full matched source matrix

The 25- and 31-channel columns identify the frozen source model configuration. Every value comes from TEP/SMD source development data, with equal weight on the two edit families and two sources per width. The worst normal risk ratio compares the largest of the four raw normal-loss ratios with the within-width R0 control. The fixed limit was 1.05. Every arm met this risk limit. Bold indicates the prespecified selection.

| Arm | 25-channel AP | 31-channel AP | Mean AP | Mean AP − R0 | Worst normal risk ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| **R0 Ordinary replay** | **0.555511** | **0.550417** | **0.552964** | **0** | **1.000000** |
| R1 rho .25, q .95 | 0.555509 | 0.550371 | 0.552940 | −0.00002410 | 1.000413 |
| R2 rho .50, q .95 | 0.555549 | 0.550286 | 0.552918 | −0.00004654 | 1.000557 |
| R3 rho .25, q .99 | 0.555598 | 0.550417 | 0.553008 | +0.00004346 | 1.000440 |
| R4 rho .50, q .99 | 0.555575 | 0.550387 | 0.552981 | +0.00001700 | 1.000572 |
| R5 Full teacher substitution | 0.555799 | 0.549857 | 0.552828 | −0.00013597 | 1.000991 |
| R6 Marginal clipping | 0.555600 | 0.550172 | 0.552886 | −0.00007811 | 1.001518 |
| R7 Whole-window weighting | 0.555620 | 0.550424 | 0.553022 | +0.00005777 | 1.000200 |

R3's mean proxy AP was 0.00001431 lower than R7's. The small ordering changes do not demonstrate a practical difference between the two. Teacher replacement and ordinary marginal clipping also failed to provide clear proxy gains. The tables preserve per-source and per-family AP and ROC-AUC, normal losses, and intervention behavior. Real target AP, ROC-AUC and VUS for a new repair are pending the formal decision. No CAL was used for a source development threshold. [Development values](development.csv) · [Proxy breakdown](proxy_panels.csv)

## What the correction changed

The q95 and q99 teacher residual cutoffs were 1.500805 and 2.174468 for width 25, and 1.621148 and 2.321404 for width 31. On held-out clean windows, q95 repair altered 4.65–4.94% of coordinates and q99 altered 1.03–1.11%. The corresponding fraction at coordinates actually changed by the amplitude proxy was 9.58–10.00% for q95 and 2.77–2.93% for q99. For shifts, the fractions were 3.24–3.57% and 0.66–0.92%. Clean correction RMS across finite variants ranged from 0.358 to 1.636 standardized units. The correction reaches normal data as well as proxy edits. The norm relative to the known clean reference can increase after repair, as it did for amplitude edits in both width-31 q99 arms. The stable post-training normal loss does not imply that the input edits were selective or small in absolute magnitude.

The separate frozen CPU diagnostic compared `teacher(edited)−teacher(clean)` to the edit at the actually changed coordinates. Median relative error was 0.656–0.759 for amplitude and 0.978–0.992 for shift. No proxy window met the descriptive criterion of an error at most one quarter of its edit norm. Most changed coordinates nevertheless stayed below the q95 activation threshold. On this panel, the low activation rate cannot be attributed to near-exact teacher copying. This diagnostic cannot establish what the teacher does to actual target anomalies. [Teacher response on paired source inputs](teacher_copy.json)

## Asset reuse and provenance

The study reused the exact archived width-specific LEFT source checkpoints, their normal source permissions, the existing amplitude/shift editor, native readout, and the whole-window weighting implementation. R0 and R7 were both trained on the new source proxy exposure, since their existing real-target runs use a different input plan. R7 retained the ranked target weighting definition and re-estimated coefficients on the new plan. All 16 weights and their source proxy results were therefore new. Source rank computations are not interchangeable with archived real target scores.

For a later formal matrix, the existing PSM/SWaT and SMD28 Z/N models and scoring banks retain their original roles and serve as public anchors. The real A-condition ordinary and window-weighted replay runs remain eligible for reuse when parent, seed, sample plan, scaler, budget and score axes match. SMD's 38-channel TEP/HAI checkpoint, all 28 roles, and nominal-normal CAL are already archived. Any new perturbation-trained weight or new paired seed requires its own target scores. Existing PSM all-attack CAL remains a limitation. The fixed M1 results and the untrained M2 design decision are separate evidence. [Two-domain replay](../left_multisource_20261126/RESULTS.md) · [Weighting and its matched control](../left_smd_reliability_20261126/permutation.md) · [Complete SMD28](../left_smd_reliability_20261126/SMD_RESULTS.md) · [M1 result](../left_proxy_rank_20261127/RESULTS.md) · [M2 design review](../left_detection_oriented_source_20261127/REVIEW.md)

## Execution and verification

Zero repair used bit-identical inputs and RNG and produced identical first-step loss in both configurations. The max parameter difference across the old and new CUDA entry points was 2.47e−7 and 1.32e−6. Source slots were bit-identical to the raw values. The repair teacher stayed frozen, outside the optimizer, and unchanged in state and curriculum. R7 reloads its learned state into the native unweighted detector for scoring. Two preliminary gate attempts caught the overly strict bitwise parameter assertion and an R7 readout that still carried training batch weights. The revised gate passed before any of the 16 adaptation jobs. Trainer settings and selection were unchanged.

Successful adaptation and proxy readout job bodies took 1,349.78 A40 seconds (0.3750 A40·hours). Summed lane wall time was 1,397 seconds (0.3881 A40·hours), including process overhead. Teacher training forwards accounted for 162.23 in-stream seconds within job bodies. The longest of four lanes took 363 seconds. Peak allocated memory was 472 MiB during formal development jobs. Preparation and passing checks took a separate 64.50 A40 job seconds. Failed gate logs covered about 32.21 seconds, separate from successful work. The additional teacher diagnostic took 15.83 CPU seconds. All lanes exited normally. The verification checks 16 checkpoint hashes, the roles and every complete window footprint, matching plan/parent/update identities, and aggregation of the per-panel metrics. The source proxy AP calculations are saved per panel, with no second computation from a raw score bank.

Raw weights, role plans and logs reside in `research_workspace/results/left_input_correction_20261127_v1` and the corresponding data-workspace results directory. Run the [four-lane entry](../../../scripts/run_left_input_correction.sh) after preparing and checking both source configurations under the [module CLI](../../../src/hybrid_tsad/left_input_correction.py). Reproduce the tables with the [CPU summary](../../../scripts/summarize_left_input_correction.py). No raw data or weights are in Git. [Receipts](receipts.json)
