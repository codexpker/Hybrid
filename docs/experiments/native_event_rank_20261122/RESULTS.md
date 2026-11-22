# Source-event-balanced learning on native variance

This is a source-supervised, zero-target-label transfer development fold from labeled public SWaT and nominal SWaT/SMD28 source data to PSM. PSM was excluded from project source training and is a historically exposed development task. Its released training prefix supplies only the existing FIT scaler for the tested TimeRadar arms. The known target CAL scores supply the fixed empirical operating thresholds.

The official decoder initialization, detector, 100-point causal endpoint, 25 PSM channels, batch-one window independence, fixed ten complementary masks and original sample variance are shared. The frozen native row reuses the exact accepted CAL and QUERY banks. Three source training arms each complete 240 updates with seed2025. All trainable parameters belong to the native decoder. Each update has the same 16 nominal SWaT and 16 nominal SMD copy windows. Both supervision arms additionally see exactly the same 15 positive and 15 normal public SWaT windows. The three consecutive groups of legal endpoints per source event stratify draws. Event balancing and point weighting differ only in weights within that paired loss. The reconstruction-gradient continuation uses native copy MSE. All three use the earliest minimum of the same event-balanced source SELECT score at steps 0, 40, …, 240.

## Table 1. PSM common causal endpoint comparison

QUERY is test-relative `[23016,87744)` with 64,728 points and 48 eligible complete events. VUS uses the same pinned LEFT evaluator with `windowSize=100` and `thre=250`. S is the strict point stream. C3 starts at the third consecutive strict alarm and has no backfill. Each operating row calibrates its own score on the same 512 CAL points using q99/higher/strict greater-than. All 512 CAL points are labeled attacks under the existing task. The table reports observed QUERY false positives and recall. This CAL is unsuitable for certifying a normal false-alarm budget. Long steady coverage starts 512 points into each of the five complete long events. Event matching uses one-to-one overlap and any-hit is shown separately. The median delay counts hit events only. False-alarm fragments and strict/C3 full event and persistence metrics are in the CSV.

| Method | AP | ROC-AUC | VUS-PR | VUS-ROC | FPR % S/C3 | Recall % S/C3 | Matched S/C3 | Any-hit S/C3 | C3 event F1 | C3 event points % | C3 long macro/pooled % | C3 longest miss | C3 median hit delay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TimeRadar frozen native V | 0.550952 | 0.735048 | 0.484578 | 0.648161 | 29.67/29.10 | 50.87/48.88 | 18/6 | 28/17 | 0.090 | 73.19 | 86.45/92.11 | 694 | 0.0 |
| TimeRadar reconstruction-gradient continuation | 0.556073 | 0.735692 | 0.494783 | 0.650822 | 29.38/28.85 | 50.42/48.44 | 18/6 | 28/17 | 0.100 | 72.51 | 84.53/90.99 | 718 | 0.0 |
| TimeRadar native V, point-weighted source supervision | 0.613415 | 0.786398 | 0.532704 | 0.672103 | 6.72/5.92 | 40.06/37.33 | 19/10 | 20/11 | 0.106 | 54.95 | 53.07/73.56 | 874 | 3.0 |
| TimeRadar native V, event-balanced source supervision | 0.588312 | 0.764587 | 0.503475 | 0.656078 | 28.67/28.23 | 55.58/53.68 | 14/5 | 25/17 | 0.088 | 76.99 | 96.09/97.72 | 477 | 0.0 |
| LEFT W192 endpoint | 0.574261 | 0.762902 | 0.579567 | 0.778901 | NA/NA | NA/NA | NA/NA | NA/NA | NA | NA | NA/NA | NA | NA |
| MLP noU | 0.606531 | 0.815061 | 0.627826 | 0.823338 | 4.33/3.88 | 32.57/31.36 | 29/20 | 31/21 | 0.226 | 49.17 | 23.82/72.39 | 1000 | 4.0 |
| MLP replay | 0.635775 | 0.828137 | 0.685514 | 0.852250 | 1.00/0.82 | 5.28/4.10 | 29/20 | 31/21 | 0.303 | 3.85 | 2.70/2.44 | 1621 | 7.0 |
| Full CANDI | 0.606531 | 0.815059 | 0.627825 | 0.823337 | 4.33/3.88 | 32.57/31.36 | 29/20 | 31/21 | 0.226 | 49.17 | 23.82/72.39 | 1000 | 4.0 |
| TimesFM 3.0 frozen | 0.344512 | 0.566754 | 0.376176 | 0.608554 | 2.24/0.07 | 6.41/0.83 | 41/14 | 41/14 | 0.292 | 0.63 | 0.24/0.11 | 3699 | 9.5 |
| Lightweight TimesFM raw forecast | 0.567902 | 0.743845 | 0.575549 | 0.764875 | 25.51/22.01 | 58.21/48.80 | 36/22 | 45/30 | 0.206 | 76.51 | 85.17/91.37 | 399 | 3.5 |
| History Mean512 | 0.303673 | 0.525589 | 0.332386 | 0.567946 | 3.02/1.11 | 5.37/2.46 | 41/20 | 41/20 | 0.154 | 3.45 | 1.94/2.37 | 1324 | 13.0 |

## Permissions for archived external rows

- LEFT W192 endpoint uses target-normal FIT, SELECT, and training-prefix statistics. Its causal ranking metrics are available on the same axis. Operating metrics for a compatible CAL are missing.
- MLP noU uses a same-domain normal initialization. Plain MLP replay and full CANDI use target mixed FIT under their archived rules, then freeze. The three use their original 10-point endpoint MSE.
- TimesFM 3.0 uses full official pretrained weights frozen throughout and a historical 512-point one-step median forecast. The lightweight TimesFM raw forecast reference uses target-normal FIT and SELECT and no official pretrained weight. History Mean512 is the historical causal reference with its existing source-CAL residual scaling.
- CATCH and other retrospective full-window baselines retain their separate original panels. Their readout permission differs from this panel.

## Table 2. Weighting ablation under matched public source supervision

Both weighted arms used the same public source labels, sampled windows, nominal reconstruction updates, initial weights, budget and earliest-minimum source SELECT rule. These differences are from one seed in one development fold. The source-supervision comparisons against reconstruction-gradient continuation also include the addition of the paired source loss and its extra window exposures.

| Stream | Metric | Point-weighted | Event-balanced | Event-balanced − point-weighted |
| --- | --- | --- | --- | --- |
| Ranking | AP | 0.613415 | 0.588312 | -0.025104 |
| Ranking | ROC_AUC | 0.786398 | 0.764587 | -0.021810 |
| Ranking | VUS_PR | 0.532704 | 0.503475 | -0.029229 |
| Ranking | VUS_ROC | 0.672103 | 0.656078 | -0.016025 |
| strict | normal_FPR (%, delta pp) | 6.724 | 28.667 | +21.943 |
| strict | Recall (%, delta pp) | 40.060 | 55.583 | +15.523 |
| strict | M | 19 | 14 | -5 |
| strict | auxiliary_any_hit_events | 20 | 25 | +5 |
| strict | one_to_one_eventF1 | 0.093 | 0.147 | +0.054 |
| strict | event_point_coverage (%, delta pp) | 56.746 | 77.326 | +20.580 |
| strict | long_steady_macro_coverage (%, delta pp) | 55.067 | 96.453 | +41.385 |
| strict | long_steady_point_coverage (%, delta pp) | 74.797 | 97.935 | +23.138 |
| strict | max_event_miss_points | 874 | 472 | -402 |
| strict | delay_hit_only_median | 0.5 | 0.0 | -0.5 |
| confirmed3 | normal_FPR (%, delta pp) | 5.919 | 28.229 | +22.311 |
| confirmed3 | Recall (%, delta pp) | 37.333 | 53.683 | +16.350 |
| confirmed3 | M | 10 | 5 | -5 |
| confirmed3 | auxiliary_any_hit_events | 11 | 17 | +6 |
| confirmed3 | one_to_one_eventF1 | 0.106 | 0.088 | -0.018 |
| confirmed3 | event_point_coverage (%, delta pp) | 54.946 | 76.995 | +22.049 |
| confirmed3 | long_steady_macro_coverage (%, delta pp) | 53.069 | 96.089 | +43.021 |
| confirmed3 | long_steady_point_coverage (%, delta pp) | 73.561 | 97.724 | +24.163 |
| confirmed3 | max_event_miss_points | 874 | 477 | -397 |
| confirmed3 | delay_hit_only_median | 3.0 | 0.0 | -3.0 |

## Source training evidence and interpretation

- Shared initialization, step 0: SELECT 0.871006, nominal reconstruction MSE 0.719039, labeled normal/positive V means 0.022998/0.156992.
- TimeRadar reconstruction-gradient continuation, step 80: SELECT 0.843823, nominal reconstruction MSE 0.636553, labeled normal/positive V means 0.027077/0.212608.
- TimeRadar native V, point-weighted source supervision, step 240: SELECT 0.602799, nominal reconstruction MSE 0.622702, labeled normal/positive V means 0.048414/1.300584.
- TimeRadar native V, event-balanced source supervision, step 240: SELECT 0.625847, nominal reconstruction MSE 0.615553, labeled normal/positive V means 0.037557/1.280709.

The runs used four A40 devices. Guarded device occupancy totalled 0.82345 GPU hours, including 0.23318 for the source scale and three trainings and 0.59027 for the target readout. Formal CPU metric evaluation took 15.20 seconds. The independent CPU review checked all sealed banks and reproduced 182 metric fields plus 30 paired differences, with a maximum absolute difference of 1.11e-16. All 19 guarded jobs completed and released their process groups.

The nominal reconstruction MSE summarizes the same 112 SWaT and 112 SMD nominal validation windows at each scheduled checkpoint. The labeled means summarize the 192 matched validation pairs from the four withheld source events. The source validation risks decreased in all arms. For both supervised arms, raw mean V increased on both sides of the pair and much more on the labeled abnormal side. The table is insufficient to infer that normal-window scores all decreased. [Full source SELECT trace](source_select.csv).

- **Source supervision improved PSM score ranking.** Point weighting increased AP by 0.057342 against reconstruction-gradient continuation and by 0.062463 against the frozen detector. Event balancing increased AP by 0.032239 and 0.037359 against the same controls. All corresponding ROC-AUC and VUS deltas were positive in this fold. The reconstruction-gradient continuation alone gained 0.005120 AP over frozen V.
- **Event balancing changed the operating trade-off.** It gave up 0.025104 AP, 0.021810 ROC-AUC and both VUS metrics against matched point weighting. Its C3 recall was 16.35 percentage points higher and long steady macro coverage was 43.02 points higher. C3 FPR was also 22.31 points higher, with five fewer one-to-one event matches and six more any-hit events. This does not establish a coverage increment at matched normal false-alarm cost.
- **Transfer was metric-dependent.** Relative to frozen native V, point weighting increased C3 matches from 6 to 10 and lowered observed C3 FPR from 29.10% to 5.92%. Normal C3 alarm fragments rose from 88 to 93. Any-hit fell from 17 to 11 and long steady macro coverage fell from 86.45% to 53.07%. Event balancing increased long steady macro coverage to 96.09% with 28.23% observed C3 FPR, 71 normal C3 alarm fragments, five matches and 17 any-hit events. This is a fixed-fold ranking gain from real source supervision with differing operational effects. Source event balancing has no established independent ranking benefit, and neither arm establishes normal-budget-controlled persistent detection or an independently confirmed Hybrid contribution. The result supports keeping the source-supervised native score comparison as a development result and giving a fixed public benchmark comparison priority over U/F or contamination studies.

The same window and label permissions gave seven prespecified checkpoint evaluations per arm. Each labeled training arm saw 3,600 positive and 3,600 normal pair-window exposures. The normal source scale was 0.02649214304983616 from the frozen detector on 4,632 complete label-normal SWaT FIT windows. PSM remains historically exposed, and the single seed does not establish repeatability.

## Reproducible artifacts

[Methods CSV](methods.csv) · [All matched deltas](paired.csv) · [Strict and C3 event details](events.csv) · [Source SELECT trace](source_select.csv) · [GPU receipts summary](compute.csv) · [Config and SHA provenance](PROVENANCE.json).

The training and readout code is pinned at `ebdb56cff970c1a610d75e08c9582583dc99a0cf`. Model checkpoints, score arrays, the source sampling plan, raw training trace, and per-shard receipts are retained on the research server at the relative locations in provenance. The repository carries aggregate values, event results, and the SHA-256 identities of the original inputs and outputs.
