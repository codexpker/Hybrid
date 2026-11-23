# Standard supervised readout on frozen TimeRadar features

## Fixed method and source permissions

TimeRadar plus supervised MLP is a project baseline. The official pretrained encoder, fractional-order network, decoder, parameters and buffers remain frozen. Each native W100, patch-five forward uses B1 and the original ten complementary masks, with isolated seed 84551. The encoder output is reshaped to [10,1,C,20,256]. Real and imaginary features concatenate to 10,240 coordinates for each copy/channel. A shared 10,240→64→1 GELU MLP produces each local logit, and their copy/channel mean gives the window score. Both learning and deployment use this window scalar. The public event label does not identify which sensor is faulty.

The fixed loss has three positive terms: 0.5 times the point-count-weighted positive softplus(−logit), 0.25 times the same-weight paired-normal softplus(logit), and 0.25 times the nominal-normal mean softplus(logit). This compares a complete discriminative learning/readout procedure against native V supervision. It changes the learned parameters and objective as well as inference, so any gap cannot be uniquely attributed to the score alone.

SOURCE roles, five FIT events, four SELECT events, normal windows, scalers and the seed-2025 sealed 240-step plan are reused. Two fixed rates, 1e-4 and 1e-3, share an identical head initialization and actual per-window frozen encoding. Each uses a separate fresh source-configured AdamW, with 24-step warmup, through 2,400 updates and ten passes of the same plan. At each step the learner sees 32 nominal windows and 15 abnormal/15 normal pair windows. The labeled source positives and paired normals retain the existing legal-positive-point-count weights. SOURCE SELECT uses the corresponding three-term balanced BCE, equal weighting across event/time strata, and the original nominal panel. All selection occurs before target scoring.

## Completed source learning

Both the formal full-range and the ≤240-step auxiliary rule chose learning rate 1e-3 at step 40, risk 0.458484785, versus step-zero risk 0.697997583. The best 1e-4 checkpoint was step 280 with risk 0.464565247. The two selected heads have equal parameter tensors; target inference computes the head once and records both specified rows. They cannot be interpreted as independent observations.

The MLP has 655,489 trainable parameters. Per learning-rate trajectory, the full search used 148,800 window exposures and 11,560 distinct endpoint windows, which can overlap in time and do not constitute additional events. The selected step-40 head used 2,480 exposures and 2,326 distinct endpoint windows. The formal search considered later checkpoints even though it ultimately selected an early state. The ≤240-step row constrains exposure at selection while retaining the two-rate development allowance. It does not establish total-compute or trainable-capacity equality with the native decoder trials.

The source task was learned with generalization limits. At lr1e-3, the final FIT loss was 0.166287 while final SELECT risk was 2.104022; the source-only rule retained the early minimum. No target metric determined this choice. This evidence permits interpreting target transfer of the chosen supervised head, with source overfitting reported separately.

[Source curves](source_curves.png) · [FIT data](source_fit.csv) · [SELECT data](source_select.csv) · [selection identities](source/ALL_SELECTION.json) · [window/event exposure](source/SOURCE_COUNTS.json) · [configuration](source/SOURCE_CONFIG.json)

## Public evaluation permissions

PSM uses its complete causal ranking axis [99,87841), 87,742 endpoints with 25 variables, and the separate 64,728-point historical suffix for operation. Its 512 CAL points all have attack labels, so q99 is an empirical operation point and does not certify a normal false-alarm budget. MSL27 and SMAP53 use sensor zero and per-entity [99,N) axes, averaging entities equally. Their small per-entity CAL sets make q99/higher the maximum. Long steady coverage has only four MSL and ten SMAP events. SMD uses all 28 separate machines and all 38 channels; its nominal training records contributed source windows. SOURCE-event supervision came from SWaT, and SMD test labels did not enter training or head selection.

Each selected head scores every legal CAL and QUERY window at its actual endpoint with raw logits. Each readout has its own threshold from the same legal CAL positions, q99/higher, strict exceedance. Confirmed alarms begin at the third consecutive crossing without backfill. Any-hit and one-to-one event matching remain separate. Coverage reports both equal-event and pooled values after the first 512 event points. Misses remain distinct from hit-only delays.

External methods keep their original training and context permissions. LEFT, CATCH and PCA can be compared on a separate PSM intersection [512,87744); the prior LEFT/HistoryMean512 panel extends to 87841. SMAP external and native results intersect at [512,N); SMD LEFT uses [99,192 floor(N/192)). The MSL historical flattened external bank has uncertified entity boundaries. Missing compatible external rows stay missing. Shared scored points do not remove the offline future context used by a comparator.
