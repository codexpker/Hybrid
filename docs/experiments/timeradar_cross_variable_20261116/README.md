# TimeRadar-native cross-variable adaptation: fixed development comparison

This experiment starts at `ab65ff6cc2becb37150a86e18a7933730b99612d`. Historical Hybrid candidates, original TimeRadar reproductions, checkpoints and results remain unchanged. It tests one decoder-input adapter, not lag learning or contamination robustness.

## Unchanged native computation

Official TimeRadar commit `4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2`; model file SHA256 `881e8912e4c5b298b202e5b7ac2fe58107c7dcf1b56acf733e28c99f2ffc0317`; pretrained checkpoint SHA256 `528aea6e803f9e6b1ea5d07e64e44162309bdb016cc7721a8ac3753d942e6c7c`. Load an immutable snapshot, not a mutable server checkout. Assets are external and are not redistributed here.

Use native FRFT/order network, complex encoder/decoder, projection, normalization, complementary masks, 10 copies, window/stride 100 and the original `cal_anomaly_score`. **L=1 weights only the native across-copy sample variance**; it is not a prediction horizon, nor variance plus MSE. PSM uses native norm=0 / batch64; SMD norm=1 / batch32. Do not substitute the old Hybrid h1/R3 scorer.

The native order estimator shares a batch/channel aggregate. Scores can depend on later observations within a window and other windows in the same batch. Save `available_index=max_input_index=batch_last_point` as well as `window_end_index`. These are **offline** point/event results. First-hit offsets describe retrospective score locations, not online detection delays.

## Only added computation

The decoder receives complex representations in `[copy,batch,channel,patch,dimension]` order. With rank16,

`h_c = GELU(W_down [Re(z_c), Im(z_c)] + b_down)`;
`m_c = h_c` for self, or `mean_{j != c}(h_j)` for cross;
`z'_c = z_c + complex_split(W_up m_c + b_up)`.

The up projection is zero-initialized. Copies, samples and patch positions are never mixed by this adapter. Self and cross have identical parameter counts/initial tensors; C=1 falls back to self. A native decoder pre-hook preserves the official forward. Disabling the adapter or using its initial state must reproduce all native copies, score and RNG consumption. The native model's returned pre-decoder representation is not changed by the hook.

## Fixed data and training

Records: PSM and the lexicographically first three complete cached SMD machines, **machine-1-1/1-2/1-3**, selected without model results. All are development assets with historical exposure.

For each original normal training record of length N: FIT `[0,floor(.6N))`, SELECT `[floor(.6N)+100,floor(.8N))`, CAL `[floor(.8N)+100,N)`. QUERY is the entire original test record. Fit a single StandardScaler on FIT only and apply identically to all four arms. This strict adaptation-comparison track differs from the historical reproduction scaler fitted on the whole original training record; historical zero-shot banks cannot replace a same-track frozen forward. Learner files contain only x/index, and evaluation labels live separately. Use role-local full nonoverlapping 100-point windows, preserve unscored tails as UNKNOWN.

Arms: pretrained zero; decoder-head adaptation; decoder+self adapter; decoder+cross adapter. Backbone stays frozen and eval, while decoder/adapter train. The objective is native MSE over **every reconstruction copy**, without target anomaly labels or a new loss. Three adaptation arms use the same FIT draws and mask streams, with RNG reset after arm construction. Seed2024; 2000 updates; AdamW lr1e-4, betas(.9,.95), eps1e-8, decay.1 with original bias/LayerNorm exclusions. Positive linear warmup200 then constant LR. SELECT reconstruction MSE at step0/every200, fixed seed84551, earlier ties; restore heterogeneous modes and CPU/CUDA/Python/NumPy RNG. No SELECT-improvement gate excluding unfavorable QUERY results. No query-based checkpoint, seed or threshold selection.

## Evaluation fixed before results

Each score uses its own legal CAL q.99 (`higher`), strict `>`; no reuse of thresholds in other units. Report point AP/ROC/P/R/F1, normal false-positive points/fragments per1000 normal points, any-hit and chronological one-to-one event metrics separately, three-consecutive confirmation coverage, complete-event coverage, longest missed span. Long events: at least512 points; steady portion starts192 points after onset. Partial/unscored events retain explicit coverage and missing indices; they are not fully observed misses. Native alarms remain offline and no physical sampling interval is invented.

Internal comparisons use identical native scored axes. External PSM comparisons use the verified common axis `[512,87744)`, with LEFT's target training/scaler permissions and offline access disclosed; HistoryMean512 is a causal statistical baseline. Cached CATCH does not qualify for the strict same-track table. Literature values and historical reproductions are reported separately, never substituted for local same-axis measurements.

## Execution and limits

Use `python -m hybrid_tsad.timeradar_cross_variable_runner {parity,train,score} --workspace WORKSPACE --config CONFIG --record RECORD --arm ARM --device cuda:0`. Four independent A40 jobs maximum; 20GiB training /30GiB inference, timeout7200s per job, preserve actual failures and GPU/PID/start/end/peak/exit/time. No cumulative GPU-hour approval cap. Native parity precedes adaptation; evaluation opens labels only after all score banks are sealed. Key results receive one role-separated independent recomputation. Source/config/traces/checkpoint hashes and raw scores remain external; publish code and nonprivate aggregates. Do not launch another recipe automatically after this comparison.

The actual CPU role-preparation source is published unchanged as `python -m hybrid_tsad.timeradar_cross_variable_data --workspace WORKSPACE`. It expects the already obtained public files in the documented workspace cache, creates a new role directory exclusively, saves FIT-only scalers and a physically separate evaluation manifest, and refuses to overwrite it. It does not download assets or score a model. Existing roles are reused for this run; do not rerun preparation to change a frozen binding. On another installation, inspect source hashes and bind that installation's manifest explicitly rather than faking this run's fixed manifest hash.

`timeradar_cross_variable_evaluation` evaluates the full native axis and the separately permissioned cached PSM external axis. `timeradar_cross_variable_baseline` reproduces the existing HistoryMean512 raw formula with strictly past512 mean/population variance plus1e-5, on the identical FIT-standardized inputs. Each role has a512-point warmup and its own CAL; the comparison axis is `[512,floor(query_length/100)*100)`. This new same-permission statistical row is separate from the historical source-calibrated/log-mapped baseline. No baseline training or new model forward is needed.
