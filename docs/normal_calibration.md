# Fixed normal calibration, three accepted LEFT checkpoints

The existing normal B1 runtime is generalized in place; there is still one
checkpoint-loading path, one native `model.infer` call and one bank collector.
The old run61 B1 snapshot stays immutable in its original preparation/results.

Only `LEFT_TEP_NORMAL_CALIBRATION` enables CAL81…100. Each seed2025/2026/2027
requires its own exact config/release and its own accepted best/scaler/receipt.
Device mapping is fixed to physical GPU0/1/2 under one visible logical cuda:0.
No training, SELECT scoring, fault query, label, model selection or refitting.

Each bank retains 401 original endpoints99…499 and raw `[401,100,3]` scores,
actual normalized input windows and raw float64 source values. The model is
freshly constructed and strict-loaded once per seed; snapshots are at each bank
boundary. Best-generation/terminal/fresh Python step remain distinct (seed2026
best1786, terminal1880, fresh0). Eval lambda=.3 is unchanged.

For each model, exactly20 banks form8020 normal scores for each fixed readout.
Order statistic `ceil((8020+1)*.99)=7941` (one based), comparison **strict >**.
Ties at the threshold do not exceed it. Save all8020 scores and run/endpoint
coordinates along with thresholds. Pooling is only a normal-score distribution;
no event axis is fabricated across run boundaries. Separate seeds/checkpoints,
scalers or extraction code/step identities cannot share a pool.

The accepted normal RUN_ASSET_INDEX JSON supplies CAL references. Candidate
construction and unit tests used only JSON and synthetic arrays. The three
formal workers subsequently ran with separate exact releases and the reviewed
300s+5s/20GiB guard. All60 actual banks, three8020-point distributions and six
fixed thresholds passed separate audit and primary review. Original source and
continuous scores are retained; no fault data or labels were consulted.
Normal point calibration does not establish event false-alarm rates or detection
gains on unseen faults/domains. No anomaly labels or AP are consulted.
