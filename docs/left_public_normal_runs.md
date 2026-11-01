# Public normal-run baseline interface

This package keeps the upstream LEFT implementation external and unchanged at
commit `3fadb4811797075e233076b9faceb0d8ec59b697`. It does not integrate TimeRadar
weights or introduce a new network. The previous TimeRadar reproductions remain
separate, completed research assets.

## Shared implementation

| Module | Responsibility |
| --- | --- |
| `normal_run_data` | Explicit independent normal runs, FIT-only float64 scaling, within-run windows and causal endpoint coordinates |
| `left_normal_public` | External native `Exp.train`/`vali`, effective H100/C3 settings, curriculum-complete checkpoint eligibility, training observers and exact run authorization |
| `normal_b1_readout` | Strict accepted-checkpoint loading, one complete window per native infer call, raw score banks, fixed last/mean32 readouts and measured runtime |

The entry points consume frozen JSON configurations and separate exact approval
records. They refuse unreleased execution. Per-run configurations, logs, source
snapshots, checkpoints, scalers and raw banks live in named research-workspace
directories; they are not additional model implementations.

```text
python -m hybrid_tsad.left_normal_public --config CONFIG.json
python -m hybrid_tsad.left_normal_public --config CONFIG.json --release RELEASE.json --execute
python -m hybrid_tsad.normal_b1_readout --config CONFIG.json --release RELEASE.json --execute
```

These module commands do not replace the reviewed process supervisor, physical
GPU identity checks or timeout. Use the separately frozen supervisor for formal
jobs. No dependency installation or upstream-source redistribution is implied.

## Fixed current TEP protocol

Only public fault-free Training runs are used for training: FIT 1–20 and 51–60;
SELECT 61–80. Columns are ordered `xmeas_9`, `xmeas_21`, `xmv_10`. Each has 500
points. H100 training uses stride 1 and B128 (12030 windows, 94 steps/epoch);
normal selection uses stride 100. Scaling uses FIT only.

The explicit baseline configuration uses at most 20 epochs, Adam/type2 LR,
warmup 123/ramp 246 and native lambda maximum .3. Only actual normal validation
at step >=369 may save a best checkpoint or consume patience. This is a declared
configuration adaptation, not the author's default TEP recipe. The original
`_basic`/THOP call and its RNG consumption are retained.

The prescribed seeds are 2025, 2026 and 2027. The actual saved checkpoint is
chosen solely by normal SELECT loss. Its generating step and the final Python
training counter are recorded separately: a native state_dict does not restore
the ordinary Python counter or optimizer.

The B1 cost gate uses SELECT run61, 401 windows with endpoints 99–499 (zero-based).
It is a runtime check on previously used normal selection data, not independent
detection validation. Continuous `[401,100,3]` native scores and input coordinates
are preserved. The last/mean32 readouts are fixed before fault evaluation. Full
state/gradient/mode/step/RNG fingerprints are taken at bank boundaries; equal
boundaries do not establish an immutable intermediate trace.

## Evidence and limits

The first normal run completed 20 epochs/1880 effective updates and passed
separate execution, audit and primary review. The other prescribed seeds and
the B1 runtime have distinct run records; consult the workspace's
`CURRENT_RESEARCH_STATE.md` and `results/START_HERE.md` for their current status.
All observed absence of native EMA calls is retained as unverified mechanism
evidence, rather than replaced by an assertion that Adam updates prove EMA.

Normal training completion does not establish anomaly-detection improvement,
weak-lag identification, industrial transfer or novelty. CAL/fault/query access
requires a separate protocol and release. No private data or test-label tuning
is authorized by these interfaces.
