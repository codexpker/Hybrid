# External official CATCH: synthetic native API admission

`python -m hybrid_tsad.catch_baseline --config /external/l0.json` previews the
fixed operation without importing Torch. `--execute` additionally requires the
exact, separately signed research release. This command is deliberately limited
to synthetic engineering admission; it does not accept real data or checkpoints.

The adapter calls the external official `CATCH.detect_fit` and `detect_score`
from the pinned source tree, attributed to decisionintelligence/CATCH commit
`3647c69be5eb56649b072596cf89098e689e20c3`. Source identity is checked by hashes;
the external tree need not contain Git metadata. Upstream code is not bundled or
redistributed. Its missing root license must not be interpreted as a grant to
redistribute it.

The one-epoch admission retains the native contiguous 80/20 split, optimizer
order, actual learning-rate behavior and random evaluation channel masks. It
uses separate deterministic synthetic train and score streams, preserves missing
tail indices, and stores continuous scores plus observational traces. It does not
save a reusable checkpoint, evaluate anomaly labels or select thresholds.

Run the integration fixtures in the existing Torch/pandas/sklearn environment
with `CATCH_SOURCE_ROOT` pointing to the fixed external tree:

```text
python -m unittest discover -s tests -p test_catch_baseline.py -v
```

The native `detect_fit` splits its first input internally; its second input is
ignored. Consequently this L0 does not implement independent-run FIT/SELECT.
That adaptation requires a separate configuration and protocol. Window scores
are available after the complete window is observed, not before each sample.
The native API admission passing is not a complete CATCH paper reproduction or
evidence of anomaly-detection gains.
