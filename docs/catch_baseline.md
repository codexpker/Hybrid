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

## Independent normal-run adaptation and fixed readout

`python -m hybrid_tsad.catch_normal_runs --config /external/train.json` is the
separate normal-run entry. It temporarily replaces only the native split and
loader globals, restores them on exit, fits the original scaler on FIT only, and
constructs every window inside its own run. The original `detect_fit` model,
losses and optimization loop remain external and unchanged. The common SELECT
window density is an explicit protocol change from native dense validation.

`python -m hybrid_tsad.catch_normal_scores --config /external/score.json` previews
the fixed checkpoint readout. Actual execution also needs `--release` and
`--execute`. Identity-filtered hooks copy the existing model reconstruction and
time/frequency criterion outputs without another forward pass. Canonical point
scores retain the native Torch channelmean-then-add order. Variant-independent
role/run seeds retain the official stochastic channel mask while allowing
paired controls. Raw components, coordinates and continuous scores are saved.

`python -m hybrid_tsad.engine.catch_normal_evaluation --config /external/eval.json`
is the CPU-only evaluation preview. It requires a separate exact label release
to execute; all 80 unique physical banks and their provenance must pass before
truth or historical comparison tables can be opened. It reuses shared metric
kernels, preserves all events and fixed comparisons, and never calls a model.

The completed C2 comparison uses three CATCH training epochs and twenty LEFT
epochs. These are not equal optimization budgets. The synthetic mechanism
comparison has not established stable gains, industrial effectiveness or a need
for a new network. Author code, weights, data and execution-path configurations
remain outside this repository; use the research result index for real runs.
