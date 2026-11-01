# Normal B1 readout

This shared module contains one native LEFT checkpoint-loading/inference path,
NumPy bank construction, strict provenance checks and fixed normal calibration.
The original preparation snapshots and failed candidates remain in their named
workspace directories. They are not additional model implementations.

## Implemented and exercised

The fixed SELECT run61 cost gate completed 401 actual B1 calls in 5.94 seconds
for accepted seed2025. Three accepted models subsequently completed separate
CAL81–100 jobs, 20 banks / 8020 calls each. Code, original raw scores, indices,
source values, resource records and thresholds passed separate audit and primary
review. These are engineering and normal calibration results, not anomaly gains.

- One complete H100 window per native infer call; no future-batch backfill.
- Raw float32 channel scores [401,100,3], actual input windows, original source
  values, starts, endpoints99–499 and availability coordinates are retained.
- Last is channel mean at window end. Mean32 averages the final32 channel-mean
  scores and is also assigned to window end. Neither is an advance forecast.
- Registered state, gradients, modes, Python step and four RNG fingerprints are
  checked at bank boundaries. Equal boundaries do not establish an immutable
  intermediate/private-cache trace. Each call still checks shape/finite/input.
- Scorer duration includes actual transfer and synchronized CPU return. First,
  steady, bank observations, IO, archive and OS time have separate records.
  Synthetic fake-scorer test timings are not GPU throughput evidence.
- The fresh model schema must exactly match the actual pre-basic science keys.
  The saved checkpoint schema must additionally match the individually recorded
  actual THOP counters. Only those extras are filtered, then strict=True loads
  all scientific tensors. No generic suffix filtering or ignored science keys.
- Best-generation, terminal-training and fresh-inference Python steps are
  different facts. A native checkpoint restores neither the ordinary counter
  nor optimizer. Eval lambda=.3; no fake training-counter recovery is claimed.

## Execution boundary

Each formal job requires an exact config/release, code/resource review and the
frozen process supervisor. Internal approval checks are execution controls, not
changes to the scientific loss or native inference formula. Pure bank functions
remain importable for synthetic tests. No training, fault query, labels or
model selection is authorized by a normal cost/calibration release.

See normal_calibration.md and left_public_normal_runs.md. Actual run paths and
primary receipts are listed in the workspace results/START_HERE.md; do not infer
current job state from this static document. The .729-hour full-call estimate
from one normal run is a compute/CPU-return point estimate, not a runtime bound.
