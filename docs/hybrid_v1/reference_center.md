# Fixed causal reference center: A-stage adapter

This is the registered two-arm center experiment, not a new architecture. The
original model, training, objective, scoring and external source snapshots are
unchanged. `reference_center.py` imports Torch lazily.

`model_class(core_module)` subclasses the pinned `HybridV1`. Its `window` arm
calls the original `_encode` directly. Its `reference` arm executes the exact
original `_encode` code object with a private globals dictionary containing an
instance-bound replacement for `visible_normalize`. No process-global function
is patched. No parameter, buffer, layer, head, precision, mask or loss is added.

For a reference prefix `[role_start, role_start+512)`, only observed values enter
the frozen mean; every channel needs at least two observations. The original
masked local mean and population variance still produce the scale, including
the original epsilon1e-5. The reference mean is returned as `stats.mean`, so the
original objective normalizes targets with it and the original decoder restores
with it. `local_mean`, `reference_mean`, `reference_count` and `reference_valid`
are tensors, compatible with the original reconstruction-group stack. All
statistics remain detached. Each reconstruction group shares the same reference
mean but uses its own original visible local scale.

`reference_window` and `reference_context` enforce the role/run identity,
512-point prefix, observed support, anchor R+H and complete window footprints at
each real engine batch. References never cross role boundaries and never use
artificially hidden current/history targets or future targets. A nonblocking
instance lock and thread identity reject concurrent/reentrant contexts, and a
finally block clears binding state. One model belongs to one job. Adaptation by
deepcopy of a model containing this lock is not a supported A-stage API.

`training_namespace(original_training)` reuses original Python code objects for
Stage1 fit, validation, lazy inference and other kernels with private window and
pass bindings. It explicitly rejects new U/F fit, update-policy and adaptation
entry points before parameter-policy mutation, optimizer creation or deepcopy.
The original legacy U/F implementations and state dictionaries remain valid and
unchanged. Z is direct frozen inference or the original zero-update fit policy;
no new target adaptation API is claimed.

Reference replay metadata records the center arm and fixed prefix. The legacy
core `normalization_scope="visible_student_only"` describes the local scale in
the reference arm, not the reference center. A-stage runner/bank metadata must
declare `fixed_role_prefix_center_and_visible_local_scale`; the original window
arm retains the legacy meaning. This is a metadata clarification, not a second
normalization formula.

`fixed_reference_record` is the one B_ref control. It uses float32 source input,
visible float32 reference/local sums, the original population-scale epsilon and
current observation eligibility. Its prediction is the fixed reference mean;
its only residual is mean valid-channel squared normalized error. There is no
neural training or score fusion. Missing current channels are invalid, never
zero residuals. Counts and per-channel errors are returned for independent
replay; source calibration uses the unchanged single-prediction recipe on the
new shared source CAL positions, with its own numerical scales/tau.

Actual CPU evidence is separate from GPU L0 approval. The first frozen candidate
was never executed; the reviewed incremental CPU gate used tiny H128/P2,
channels1/3 and official source modules, with no real data or checkpoint load.
All14 tests passed, including original-window exact output/objective parity,
reference/target/restore interface, hidden/future isolation, per-group scale,
detachment, active gradients, Z freezing, U/F refusal, eval seeds/modes and the
unchanged calibration math. Formal A remains H512/P32. No efficacy claim follows
from these engineering checks.
