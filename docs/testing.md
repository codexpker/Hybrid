# Test profiles and external integration boundaries

The public review snapshot is a research archive, not a self-contained historical
runtime. The original 622-test log remains unchanged in
`review_artifacts/VALIDATION.json`. Root `unittest discover -s tests` does not
recurse into `tests/hybrid_v1` because it is not a Python package. The profile
runner explicitly includes that directory; its totals therefore differ.

Install CPU test prerequisites with `pip install -e '.[test,test-torch]'`.
`test` supplies NumPy/Pandas/SciPy/scikit-learn/psutil/threadpoolctl;
`test-torch` adds Torch/einops. Model integrations can additionally require
`.[model]` (including torch-frft), author checkouts and their own requirements.
These are dependency declarations, not a lockfile or an installation performed by
this repair. CUDA execution is not part of the normal test profile.

```
python scripts/run_test_profiles.py --profile unit --output /new/unit.json
python scripts/run_test_profiles.py --profile integration --output /new/integration.json
python scripts/run_test_profiles.py --profile all --output /new/all.json
```

`tests/profiles.json` is the inspectable classification. Unlisted tests default to
unit: an unknown failure cannot silently become an exclusion. Each module runs
in a fresh CPU process so metadata tests can truthfully assert that Torch was not
imported. Two mixed modules (`test_candidate_experiment`, `test_shared_engine`)
use a process per case: loading sklearn before Torch in the same Windows process
triggered c10.dll initialization errors. This isolates state without pre-importing
Torch or weakening any assertion. Every exclusion is enumerated as
`INTEGRATION_NOT_RUN`, never counted as a passing test; an all-excluded module is
likewise not labeled PASS. Tests' original skip decorators are reported separately.
Raw `python -m unittest ...` behavior is unchanged.

`test_stage2_readout` also runs per case. Its complete synthetic chain repeatedly
checks the 18-model archive, calibration provenance, source/runtime hashes and
semantic attacks. The original 180-second MODULE budget cut off this cumulative
work. Each case now has the SAME 180-second limit, enforced by the supervising
process; discovery is separately bounded. There is no enclosing 180-second
module timer that kills otherwise healthy cases. Reports identify individual
timeouts/process errors, retain them as failures, and continue to the next case.
No verification or expected hash is cached away or disabled to reduce runtime.
The fixed follow-up completed with **6 passed / 1 external-source skip**:
the complete-chain case took 120.391 seconds; all cases together took 215.859
seconds. This explains why the former 180-second module cap was insufficient
without requiring a larger single-case timeout. Timed native-stack sampling in
the first diagnostic process exited with a Windows access violation; that failure
is retained, its native cause is not claimed, and it did not recur in the later
passing profiling/ordinary runs.

External integrations include pinned LEFT/CATCH/RoCA/TimeRadar/TimesFM sources,
original initial weights, frozen configs and archived child/runtime layouts.
Relevant environment bindings are recorded per entry. An unset binding is a
specific prerequisite blocker, not a software test success. Operational L0 tests
require their own real release/assets and are not plain unit tests.

Historical frozen identity assertions remain enabled. In particular, old
seven-file/event-data and pre-chunk-readout fixtures compare frozen identities
against current source; F04's new identity API deliberately changes that source.
These are `INCOMPATIBLE_CURRENT_CODE`, not missing data and not a reason to re-sign
an old manifest. The legacy Stage2 metric/event/score oracle likewise remains
unchanged; bind the exact archived runtime for historical reproduction, rather
than pretending that the repaired package is byte-identical.

Bounded portability repairs: POSIX resource imports are deferred until actual
production extraction/wave execution (the original resource enforcement still
applies there); pure statistics/metadata imports now work on Windows. The physical
reference test explicitly closes its read-only memory map before temporary-file
cleanup; the production preprocessing formula and integrity checks are unchanged.
No tests train a new scientific candidate or evaluate real QUERY data as part of
this repair. The optional synthetic CUDA RNG check is opt-in via
`HYBRID_AUDIT_CUDA_RNG=1`; this runner removes that variable and hides CUDA.

Actual bounded repair results and remaining cases: [F06_TEST_RESULTS.json](audit_c600125/F06_TEST_RESULTS.json).
The recursive first run remains recorded as not green. The Stage2 readout
follow-up identifies the expensive complete-chain test and its repeated
`validate_definition -> Assets.file/resolve_path/sha256` calls, rather than
attributing the CPU cases to absent external data. Its real-core transport case
still requires `HYBRID_OFFICIAL_FILE` (unbound locally), separately from installed
Torch. The exact old runtime exists locally and its COPY/identity agree; among
its five pinned kernels only current engine/evaluation.py differs. See
[kernel hashes](audit_c600125/F06_frozen_kernel_check.json). No integrity oracle
was reset; F06 remains partially closed overall.
