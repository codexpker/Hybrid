# CPU-only public handoff validation — 2026-11-14

No GPU training, new model candidate or historical metric recomputation was performed. These checks test packaging and existing synthetic fixtures.

- Python3.12.12, local packages recorded in `VALIDATION.json`; this is the CPU handoff environment, **not a fabricated lockfile for all historical GPU experiments**.
- `python -m compileall -q src scripts tests`: passed. Final `compileall` including `review_artifacts/code` also passed. CLI help and the 89-file export restoration check passed without materializing an external workspace.
- Default single-process `python -m unittest discover -s tests -v`: **622 tests reported, 17 failures, 21 errors, 16 skipped**, 296.663s, exit1. This snapshot is not represented as a clean, self-contained test suite.
- Two relevant failed import-order assertions were checked in separate fresh CPU processes: `test_r2_learnability.py` (6 tests) and `test_stage2_u_evaluation.py` (8 tests), both passed. This supports the test-isolation diagnosis; it does not erase the initial failed run or prove all remaining failures resolved.

Observed limitations retained for the independent reviewer:

1. Some tests deliberately require fixed external LEFT/CATCH/RoCA/TimeRadar sources, runtime configuration or original reference fixtures; missing assets are not silently substituted. Variables include `LEFT_OFFICIAL_ROOT`, `CATCH_SOURCE_ROOT`, `ROCA_SOURCE_ROOT` and `HYBRID_V1_WORKSPACE`.
2. Several metadata-only tests assert Torch has never been imported into their Python process. A combined discovery run also imports model tests, causing order-dependent failures. Run those test modules in fresh processes; do not weaken their checks merely for a green summary.
3. Some historical helpers require Linux's `resource` module; one Windows test also encountered temporary memmap file locking.
4. A historical Stage2 frozen metric/oracle identity check failed. No kernel hash was reset and no scientific calculation was changed to suppress that failure.
5. Missing external dependencies and upstream fixture checks remain visible, including skipped cases. No download or external-code redistribution was performed for tests.

The compact JSON records test names and categories without private paths or giant `sys.modules` dumps. Full local logs are deliberately excluded from the public repository. Exact aggregate experiment conclusions are separate from these engineering checks.
