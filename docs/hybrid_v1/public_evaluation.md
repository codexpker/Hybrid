# Fixed processed-MSL evaluation adapter

`hybrid_tsad.hybrid_v1.public_evaluate_cli` adds public-domain metadata admission,
not a new metric implementation. The fixed objects are three original source
models (2025/2026/2027) and persistence/history-mean controls, each with its own
unchanged source calibration. All six model-minus-control pairs are retained.

The processed single-channel MSL axis has 73729 points and 73217 scores after
H=512 warmup. The last 31 arrived targets are scored; the last chunk64 has one
record. Native entity boundaries and physical time are not certified. This
benchmark cannot establish a cross-variable lag mechanism.

Order: whole-five-bank root seal, every bank/chunk/CAL identity, all five scalar
axis replays, then the sole old-NPZ label-key loader. No query-label I/O is
authorized by engineering tests or a query/cost release. Label rows are never
used to select a seed, threshold, checkpoint, structure or calibration.

The module reuses the accepted H5 `_save_results` array serializer, which calls
the unchanged evaluator, event matching and official Aff wrapper. Its historical
`h5_evaluate_cli.py` filename is archived as a reused helper, not represented as
a new MSL metric definition. Original full per-method result JSONs feed the
unchanged three-seed `evaluate_cli.aggregate`; `MODEL_SEED_SUMMARY.json` reports
mean/sample-SD/worst seed separately from the two deterministic controls. The
human SUMMARY retains the helper's `model_seed_mean=null` field; the separate
seed-summary artifact is authoritative for those statistics. There is no score
ensemble, event bootstrap or newly invented CI. Existing NA limits remain.

The new metadata8 tests use only fake bank metadata and synthetic labels plus a
serializer stub. They check gates, orchestration and aggregate schema; they do
not claim neural forward or numerical metric parity. Production execution needs
a separately signed full code closure, exact bank seal, label file and bounded
CPU resource wrapper after all raw-bank acceptances.
