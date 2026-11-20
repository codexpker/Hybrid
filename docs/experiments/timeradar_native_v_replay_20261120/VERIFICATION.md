# Bounded checks and public export

All19 affected CPU cases passed, without a full-project audit or neural QUERY rerun:

```text
PYTHONPATH=src python -m pytest -q tests/test_native_v_replay.py tests/test_native_v_conditioning_evaluation.py
15 passed in9.39s
PYTHONPATH=src python -m pytest -q tests/test_native_v_conditioning.py
4 passed in1.60s
```

The only warning is the existing anyio pytest plugin CallSpec2 deprecation. Real normal-data GPU gates separately confirmed frozen states, nonzero decoder gradients, per-window order/masks, own initialization, loss scaling and RNG restoration before the six runs. Training/inference were not repeated to generate this record.

Key-result review:16458 checked fields, maxdifference3.3306690738754696e-16, no scientific discrepancy. Original FINAL_AUDIT.json SHA0517d8147d9f13d1c9c8ac7a55022653f3503d51eb9f95e255b82b093c0294de. Same-review finite public export check:33446 checks,0 differences; PUBLIC_EXPORT_CHECK.json SHA2a18bad1ab1213facd00137817250ca065812147b83db2f96bbc3315a03a6eaa,0.14895s CPU. Not external double blind. METRICS and the original audit are not rewritten to avoid circular hash bindings.

The export check records exact local working-file bytes, including pre-existing user Introduction indentation; that two-space user edit remains unpublished. Git may normalize line endings. Numeric CSV content and reviewer-checked prose are unchanged; raw SHA bindings identify the original physical artifacts rather than normalized Git blobs. Full raw banks, labels, scaler, full pretrained-containing checkpoints/optimizers and runtime logs remain in the local results tree, not public Git.

No GPU process or automatic successor remains. Historical V-condition B/C, their R and all other stopped branches stay closed.
