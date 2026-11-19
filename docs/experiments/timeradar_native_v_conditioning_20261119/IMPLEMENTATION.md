# Fixed native-V conditional validation

Execution authorized from protocol commit `1be5e0001ae80b787302c2fc62f58860e101cd11`.
Only public SWaT31 and PSM25, seed2025, normal SOURCE_FIT/SOURCE_SELECT.
No R, pollution, new seeds, private data or former candidate restoration.

`native_v_conditioning.py` loads the exact official TimeRadar source/weights,
freezes all native tensors, and exposes three new normal-source predictors.
Each teacher call is serial B=1 with an isolated CPU/CUDA mask RNG seeded84551;
eval alone is insufficient. The actual ten complementary masks, native order,
terminal10×C reconstructions, original/native-scaled observations and footprints
are retained outside Git. A is the official correction=1 copy variance, averaged
over channels at position99, not an energy or reconstruction-error proxy.

B uses a 512×C history MLP32/32. C reuses the official TimesFM3 residual and
mixing transformer code at width128/layers2/heads4/patch16 with causal temporal
and contemporaneous variate attention. There are no official TimesFM weights.
The existing thin temporal/provenance loader is reused, not the old V1 model.
The final ordered C×128 tokens preserve channel identity. This is not explicit
directed Cross-lag. D shares C's actual encoder initial tensors and predicts
normalized current raw variables with its own linear head and MSE, without V
in its loss or criterion. B/C/D are not an equal-parameter score ablation.

For B/C, `sV=max(FIT_quantile95(V),1e-12)`; train the three normalized quantiles
using equally weighted pinball. Zero final weights and FP32 FIT quantile biases
give the step0 global-constant reference. Stable sorting permits ties; no forced
positive width is introduced. Archived FP32 predictions are promoted to FP64:

`score=(V/sV-q95)/max(q95-q05,1e-3)`.

Negative scores remain. No truncation, two-tail combination, extra evidence or
threshold search. D's per-channel FP32 prediction/observation is promoted before
squaring and channel averaging. The current observation only enters scoring;
predictor input is exactly `[t-512,t)`. Native V uses `[t-99,t+1)`, available t.

Both domains share fixed no-replacement20-epoch plans across B/C/D, batch128,
Adam1e-4, FP32, retain the partial batch. SWaT3880 and PSM1040 actual updates
follow the20epoch/4000update first-limit rule. Full normal SELECT is weighted
by sample count; evaluate epoch0 and each epoch, retain earliest strictly
minimum own-task loss. CAL/QUERY values are guarded until all six selections
freeze. Labels are joined only after all16 final CAL/QUERY banks seal.

Run from a workspace containing the SHA-bound external assets, with this repo's
`src` on PYTHONPATH. The fixed config is
`configs/native-v-conditioning-20261119.json`. Existing valid phases are reused;
the output directories are exclusive-create, not overwrite/retry switches.

```text
python -m hybrid_tsad.native_v_conditioning_runner prepare --workspace WORKSPACE
python -m hybrid_tsad.native_v_conditioning_queue normal --workspace WORKSPACE
python -m hybrid_tsad.native_v_conditioning_queue remaining --workspace WORKSPACE
python -m hybrid_tsad.native_v_conditioning_evaluation --workspace WORKSPACE --root RAW_ROOT --out METRICS_ROOT
python -m hybrid_tsad.native_v_conditioning_curves --workspace WORKSPACE --root RAW_ROOT --metrics METRICS_ROOT/METRICS.json --out CURVE_ROOT
```

The owned server runtime and its source bytes are frozen separately for caching
and training. Their manifests bind the actual executed files; Git line-ending
normalization is not a substitute for those byte hashes. Per-job GPU/PID/exit,
cleanup, occupancy, body timing and allocated/reserved memory are archived.

Evaluation keeps the exact512 old mixed CAL points, own q99/higher/strict `>`;
PSM CAL is all attack-labeled, with no normal1%false-alarm guarantee. The QUERY
axes are SWaT `[608535,944856)` and PSM test `[23016,87744)`. Full record identity
distinguishes PSM train/test even when numeric indices overlap. External LEFT,
MLP, CANDI, history and old native TimeRadar caches retain their differing
training/scaler/context/availability permissions; ranking-only results acquire
no invented new CAL threshold. This is historically exposed retrospective
public-data validation with causal endpoint footprints, not a live field trial.

Only affected code tests and one bounded role-independent scientific
recomputation are required. The final results, costs and checkpoint/cache hashes
are reported after completion. No follow-on method or adaptation is automatic.
