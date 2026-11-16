# Fixed receiver-selective context adaptation

One bounded aggregation experiment on the unchanged official TimeRadar backbone, decoder and scorer. This is a hypothesis, not an established diagnosis that irrelevant channels caused earlier failures.

## Frozen mechanism

For complex encoder tokens `z[K,B,C,P,D]`, retain the original layout. With D256 and rank r16:

```text
h_i = GELU(W_down [Re(z_i), Im(z_i)] + b_down)
q_i = Q h_i; k_j = K h_j
a_ij = softmax over j!=i of (q_i^T k_j / sqrt(r))
context_i = sum_{j!=i} a_ij h_j
z'_i = z_i + complex_split(W_up context_i + b_up)
```

Q and K are bias-free16x16 maps; the final up projection initializes to zero. The native `z_i` residual always preserves the own-variable path. This is not a forced positive diagonal attention probability. There is no new V projection, dropout, hard channel mask, top-k, lag search, head, loss or score. Aggregation occurs only within one K/B/P group. One-channel input uses h_i; with two channels Q/K cannot identify a choice among donors.

The exact-parameter capacity control uses the same initialized down/up/Q/K tensors and native residual, independently for each variable:

```text
context_i = h_i + Q GELU(K h_i)
```

All512 extra parameters are active in this control. It matches parameter count, not exact FLOPs or function class. Old Self/mean Cross adapters have16912 parameters; each new adapter has17424, with811034 decoder-plus-adapter trainables (verify numerically). The capacity control is not a second proposed method. Compatible old Head/Self/mean-Cross results are reused.

## Reference and attribution

CATCH, ICLR2025, [paper](https://arxiv.org/abs/2410.12261), [official fixed code](https://github.com/decisionintelligence/CATCH/tree/3647c69be5eb56649b072596cf89098e689e20c3), supplies the principle of receiver-specific channel selection. Its actual method includes learned patchwise Gumbel masks, diagonal protection, Q/K/V attention and separate channel-discovery optimization. This experiment uses independently written soft-weighting code, without those extra objectives or the complete CATCH architecture. Upstream source and weights remain external. Attention is not evidence of causal relations or time lags.

## Fixed experiment

- PSM and SMD1-1/1-2/1-3, seed2024 only. Exactly two new trained arms: `selective` and `capacity_self`; no new entity, seed, pollution or F experiment.
- Scientific baseline7480d5f; engineering start ef213b3. Original model/runner bytes, source weights, physical data binding d10cd11, FIT-only scaler and all role boundaries remain unchanged.
- Same native initialization, frozen body/eval policy, decoder updates,2000 updates, AdamW1e-4,200 warmup and original reconstruction MSE. Normal SELECT at0/every200 with earliest minimum; no optimization search or QUERY feedback.
- Same down/up initialization and within-seed FIT/mask streams. New arms also share Q/K initial tensors. Reseed after construction, as in the accepted runner.
- Preserve original score seed2024, CAL-to-QUERY RNG stream, H100/stride100, copies10, PSM B64/norm0 and SMD B32/norm1, native L1 copy variance and each score's own CAL q99higher strict `>` threshold.
- Main comparison reuses the16 old native-axis rows and appends8 new rows. External comparison reuses24 accepted LEFT/H512 common-axis rows and appends8. Native window and batch dependencies remain offline; C3 does not make them strict online scores. LEFT's unfinished SMD curriculum remains disclosed and is not extended per QUERY.
- Seal all16 new CAL/QUERY banks before label evaluation. Report AP/ROC, strict/C3 point metrics, any-hit and one-to-one events, normal false-positive points/fragments, delay offsets, sustained/long-event coverage, longest misses and UNKNOWN.

## Necessary checks and stopping boundary

Zero/disabled equivalence, complex layout, K/B/P isolation, receiver-specific context, finite single-channel fallback, active Q/K gradients after up updates, and exact capacity matching precede real training. Fixed FIT-only diagnostics use the first four complete windows, compare selective versus forced-uniform context with identical weights/RNG, and intervene on a named donor value. Check effects on actual residuals, decoder outputs and native scores. Joint channel permutation is equivariant and is not an effective falsification. No forced-uniform QUERY scoring candidate is added.

Use at most four existing A40 tasks, training20GiB/scoring30GiB, fresh device checks and bounded job timeouts. Record actual occupied time, PIDs, failures and cleanup; preserve other users' work. One bounded key-result audit follows completion. Keep all negative evidence and do not automatically expand local variants if the increment is insufficient.

## Actual execution identity

The first FIT integration run completed its numerical checks but failed when writing a receipt: the descriptive `interventions` keyword collided with the statistics dictionary. Four failed FIT jobs and their actual compute remain archived; no formal training or QUERY evaluation occurred in that run. The sole repair renames that descriptive field to `intervention_definition`.

The execution config is therefore `configs/timeradar-selective-context-v2-20261116.json`, whose only differences from the original config are `id` and `output`. This is an output-identity repair, not a second scientific candidate. The evaluator must explicitly receive this config. Both original failed and repaired-run identities are retained; the aggregation formula and all training/scoring settings are unchanged.
