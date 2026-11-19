# Frozen response geometry / conditional energy: fixed A–B experiment

Status: actual source training running, no QUERY labels opened. This is a new,
public-data-only experiment, not a restoration of V1 or its stopped adapters.

## Fixed computation

The trainable generator uses pinned TimesFM3 ResidualBlock and two
MixingTransformer layers (width128, heads4, patch16, variate attention enabled).
It reads exactly512 past values. Shared8/per-variate8 Gaussian noise produces
K=4 conditionally independent current-value samples. Its612626 parameters are
identical at initialization between A/B. It does not load TimesFM3 weights.

For each current candidate v and observed y, separately call the pinned native
TimeRadar model on `[past99, v]` or `[past99, y]`, B=1, norm0, ten complementary
masked reconstructions. Official1,636,043 parameters are frozen; no historical
adapter is installed. The generated-input path through FRFT, encoding,
decoding, iFRFT and abs is differentiable. Only the observation path may use
no_grad. Variable sharing *inside* one window follows the official model;
different windows or candidates never share their estimated order.

A embeds `[v/sqrt(C), vec(response(v))/(s_r*sqrt(10*100*C))]`. B embeds only
`v/sqrt(C)`. Both use the same unbiased energy formula:

`mean_k ||z_k-z_y|| - sum_{k<l} ||z_k-z_l||/[K*(K-1)]`.

This is half the author's JMLR energy convention, uniformly for A and B.
The published API returns a batch aggregate even with mean=False. Our thin
per-observation implementation is checked against the pinned original method
for values and gradients; no scalar batch loss is broadcast to time points.
Finite U-estimates are not clipped. The ten mask copies are feature
coordinates, not the independent K prediction samples.

JMLR reference commit `6f28942f6a703c2b52501d01258ca2708539f209`,
`src/scoring_rules.py` SHA256
`8d0e065143385c77e0e1f91970385bb80c72e9780643275e5385855e5e41f7bf`.
No explicit redistribution license was found in the fixed root/README. The
original file is an external local regression reference, not bundled in this
public repository. The implementation cites the mathematical source rather
than claiming an unchanged official training program.

## Randomness, scales and selection fixed before QUERY

- FIT-only per-channel population mean/std; std<1e-8 becomes1; no target fitting.
- s_r: aggregate RMS of every complete, FIT-anchored nonoverlapping100-point
  native response, all ten copies/all elements; floor1e-6. SWaT3960 windows,
  s_r=.4159373753146716; PSM1059 windows, s_r=.502003460454408.
- Mask seed84551 is reset in an isolated CPU/CUDA RNG context for every B1
  call. eval() alone does not remove native random masks.
- Separate training-noise generator2025 is reset at each source/R stage;
  SELECT noise92025 resets for each validation; scoring noise uses
  `192025 + absolute_index`. A/B use the same actual plans.
- Source:2000 updates, logical batch16 via microbatch1 accumulation, Adam1e-4,
  normal SELECT every100 including step0; strict lower loss selects a weight.
- R: own selected source state, fresh Adam,2000 endpoint updates,
  `[8 target,8 source]`; mature replay is shared, not a claimed new mechanism.
- Same-domain normal initialization and natural mixed adaptation only. No
  heterogeneous multi-source pretraining, F, extra contamination or SMD run.

## Axes and comparators

SWaT QUERY `[608535,944856)`, CAL score range `[586056,607320)`;
PSM QUERY `[23016,87744)`, CAL `[18624,21888)`. Each CAL selects512 fixed
equally spaced points; each score uses its own q99/higher/strict> threshold.
The PSM role remains attack-only CAL, not a certified normal false-alarm budget.
History512 plus current explains the360-point PSM reduction from the old axis.
Original full-axis results and old thresholds are never overwritten.

Compatible LEFT/MLP/CANDI and simple caches are cropped and recalibrated on
these exact points. Their native context, available_index, source statistics,
batch/exposure and objectives remain explicit. In particular LEFT's192-point
whole-window scores are retrospective, not causal alarm competitors.
Unmatched TimeRadar/other CAL banks are marked incompatible instead of rerun.

The actual CUDA gate verifies response-only gradients, A/raw gradient
difference, frozen native tensors, isolated masks/RNG and sample independence.
It is an engineering check, not a scientific performance gate. Raw gates,
failed starts, selected weights, endpoints, per-point banks and actual
occupancy remain outside Git. Key final results receive one role-independent
same-family recomputation. No successor variant is authorized automatically.
