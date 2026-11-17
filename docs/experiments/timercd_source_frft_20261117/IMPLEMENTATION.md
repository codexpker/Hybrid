# Source-supervised patch representation experiment

This is a new Hybrid candidate, not the closed TimeRadar reconstruction adapter.
Pinned Time-RCD `b031773e323bc0e6c9dcd8c9119b800ec4d70f8c` and the archived
official multivariate checkpoint are shared by all three arms. Only the existing
`ts_encoder.embedding_layer` is wrapped. Original positions, variable identities,
context encoder, two heads, normalization and inference window remain unchanged.

For each original length-16 patch p, independently of other patches and samples:

* native: e = Wp+b;
* time control: e = Wp+b + W2 GELU(W1 [p; GELU(a p)]+b1)+b2;
* Hybrid: e = Wp+b + W2 GELU(W1 [Re(D_a p); Im(D_a p)]+b1)+b2.

`D_a` is `torch_frft.dfrft_module.dfrft`, applied only along the final patch axis.
`a=2 sigmoid(raw_order)` starts at 0.5; the scalar is shared, not estimated by
mixing samples. Fractional coordinates are not physical lag coordinates. Both
controls have hidden width32 and exactly equal parameter counts, including the
scalar. W2 and b2 start at zero; disabling the wrapper calls the native embedding
directly. Real official full-model/head CPU checks include B2/T33/C2 and padding,
and are bitwise equal at initialization and when disabled. TimeRadar supplies the
learnable-FRFT technical motivation; no second network or TimeRadar weights are
loaded. This local DFRFT residual is project code, not an official TimeRadar module.

All original backbone and both heads update in every arm. Training minimizes
valid-time binary cross entropy of mean-channel logits plus masked reconstruction
MSE of the original synthetic input. The mask uses actual checkpoint patch16 and
15% patches, with the author's small Gaussian replacement. Original padding labels
are excluded before binarization; reconstruction also excludes padding. These
two integration safeguards apply identically to all arms; upstream is unchanged.
The reconstruction target is the original input, not the generator's uncorrupted
counterpart. The latter is not used as privileged training supervision.

Source instances come from the official generator pinned to
`73eb733f3026e32bf9edfde1d4c3b787358908e7`, with fixed per-instance seeds and
disjoint instance-level FIT/SELECT. Config/plan and source hashes identify actual
exposure. This is synthetic supervised anomaly learning plus auxiliary masked
self-supervision, not pure unsupervised learning or demonstrated contamination
robustness. The bounded continuation is not a full official pretraining reproduction.
Its public config fixes length512/C8,1536 FIT/256 SELECT, three epochs, batch3,
AdamW1e-4/wd1e-5; source SELECT joint loss alone chooses checkpoints including
step0. The public code default supplies lr1e-4/batch3; the paper's full pretraining
uses lr5e-4/dynamic batch and a much larger corpus. No real QUERY chooses these.

Reference for joint training and the reconstruction role: Time-RCD paper,
[method and training details](https://arxiv.org/html/2509.21190v3).
Public API source lacks a complete pretraining loop; the project runner explicitly
wires its supplied CE and reconstruction definitions with their summed objective.
Masks and source instance order are paired; training RNG resets after construction.

Evaluation keeps all PSM/SMD28 objects, window5000/batch1 and original axes.
B uses original FIT statistics once; A retains whole-role statistics and is an
offline parallel reference. Every new score is abnormal-minus-normal raw logit;
each own CAL uses q99/higher and strict greater-than. Old accepted external tables
are hash-reused, never replaced by per-object preferred values. All are development
results, not independent confirmation or evidence about physical causality/lags.
