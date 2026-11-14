# CAROTS SOC published-method control

Source: ICML 2025 CAROTS, official default branch `master`, commit
`63a41f45a2ae680d5cdbb2d5739e7a11264867cb`:
<https://github.com/kimanki/CAROTS/blob/63a41f45a2ae680d5cdbb2d5739e7a11264867cb/models/carots/loss.py>.
Formal paper: <https://proceedings.mlr.press/v267/kim25aa.html>.

`upstream_loss.py` is byte-identical to the author file. `LICENSE` is the
author's Modified MIT Non-Commercial with Permission, **not unrestricted MIT**.
This license notice covers the vendored code and its local adapter; no claim
is made to relicense it for commercial use. It must remain with distribution.

`soc_objective` accepts already-extracted embeddings only, ordered `[G_P,G_N]`
with corresponding negative rows. It adds no parameters, projection head,
augmentation, causal discoverer, trainer or score. It is disabled by default.
The caller must bind feature identity and predefine lawful augmentation,
training roles and thresholds. Same-feature identity projection is required
for any paired control; this function cannot create those samples or verify
their semantics. It does not establish causality or contamination robustness.

With valid positive support the **original** loss function is called unchanged.
No nonself positive support returns a differentiable zero and explicit metadata;
this is a safety repair to avoid the upstream zero denominator. Unsupported
self-pair configurations, zero-norm rows and non-finite inputs/loss raise instead of silently
changing the formula. Finite loss is checked explicitly; extreme temperatures
can still overflow the author's exponentials and must fail, not be retuned on
query labels. Defaults retain the author's .1 temperature and .5 similarity.

This is a named published learning-objective control, not a claimed innovation
or full CAROTS reproduction. No training, model forward, score or protocol is
activated by adding these files. Augmentor code is deliberately not copied:
the author's default CPA time slice does not feed the perturbed endpoint into
its forecaster and requires separate, explicit time semantics if later adapted.
