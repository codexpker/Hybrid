# Six project-trained normal-source predictors

These are this experiment's selected B/C/D checkpoints, not official TimeRadar
or TimesFM pretrained weights, a finished deployed detector, or evidence of a
positive scientific outcome. The byte-identical originals remain in the raw run.
`MODEL_MANIFEST.json` gives each SHA256, selected epoch/update, own normal SELECT
loss, parameter count and frozen scaler/sampling identity. All20 epochs were
completed; the selected checkpoint can precede the endpoint.

Each file contains only `model` tensors, `epoch`, `updates`, `loss`; it includes
neither raw observations, annotations, optimizer state, paths/credentials nor
third-party pretrained parameters. Source data are the authorized public SWaT31
and PSM25 normal training roles defined in the protocol. No laboratory/private
data, target labels, CAL or QUERY enter these weights. Obtain datasets separately
under their original terms; publishing these weights does not relicense data.

B is a simple MLP. C/D use the official TimesFM3 code modules under Apache2.0,
lightly configured and initialized/trained by this project. Preserve the included
TimesFM code license and provenance. No official TimesFM model is loaded. Do not
imply Google endorsement, official pretrained capability, or an unrestricted
commercial license for the entire repository.

Load only after constructing the matching model/domain/config, with a supported
PyTorch version and `torch.load(..., weights_only=True)`. Read `model` with strict
state loading. B/C require the separately authorized official frozen TimeRadar
source/checkpoint to produce the native V and the exact source scaler/sV.
D's criterion uses its original-variable forecast only. Scaler and source/global
quantile identities are provided by the artifact manifest; original numeric
assets stay outside Git. Recompute them only from authorized source FIT when
reproducing, never from SELECT/CAL/QUERY. No deployment or safety guarantee is
made by a source-selected model file.
