# Bounded source and writing record — 15 November 2026

The new editable draft is `draft.md`. The prior `manuscript/next_generation_20261115/draft.md`, including the user's uncommitted edit, was read but not modified. This writing pass ran no model, opened no QUERY labels, and did not select a new configuration. Method definitions follow `configs/normal-evolution-decisions-20261115.md`.

## Newly read primary materials

| Work | Reading and official implementation | Scope used in the draft |
|---|---|---|
| Balsells-Rodas, Wang and Li, ICML 2024, PMLR235:2639–2672, [formal record](https://proceedings.mlr.press/v235/balsells-rodas24a.html) | Formal 34-page PDF obtained; §§2–4 and §6 read, theorem assumptions and inference equations checked. [Author repository](https://github.com/charlio23/identifiable-SDS) fixed at `b6c11a5f76b3a7bedd0506d65ff843a9ed2a8a48`, MIT. Actual `models/MSM.py::_forward`, `LogLikelihood`, and `models/NeuralMSM.py::_compute_local_evidence`, `_maximization`, `fit` read. | Mature switching/AR distribution and log-space forward recursion. Their neural MSM is trained using generalized EM and paired-marginal Q updates; Hybrid's differentiable exact forward likelihood is an independent implementation, not an official reproduction. Identifiability assumptions do not certify real operating-state semantics or detection performance. |
| Baumgartner, da Silva and Urteaga, **UAI 2026**, PMLR337:448–468, [formal record](https://proceedings.mlr.press/v337/baumgartner26a.html) | Formal 21-page PDF obtained; §§3–4, experimental/threshold details read, earlier arXiv HTML cross-read. Formal paper's footnote identifies [official code](https://github.com/iurteagalab/probML_anomaly_detection), fixed at `559db8f7c93d7f48d29bb675e1cdf66acb263f17`, GPL-3.0. Actual `src/canary_blocks.py` RealNVP/CondDiagGaussian, `generic_util.py` train/readout, and `real_main_experiment.py::extract_component_statistics` read. | Latent dynamics plus compliance is prior art. Their code trains conditional flow NLL, whitens latents and calls MV-KS; it includes broader configuration/search and threshold procedures not inherited here. This paper must no longer be described only as an unaccepted 2026 preprint. |
| Rosenblatt, 1952, Annals of Mathematical Statistics23(3):470–472, [DOI](https://doi.org/10.1214/aoms/1177729394) | Classic conditional-CDF transform; bibliographic identity checked against publisher redirect and existing bibliographic records. Publisher HTML did not expose full article text in this pass; no claim of newly reading its full proof. | Credit the successive conditional probability transform. The mixture formula in the draft is obtained directly by conditioning the stated Gaussian mixture; it is not a new probability-transform theorem. |

Official-code snapshots and the two full PDFs are local research evidence only at `research_workspace/literature/evolution_20261115/`. The GPL source was **not copied into the Hybrid package**. Snapshot names: `sds_models_MSM.py`, `sds_models_NeuralMSM.py`, `sds_LICENSE`, and `uai_src_canary_blocks.py`, `uai_src_generic_util.py`, `uai_src_real_main_experiment.py`, `uai_src_generic_scoring.py`. They document inspected functions, not a claim that either upstream repository was executed.

## Reused existing reading evidence

LEFT/TimeRadar/CATCH bibliographic identities and original-code reading were reused from the existing manuscript bibliography and project literature records. GANF, MTGFlow, Real NVP, and Wavelet Flow evidence was reused from `literature/normality_density_20261115/EVIDENCE.md` and its pinned source snapshots. No old model was retrained or rebranded. MTGFlow's cited item remains explicitly an extension preprint where a formal identity was not relied upon.

## Claims not made

- No assertion that this composition is first, that each mathematical component is original, or that experimental improvement exists before results are accepted.
- Learned regimes are not authenticated plant commands. Normal-mode coverage and observable distinguishability are assumptions to test.
- The coordinate-tail score is not a complete GOF test and may miss central concentration/variance-collapse anomalies. It is not the upstream MV-KS score.
- No calibrated population false-alarm guarantee from fitted PIT; no causal-discovery, contamination-robustness, or independent-confirmation claim.
- No results or tables were inserted into this draft before the current experiment completed.
