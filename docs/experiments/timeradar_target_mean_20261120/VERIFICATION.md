# Bounded implementation/result verification

The initial implementation8751199ddc149f60322fd852ce400024098f5047 passed all22 affected CPU cases:

```text
PYTHONPATH=src python -m pytest -q tests/test_native_v_replay.py tests/test_native_v_conditioning_evaluation.py tests/test_native_v_conditioning.py
22 passed in2.88s; one existing anyio CallSpec2 deprecation warning
```

No later scientific code change or full-project re-audit was made. The two real normal-source CUDA gates checked scalar and decoder-gradient identities, frozen/eval state, original plans and RNG restoration; no QUERY labels were opened. The gates were discarded rather than used as model initializations.

Original key review FINAL_AUDIT.json SHA4c312683182c43f395d9509263807909c52fa1a66e11c22a62985cad9a424cac:45048 checks, maxdifference2.220446049250313e-16, no scientific discrepancy. Two actual endpoints/optimizers/4000 traces and frozen tensors were checked; new four score banks/10part masks bind their correct endpoints. Only two new metric states were recomputed, with shared official VUS code explicitly disclosed. Stored nativeFP32V remains canonical; no CPU copy-variance replacement occurs. Correlated complementary masks justify calling the correction1 formula a copy variance, not certified unbiased uncertainty.

The same review's limited public export check passed90706 fields with0 differences:10 ranking/20 operating/16 single-factor/17 external ranking/10 manuscript rows,1070 event rows,140 eligible event pairs,4000 training-component rows/four summaries, and27 original CAL curve operating points. PUBLIC_EXPORT_CHECK.json SHA825425299dea07af511911e88760ea2052267ecde20280be154ef9334922c1f5. Its original audit hash is not rewritten, avoiding circular identities. This is same-family role separation, not external double blind.

One wording correction removed “unbiased” from the new manuscript's full-window correction1 copy-variance description. Formula denominator9, training, scoring and all metrics remain unchanged. Actual runtime/config bytes and metadata-only REUSE_BINDING are retained outsideGit. Public export checks bind physical local file bytes; Git text normalization can change blob bytes. The working manuscript includes the user's pre-existing two-space Introduction indent; that edit is preserved locally and excluded from publication, without changing reviewer-checked numerical text.

Costs:14owned GPU jobs2.747929552027425GPUh, allOS0/cleanup. Main CPU evaluation36.3143533s; cached curves30.8441185s. Independent later metadata/NumPy plus sharedVUS29.1136s includes its6.4949s metadata subset; prep.0148s/endpoint.2093s/brief CLI overhead separately recorded. Export check body.3745s/CLI.9267s; no exact all-command CPU sum is asserted. No review neural forward or GPU job occurred.

Full checkpoint/optimizer includes upstream pretrained tensors and is not uploaded. Raw data/labels, scalers, full banks and numeric full-curve arrays stay outsideGit. Hashes, fixed config, code, aggregate/event results and figures are published. No ongoing GPU process or successor is scheduled.
