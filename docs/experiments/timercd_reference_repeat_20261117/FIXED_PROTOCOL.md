# Fixed same-recipe training-seed repetition

The accepted seed2025 model, scores and evaluation remain immutable. This run adds only seeds2026 and2027, each with batch-statistics and past-reference-statistics training arms. All four start from the same official multivariate checkpoint and read the original1792 source instances by verified SHA; no data generation is invoked.

For each new seed, RandomState(seed) produces three consecutive permutations of1536 FIT instance IDs, grouped into batches of3:1536 updates and4608 sample exposures. Its two arms share this exact plan and training masks. Training mask bases are8456100 and8457100 respectively. Source SELECT always uses the original mask base8455100 plus900000 and batch start, per-instance past-R statistics, the same joint loss, and the same step0-inclusive minimum-loss checkpoint rule. R is not additional encoder context.

Model, all-parameter update scope, AdamW settings, losses and update budget remain unchanged. Each selected model is evaluated on PSM and all28 SMD records using only the existing FIT-statistics B permission, original5000-window inference, raw class-logit difference and its own lawful CAL q99/higher/strict-greater threshold. No QUERY label selects a checkpoint, threshold, seed or entity. Evaluation randomness remains fixed. These are development evaluations and SMD entities are not independent dataset domains.

RUN_BINDING.json records the pre-result configuration and source-plan identities. The two public plans contain generated-source instance indices only, with no private data. Resource logs include process startup and any failures; unrelated server jobs are not evicted.
