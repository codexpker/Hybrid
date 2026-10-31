# Research engineering rules

- Read README, docs/research_scope.md and provenance records before changing algorithms.
- This is an independent candidate, not an official TimeRadar or TimesFM implementation. Never claim anomaly-contamination robustness or model improvements without appropriate evidence.
- When official source exists, inspect its actual code and pin its revision before implementing a related change. Separate reused code, independently written code and conceptual references. Do not redistribute unlicensed upstream code or weights.
- Use the package API, shared modules and versioned configurations. Do not create one new Python driver per experiment. Preserve immutable historical runs externally.
- Engineering refactors must preserve core hashes or demonstrate checkpoint/output equivalence. Do not silently alter architecture, loss, scores, precision, random masks or data permissions.
- Train/validation/calibration/query roles, full window footprints and independent event boundaries must be explicit. Never select thresholds, weights, seeds or numerical variants using query labels.
- Keep raw data, weights, private documents, credentials and personal server configuration outside Git. Use local configuration paths.
- Commit runnable baselines and negative results as well as improvements. An improvement commit must reference fair evaluation evidence; a larger model or lower prediction loss is not itself a detection improvement.
- Do not launch training, download large assets or publish additional private material merely because an entry point exists. Follow the current research contract and resource budget.
