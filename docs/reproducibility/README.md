# Reproducibility files

- `generation_prompts_and_schema.md` and `.json`: the primary generation prompt, output schema, and generation settings. The prompt is also embedded in `scripts/run_three_llm_full_dataset_pilot.py`.
- `support_checking_rules.md`: definitions of the fixed feature/citation checks and derived metrics.
- `claim_audit_examples.json` and `.csv`: representative claim-level examples.
- `ontology_graph_retrieval.md`: a description of the OGR method as reported in the study.
- `../../supplementary/ESM_1.tex` and `.pdf`: the supplementary source and rendered document, including the reported OGR summary metrics and supplementary result tables.

The supplementary document reports OGR and supplementary results, but the current release does not contain the OGR runner, GADO ontology/mapping inputs, or primary OGR full-run records. It also does not contain every script and run-level output for the supplementary analyses. The OGR note is documentation, not an executable reproduction. See the root README for the exact limits of what can be rerun from this repository.
