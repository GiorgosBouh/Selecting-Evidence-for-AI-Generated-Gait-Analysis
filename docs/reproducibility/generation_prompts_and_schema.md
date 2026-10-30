# Generation Prompts and Output Schema

This file records the exact prompt templates used for the gait-explanation experiments. Placeholders in braces were filled at runtime with case features, retrieved contexts, pathology scope notes, and the relevant evidence-id key.

## Recommended Citation in Manuscript/Supplement

The exact generation prompts and JSON output schema are archived in `docs/reproducibility/generation_prompts_and_schema.json` and reproduced in readable form here.

## Primary Full-Run FR/TGR/OGR Prompt

Source: `scripts/run_three_llm_full_dataset_pilot.py`, function `answer_from_context`.

Used for the primary full-run FR and TGR outputs. Earlier project notes report that the final OGR rerun used this prompt, but its run file was not found for this release. This statement applies to the primary FR/TGR/OGR comparison only; audit and sensitivity analyses used the separate audit/sensitivity prompt documented below.

```text
Answer based only on the case evidence and retrieved context.

Case ID: {case_id}
Case evidence:
{case_evidence}

Retrieved context:
{retrieved_context_json}

Write concise, clinically useful gait-interpretation findings.
For biomechanical cases, explain what the measured feature means for gait function or protocol interpretation: forward progression, cadence/timing, swing phase, stance phase, double support/stability, left-right asymmetry, variability, turning performance, speed segments, repeated task segments, prosthetic-vs-intact limb provenance, sagittal-plane joint ROM, joint moment asymmetry, propulsion/push-off, or measurement provenance.
Relate the pattern to the dataset pathology label only when the retrieved context supports that interpretation.
This is not a diagnosis task. Do not make a new diagnosis, infer pain severity, prescribe treatment, or infer joint angles/step width/ROM unless explicitly present in the case evidence.
Every claim must preserve exact case values and healthy-reference z-scores for the features it uses.
If a feature is marked approximate=true, call it approximate in the claim.
Do not write metadata-only claims such as "clinical score available", "visual gait impairment was noted", "gait events were available", or "U-turn boundaries were available" unless the claim also interprets a biomechanical gait finding.
For each claim, copy all relevant feature IDs from the case evidence.
The final claims must cover every case feature listed in the retrieved context matched_features fields.
Generate one claim per matched case feature unless two closely related features are combined; if combined, the claim must name every combined feature and include exact values for each.
Use only source_id values that appear in the retrieved context.
Do not invent unsupported pathology labels or measurements.
{pathology_scope_note}

Return only JSON:
{
  "summary": "short paragraph",
  "claims": [
    {
      "text": "claim grounded in the context",
      "features": ["feature_id"],
      "{context_key}": "supporting id",
      "citation": "citation string from context"
    }
  ]
}

```

## Audit/Sensitivity Prompt

Recorded from earlier project notes as `scripts/run_revision_ablation_experiment.py`, function `answer_from_context`. That source file is not present in this release draft, so the audit/sensitivity template is documented but cannot currently be rerun from this repository.

Used for coverage-control ablations, modern retriever sensitivity runs, the earlier audit-prompt OGR/GADO run, and the OGR coverage-repair diagnostic. It is not the prompt used for the final primary FR/TGR/OGR comparison.

```text
Answer based only on the case evidence and retrieved context.

Case ID: {case_id}
Case features:
{case_features_json}

Retrieved context:
{retrieved_context_json}

Write concise, audit-ready gait-interpretation findings.
Rules:
- Use only the case evidence and retrieved context.
- Generate one claim per matched case feature unless two closely related features are combined.
- Preserve exact case values and feature IDs for every claim.
- Use only source_id values that appear in retrieved context.
- Do not add diagnosis, treatment, pain, causality, severity, or unmeasured variables.
- If evidence supports only a measured pattern, state only that measured pattern.

Return only JSON:
{
  "summary": "short paragraph",
  "claims": [
    {
      "text": "claim grounded in the context",
      "features": ["feature_id"],
      "source_id": "supporting id",
      "citation": "citation string from context"
    }
  ]
}

```

## Generation Settings

```json
{
  "format": "json",
  "stream": false,
  "temperature": 0,
  "think": false
}
```

## Output Schema

```json
{
  "summary": "short paragraph",
  "claims": [
    {
      "text": "claim grounded in the context",
      "features": [
        "feature_id"
      ],
      "source_id_or_context_id": "supporting retrieved evidence id",
      "citation": "citation string from context"
    }
  ]
}
```
