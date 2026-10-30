# Support-Checking Rules

Primary grounding metrics were deterministic. They were not assigned by an LLM judge.

## Claim Parsing

Each model response was expected to contain:

- `summary`: a short paragraph.
- `claims`: a list of claim objects.
- Each claim object contained `text`, `features`, a retrieved evidence id (`source_id` or `context_id` depending on the run), and `citation`.

## Supported Claim Rule

A generated claim was counted as supported only when all of the following were true:

1. The claim referenced at least one feature id present in the case.
2. The claim cited a retrieved evidence id available in that run's selected context.
3. The retrieved evidence id was allowed to support the referenced feature according to the feature-to-evidence support map.
4. The claim did not violate the dataset pathology-scope rules.

Claims failing these checks were retained as unsupported claims.

## Metrics

- `feature_coverage`: fraction of case features mentioned by supported claims.
- `faithfulness`: supported claims divided by total generated claims.
- `hallucination_rate`: unsupported claims divided by total generated claims.
- `citation_accuracy`: supported claim citations that pointed to retrieved evidence.
- `quality_score`: `feature_coverage + faithfulness + citation_accuracy - hallucination_rate`. Faithfulness and hallucination rate are complementary when each claim is either supported or unsupported. The composite score is a descriptive summary, not an independent validation of clinical correctness; its component metrics should be reported alongside it.

## Code locations in this release

- Claim construction and support filtering: `scripts/run_three_llm_full_dataset_pilot.py`, function `build_explanation`.
- Metric computation: `gait_rag/metrics.py`, function `evaluate_explanation`.
- Paired summaries and composite-score calculation: `scripts/summarize_full_dataset_batch.py`.

The supplementary ablation and modern-retriever analysis scripts are not included in this release draft; their quality-score implementation must be added and checked before those sensitivity analyses are described as reproducible from this repository.
