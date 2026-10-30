from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METHODS = ("embedding_rag", "vectorless_tree_rag")
METRIC_FIELDS = (
    "feature_coverage",
    "faithfulness",
    "hallucination_rate",
    "citation_accuracy",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export method-level and claim-level CSV tables from batch JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-method-output", type=Path, required=True)
    parser.add_argument("--pathology-model-method-output", type=Path, required=True)
    parser.add_argument("--claims-output", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    export_metrics(records, args.model_method_output, group_fields=("model",))
    export_metrics(records, args.pathology_model_method_output, group_fields=("pathologyKey", "model"))
    export_claims(records, args.claims_output)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
    return [record for record in records if record.get("status") == "ok"]


def export_metrics(records: list[dict[str, Any]], output_path: Path, group_fields: tuple[str, ...]) -> None:
    rows: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = record.get("patient_metadata", {})
        group_values = tuple(group_value(record, metadata, field) for field in group_fields)
        for method in METHODS:
            rows[group_values + (method,)].append(record)

    fieldnames = list(group_fields) + [
        "method",
        "n_records",
        "feature_coverage_mean",
        "faithfulness_mean",
        "hallucination_rate_mean",
        "citation_accuracy_mean",
        "quality_score_mean",
        "mean_claims_per_case",
        "total_claims",
        "unsupported_claims",
        "records_with_unsupported",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows):
            *group_values, method = key
            group_records = rows[key]
            row = {field: value for field, value in zip(group_fields, group_values)}
            row.update(method_metrics(group_records, method))
            writer.writerow(row)


def method_metrics(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    evaluations = [record.get(method, {}).get("evaluation", {}) for record in records]
    explanations = [record.get(method, {}).get("explanation", {}) for record in records]
    claim_counts = [len(explanation.get("claims", [])) for explanation in explanations]
    unsupported_counts = [len(explanation.get("unsupported_claims", [])) for explanation in explanations]
    return {
        "method": method,
        "n_records": len(records),
        "feature_coverage_mean": rounded_mean(evaluation.get("feature_coverage", 0.0) for evaluation in evaluations),
        "faithfulness_mean": rounded_mean(evaluation.get("faithfulness", 0.0) for evaluation in evaluations),
        "hallucination_rate_mean": rounded_mean(evaluation.get("hallucination_rate", 0.0) for evaluation in evaluations),
        "citation_accuracy_mean": rounded_mean(evaluation.get("citation_accuracy", 0.0) for evaluation in evaluations),
        "quality_score_mean": rounded_mean(quality_score(evaluation) for evaluation in evaluations),
        "mean_claims_per_case": rounded_mean(claim_counts),
        "total_claims": sum(claim_counts),
        "unsupported_claims": sum(unsupported_counts),
        "records_with_unsupported": sum(1 for count in unsupported_counts if count),
    }


def export_claims(records: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "dataset",
        "model",
        "case_id",
        "pathologyKey",
        "feature_count",
        "method",
        "claim_type",
        "claim_index",
        "feature",
        "text",
        "source_citations",
        "evidence_ids",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            metadata = record.get("patient_metadata", {})
            for method in METHODS:
                explanation = record.get(method, {}).get("explanation", {})
                for index, claim in enumerate(explanation.get("claims", []), start=1):
                    writer.writerow(claim_row(record, metadata, method, "supported", index, claim))
                for index, claim in enumerate(explanation.get("unsupported_claims", []), start=1):
                    writer.writerow(claim_row(record, metadata, method, "unsupported", index, claim))


def claim_row(
    record: dict[str, Any],
    metadata: dict[str, Any],
    method: str,
    claim_type: str,
    index: int,
    claim: dict[str, Any] | str,
) -> dict[str, Any]:
    if isinstance(claim, dict):
        text = claim.get("text", "")
        feature = claim.get("feature", "")
        source_citations = claim.get("source_citations", [])
        evidence_ids = claim.get("evidence_ids", [])
    else:
        text = str(claim)
        feature = ""
        source_citations = []
        evidence_ids = []
    return {
        "dataset": record.get("dataset", ""),
        "model": record.get("model", ""),
        "case_id": record.get("case_id", ""),
        "pathologyKey": metadata.get("pathologyKey", ""),
        "feature_count": record.get("feature_count", ""),
        "method": method,
        "claim_type": claim_type,
        "claim_index": index,
        "feature": feature,
        "text": text,
        "source_citations": "; ".join(source_citations),
        "evidence_ids": "; ".join(evidence_ids),
    }


def group_value(record: dict[str, Any], metadata: dict[str, Any], field: str) -> str:
    if field in metadata:
        return str(metadata.get(field, ""))
    return str(record.get(field, ""))


def quality_score(evaluation: dict[str, Any]) -> float:
    return (
        float(evaluation.get("feature_coverage", 0.0))
        + float(evaluation.get("faithfulness", 0.0))
        + float(evaluation.get("citation_accuracy", 0.0))
        - float(evaluation.get("hallucination_rate", 0.0))
    )


def rounded_mean(values: Any) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return round(mean(values_list), 6)


if __name__ == "__main__":
    main()
