from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

DIMENSIONS = [
    "relevance_to_case",
    "clinical_interpretability",
    "specificity",
    "actionability_for_clinician",
    "conciseness_nonredundancy",
    "safety_no_overclaim",
    "overall_usefulness",
]
METHODS = ["embedding_rag", "vectorless_tree_rag"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed blinded human clinical claim reviews.")
    parser.add_argument("--review-sheet", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/human_review_round1/human_review_summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("outputs/human_review_round1/human_review_summary.md"))
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    args = parser.parse_args()

    review_rows = read_csv(args.review_sheet)
    key_rows = {row["sample_id"]: row for row in read_csv(args.answer_key)}
    records = normalize_reviews(review_rows, key_rows)
    summary = build_summary(records, args.bootstrap_reps)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_reviews(review_rows: list[dict[str, str]], key_rows: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in review_rows:
        sample_id = row.get("sample_id", "")
        key = key_rows.get(sample_id)
        if not key:
            continue
        if not has_required_scores(row):
            continue
        by_label = {
            "A": {dimension: to_float(row.get(f"A_{dimension}_0_5")) for dimension in DIMENSIONS},
            "B": {dimension: to_float(row.get(f"B_{dimension}_0_5")) for dimension in DIMENSIONS},
        }
        order = {"A": key["A_method"], "B": key["B_method"]}
        by_method = {method: by_label[label] for label, method in order.items()}
        winner_label = normalize_winner(row.get("winner_A_B_tie", "tie"))
        winner_method = "tie" if winner_label == "tie" else order[winner_label]
        records.append(
            {
                "sample_id": sample_id,
                "reviewer_id": row.get("reviewer_id", ""),
                "dataset": key.get("dataset", row.get("dataset", "")),
                "case_id": key.get("case_id", ""),
                "generator_model": key.get("generator_model", ""),
                "order": order,
                "scores_by_label": by_label,
                "scores_by_method": by_method,
                "winner_label": winner_label,
                "winner_method": winner_method,
                "reviewer_confidence": to_float(row.get("reviewer_confidence_0_5")),
                "reviewer_notes": row.get("reviewer_notes", ""),
            }
        )
    return records


def has_required_scores(row: dict[str, str]) -> bool:
    required = [f"{label}_{dimension}_0_5" for label in ["A", "B"] for dimension in DIMENSIONS]
    return all(str(row.get(column, "")).strip() != "" for column in required)


def normalize_winner(value: str) -> str:
    cleaned = str(value).strip().lower()
    if cleaned == "a":
        return "A"
    if cleaned == "b":
        return "B"
    return "tie"


def to_float(value: object) -> float:
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(5.0, score))


def build_summary(records: list[dict[str, Any]], bootstrap_reps: int) -> dict[str, Any]:
    specs = {
        "overall": [],
        "by_dataset": ["dataset"],
        "by_generator_model": ["generator_model"],
        "by_dataset_generator_model": ["dataset", "generator_model"],
        "by_reviewer": ["reviewer_id"],
    }
    groups = {}
    for group_name, fields in specs.items():
        for key, group_records in grouped(records, fields).items():
            label = group_name if not fields else " | ".join(f"{field}={value}" for field, value in zip(fields, key))
            groups[label] = summarize_group(label, group_records, bootstrap_reps)
    return {
        "reviewed_record_count": len(records),
        "groups": groups,
        "notes": [
            "Human review scores are blinded ratings from 0 to 5.",
            "Paired differences are vectorless_tree_rag minus embedding_rag within the same sample.",
            "Rows without complete A/B scores are excluded.",
        ],
    }


def grouped(records: list[dict[str, Any]], fields: list[str]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    if not fields:
        return {("overall",): records}
    result: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        result[tuple(str(record.get(field, "")) for field in fields)].append(record)
    return dict(sorted(result.items(), key=lambda item: item[0]))


def summarize_group(label: str, records: list[dict[str, Any]], bootstrap_reps: int) -> dict[str, Any]:
    method_scores = {method: {dimension: [] for dimension in DIMENSIONS} for method in METHODS}
    diffs = {dimension: [] for dimension in DIMENSIONS}
    for record in records:
        by_method = record["scores_by_method"]
        for dimension in DIMENSIONS:
            emb = float(by_method["embedding_rag"][dimension])
            vec = float(by_method["vectorless_tree_rag"][dimension])
            method_scores["embedding_rag"][dimension].append(emb)
            method_scores["vectorless_tree_rag"][dimension].append(vec)
            diffs[dimension].append(vec - emb)
    return {
        "label": label,
        "n": len(records),
        "winner_counts": dict(Counter(record["winner_method"] for record in records)),
        "reviewers": dict(Counter(record.get("reviewer_id", "") for record in records)),
        "datasets": dict(Counter(record.get("dataset", "") for record in records)),
        "generator_models": dict(Counter(record.get("generator_model", "") for record in records)),
        "confidence": numeric_summary([record["reviewer_confidence"] for record in records]),
        "method_scores": {
            method: {dimension: numeric_summary(values) for dimension, values in scores.items()}
            for method, scores in method_scores.items()
        },
        "paired_differences": {
            dimension: paired_summary(values, bootstrap_reps) for dimension, values in diffs.items()
        },
    }


def paired_summary(values: list[float], bootstrap_reps: int) -> dict[str, Any]:
    positives = sum(1 for value in values if value > 1e-12)
    negatives = sum(1 for value in values if value < -1e-12)
    ties = len(values) - positives - negatives
    return {
        "mean_diff_vectorless_minus_embedding": round_float(mean(values) if values else 0.0),
        "sd": round_float(pstdev(values) if len(values) > 1 else 0.0),
        "bootstrap_ci95": [round_float(value) for value in bootstrap_ci(values, bootstrap_reps)],
        "positive": positives,
        "ties": ties,
        "negative": negatives,
        "sign_test_p_two_sided": sign_test_p(positives, negatives),
    }


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round_float(mean(values)),
        "sd": round_float(pstdev(values) if len(values) > 1 else 0.0),
        "min": round_float(min(values)),
        "max": round_float(max(values)),
    }


def bootstrap_ci(values: list[float], reps: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1 or reps <= 0:
        return [values[0], values[0]]
    rng = random.Random(20260614)
    estimates = []
    n = len(values)
    for _ in range(reps):
        estimates.append(mean(values[rng.randrange(n)] for _ in range(n)))
    estimates.sort()
    return [estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]]


def sign_test_p(positives: int, negatives: int) -> float:
    n = positives + negatives
    if n == 0:
        return 1.0
    k = min(positives, negatives)
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        for i in range(k + 1)
    ]
    max_log = max(log_terms)
    log_tail = max_log + math.log(sum(math.exp(term - max_log) for term in log_terms)) - n * math.log(2.0)
    return round_float(min(1.0, 2.0 * math.exp(log_tail)))


def round_float(value: float) -> float:
    return round(float(value), 6)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Human Clinical Claim Review Summary",
        "",
        f"Reviewed complete rows: `{summary['reviewed_record_count']}`",
        "",
        "Scores are 0-5. Diff is vectorless_tree_rag minus embedding_rag.",
        "",
        "| Group | N | Method | Overall usefulness | Interpretability | Specificity | Actionability | Conciseness | Safety | Winner counts | Overall diff | p |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for label in sorted(summary["groups"], key=group_sort_key):
        if label != "overall" and not (label.startswith("dataset=") or label.startswith("generator_model=")):
            continue
        group = summary["groups"][label]
        winners = ", ".join(f"{key}:{value}" for key, value in sorted(group["winner_counts"].items()))
        diff = group["paired_differences"]["overall_usefulness"]
        for method in METHODS:
            scores = group["method_scores"][method]
            lines.append(
                f"| {md(label)} | {group['n']} | `{method}` | "
                f"{scores['overall_usefulness']['mean']:.3f} | "
                f"{scores['clinical_interpretability']['mean']:.3f} | "
                f"{scores['specificity']['mean']:.3f} | "
                f"{scores['actionability_for_clinician']['mean']:.3f} | "
                f"{scores['conciseness_nonredundancy']['mean']:.3f} | "
                f"{scores['safety_no_overclaim']['mean']:.3f} | "
                f"{md(winners)} | {diff['mean_diff_vectorless_minus_embedding']:.3f} | {diff['sign_test_p_two_sided']:.4f} |"
            )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in summary["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def group_sort_key(label: str) -> tuple[int, str]:
    if label == "overall":
        return (0, label)
    if label.startswith("dataset="):
        return (1, label)
    if label.startswith("generator_model="):
        return (2, label)
    return (3, label)


def md(value: object) -> str:
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
