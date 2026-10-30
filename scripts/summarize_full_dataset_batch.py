from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


METRICS = [
    "feature_coverage",
    "faithfulness",
    "hallucination_rate",
    "citation_accuracy",
    "claim_count",
    "quality_score",
]

HIGHER_IS_BETTER = {
    "feature_coverage": True,
    "faithfulness": True,
    "hallucination_rate": False,
    "citation_accuracy": True,
    "claim_count": None,
    "quality_score": True,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize full-dataset paired RAG results with statistics.")
    parser.add_argument("--input", type=Path, default=Path("outputs/full_dataset_batch_perry_first.jsonl"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/full_dataset_batch_perry_first_summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("outputs/full_dataset_batch_perry_first_summary.md"))
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    summary = build_summary(records, args.bootstrap_reps)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")


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
    return records


def build_summary(records: list[dict[str, Any]], bootstrap_reps: int) -> dict[str, Any]:
    status_counts = Counter(str(record.get("status", "missing")) for record in records)
    dataset_counts = Counter(str(record.get("dataset", "")) for record in records)
    model_counts = Counter(str(record.get("model", "")) for record in records)
    analyzable = [record for record in records if record.get("status") == "ok" and record.get("feature_count", 0) > 0]

    group_specs = {
        "overall": [],
        "by_dataset": ["dataset"],
        "by_model": ["model"],
        "by_dataset_model": ["dataset", "model"],
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for group_name, fields in group_specs.items():
        for key, group_records in grouped(analyzable, fields).items():
            label = group_name if not fields else " | ".join(f"{field}={value}" for field, value in zip(fields, key))
            groups[label] = summarize_group(label, group_records, bootstrap_reps)

    return {
        "input_record_count": len(records),
        "analyzable_record_count": len(analyzable),
        "status_counts": dict(status_counts),
        "dataset_record_counts": dict(dataset_counts),
        "model_record_counts": dict(model_counts),
        "groups": groups,
        "notes": [
            "All comparisons are paired within the same case and LLM.",
            "Mean difference is vectorless_tree_rag minus embedding_rag.",
            "For hallucination_rate, negative mean difference favors vectorless_tree_rag because lower is better.",
            "Sign-test p-values are exact two-sided tests over non-tied paired differences.",
            "Bootstrap CIs are percentile intervals for the mean paired difference.",
            "Records with status=no_features are counted but excluded from inferential paired tests.",
        ],
    }


def grouped(records: list[dict[str, Any]], fields: list[str]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    if not fields:
        return {("overall",): records}
    result: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(str(record.get(field, "")) for field in fields)
        result[key].append(record)
    return dict(sorted(result.items(), key=lambda item: item[0]))


def summarize_group(label: str, records: list[dict[str, Any]], bootstrap_reps: int) -> dict[str, Any]:
    metric_summaries = {metric: paired_metric_summary(records, metric, bootstrap_reps) for metric in METRICS}
    return {
        "label": label,
        "n": len(records),
        "datasets": dict(Counter(str(record.get("dataset", "")) for record in records)),
        "models": dict(Counter(str(record.get("model", "")) for record in records)),
        "feature_count": numeric_summary([float(record.get("feature_count", 0)) for record in records]),
        "metrics": metric_summaries,
        "top_embedding_contexts": counter_top(
            context for record in records for context in record.get("embedding_rag", {}).get("retrieved_contexts", [])
        ),
        "top_vectorless_nodes": counter_top(
            node for record in records for node in record.get("vectorless_tree_rag", {}).get("selected_nodes", [])
        ),
    }


def paired_metric_summary(records: list[dict[str, Any]], metric: str, bootstrap_reps: int) -> dict[str, Any]:
    embedding_values = [metric_value(record, "embedding_rag", metric) for record in records]
    vectorless_values = [metric_value(record, "vectorless_tree_rag", metric) for record in records]
    diffs = [v - e for e, v in zip(embedding_values, vectorless_values)]
    sign = sign_test(diffs)
    better = better_counts(embedding_values, vectorless_values, metric)
    return {
        "n": len(records),
        "embedding_mean": round_float(mean_or_zero(embedding_values)),
        "vectorless_mean": round_float(mean_or_zero(vectorless_values)),
        "mean_diff_vectorless_minus_embedding": round_float(mean_or_zero(diffs)),
        "diff_sd": round_float(pstdev(diffs) if len(diffs) > 1 else 0.0),
        "bootstrap_ci95": [round_float(value) for value in bootstrap_ci(diffs, bootstrap_reps)],
        "wins_ties_losses": better,
        "exact_sign_test": sign,
    }


def metric_value(record: dict[str, Any], method: str, metric: str) -> float:
    if metric == "claim_count":
        return float(len(record.get(method, {}).get("explanation", {}).get("claims", [])))
    if metric == "quality_score":
        evaluation = record.get(method, {}).get("evaluation", {})
        return float(evaluation.get("feature_coverage", 0.0)) + float(evaluation.get("faithfulness", 0.0)) + float(
            evaluation.get("citation_accuracy", 0.0)
        ) - float(evaluation.get("hallucination_rate", 0.0))
    return float(record.get(method, {}).get("evaluation", {}).get(metric, 0.0))


def better_counts(embedding_values: list[float], vectorless_values: list[float], metric: str) -> dict[str, int | str | None]:
    higher = HIGHER_IS_BETTER[metric]
    vectorless = embedding = ties = 0
    for emb, vec in zip(embedding_values, vectorless_values):
        if math.isclose(emb, vec, abs_tol=1e-12):
            ties += 1
        elif higher is None:
            # Claim count has no default better direction; still record raw larger/smaller counts.
            if vec > emb:
                vectorless += 1
            else:
                embedding += 1
        elif higher:
            if vec > emb:
                vectorless += 1
            else:
                embedding += 1
        else:
            if vec < emb:
                vectorless += 1
            else:
                embedding += 1
    return {
        "vectorless_better_or_larger": vectorless,
        "ties": ties,
        "embedding_better_or_larger": embedding,
        "direction": "higher_is_better" if higher else ("lower_is_better" if higher is False else "count_only"),
    }


def sign_test(diffs: list[float]) -> dict[str, Any]:
    positives = sum(1 for diff in diffs if diff > 1e-12)
    negatives = sum(1 for diff in diffs if diff < -1e-12)
    ties = len(diffs) - positives - negatives
    n = positives + negatives
    if n == 0:
        return {"n_non_tied": 0, "positive": positives, "negative": negatives, "ties": ties, "p_two_sided": 1.0}
    k = min(positives, negatives)
    probability = 2.0 * sum(math.comb(n, index) for index in range(k + 1)) / (2**n)
    return {
        "n_non_tied": n,
        "positive": positives,
        "negative": negatives,
        "ties": ties,
        "p_two_sided": round_float(min(1.0, probability)),
    }


def bootstrap_ci(values: list[float], reps: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1 or reps <= 0:
        value = mean(values)
        return [value, value]
    rng = random.Random(20260613)
    estimates = []
    n = len(values)
    for _ in range(reps):
        estimates.append(mean(values[rng.randrange(n)] for _ in range(n)))
    estimates.sort()
    lo = estimates[int(0.025 * (len(estimates) - 1))]
    hi = estimates[int(0.975 * (len(estimates) - 1))]
    return [lo, hi]


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round_float(mean(values)),
        "sd": round_float(pstdev(values) if len(values) > 1 else 0.0),
        "min": round_float(min(values)),
        "max": round_float(max(values)),
    }


def counter_top(values: Iterable[str], limit: int = 10) -> list[dict[str, Any]]:
    return [{"id": key, "count": count} for key, count in Counter(values).most_common(limit)]


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def round_float(value: float) -> float:
    return round(float(value), 6)


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Full Dataset RAG Batch Summary")
    lines.append("")
    lines.append("## Record Counts")
    lines.append("")
    lines.append(f"- Input records: `{summary['input_record_count']}`")
    lines.append(f"- Analyzable paired records: `{summary['analyzable_record_count']}`")
    lines.append(f"- Status counts: `{summary['status_counts']}`")
    lines.append("")
    lines.append("## Primary Paired Results")
    lines.append("")
    lines.append("Mean diff is `vectorless_tree_rag - embedding_rag`. For hallucination, negative favors vectorless.")
    lines.append("")
    lines.append("| Group | N | Metric | Emb mean | Vec mean | Diff | 95% bootstrap CI | sign-test p | Vec better | Ties | Emb better |")
    lines.append("|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|")
    ordered_labels = sorted(summary["groups"], key=group_sort_key)
    for label in ordered_labels:
        group = summary["groups"][label]
        if label == "overall" or "dataset=" in label and "model=" in label:
            for metric in METRICS:
                metric_summary = group["metrics"][metric]
                wins = metric_summary["wins_ties_losses"]
                ci = metric_summary["bootstrap_ci95"]
                p = metric_summary["exact_sign_test"]["p_two_sided"]
                safe_label = markdown_cell(label)
                lines.append(
                    f"| {safe_label} | {group['n']} | {metric} | {metric_summary['embedding_mean']:.4f} | "
                    f"{metric_summary['vectorless_mean']:.4f} | {metric_summary['mean_diff_vectorless_minus_embedding']:.4f} | "
                    f"[{ci[0]:.4f}, {ci[1]:.4f}] | {p:.4f} | "
                    f"{wins['vectorless_better_or_larger']} | {wins['ties']} | {wins['embedding_better_or_larger']} |"
                )
    lines.append("")
    lines.append("## Retrieval Frequencies")
    lines.append("")
    for label in ordered_labels:
        if label != "overall" and not ("dataset=" in label and "model=" in label):
            continue
        group = summary["groups"][label]
        lines.append(f"### {label}")
        lines.append("")
        lines.append("Top embedding contexts:")
        for item in group["top_embedding_contexts"][:8]:
            lines.append(f"- `{item['id']}`: {item['count']}")
        lines.append("")
        lines.append("Top vectorless nodes:")
        for item in group["top_vectorless_nodes"][:8]:
            lines.append(f"- `{item['id']}`: {item['count']}")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in summary["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def group_sort_key(label: str) -> tuple[int, str]:
    if label == "overall":
        return (0, label)
    if "dataset=" in label and "model=" in label:
        return (1, label)
    if label.startswith("dataset="):
        return (2, label)
    if label.startswith("model="):
        return (3, label)
    return (4, label)


if __name__ == "__main__":
    main()
