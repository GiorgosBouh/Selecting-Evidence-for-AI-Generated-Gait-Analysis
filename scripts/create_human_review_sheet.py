from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

METHODS = ["embedding_rag", "vectorless_tree_rag"]
RUBRIC_COLUMNS = [
    "A_relevance_to_case_0_5",
    "A_clinical_interpretability_0_5",
    "A_specificity_0_5",
    "A_actionability_for_clinician_0_5",
    "A_conciseness_nonredundancy_0_5",
    "A_safety_no_overclaim_0_5",
    "A_overall_usefulness_0_5",
    "B_relevance_to_case_0_5",
    "B_clinical_interpretability_0_5",
    "B_specificity_0_5",
    "B_actionability_for_clinician_0_5",
    "B_conciseness_nonredundancy_0_5",
    "B_safety_no_overclaim_0_5",
    "B_overall_usefulness_0_5",
    "winner_A_B_tie",
    "reviewer_confidence_0_5",
    "reviewer_notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a blinded human clinical-usefulness review package.")
    parser.add_argument("--input", type=Path, default=Path("outputs/full_dataset_batch_perry_first.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/human_review_round1"))
    parser.add_argument("--sample-per-dataset-model", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--reviewer-id", default="reviewer_01")
    args = parser.parse_args()

    records = load_records(args.input)
    sampled = stratified_sample(records, args.sample_per_dataset_model, args.seed)
    package = build_package(sampled, args.reviewer_id, args.seed)
    write_package(package, args.output_dir)
    print(
        f"Wrote blinded review package to {args.output_dir} with "
        f"{len(package['review_rows'])} items."
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "ok":
                continue
            if int(record.get("feature_count", 0)) <= 0:
                continue
            if any(method not in record for method in METHODS):
                continue
            records.append(record)
    return records


def stratified_sample(records: list[dict[str, Any]], per_stratum: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[(str(record.get("dataset")), str(record.get("model")))].append(record)
    sampled: list[dict[str, Any]] = []
    for key in sorted(strata):
        candidates = sorted(strata[key], key=lambda item: str(item.get("case_id")))
        if len(candidates) > per_stratum:
            candidates = rng.sample(candidates, per_stratum)
        sampled.extend(candidates)
    rng.shuffle(sampled)
    return sampled


def build_package(records: list[dict[str, Any]], reviewer_id: str, seed: int) -> dict[str, Any]:
    review_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        sample_id = f"HR1_{index:04d}"
        order = anonymized_order(sample_id, record, seed)
        claim_sets = {label: claim_text(record, method) for label, method in order.items()}
        row = {
            "sample_id": sample_id,
            "reviewer_id": reviewer_id,
            "dataset": record.get("dataset"),
            "case_evidence": feature_evidence_text(record),
            "claim_set_A": claim_sets["A"],
            "claim_set_B": claim_sets["B"],
        }
        for column in RUBRIC_COLUMNS:
            row[column] = ""
        review_rows.append(row)
        key_rows.append(
            {
                "sample_id": sample_id,
                "dataset": record.get("dataset"),
                "case_id": record.get("case_id"),
                "generator_model": record.get("model"),
                "A_method": order["A"],
                "B_method": order["B"],
                "feature_count": record.get("feature_count"),
                "A_claim_count": len(record.get(order["A"], {}).get("explanation", {}).get("claims", [])),
                "B_claim_count": len(record.get(order["B"], {}).get("explanation", {}).get("claims", [])),
            }
        )
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "dataset": record.get("dataset"),
                "feature_count": record.get("feature_count"),
                "A_claim_count": len(record.get(order["A"], {}).get("explanation", {}).get("claims", [])),
                "B_claim_count": len(record.get(order["B"], {}).get("explanation", {}).get("claims", [])),
            }
        )
    return {"review_rows": review_rows, "key_rows": key_rows, "manifest_rows": manifest_rows}


def anonymized_order(sample_id: str, record: dict[str, Any], seed: int) -> dict[str, str]:
    token = f"{seed}|{sample_id}|{record.get('dataset')}|{record.get('model')}|{record.get('case_id')}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    methods = METHODS[:]
    if int(digest[:2], 16) % 2:
        methods.reverse()
    return {"A": methods[0], "B": methods[1]}


def feature_evidence_text(record: dict[str, Any]) -> str:
    items = []
    for feature in record.get("features", []):
        name = str(feature.get("name", ""))
        evidence = str(feature.get("evidence", ""))
        items.append(f"- {name}: {evidence}")
    return "\n".join(items)


def claim_text(record: dict[str, Any], method: str) -> str:
    claims = record.get(method, {}).get("explanation", {}).get("claims", [])
    if not claims:
        return "No claims generated."
    return "\n".join(f"{idx}. {claim.get('text', '')}" for idx, claim in enumerate(claims, start=1))


def write_package(package: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "clinical_claim_review_sheet_blinded.csv", package["review_rows"])
    write_csv(output_dir / "clinical_claim_review_answer_key_DO_NOT_SHARE.csv", package["key_rows"])
    write_csv(output_dir / "clinical_claim_review_manifest_blinded.csv", package["manifest_rows"])
    (output_dir / "clinical_claim_review_instructions.md").write_text(instructions_text(), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def instructions_text() -> str:
    return """# Blinded Clinical Claim Review Instructions

## Goal

Evaluate which claim set is more clinically useful for interpreting the gait case.

The two claim sets are anonymized as `A` and `B`. Do not try to infer which RAG method produced them.

## What You See

Each row contains:

- `sample_id`: row identifier.
- `dataset`: dataset name.
- `case_evidence`: measured case features and evidence.
- `claim_set_A`: first anonymized claim set.
- `claim_set_B`: second anonymized claim set.
- empty scoring columns.

## Scoring Scale

Use 0-5 for each dimension:

- 0 = absent, misleading, or harmful
- 1 = poor
- 2 = limited
- 3 = acceptable
- 4 = good
- 5 = excellent

## Dimensions

- `relevance_to_case`: claims address the provided case features.
- `clinical_interpretability`: claims translate raw gait features into clinically meaningful gait interpretation.
- `specificity`: claims preserve exact case values such as biomechanical feature value, unit, healthy-reference z-score, approximate-feature flag, pathology label, visual gait score, or clinical score when relevant. Penalize generic wording such as 'impairment noted', 'score available', or 'abnormal gait was observed' without the actual values.
- `actionability_for_clinician`: claims help a clinician know what to inspect, prioritize, or follow up, without inventing treatment.
- `conciseness_nonredundancy`: claims are compact and not repetitive while preserving meaning.
- `safety_no_overclaim`: claims avoid unsupported diagnosis, severity, causality, normality, or treatment recommendations.
- `overall_usefulness`: overall usefulness of the claim set for clinical interpretation.

## Winner

In `winner_A_B_tie`, enter exactly one of:

```text
A
B
tie
```

Use `tie` when neither set is meaningfully more useful.

## Reviewer Confidence

In `reviewer_confidence_0_5`, rate how confident you are in your judgment:

- 0 = cannot judge
- 5 = very confident

## Notes

Use `reviewer_notes` for brief comments about missing exact values, missing clinical interpretation, overclaiming, redundancy, or particularly useful phrasing.

## Important

This is not a clinical diagnosis task and not a treatment recommendation task. Judge only whether the claims are useful and safe for interpreting the provided gait evidence.
"""


if __name__ == "__main__":
    main()
