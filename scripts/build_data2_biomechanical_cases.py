from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_data2_biomechanical_features import DEFAULT_ZIP, TrialFeatures, load_trial_features


DEFAULT_JSONL = Path("outputs/data2_biomechanical_cases_2026-06-15.jsonl")
DEFAULT_CSV = Path("outputs/data2_biomechanical_cases_2026-06-15.csv")
DEFAULT_SUMMARY = Path("outputs/data2_biomechanical_cases_summary_2026-06-15.md")


FEATURE_RULES = [
    {
        "metric": "approximate_straight_speed_m_s",
        "low_feature": "reduced_approximate_walking_speed",
        "unit": "m/s",
        "approximate": True,
        "domain": "spatiotemporal_progression",
        "evidence_note": "approximated as 17 m straight walking distance divided by straight walking duration",
    },
    {
        "metric": "approximate_step_length_m",
        "low_feature": "reduced_approximate_step_length",
        "unit": "m",
        "approximate": True,
        "domain": "spatiotemporal_progression",
        "evidence_note": "approximated as 17 m straight walking distance divided by straight-phase step count",
    },
    {
        "metric": "approximate_stride_length_m",
        "low_feature": "reduced_approximate_stride_length",
        "unit": "m",
        "approximate": True,
        "domain": "spatiotemporal_progression",
        "evidence_note": "approximated as twice the average approximate step length",
    },
    {
        "metric": "approximate_cadence_steps_min",
        "low_feature": "reduced_cadence",
        "high_feature": "increased_cadence",
        "unit": "steps/min",
        "approximate": False,
        "domain": "spatiotemporal_timing",
        "evidence_note": "computed from straight-phase step events and straight walking duration",
    },
    {
        "metric": "mean_stride_time_s",
        "high_feature": "prolonged_stride_time",
        "unit": "s",
        "approximate": False,
        "domain": "temporal_gait_cycle",
        "evidence_note": "computed from consecutive ipsilateral heel-strike events",
    },
    {
        "metric": "stride_time_cv_pct",
        "high_feature": "increased_stride_time_variability",
        "unit": "%",
        "approximate": False,
        "domain": "temporal_variability",
        "evidence_note": "coefficient of variation of left and right stride intervals",
    },
    {
        "metric": "stride_time_asymmetry_pct",
        "high_feature": "increased_stride_time_asymmetry",
        "unit": "%",
        "approximate": False,
        "domain": "left_right_asymmetry",
        "evidence_note": "absolute left-right difference normalized by side mean",
    },
    {
        "metric": "mean_swing_time_s",
        "high_feature": "prolonged_swing_time",
        "unit": "s",
        "approximate": False,
        "domain": "gait_phase_timing",
        "evidence_note": "toe-off to heel-strike interval; official Data_2 plot script labels this interval as swing",
    },
    {
        "metric": "swing_time_asymmetry_pct",
        "high_feature": "increased_swing_time_asymmetry",
        "unit": "%",
        "approximate": False,
        "domain": "left_right_asymmetry",
        "evidence_note": "absolute left-right swing-time difference normalized by side mean",
    },
    {
        "metric": "mean_stance_time_s",
        "high_feature": "prolonged_stance_time",
        "unit": "s",
        "approximate": False,
        "domain": "gait_phase_timing",
        "evidence_note": "heel-strike to next toe-off interval; official Data_2 plot script labels this interval as stance",
    },
    {
        "metric": "stance_time_asymmetry_pct",
        "high_feature": "increased_stance_time_asymmetry",
        "unit": "%",
        "approximate": False,
        "domain": "left_right_asymmetry",
        "evidence_note": "absolute left-right stance-time difference normalized by side mean",
    },
    {
        "metric": "double_support_pct_gait",
        "high_feature": "increased_double_support_percentage",
        "unit": "%",
        "approximate": False,
        "domain": "support_phase_stability",
        "evidence_note": "overlap between left and right stance intervals as percentage of gait duration",
    },
    {
        "metric": "uturn_duration_s",
        "high_feature": "prolonged_uturn_duration",
        "unit": "s",
        "approximate": False,
        "domain": "turning_performance",
        "evidence_note": "computed from metadata U-turn boundaries and sampling frequency",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build participant-level Data_2 biomechanical gait cases.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--min-abs-z", type=float, default=1.5)
    args = parser.parse_args()

    trial_rows = load_trial_features(args.zip)
    cases = build_cases(trial_rows, min_abs_z=args.min_abs_z)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(cases, args.output_jsonl)
    write_case_csv(cases, args.output_csv)
    args.output_summary.write_text(render_summary(cases, trial_rows, args.output_jsonl, args.output_csv, args.min_abs_z), encoding="utf-8")
    print(f"Wrote {args.output_jsonl}, {args.output_csv}, and {args.output_summary}")


def build_cases(trial_rows: list[TrialFeatures], min_abs_z: float) -> list[dict[str, Any]]:
    subject_trials: dict[str, list[TrialFeatures]] = defaultdict(list)
    for row in trial_rows:
        subject_trials[row.subject].append(row)

    subject_aggregates = [aggregate_subject(subject, rows) for subject, rows in sorted(subject_trials.items())]
    reference = healthy_reference(subject_aggregates)
    cases = [build_case(aggregate, reference, min_abs_z) for aggregate in subject_aggregates]
    return cases


def aggregate_subject(subject: str, rows: list[TrialFeatures]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda item: item.trial)
    first = rows[0]
    measurements: dict[str, float] = {}
    measurement_sd: dict[str, float] = {}
    measurement_n: dict[str, int] = {}
    metrics = metric_names()
    derived_rows = [trial_metric_values(row) for row in rows]
    for metric in metrics:
        values = [item[metric] for item in derived_rows if math.isfinite(item.get(metric, math.nan))]
        if not values:
            continue
        measurements[metric] = statistics.mean(values)
        measurement_sd[metric] = statistics.pstdev(values) if len(values) > 1 else 0.0
        measurement_n[metric] = len(values)

    clinical_score_values = [row.clinical_score_value for row in rows if row.clinical_score_value not in {"", "None", "Not evaluated"}]
    visual_values = [parse_float(row.visual_gait_assessment) for row in rows]
    visual_values = [value for value in visual_values if value is not None]
    return {
        "case_id": subject,
        "dataset": "Clinical Gait Signals",
        "aggregation_level": "participant_mean_across_trials",
        "n_trials": len(rows),
        "trial_ids": [row.case_id for row in rows],
        "trial_paths": [row.zip_path for row in rows],
        "metadata": {
            "subject": subject,
            "group": first.group,
            "pathologyKey": first.pathology_key,
            "pathology": first.pathology,
            "clinical_score_name": first.clinical_score_name,
            "clinical_score_value": clinical_score_values[0] if clinical_score_values else first.clinical_score_value,
            "visual_gait_assessment_mean": statistics.mean(visual_values) if visual_values else None,
            "visual_gait_assessment_values": visual_values,
        },
        "measurements": measurements,
        "measurement_sd_across_trials": measurement_sd,
        "measurement_n_trials": measurement_n,
    }


def trial_metric_values(row: TrialFeatures) -> dict[str, float]:
    mean_stride = mean_finite([row.left_stride_time_mean_s, row.right_stride_time_mean_s])
    mean_swing = mean_finite([row.left_swing_time_mean_s, row.right_swing_time_mean_s])
    mean_stance = mean_finite([row.left_stance_time_mean_s, row.right_stance_time_mean_s])
    return {
        "approximate_straight_speed_m_s": row.approximate_straight_speed_m_s,
        "approximate_cadence_steps_min": row.approximate_cadence_steps_min,
        "approximate_step_length_m": row.approximate_step_length_m,
        "approximate_stride_length_m": row.approximate_stride_length_m,
        "mean_stride_time_s": mean_stride,
        "stride_time_asymmetry_pct": row.stride_time_asymmetry_pct,
        "mean_swing_time_s": mean_swing,
        "swing_time_asymmetry_pct": row.swing_time_asymmetry_pct,
        "mean_stance_time_s": mean_stance,
        "stance_time_asymmetry_pct": row.stance_time_asymmetry_pct,
        "double_support_pct_gait": row.double_support_pct_gait,
        "stride_time_cv_pct": row.stride_time_cv_pct,
        "uturn_duration_s": row.uturn_duration_s,
    }


def healthy_reference(subject_aggregates: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    healthy = [item for item in subject_aggregates if item["metadata"].get("pathologyKey") == "HS"]
    reference: dict[str, dict[str, float]] = {}
    for metric in metric_names():
        values = [item["measurements"][metric] for item in healthy if metric in item["measurements"]]
        if not values:
            continue
        reference[metric] = {
            "mean": statistics.mean(values),
            "sd": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return reference


def build_case(aggregate: dict[str, Any], reference: dict[str, dict[str, float]], min_abs_z: float) -> dict[str, Any]:
    measurements = aggregate["measurements"]
    z_scores: dict[str, dict[str, float]] = {}
    abnormal_features: list[dict[str, Any]] = []
    for rule in FEATURE_RULES:
        metric = rule["metric"]
        if metric not in measurements or metric not in reference:
            continue
        value = measurements[metric]
        ref = reference[metric]
        sd = ref["sd"]
        if sd <= 0:
            continue
        z_score = (value - ref["mean"]) / sd
        z_scores[metric] = {
            "value": round_float(value),
            "healthy_reference_mean": round_float(ref["mean"]),
            "healthy_reference_sd": round_float(sd),
            "healthy_reference_n": int(ref["n"]),
            "z_score": round_float(z_score),
        }
        abnormal_features.extend(features_from_rule(rule, value, z_score, ref, min_abs_z, aggregate))

    abnormal_features.sort(key=lambda item: abs(item["z_score"]), reverse=True)
    return {
        **aggregate,
        "healthy_reference": z_scores,
        "abnormal_features": abnormal_features,
        "case_feature_packet": build_feature_packet(aggregate, abnormal_features),
    }


def features_from_rule(
    rule: dict[str, Any],
    value: float,
    z_score: float,
    ref: dict[str, float],
    min_abs_z: float,
    aggregate: dict[str, Any],
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    if "low_feature" in rule and z_score <= -min_abs_z:
        features.append(feature_payload(rule, rule["low_feature"], "low", value, z_score, ref, min_abs_z, aggregate))
    if "high_feature" in rule and z_score >= min_abs_z:
        features.append(feature_payload(rule, rule["high_feature"], "high", value, z_score, ref, min_abs_z, aggregate))
    return features


def feature_payload(
    rule: dict[str, Any],
    feature_id: str,
    direction: str,
    value: float,
    z_score: float,
    ref: dict[str, float],
    min_abs_z: float,
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    threshold = ref["mean"] + ((-min_abs_z if direction == "low" else min_abs_z) * ref["sd"])
    unit = rule["unit"]
    metric = rule["metric"]
    severity = "marked" if abs(z_score) >= 2.0 else "mild_to_moderate"
    approximate = bool(rule["approximate"])
    evidence = (
        f"{metric}={value:.3f} {unit}; healthy_ref_mean={ref['mean']:.3f} {unit}; "
        f"healthy_ref_sd={ref['sd']:.3f}; z={z_score:.2f}; "
        f"{direction}_threshold={threshold:.3f} {unit}; n_trials={aggregate['n_trials']}; "
        f"approximate={str(approximate).lower()}; computation={rule['evidence_note']}"
    )
    return {
        "feature_id": feature_id,
        "metric": metric,
        "value": round_float(value),
        "unit": unit,
        "direction": direction,
        "z_score": round_float(z_score),
        "severity": severity,
        "healthy_reference_mean": round_float(ref["mean"]),
        "healthy_reference_sd": round_float(ref["sd"]),
        "healthy_reference_n": int(ref["n"]),
        "threshold": round_float(threshold),
        "domain": rule["domain"],
        "is_approximate": approximate,
        "evidence": evidence,
    }


def build_feature_packet(aggregate: dict[str, Any], abnormal_features: list[dict[str, Any]]) -> str:
    metadata = aggregate["metadata"]
    lines = [
        f"Subject {aggregate['case_id']} ({metadata.get('pathologyKey')}, {metadata.get('pathology')})",
        f"Group: {metadata.get('group')}; trials aggregated: {aggregate['n_trials']}",
    ]
    score_name = metadata.get("clinical_score_name")
    score_value = metadata.get("clinical_score_value")
    if score_name and score_value not in {None, "", "None"}:
        lines.append(f"Clinical score: {score_name}={score_value}")
    visual = metadata.get("visual_gait_assessment_mean")
    if visual is not None:
        lines.append(f"Visual gait assessment mean: {visual:.2f}")
    if not abnormal_features:
        lines.append("No biomechanical feature exceeded the healthy-reference z-score threshold.")
        return "\n".join(lines)
    lines.append("Abnormal biomechanical features relative to healthy subject reference:")
    for feature in abnormal_features:
        lines.append(f"- {feature['feature_id']}: {feature['evidence']}")
    return "\n".join(lines)


def write_jsonl(cases: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(clean_json(case), ensure_ascii=False) + "\n")


def write_case_csv(cases: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "case_id",
        "pathologyKey",
        "pathology",
        "group",
        "n_trials",
        "n_abnormal_features",
        "abnormal_feature_ids",
        "approximate_straight_speed_m_s",
        "approximate_step_length_m",
        "approximate_stride_length_m",
        "approximate_cadence_steps_min",
        "double_support_pct_gait",
        "stride_time_asymmetry_pct",
        "stance_time_asymmetry_pct",
        "stride_time_cv_pct",
        "uturn_duration_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            measurements = case["measurements"]
            features = case["abnormal_features"]
            metadata = case["metadata"]
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "pathologyKey": metadata.get("pathologyKey", ""),
                    "pathology": metadata.get("pathology", ""),
                    "group": metadata.get("group", ""),
                    "n_trials": case["n_trials"],
                    "n_abnormal_features": len(features),
                    "abnormal_feature_ids": ";".join(feature["feature_id"] for feature in features),
                    **{metric: round_float(measurements.get(metric, math.nan)) for metric in fieldnames[7:]},
                }
            )


def render_summary(
    cases: list[dict[str, Any]],
    trial_rows: list[TrialFeatures],
    jsonl_path: Path,
    csv_path: Path,
    min_abs_z: float,
) -> str:
    pathology_counts = Counter(case["metadata"].get("pathologyKey", "") for case in cases)
    feature_counts = Counter(feature["feature_id"] for case in cases for feature in case["abnormal_features"])
    cases_with_features = sum(1 for case in cases if case["abnormal_features"])
    lines = [
        "# Data_2 Biomechanical Participant Cases",
        "",
        f"JSONL: `{jsonl_path}`",
        f"CSV: `{csv_path}`",
        "",
        "## Design",
        "",
        "Participant-level cases were built by aggregating all available Data_2 trials for each subject.",
        "Healthy-reference z-scores were computed from participant-level healthy-control aggregates, not from repeated trial rows.",
        f"Abnormal labels use `abs(z) >= {min_abs_z}` with direction-specific rules.",
        "",
        "## Counts",
        "",
        f"- Trial-level rows available: `{len(trial_rows)}`",
        f"- Participant-level cases: `{len(cases)}`",
        f"- Cases with at least one abnormal biomechanical feature: `{cases_with_features}`",
        "",
        "### Cases By Pathology",
        "",
    ]
    for key, count in sorted(pathology_counts.items()):
        lines.append(f"- `{key}`: `{count}`")
    lines.extend(["", "### Top Abnormal Feature Labels", ""])
    for feature_id, count in feature_counts.most_common(20):
        lines.append(f"- `{feature_id}`: `{count}` cases")
    lines.extend(
        [
            "",
            "## Important Limits",
            "",
            "- `approximate_straight_speed_m_s`, `approximate_step_length_m`, and `approximate_stride_length_m` are distance/protocol approximations.",
            "- The cases do not include true step width or joint angles.",
            "- Joint-angle or ROM claims should not be generated from this case file unless a later validated kinematic extraction step is added.",
        ]
    )
    return "\n".join(lines) + "\n"


def metric_names() -> list[str]:
    return [str(rule["metric"]) for rule in FEATURE_RULES]


def mean_finite(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.mean(clean) if clean else math.nan


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def round_float(value: Any) -> float | str:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return ""
    return round(float(value), 6)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    return value


if __name__ == "__main__":
    main()
