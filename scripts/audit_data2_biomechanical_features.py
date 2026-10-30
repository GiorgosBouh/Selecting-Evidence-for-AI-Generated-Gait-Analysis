from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ZIP = Path("Data_2_Clinical_Gait_Signals_with_Wearable_Sensors_from_Healthy_Neurological/dataset.zip")


@dataclass(frozen=True)
class TrialFeatures:
    zip_path: str
    case_id: str
    subject: str
    trial: int
    group: str
    pathology_key: str
    pathology: str
    freq: float
    n_left_events: int
    n_right_events: int
    gait_duration_s: float
    straight_duration_s: float
    uturn_duration_s: float
    approximate_straight_speed_m_s: float
    approximate_cadence_steps_min: float
    approximate_step_length_m: float
    approximate_stride_length_m: float
    left_stride_time_mean_s: float
    right_stride_time_mean_s: float
    stride_time_asymmetry_pct: float
    left_swing_time_mean_s: float
    right_swing_time_mean_s: float
    swing_time_asymmetry_pct: float
    left_stance_time_mean_s: float
    right_stance_time_mean_s: float
    stance_time_asymmetry_pct: float
    double_support_total_s: float
    double_support_pct_gait: float
    stride_time_cv_pct: float
    clinical_score_name: str
    clinical_score_value: str
    visual_gait_assessment: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Data_2 metadata-derived biomechanical gait features.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/data2_biomechanical_feature_audit_2026-06-15.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("outputs/data2_biomechanical_feature_audit_summary_2026-06-15.md"))
    args = parser.parse_args()

    features = load_trial_features(args.zip)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(features, args.output_csv)
    args.output_md.write_text(render_summary(features, args.output_csv), encoding="utf-8")
    print(f"Wrote {args.output_csv} and {args.output_md}")


def load_trial_features(zip_path: Path) -> list[TrialFeatures]:
    rows: list[TrialFeatures] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith("_meta.json"):
                continue
            metadata = json.loads(zf.read(name))
            row = features_from_metadata(name, metadata)
            if row is not None:
                rows.append(row)
    return rows


def features_from_metadata(zip_name: str, metadata: dict[str, Any]) -> TrialFeatures | None:
    freq = float(metadata.get("freq") or 0.0)
    if freq <= 0:
        return None
    left = event_pairs(metadata.get("leftGaitEvents") or [])
    right = event_pairs(metadata.get("rightGaitEvents") or [])
    if len(left) < 2 or len(right) < 2:
        return None

    all_events = left + right
    gait_start = min(to for to, _ in all_events)
    gait_end = max(hs for _, hs in all_events)
    uturn = metadata.get("uturnBoundaries") or [math.nan, math.nan]
    uturn_start = int(uturn[0]) if len(uturn) >= 2 and is_number(uturn[0]) else gait_start
    uturn_end = int(uturn[1]) if len(uturn) >= 2 and is_number(uturn[1]) else gait_end

    go_duration = max(0.0, (uturn_start - gait_start) / freq)
    back_duration = max(0.0, (gait_end - uturn_end) / freq)
    straight_duration = go_duration + back_duration
    gait_duration = max(0.0, (gait_end - gait_start) / freq)
    uturn_duration = max(0.0, (uturn_end - uturn_start) / freq)

    straight_steps = count_straight_steps(left, right, uturn_start, uturn_end)
    approximate_speed = 17.0 / straight_duration if straight_duration > 0 else math.nan
    approximate_cadence = 60.0 * straight_steps / straight_duration if straight_duration > 0 else math.nan
    approximate_step_length = 17.0 / straight_steps if straight_steps > 0 else math.nan
    approximate_stride_length = 2.0 * approximate_step_length if math.isfinite(approximate_step_length) else math.nan

    left_hs = [hs for _, hs in left]
    right_hs = [hs for _, hs in right]
    left_stride = intervals(left_hs, freq)
    right_stride = intervals(right_hs, freq)
    left_swing = [(hs - to) / freq for to, hs in left if hs > to]
    right_swing = [(hs - to) / freq for to, hs in right if hs > to]
    left_stance = stance_intervals(left, freq)
    right_stance = stance_intervals(right, freq)
    left_stance_ranges = stance_ranges(left)
    right_stance_ranges = stance_ranges(right)
    double_support = overlap_total(left_stance_ranges, right_stance_ranges) / freq

    subject = str(metadata.get("subject", ""))
    trial = int(metadata.get("trial") or 0)
    return TrialFeatures(
        zip_path=zip_name,
        case_id=f"{subject}_{trial}",
        subject=subject,
        trial=trial,
        group=str(metadata.get("group", "")),
        pathology_key=str(metadata.get("pathologyKey", "")),
        pathology=str(metadata.get("pathology", "")),
        freq=freq,
        n_left_events=len(left),
        n_right_events=len(right),
        gait_duration_s=gait_duration,
        straight_duration_s=straight_duration,
        uturn_duration_s=uturn_duration,
        approximate_straight_speed_m_s=approximate_speed,
        approximate_cadence_steps_min=approximate_cadence,
        approximate_step_length_m=approximate_step_length,
        approximate_stride_length_m=approximate_stride_length,
        left_stride_time_mean_s=mean_or_nan(left_stride),
        right_stride_time_mean_s=mean_or_nan(right_stride),
        stride_time_asymmetry_pct=asymmetry_pct(mean_or_nan(left_stride), mean_or_nan(right_stride)),
        left_swing_time_mean_s=mean_or_nan(left_swing),
        right_swing_time_mean_s=mean_or_nan(right_swing),
        swing_time_asymmetry_pct=asymmetry_pct(mean_or_nan(left_swing), mean_or_nan(right_swing)),
        left_stance_time_mean_s=mean_or_nan(left_stance),
        right_stance_time_mean_s=mean_or_nan(right_stance),
        stance_time_asymmetry_pct=asymmetry_pct(mean_or_nan(left_stance), mean_or_nan(right_stance)),
        double_support_total_s=double_support,
        double_support_pct_gait=(100.0 * double_support / gait_duration if gait_duration > 0 else math.nan),
        stride_time_cv_pct=cv_pct(left_stride + right_stride),
        clinical_score_name=str(metadata.get("evaluationScoreName", "")),
        clinical_score_value=str(metadata.get("evaluationScoreValue", "")),
        visual_gait_assessment=str(metadata.get("visualGaitAssessment", "")),
    )


def event_pairs(raw: list[Any]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        to, hs = item
        if not is_number(to) or not is_number(hs):
            continue
        pairs.append((int(to), int(hs)))
    return sorted(pairs)


def is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def count_straight_steps(
    left: list[tuple[int, int]], right: list[tuple[int, int]], uturn_start: int, uturn_end: int
) -> int:
    return sum(1 for _, hs in left + right if hs < uturn_start or hs > uturn_end)


def intervals(samples: list[int], freq: float) -> list[float]:
    return [(b - a) / freq for a, b in zip(samples, samples[1:]) if b > a]


def stance_intervals(events: list[tuple[int, int]], freq: float) -> list[float]:
    values: list[float] = []
    for (_, hs), (next_to, _) in zip(events, events[1:]):
        if next_to > hs:
            values.append((next_to - hs) / freq)
    return values


def stance_ranges(events: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for (_, hs), (next_to, _) in zip(events, events[1:]):
        if next_to > hs:
            ranges.append((hs, next_to))
    return ranges


def overlap_total(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
    total = 0
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def mean_or_nan(values: list[float]) -> float:
    return statistics.mean(values) if values else math.nan


def cv_pct(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    avg = statistics.mean(values)
    if avg == 0:
        return math.nan
    return 100.0 * statistics.pstdev(values) / avg


def asymmetry_pct(left: float, right: float) -> float:
    if not math.isfinite(left) or not math.isfinite(right):
        return math.nan
    denom = (abs(left) + abs(right)) / 2.0
    if denom == 0:
        return math.nan
    return 100.0 * abs(left - right) / denom


def write_csv(rows: list[TrialFeatures], path: Path) -> None:
    fieldnames = list(TrialFeatures.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(getattr(row, field)) for field in fieldnames})


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return round(value, 6)
    return value


def render_summary(rows: list[TrialFeatures], csv_path: Path) -> str:
    by_pathology = Counter(row.pathology_key for row in rows)
    by_subject = unique_subject_counts(rows)
    participant_rows = first_trial_per_subject(rows)
    feature_names = [
        "approximate_straight_speed_m_s",
        "approximate_cadence_steps_min",
        "approximate_step_length_m",
        "approximate_stride_length_m",
        "left_stride_time_mean_s",
        "right_stride_time_mean_s",
        "stride_time_asymmetry_pct",
        "left_swing_time_mean_s",
        "right_swing_time_mean_s",
        "left_stance_time_mean_s",
        "right_stance_time_mean_s",
        "stance_time_asymmetry_pct",
        "double_support_pct_gait",
        "stride_time_cv_pct",
        "uturn_duration_s",
    ]
    total_unique_subjects = len({row.subject for row in rows})
    lines = [
        "# Data_2 Biomechanical Feature Audit",
        "",
        f"CSV: `{csv_path}`",
        "",
        "## Scope",
        "",
        "This audit uses metadata-derived gait events from Data_2. It does not infer joint angles.",
        "The event pairs are interpreted according to the official plotting script: toe-off to heel-strike is swing; heel-strike to next toe-off is stance.",
        "",
        "## Counts",
        "",
        f"- Trials with usable left/right gait events: `{len(rows)}`",
        f"- Unique subjects: `{total_unique_subjects}`",
        f"- Participant-level first-trial rows: `{len(participant_rows)}`",
        "",
        "### Trial Counts By Pathology",
        "",
    ]
    for key, count in sorted(by_pathology.items()):
        lines.append(f"- `{key}`: `{count}` trials, `{by_subject.get(key, 0)}` subjects")
    lines.extend(["", "## Candidate Features", ""])
    for feature in feature_names:
        values = [float(getattr(row, feature)) for row in rows if math.isfinite(float(getattr(row, feature)))]
        lines.append(f"- `{feature}`: {describe(values)}")
    lines.extend(
        [
            "",
            "## Interpretation For Revised Experiment",
            "",
            "Supported from Data_2 events:",
            "",
            "- temporal stride time and stride-time variability;",
            "- swing time and stance time per side;",
            "- left-right temporal asymmetry;",
            "- double-support overlap as percentage of gait duration;",
            "- U-turn duration;",
            "- approximate cadence;",
            "- approximate straight walking speed using the known 8.5 m out-and-back protocol.",
            "",
            "Not directly supported without additional validated algorithms:",
            "",
            "- joint angles and joint range of motion;",
            "- true step width;",
            "- true side-specific step length from sensor trajectories;",
            "- true stride length from foot displacement.",
            "",
            "Possible but should be labeled as approximate:",
            "",
            "- average step length can be approximated from known straight distance divided by straight-phase step count;",
            "- average stride length can be approximated as twice average step length;",
            "- IMU-derived foot angular-velocity features can be used as signal descriptors, not joint-angle measures.",
        ]
    )
    return "\n".join(lines) + "\n"


def unique_subject_counts(rows: list[TrialFeatures]) -> dict[str, int]:
    subjects: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        subjects[row.pathology_key].add(row.subject)
    return {key: len(value) for key, value in subjects.items()}


def first_trial_per_subject(rows: list[TrialFeatures]) -> list[TrialFeatures]:
    first: dict[str, TrialFeatures] = {}
    for row in sorted(rows, key=lambda item: (item.subject, item.trial)):
        first.setdefault(row.subject, row)
    return list(first.values())


def describe(values: list[float]) -> str:
    if not values:
        return "not available"
    return (
        f"n={len(values)}, mean={statistics.mean(values):.3f}, "
        f"sd={(statistics.pstdev(values) if len(values) > 1 else 0.0):.3f}, "
        f"min={min(values):.3f}, max={max(values):.3f}"
    )


if __name__ == "__main__":
    main()
