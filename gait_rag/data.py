from __future__ import annotations

from dataclasses import dataclass, field, replace
import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


@dataclass(frozen=True)
class GaitFeature:
    name: str
    value: float
    direction: str
    threshold: float
    source_column: str
    evidence: str


@dataclass(frozen=True)
class GaitCase:
    case_id: str
    measurements: dict[str, float]
    abnormal_features: list[GaitFeature] = field(default_factory=list)
    estimated_measurements: dict[str, float] = field(default_factory=dict)
    patient_metadata: dict[str, float] = field(default_factory=dict)
    asset_counts: dict[str, int] = field(default_factory=dict)


FEATURE_RULES = {
    "Step_UGS": ("reduced_step_length_ugs", "low"),
    "Step_FGS": ("reduced_step_length_fgs", "low"),
    "Stride_UGS": ("reduced_stride_length_ugs", "low"),
    "Stride_FGS": ("reduced_stride_length_fgs", "low"),
    "Cadence_UGS": ("increased_cadence_ugs", "high"),
    "Cadence_FGS": ("increased_cadence_fgs", "high"),
    "MonoSP_UGS": ("reduced_single_support_ugs", "low"),
    "MonoSP_FGS": ("reduced_single_support_fgs", "low"),
    "BiSP_UGS": ("prolonged_double_support_ugs", "high"),
    "BiSP_FGS": ("prolonged_double_support_fgs", "high"),
    "Velocity_UGS": ("reduced_velocity_ugs", "low"),
    "Velocity_FGS": ("reduced_velocity_fgs", "low"),
}


def load_dataset_cases(dataset_root: Path, min_abs_z: float = 1.0) -> list[GaitCase]:
    """Load the available dataset as gait cases.

    The measured gait parameters are the source of abnormal feature labels.
    Estimated gait parameters, patient measures, and raw asset counts are attached
    as case metadata so the output remains traceable to the actual dataset.
    """
    measured_path = dataset_root / "gait_parameters.csv"
    cases = load_gait_cases(measured_path, min_abs_z=min_abs_z)
    estimated = _rows_by_id(dataset_root / "gait_parameters_estimation.csv")
    patient_metadata = _rows_by_id(dataset_root / "patients_measures.csv")

    enriched: list[GaitCase] = []
    for case in cases:
        enriched.append(
            replace(
                case,
                estimated_measurements=estimated.get(case.case_id, {}),
                patient_metadata=patient_metadata.get(case.case_id, {}),
                asset_counts=_asset_counts(dataset_root, case.case_id),
            )
        )
    return enriched


def load_gait_cases(path: Path, min_abs_z: float = 1.0) -> list[GaitCase]:
    """Load measured gait cases and derive abnormal labels from numeric columns.

    If a column has too few rows or zero variance, the prototype uses coarse
    fallback thresholds. With a larger dataset, the same function uses z-score
    thresholds per gait parameter.
    """
    rows = _read_numeric_csv(path)
    if not rows:
        return []

    columns = sorted({key for row in rows for key in row if key != "ID"})
    stats = _column_stats(rows, columns)

    cases: list[GaitCase] = []
    for row in rows:
        case_id = str(row.get("ID", f"Case_{len(cases) + 1:03d}"))
        measurements = {
            key: value for key, value in row.items() if key != "ID" and isinstance(value, float)
        }
        features = _derive_abnormal_features(measurements, stats, min_abs_z)
        cases.append(GaitCase(case_id=case_id, measurements=measurements, abnormal_features=features))
    return cases


def _rows_by_id(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    rows = _read_numeric_csv(path)
    by_id: dict[str, dict[str, float]] = {}
    for row in rows:
        case_id = row.get("ID")
        if not isinstance(case_id, str):
            continue
        by_id[case_id] = {
            key: value for key, value in row.items() if key != "ID" and isinstance(value, float)
        }
    return by_id


def _asset_counts(dataset_root: Path, case_id: str) -> dict[str, int]:
    modalities = {
        "pose_json_files": dataset_root / "pose" / case_id,
        "silhouette_frames": dataset_root / "silhouette" / case_id,
        "semantic_segmentation_frames": dataset_root / "semantic_segmentation" / case_id,
        "optical_flow_frames": dataset_root / "optical_flow" / case_id,
    }
    counts = {name: _count_files(path) for name, path in modalities.items()}
    counts["total_raw_asset_files"] = sum(counts.values())
    return counts


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for candidate in path.rglob("*") if candidate.is_file())


def _read_numeric_csv(path: Path) -> list[dict[str, str | float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str | float]] = []
        for raw_row in reader:
            row: dict[str, str | float] = {}
            for key, value in raw_row.items():
                if key == "ID":
                    row[key] = value or ""
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    continue
            rows.append(row)
    return rows


def _column_stats(
    rows: Iterable[dict[str, str | float]], columns: Iterable[str]
) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    row_list = list(rows)
    for column in columns:
        values = [row[column] for row in row_list if isinstance(row.get(column), float)]
        if not values:
            continue
        stats[column] = (mean(values), pstdev(values) if len(values) > 1 else 0.0)
    return stats


def _derive_abnormal_features(
    measurements: dict[str, float],
    stats: dict[str, tuple[float, float]],
    min_abs_z: float,
) -> list[GaitFeature]:
    derived: list[GaitFeature] = []
    for column, (feature_name, abnormal_direction) in FEATURE_RULES.items():
        if column not in measurements or column not in stats:
            continue
        value = measurements[column]
        center, spread = stats[column]
        if spread > 0:
            z_score = (value - center) / spread
            is_abnormal = z_score <= -min_abs_z if abnormal_direction == "low" else z_score >= min_abs_z
            threshold = center + (-min_abs_z if abnormal_direction == "low" else min_abs_z) * spread
            evidence = f"{column}={value:.3f}, z={z_score:.2f}, threshold={threshold:.3f}"
        else:
            # Single-subject samples cannot support distributional thresholds.
            threshold = _fallback_threshold(column, value, abnormal_direction)
            is_abnormal = value <= threshold if abnormal_direction == "low" else value >= threshold
            evidence = f"{column}={value:.3f}, fallback_threshold={threshold:.3f}"
        if is_abnormal:
            derived.append(
                GaitFeature(
                    name=feature_name,
                    value=value,
                    direction=abnormal_direction,
                    threshold=threshold,
                    source_column=column,
                    evidence=evidence,
                )
            )
    return derived


def _fallback_threshold(column: str, value: float, direction: str) -> float:
    defaults = {
        "Step_UGS": 60.0,
        "Step_FGS": 65.0,
        "Stride_UGS": 120.0,
        "Stride_FGS": 130.0,
        "Cadence_UGS": 125.0,
        "Cadence_FGS": 130.0,
        "MonoSP_UGS": 0.36,
        "MonoSP_FGS": 0.36,
        "BiSP_UGS": 0.16,
        "BiSP_FGS": 0.16,
        "Velocity_UGS": 1.2,
        "Velocity_FGS": 1.4,
    }
    if column in defaults:
        return defaults[column]
    return value * (0.95 if direction == "low" else 1.05)
