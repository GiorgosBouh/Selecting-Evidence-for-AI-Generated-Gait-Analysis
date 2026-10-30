from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import sys
import time
import zipfile
from io import BytesIO
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.io as scipy_io

from gait_rag.data import GaitCase, GaitFeature, load_gait_cases
from gait_rag.explain import Explanation
from gait_rag.graph import load_graph
from gait_rag.metrics import evaluate_explanation

# Import the existing LLM RAG helpers from the pilot runner.
from run_three_llm_full_dataset_pilot import (  # type: ignore[import-not-found]
    DEFAULT_MODELS,
    answer_from_context,
    build_explanation,
    flatten_tree,
    is_float,
    load_contexts,
    load_tree,
    read_csv_by_id,
    retrieve_embedding_contexts,
    select_tree_nodes,
)


HEALTH_DATASET = "Health&Gait"
WEARABLE_DATASET = "Clinical Gait Signals"
GAITEX_DATASET = "GAITEX"
AMPUTEE_DATASET = "Above-Knee Amputee Gait"

WEARABLE_ZIP = Path("Data_2_Clinical_Gait_Signals_with_Wearable_Sensors_from_Healthy_Neurological/dataset.zip")
GAITEX_ZIP = Path("Data_3_GAITEX/data.zip")
AMPUTEE_ZIP_DIR = Path("Data_4_Above_Knee_Amputee_Gait/subject_zips")
AMPUTEE_SUBJECT_INFO = Path("Data_4_Above_Knee_Amputee_Gait/docs/Subject Information.xlsx")
HEALTH_GAIT_CSV = Path("prepared_data/health_gait/gait_parameters.csv")
HEALTH_PATIENTS_CSV = Path("prepared_data/health_gait/patients_measures.csv")
HEALTH_ESTIMATED_CSV = Path("prepared_data/health_gait/gait_parameters_estimation.csv")
DATA2_BIOMECHANICAL_CASES_JSONL = Path("outputs/data2_biomechanical_cases_2026-06-15.jsonl")

GAITEX_CONDITION_LABELS = {
    "ng": "normal gait",
    "gwo": "gait with orthosis",
    "rd": "rotation/displacement task",
    "rgs": "range-of-motion gait sequence",
}

GROUP_FEATURES = {
    "healthy": "healthy_cohort",
    "neuro": "neurological_cohort",
    "ortho": "orthopedic_cohort",
}

PATHOLOGY_FEATURES = {
    "HS": "healthy_subject",
    "PD": "pathology_pd",
    "CVA": "pathology_cva",
    "CIPN": "pathology_cipn",
    "RIL": "pathology_ril",
    "ACL": "pathology_acl",
    "KOA": "pathology_koa",
    "HOA": "pathology_hoa",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a resumable all-case LLM comparison: embedding_rag vs vectorless_tree_rag."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=["health", "wearable"], choices=["health", "wearable", "gaitex", "amputee"])
    parser.add_argument("--wearable-level", choices=["participant", "trial"], default="participant")
    parser.add_argument(
        "--wearable-source",
        choices=["metadata", "biomechanical_cases"],
        default="metadata",
        help="Use original Data_2 metadata features or participant-level biomechanical feature packets.",
    )
    parser.add_argument("--biomechanical-cases", type=Path, default=DATA2_BIOMECHANICAL_CASES_JSONL)
    parser.add_argument("--health-min-abs-z", type=float, default=1.0)
    parser.add_argument("--embedding-limit", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=None, help="Limit total cases after dataset selection; useful for smoke tests.")
    parser.add_argument("--max-cases-per-dataset", type=int, default=None)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--case-ids", nargs="+", default=None, help="Run only these case IDs, preserving the requested order.")
    parser.add_argument("--output", type=Path, default=Path("outputs/full_dataset_batch_perry_first.jsonl"))
    parser.add_argument("--no-resume", action="store_true", help="Do not skip records already present in the output JSONL.")
    parser.add_argument("--no-tree-selection-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only list counts; do not call LLMs.")
    args = parser.parse_args()

    cases = load_cases(args)
    if args.case_ids:
        cases = select_case_ids(cases, args.case_ids)
    if args.case_offset:
        cases = cases[args.case_offset :]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    print_case_manifest(cases, args.models, args.dry_run)
    if args.dry_run:
        return

    graph = load_graph(Path("resources/biomechanical_graph.json"))
    corpus = load_contexts(Path("resources/full_dataset_literature_corpus.json"))
    tree = load_tree(Path("resources/full_dataset_document_tree.json"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(args.output) if not args.no_resume else set()
    tree_selection_cache: dict[str, dict[str, Any]] = {}

    total = len(cases) * len(args.models)
    written = 0
    skipped = 0
    start = time.time()
    with args.output.open("a", encoding="utf-8") as handle:
        for model in args.models:
            for case in cases:
                key = record_key(model, case)
                if key in completed:
                    skipped += 1
                    continue
                record_start = time.time()
                try:
                    if not case.abnormal_features:
                        record = no_feature_record(model, case, graph)
                    else:
                        record = run_one_case(
                            model=model,
                            case=case,
                            corpus=corpus,
                            tree=tree,
                            graph=graph,
                            embedding_limit=args.embedding_limit,
                            tree_selection_cache=(None if args.no_tree_selection_cache else tree_selection_cache),
                        )
                    record["status"] = record.get("status", "ok")
                except Exception as exc:  # Keep long runs resumable even when one case fails.
                    record = base_record(model, case)
                    record.update({"status": "error", "error": repr(exc)})
                record["elapsed_sec"] = round(time.time() - record_start, 3)
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                handle.flush()
                written += 1
                done_now = len(completed) + skipped + written
                print(
                    f"[{done_now}/{total}] {record['status']} model={model} "
                    f"dataset={record['dataset']} case={record['case_id']} "
                    f"features={record['feature_count']} elapsed={record['elapsed_sec']}s",
                    flush=True,
                )
    print(
        f"Wrote {written} new records to {args.output}; skipped {skipped}; "
        f"wall_time_sec={round(time.time() - start, 2)}"
    )


def load_cases(args: argparse.Namespace) -> list[GaitCase]:
    cases: list[GaitCase] = []
    if "health" in args.datasets:
        health_cases = load_health_cases(args.health_min_abs_z)
        if args.max_cases_per_dataset is not None:
            health_cases = health_cases[: args.max_cases_per_dataset]
        cases.extend(health_cases)
    if "wearable" in args.datasets:
        if args.wearable_source == "biomechanical_cases":
            wearable_cases = load_biomechanical_wearable_cases(args.biomechanical_cases)
        else:
            wearable_cases = load_wearable_cases(args.wearable_level)
        if args.max_cases_per_dataset is not None:
            wearable_cases = wearable_cases[: args.max_cases_per_dataset]
        cases.extend(wearable_cases)
    if "gaitex" in args.datasets:
        gaitex_cases = load_gaitex_cases(GAITEX_ZIP)
        if args.max_cases_per_dataset is not None:
            gaitex_cases = gaitex_cases[: args.max_cases_per_dataset]
        cases.extend(gaitex_cases)
    if "amputee" in args.datasets:
        amputee_cases = load_amputee_cases(AMPUTEE_ZIP_DIR, AMPUTEE_SUBJECT_INFO)
        if args.max_cases_per_dataset is not None:
            amputee_cases = amputee_cases[: args.max_cases_per_dataset]
        cases.extend(amputee_cases)
    return cases


def load_health_cases(min_abs_z: float) -> list[GaitCase]:
    raw_cases = load_gait_cases(HEALTH_GAIT_CSV, min_abs_z=min_abs_z)
    patient_meta = read_csv_by_id(HEALTH_PATIENTS_CSV)
    estimated = read_csv_by_id(HEALTH_ESTIMATED_CSV)
    enriched: list[GaitCase] = []
    for case in sorted(raw_cases, key=lambda item: natural_key(item.case_id)):
        estimated_values = {
            key: float(value)
            for key, value in estimated.get(case.case_id, {}).items()
            if key != "ID" and is_float(value)
        }
        enriched.append(
            GaitCase(
                case_id=case.case_id,
                measurements=case.measurements,
                abnormal_features=case.abnormal_features,
                estimated_measurements=estimated_values,
                patient_metadata={"dataset_name": HEALTH_DATASET, **patient_meta.get(case.case_id, {})},
                asset_counts={},
            )
        )
    return enriched


def load_wearable_cases(level: str) -> list[GaitCase]:
    with zipfile.ZipFile(WEARABLE_ZIP) as zf:
        meta_names = sorted(
            [name for name in zf.namelist() if name.startswith("dataset/data/") and name.endswith("_meta.json")],
            key=natural_key,
        )
        if level == "participant":
            grouped: dict[str, list[str]] = defaultdict(list)
            for name in meta_names:
                subject = subject_from_meta_name(name)
                grouped[subject].append(name)
            selected = [min(names, key=trial_sort_key) for _, names in sorted(grouped.items(), key=lambda item: natural_key(item[0]))]
        else:
            selected = meta_names
        cases = []
        trial_counts = Counter(subject_from_meta_name(name) for name in meta_names)
        for name in selected:
            meta = json.loads(zf.read(name).decode("utf-8"))
            cases.append(wearable_case_from_meta(meta, name, trial_counts[subject_from_meta_name(name)]))
    return cases


def load_biomechanical_wearable_cases(path: Path) -> list[GaitCase]:
    cases: list[GaitCase] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            metadata = dict(item.get("metadata") or {})
            metadata.update(
                {
                    "dataset_name": WEARABLE_DATASET,
                    "biomechanical_case_source": str(path),
                    "aggregation_level": item.get("aggregation_level", ""),
                    "n_trials": item.get("n_trials", ""),
                    "trial_ids": item.get("trial_ids", []),
                    "case_feature_packet": item.get("case_feature_packet", ""),
                }
            )
            features = [biomechanical_feature_from_json(feature) for feature in item.get("abnormal_features", [])]
            measurements = {
                key: float(value)
                for key, value in (item.get("measurements") or {}).items()
                if numeric_or_none(value) is not None
            }
            cases.append(
                GaitCase(
                    case_id=str(item.get("case_id", "")),
                    measurements=measurements,
                    abnormal_features=features,
                    estimated_measurements={},
                    patient_metadata=metadata,
                    asset_counts={},
                )
            )
    return sorted(cases, key=lambda case: natural_key(case.case_id))



def load_gaitex_cases(path: Path) -> list[GaitCase]:
    cases: list[GaitCase] = []
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        timestamp_names = sorted(
            [
                name
                for name in names
                if name.startswith("data/") and re.search(r"/timestamps_[^/]+\.csv$", name)
            ],
            key=natural_key,
        )
        for timestamp_name in timestamp_names:
            parts = timestamp_name.split("/")
            if len(parts) < 4:
                continue
            participant = parts[1]
            condition = parts[2]
            stem = f"{participant}_{condition}"
            metadata_name = f"data/{participant}/{condition}/metadata_{stem}.json"
            metadata = json.loads(zf.read(metadata_name).decode("utf-8")) if metadata_name in names else {}
            timestamp_rows = read_gaitex_timestamps(zf.read(timestamp_name).decode("utf-8"))
            cases.append(gaitex_case_from_rows(stem, participant, condition, timestamp_name, metadata, timestamp_rows))
    return sorted(cases, key=lambda case: natural_key(case.case_id))


def read_gaitex_timestamps(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def gaitex_case_from_rows(
    case_id: str,
    participant: str,
    condition: str,
    timestamp_name: str,
    metadata: dict[str, Any],
    rows: list[dict[str, str]],
) -> GaitCase:
    labels = [str(row.get("label") or "") for row in rows if row.get("label")]
    durations: list[float] = []
    velocities: list[float] = []
    for row in rows:
        start = numeric_or_none(row.get("temporal_segment_start_[s]"))
        end = numeric_or_none(row.get("temporal_segment_end_[s]"))
        if start is not None and end is not None and end > start:
            durations.append(end - start)
        velocity = numeric_or_none(row.get("velocities_[km_h]"))
        if velocity is not None:
            velocities.append(velocity)

    label_counts = Counter(labels)
    unique_labels = sorted(label_counts, key=natural_key)
    condition_label = GAITEX_CONDITION_LABELS.get(condition, condition)
    features = [
        GaitFeature(
            "gaitex_motion_capture_trial",
            1.0,
            "present",
            1.0,
            "metadata/timestamps",
            (
                f"participant={participant}, condition={condition} ({condition_label}), "
                f"segments={len(rows)}, labels={','.join(unique_labels[:8])}"
            ),
        )
    ]
    if velocities:
        features.append(
            GaitFeature(
                "gaitex_speed_protocol",
                mean_float(velocities),
                "present",
                0.0,
                "velocities_[km_h]",
                (
                    f"velocity_km_h mean={mean_float(velocities):.2f}, "
                    f"min={min(velocities):.2f}, max={max(velocities):.2f}, n={len(velocities)}"
                ),
            )
        )
        if max(velocities) - min(velocities) >= 0.5:
            features.append(
                GaitFeature(
                    "gaitex_multi_speed_progression",
                    max(velocities) - min(velocities),
                    "range",
                    0.5,
                    "velocities_[km_h]",
                    f"velocity_range_km_h={max(velocities) - min(velocities):.2f} across labels={','.join(unique_labels[:8])}",
                )
            )
    if len(rows) >= 10 or len(unique_labels) >= 2:
        evidence = f"segments={len(rows)}, unique_labels={len(unique_labels)}"
        if durations:
            evidence += f", mean_segment_duration_s={mean_float(durations):.2f}"
        features.append(
            GaitFeature(
                "gaitex_repeated_task_segments",
                float(len(rows)),
                "present",
                1.0,
                "timestamps",
                evidence,
            )
        )

    measurements = {
        "segment_count": float(len(rows)),
        "unique_label_count": float(len(unique_labels)),
    }
    if durations:
        measurements["mean_segment_duration_s"] = mean_float(durations)
    if velocities:
        measurements.update(
            {
                "mean_velocity_km_h": mean_float(velocities),
                "min_velocity_km_h": min(velocities),
                "max_velocity_km_h": max(velocities),
            }
        )

    packet = {
        "dataset": GAITEX_DATASET,
        "case_id": case_id,
        "participant": participant,
        "condition": condition,
        "condition_label": condition_label,
        "timestamp_file": timestamp_name,
        "features": [
            {"name": feature.name, "value": feature.value, "evidence": feature.evidence}
            for feature in features
        ],
        "segment_count": len(rows),
        "labels": dict(label_counts),
        "velocity_km_h": {
            "mean": round(mean_float(velocities), 3) if velocities else None,
            "min": round(min(velocities), 3) if velocities else None,
            "max": round(max(velocities), 3) if velocities else None,
            "n": len(velocities),
        },
        "mean_segment_duration_s": round(mean_float(durations), 3) if durations else None,
        "metadata_keys": sorted(metadata.keys()),
    }
    return GaitCase(
        case_id=case_id,
        measurements=measurements,
        abnormal_features=features,
        estimated_measurements={},
        patient_metadata={
            "dataset_name": GAITEX_DATASET,
            "participant": participant,
            "condition": condition,
            "condition_label": condition_label,
            "timestamp_path": timestamp_name,
            "segment_count": len(rows),
            "segment_labels": unique_labels,
            "velocity_mean_km_h": round(mean_float(velocities), 3) if velocities else "",
            "velocity_min_km_h": round(min(velocities), 3) if velocities else "",
            "velocity_max_km_h": round(max(velocities), 3) if velocities else "",
            "case_feature_packet": json.dumps(packet, ensure_ascii=True),
        },
        asset_counts={},
    )



def load_amputee_cases(zip_dir: Path, subject_info_path: Path) -> list[GaitCase]:
    subject_info = read_amputee_subject_info(subject_info_path)
    cases: list[GaitCase] = []
    for zip_path in sorted(zip_dir.glob("TF*.zip"), key=lambda path: natural_key(path.stem)):
        subject_id = zip_path.stem
        with zipfile.ZipFile(zip_path) as zf:
            mat_name = f"{subject_id}/Matlab Workspace/{subject_id}_DATA.mat"
            if mat_name not in zf.namelist():
                continue
            mat = scipy_io.loadmat(BytesIO(zf.read(mat_name)), squeeze_me=True, struct_as_record=False)
        root = mat.get(subject_id)
        if root is None or not hasattr(root, "data"):
            continue
        info = dict(subject_info.get(subject_id, {}))
        info.update(mat_info_to_metadata(getattr(root, "info", None)))
        for speed_field in sorted(mat_fields(root.data), key=natural_key):
            speed_struct = getattr(root.data, speed_field)
            if not hasattr(speed_struct, "ipsilateral") or not hasattr(speed_struct, "contralateral"):
                continue
            speed_mps = speed_field_to_float(speed_field)
            measurements = amputee_speed_measurements(speed_struct)
            features = amputee_features_from_measurements(measurements, speed_mps)
            packet = amputee_case_packet(subject_id, speed_field, speed_mps, info, measurements, features)
            metadata = {
                "dataset_name": AMPUTEE_DATASET,
                "subject": subject_id,
                "speed_field": speed_field,
                "speed_mps": speed_mps,
                "amputation_side": info.get("amputation_side") or info.get("Amputation Side", ""),
                "reason_for_amputation": info.get("Reason for Amputation", ""),
                "k_level": info.get("k_level") or info.get("K-Level", ""),
                "knee_prosthesis": info.get("Knee Prosthesis", ""),
                "foot_prosthesis": info.get("Foot Prosthesis", ""),
                "handrails": info.get("Handrails?", ""),
                "case_feature_packet": json.dumps(packet, ensure_ascii=True),
            }
            cases.append(
                GaitCase(
                    case_id=f"{subject_id}_{speed_field.replace('speed_', '')}",
                    measurements=measurements,
                    abnormal_features=features,
                    estimated_measurements={},
                    patient_metadata=metadata,
                    asset_counts={},
                )
            )
    return sorted(cases, key=lambda case: natural_key(case.case_id))


def read_amputee_subject_info(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        shared_xml = zf.read("xl/sharedStrings.xml").decode("utf-8")
        for item in re.findall(r"<si.*?</si>", shared_xml, flags=re.S):
            text = "".join(re.findall(r"<t[^>]*>(.*?)</t>", item, flags=re.S))
            shared.append(html.unescape(text))
        sheet_xml = zf.read("xl/worksheets/sheet1.xml").decode("utf-8")
    rows: list[list[str]] = []
    for row_xml in re.findall(r"<row[^>]*>(.*?)</row>", sheet_xml, flags=re.S):
        cells: dict[int, str] = {}
        for attrs, body in re.findall(r"<c([^>]*)>(.*?)</c>", row_xml, flags=re.S):
            ref = re.search(r"r=\"([A-Z]+)\d+\"", attrs)
            index = excel_column_index(ref.group(1)) if ref else len(cells)
            value_match = re.search(r"<v>(.*?)</v>", body)
            value = value_match.group(1) if value_match else ""
            if 't="s"' in attrs and value:
                value = shared[int(value)]
            cells[index] = html.unescape(value)
        if cells:
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    if not rows:
        return {}
    header = rows[0]
    result: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        if not row or not str(row[0]).startswith("TF"):
            continue
        item = {header[i]: normalize_excel_value(row[i]) for i in range(min(len(header), len(row)))}
        result[str(row[0])] = item
    return result


def excel_column_index(column: str) -> int:
    value = 0
    for char in column:
        value = value * 26 + (ord(char.upper()) - 64)
    return value - 1


def normalize_excel_value(value: str) -> Any:
    if value == "":
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def mat_fields(obj: Any) -> list[str]:
    return [name for name in dir(obj) if not name.startswith("_")]


def mat_info_to_metadata(info: Any) -> dict[str, Any]:
    if info is None:
        return {}
    result: dict[str, Any] = {}
    for name in mat_fields(info):
        value = getattr(info, name)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        result[name] = value.item() if hasattr(value, "item") else value
    return result


def speed_field_to_float(speed_field: str) -> float:
    return float(speed_field.replace("speed_", "").replace("p", "."))


def as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def series_rom(value: Any) -> float:
    array = as_float_array(value)
    return float(np.nanmax(array) - np.nanmin(array))


def peak_abs(value: Any) -> float:
    array = as_float_array(value)
    return float(np.nanmax(np.abs(array)))


def ankle_plantarflexion_moment_peak(value: Any) -> float:
    array = as_float_array(value)
    return float(abs(np.nanmin(array)))


def joint_value(speed_struct: Any, side: str, joint: str, value: str, statistic: str = "avg") -> Any:
    return getattr(getattr(getattr(getattr(speed_struct, side), joint), value), statistic)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def amputee_speed_measurements(speed_struct: Any) -> dict[str, float]:
    measurements: dict[str, float] = {}
    for joint in ["ankle", "knee", "hip"]:
        ipsi_position = joint_value(speed_struct, "ipsilateral", joint, "position")
        contra_position = joint_value(speed_struct, "contralateral", joint, "position")
        ipsi_moment = joint_value(speed_struct, "ipsilateral", joint, "moment")
        contra_moment = joint_value(speed_struct, "contralateral", joint, "moment")
        ipsi_rom = series_rom(ipsi_position)
        contra_rom = series_rom(contra_position)
        ipsi_peak = peak_abs(ipsi_moment)
        contra_peak = peak_abs(contra_moment)
        measurements[f"{joint}_position_ipsilateral_rom_deg"] = ipsi_rom
        measurements[f"{joint}_position_contralateral_rom_deg"] = contra_rom
        measurements[f"{joint}_position_rom_ratio_ipsi_contra"] = safe_ratio(ipsi_rom, contra_rom)
        measurements[f"{joint}_moment_ipsilateral_peak_nmkg"] = ipsi_peak
        measurements[f"{joint}_moment_contralateral_peak_nmkg"] = contra_peak
        measurements[f"{joint}_moment_peak_ratio_ipsi_contra"] = safe_ratio(ipsi_peak, contra_peak)
    ipsi_pf = ankle_plantarflexion_moment_peak(joint_value(speed_struct, "ipsilateral", "ankle", "moment"))
    contra_pf = ankle_plantarflexion_moment_peak(joint_value(speed_struct, "contralateral", "ankle", "moment"))
    measurements["ankle_plantarflexion_moment_ipsilateral_peak_nmkg"] = ipsi_pf
    measurements["ankle_plantarflexion_moment_contralateral_peak_nmkg"] = contra_pf
    measurements["ankle_plantarflexion_moment_ratio_ipsi_contra"] = safe_ratio(ipsi_pf, contra_pf)
    return measurements


def amputee_features_from_measurements(measurements: dict[str, float], speed_mps: float) -> list[GaitFeature]:
    features: list[GaitFeature] = []
    add = features.append
    ankle_pf_ratio = measurements["ankle_plantarflexion_moment_ratio_ipsi_contra"]
    if ankle_pf_ratio < 0.75:
        add(amputee_feature(
            "prosthetic_ankle_reduced_push_off_moment",
            ankle_pf_ratio,
            "low",
            0.75,
            "ankle_plantarflexion_moment_ratio_ipsi_contra",
            speed_mps,
            measurements,
            "prosthetic ankle plantarflexion moment peak is reduced versus intact side",
            ["ankle_plantarflexion_moment_ipsilateral_peak_nmkg", "ankle_plantarflexion_moment_contralateral_peak_nmkg"],
        ))
    ankle_rom_ratio = measurements["ankle_position_rom_ratio_ipsi_contra"]
    if ankle_rom_ratio < 0.80 or ankle_rom_ratio > 1.25:
        add(amputee_feature(
            "prosthetic_ankle_rom_asymmetry",
            ankle_rom_ratio,
            "asymmetric",
            0.80 if ankle_rom_ratio < 0.80 else 1.25,
            "ankle_position_rom_ratio_ipsi_contra",
            speed_mps,
            measurements,
            "prosthetic ankle sagittal-plane range of motion differs from intact side",
            ["ankle_position_ipsilateral_rom_deg", "ankle_position_contralateral_rom_deg"],
        ))
    knee_rom_ratio = measurements["knee_position_rom_ratio_ipsi_contra"]
    if knee_rom_ratio < 0.80:
        add(amputee_feature(
            "prosthetic_knee_reduced_rom",
            knee_rom_ratio,
            "low",
            0.80,
            "knee_position_rom_ratio_ipsi_contra",
            speed_mps,
            measurements,
            "prosthetic-side knee sagittal-plane range of motion is reduced versus intact side",
            ["knee_position_ipsilateral_rom_deg", "knee_position_contralateral_rom_deg"],
        ))
    hip_rom_ratio = measurements["hip_position_rom_ratio_ipsi_contra"]
    if hip_rom_ratio > 1.15:
        add(amputee_feature(
            "prosthetic_hip_compensation_rom",
            hip_rom_ratio,
            "high",
            1.15,
            "hip_position_rom_ratio_ipsi_contra",
            speed_mps,
            measurements,
            "prosthetic-side hip sagittal-plane range of motion is higher than intact side",
            ["hip_position_ipsilateral_rom_deg", "hip_position_contralateral_rom_deg"],
        ))
    knee_moment_ratio = measurements["knee_moment_peak_ratio_ipsi_contra"]
    if knee_moment_ratio < 0.75 or knee_moment_ratio > 1.25:
        add(amputee_feature(
            "prosthetic_knee_moment_asymmetry",
            knee_moment_ratio,
            "asymmetric",
            0.75 if knee_moment_ratio < 0.75 else 1.25,
            "knee_moment_peak_ratio_ipsi_contra",
            speed_mps,
            measurements,
            "prosthetic-side knee moment peak differs from intact side",
            ["knee_moment_ipsilateral_peak_nmkg", "knee_moment_contralateral_peak_nmkg"],
        ))
    hip_moment_ratio = measurements["hip_moment_peak_ratio_ipsi_contra"]
    if hip_moment_ratio < 0.75 or hip_moment_ratio > 1.25:
        add(amputee_feature(
            "prosthetic_hip_moment_asymmetry",
            hip_moment_ratio,
            "asymmetric",
            0.75 if hip_moment_ratio < 0.75 else 1.25,
            "hip_moment_peak_ratio_ipsi_contra",
            speed_mps,
            measurements,
            "prosthetic-side hip moment peak differs from intact side",
            ["hip_moment_ipsilateral_peak_nmkg", "hip_moment_contralateral_peak_nmkg"],
        ))
    return features


def amputee_feature(
    name: str,
    value: float,
    direction: str,
    threshold: float,
    source_column: str,
    speed_mps: float,
    measurements: dict[str, float],
    interpretation: str,
    detail_keys: list[str],
) -> GaitFeature:
    details = ", ".join(f"{key}={measurements[key]:.3f}" for key in detail_keys)
    evidence = (
        f"speed={speed_mps:.1f} m/s, {source_column}={value:.3f}, "
        f"threshold={threshold:.3f}, {details}; {interpretation}"
    )
    return GaitFeature(name, float(value), direction, float(threshold), source_column, evidence)


def amputee_case_packet(
    subject_id: str,
    speed_field: str,
    speed_mps: float,
    info: dict[str, Any],
    measurements: dict[str, float],
    features: list[GaitFeature],
) -> dict[str, Any]:
    return {
        "dataset": AMPUTEE_DATASET,
        "case_id": f"{subject_id}_{speed_field.replace('speed_', '')}",
        "subject": subject_id,
        "speed_mps": speed_mps,
        "side_provenance": {
            "ipsilateral": "prosthetic/amputated limb side",
            "contralateral": "sound limb side",
            "amputation_side": info.get("amputation_side") or info.get("Amputation Side", ""),
        },
        "clinical_metadata": {
            "k_level": info.get("k_level") or info.get("K-Level", ""),
            "reason_for_amputation": info.get("Reason for Amputation", ""),
            "knee_prosthesis": info.get("Knee Prosthesis", ""),
            "foot_prosthesis": info.get("Foot Prosthesis", ""),
            "handrails": info.get("Handrails?", ""),
        },
        "features": [
            {"name": feature.name, "value": round(feature.value, 4), "evidence": feature.evidence}
            for feature in features
        ],
        "measurements": {key: round(value, 4) for key, value in sorted(measurements.items())},
    }

def mean_float(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def biomechanical_feature_from_json(item: dict[str, Any]) -> GaitFeature:
    value = numeric_or_none(item.get("value"))
    threshold = numeric_or_none(item.get("threshold"))
    return GaitFeature(
        name=str(item.get("feature_id", "")),
        value=float(value if value is not None else 0.0),
        direction=str(item.get("direction", "")),
        threshold=float(threshold if threshold is not None else 0.0),
        source_column=str(item.get("metric", "")),
        evidence=str(item.get("evidence", "")),
    )


def wearable_case_from_meta(meta: dict[str, Any], meta_name: str, available_trial_count: int) -> GaitCase:
    left_events = meta.get("leftGaitEvents") or []
    right_events = meta.get("rightGaitEvents") or []
    total_events = len(left_events) + len(right_events)
    features: list[GaitFeature] = [
        GaitFeature(
            "wearable_imu_trial",
            1.0,
            "present",
            1.0,
            "protocol/sensor/freq",
            f"protocol={meta.get('protocol')}, freq={meta.get('freq')}, sensor={meta.get('sensor')}",
        )
    ]
    if meta.get("uturnBoundaries"):
        features.append(
            GaitFeature(
                "u_turn_available",
                1.0,
                "present",
                1.0,
                "uturnBoundaries",
                f"uturnBoundaries={meta.get('uturnBoundaries')}",
            )
        )
    if total_events > 0:
        features.append(
            GaitFeature(
                "gait_events_available",
                float(total_events),
                "present",
                1.0,
                "left/right gait events",
                f"left_events={len(left_events)}, right_events={len(right_events)}",
            )
        )
    group_feature = GROUP_FEATURES.get(str(meta.get("group")))
    if group_feature:
        features.append(
            GaitFeature(group_feature, 1.0, "present", 1.0, "group", f"group={meta.get('group')}")
        )
    pathology_feature = PATHOLOGY_FEATURES.get(str(meta.get("pathologyKey")))
    if pathology_feature:
        features.append(
            GaitFeature(
                pathology_feature,
                1.0,
                "present",
                1.0,
                "pathologyKey/pathology",
                f"pathologyKey={meta.get('pathologyKey')}, pathology={meta.get('pathology')}",
            )
        )
    vga = numeric_or_none(meta.get("visualGaitAssessment"))
    if vga is not None and vga >= 1:
        features.append(
            GaitFeature(
                "visual_gait_impairment",
                vga,
                "high",
                1.0,
                "visualGaitAssessment",
                f"visualGaitAssessment={vga} on a 0-4 scale",
            )
        )
    score_value = numeric_or_none(meta.get("evaluationScoreValue"))
    score_name = str(meta.get("evaluationScoreName") or "")
    if score_name and score_value is not None:
        score_feature = "clinical_score_updrs_iii" if "UPDRS" in score_name.upper() else "clinical_score_available"
        features.append(
            GaitFeature(
                score_feature,
                score_value,
                "present",
                0.0,
                "evaluationScoreValue",
                f"{score_name}={score_value}",
            )
        )
    metadata = {
        key: value
        for key, value in meta.items()
        if key not in {"leftGaitEvents", "rightGaitEvents"}
    }
    metadata.update(
        {
            "dataset_name": WEARABLE_DATASET,
            "meta_path": meta_name,
            "available_trial_count_for_subject": available_trial_count,
        }
    )
    return GaitCase(
        case_id=f"{meta.get('subject')}_{meta.get('trial')}",
        measurements={
            "age": float(meta.get("age") or 0.0),
            "BMI": float(meta.get("BMI") or 0.0),
            "visualGaitAssessment": float(vga or 0.0),
            "leftGaitEvents": float(len(left_events)),
            "rightGaitEvents": float(len(right_events)),
        },
        abnormal_features=features,
        patient_metadata=metadata,
        asset_counts={},
    )


def run_one_case(
    model: str,
    case: GaitCase,
    corpus: Any,
    tree: Any,
    graph: Any,
    embedding_limit: int,
    tree_selection_cache: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    features = [feature.name for feature in case.abnormal_features]
    embedding_context = retrieve_embedding_contexts(features, corpus, limit=embedding_limit)
    embedding_answer = answer_from_context(
        model=model,
        case=case,
        contexts=[
            {
                "source_id": item.context_id,
                "text": item.text,
                "citation": item.citation,
                "matched_features": [feature for feature in features if feature in item.supports],
            }
            for item in embedding_context
        ],
        context_key="source_id",
    )
    embedding_explanation = build_explanation(
        pipeline="embedding_rag",
        case=case,
        answer=embedding_answer,
        evidence_contexts={
            item.context_id: evidence_context(item.context_id, item.text, item.citation, "context")
            for item in embedding_context
        },
        context_supports={item.context_id: set(item.supports) for item in embedding_context},
        context_prefix="context",
        context_key="source_id",
    )

    tree_search = cached_tree_search(model, case, tree, tree_selection_cache)
    tree_search = enforce_exact_tree_coverage(tree_search, case, tree)
    selected_nodes = [node for node in flatten_tree(tree) if node.node_id in tree_search.get("node_list", [])]
    tree_answer = answer_from_context(
        model=model,
        case=case,
        contexts=[
            {
                "source_id": node.node_id,
                "title": node.title,
                "text": node.text,
                "citation": node.citation,
                "matched_features": [feature for feature in features if feature in node.supports],
            }
            for node in selected_nodes
            if any(feature in node.supports for feature in features)
        ],
        context_key="source_id",
    )
    tree_explanation = build_explanation(
        pipeline="vectorless_tree_rag",
        case=case,
        answer=tree_answer,
        evidence_contexts={
            node.node_id: evidence_context(node.node_id, f"{node.title}: {node.text}", node.citation, "node")
            for node in selected_nodes
        },
        context_supports={node.node_id: set(node.supports) for node in selected_nodes},
        context_prefix="node",
        context_key="source_id",
    )

    record = base_record(model, case)
    record.update(
        {
            "status": "ok",
            "embedding_rag": {
                "retrieved_contexts": [item.context_id for item in embedding_context],
                "llm_answer": embedding_answer,
                "explanation": embedding_explanation.to_dict(),
                "evaluation": evaluate_explanation(case, embedding_explanation, graph).to_dict(),
            },
            "vectorless_tree_rag": {
                "tree_search": tree_search,
                "selected_nodes": [node.node_id for node in selected_nodes],
                "llm_answer": tree_answer,
                "explanation": tree_explanation.to_dict(),
                "evaluation": evaluate_explanation(case, tree_explanation, graph).to_dict(),
            },
        }
    )
    return record


def evidence_context(identifier: str, text: str, citation: str, prefix: str):
    from gait_rag.explain import Evidence

    kind = "document_tree_node" if prefix == "node" else "text_chunk"
    return Evidence(evidence_id=f"{prefix}:{identifier}", kind=kind, text=f"{text} | source: {citation}")


def cached_tree_search(
    model: str,
    case: GaitCase,
    tree: Any,
    cache: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if cache is None:
        return select_tree_nodes(model, case, tree)
    key = json.dumps(
        {
            "model": model,
            "features": sorted(feature.name for feature in case.abnormal_features),
        },
        sort_keys=True,
    )
    if key not in cache:
        cache[key] = select_tree_nodes(model, case, tree)
    result = dict(cache[key])
    result["cache_key"] = key
    return result



def enforce_exact_tree_coverage(tree_search: dict[str, Any], case: GaitCase, tree: Any) -> dict[str, Any]:
    """Keep only exact feature-tag nodes and add missing exact-match nodes deterministically."""
    features = {feature.name for feature in case.abnormal_features}
    nodes = flatten_tree(tree)
    by_id = {node.node_id: node for node in nodes}
    selected: list[str] = []
    covered: set[str] = set()
    for node_id in tree_search.get("node_list", []):
        node = by_id.get(str(node_id))
        if node is None:
            continue
        node_features = features & set(node.supports)
        if not node_features:
            continue
        selected.append(node.node_id)
        covered.update(node_features)
    for node in nodes:
        node_features = features & set(node.supports)
        if node_features - covered:
            selected.append(node.node_id)
            covered.update(node_features)
    result = dict(tree_search)
    result["node_list"] = list(dict.fromkeys(selected))
    missing = sorted(features - covered)
    if missing:
        result["missing_exact_feature_tags"] = missing
    if result.get("node_list") != tree_search.get("node_list"):
        result["postprocessed_exact_feature_coverage"] = True
    return result


def no_feature_record(model: str, case: GaitCase, graph: Any) -> dict[str, Any]:
    record = base_record(model, case)
    embedding = empty_explanation("embedding_rag", case.case_id)
    tree = empty_explanation("vectorless_tree_rag", case.case_id)
    record.update(
        {
            "status": "no_features",
            "embedding_rag": {
                "retrieved_contexts": [],
                "llm_answer": {},
                "explanation": embedding.to_dict(),
                "evaluation": evaluate_explanation(case, embedding, graph).to_dict(),
            },
            "vectorless_tree_rag": {
                "tree_search": {},
                "selected_nodes": [],
                "llm_answer": {},
                "explanation": tree.to_dict(),
                "evaluation": evaluate_explanation(case, tree, graph).to_dict(),
            },
        }
    )
    return record


def empty_explanation(pipeline: str, case_id: str) -> Explanation:
    return Explanation(
        pipeline=pipeline,  # type: ignore[arg-type]
        case_id=case_id,
        mentioned_features=[],
        claims=[],
        supporting_evidence=[],
        unsupported_claims=[],
        summary="No abnormal/supported features were selected for explanation.",
    )


def base_record(model: str, case: GaitCase) -> dict[str, Any]:
    return {
        "model": model,
        "case_id": case.case_id,
        "dataset": case.patient_metadata.get("dataset_name", ""),
        "feature_count": len(case.abnormal_features),
        "features": [
            {"name": feature.name, "value": feature.value, "evidence": feature.evidence}
            for feature in case.abnormal_features
        ],
        "patient_metadata": compact_metadata(case.patient_metadata),
    }


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "dataset_name",
        "ID",
        "AGE",
        "BMI",
        "SEX",
        "age",
        "BMI",
        "sex",
        "group",
        "pathologyKey",
        "pathology",
        "subject",
        "trial",
        "protocol",
        "sensor",
        "freq",
        "visualGaitAssessment",
        "evaluationScoreName",
        "evaluationScoreValue",
        "meta_path",
        "available_trial_count_for_subject",
        "aggregation_level",
        "n_trials",
        "clinical_score_name",
        "clinical_score_value",
        "visual_gait_assessment_mean",
        "biomechanical_case_source",
        "participant",
        "condition",
        "condition_label",
        "timestamp_path",
        "segment_count",
        "segment_labels",
        "velocity_mean_km_h",
        "velocity_min_km_h",
        "velocity_max_km_h",
        "speed_field",
        "speed_mps",
        "amputation_side",
        "reason_for_amputation",
        "k_level",
        "knee_prosthesis",
        "foot_prosthesis",
        "handrails",
    ]
    return {key: metadata[key] for key in keep if key in metadata}


def load_completed_keys(path: Path) -> set[tuple[str, str, str]]:
    completed: set[tuple[str, str, str]] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") in {"ok", "no_features"}:
                completed.add((str(record.get("model")), str(record.get("dataset")), str(record.get("case_id"))))
    return completed


def record_key(model: str, case: GaitCase) -> tuple[str, str, str]:
    return (model, str(case.patient_metadata.get("dataset_name", "")), case.case_id)


def select_case_ids(cases: list[GaitCase], requested_ids: list[str]) -> list[GaitCase]:
    by_id = {case.case_id: case for case in cases}
    missing = [case_id for case_id in requested_ids if case_id not in by_id]
    if missing:
        available_preview = ", ".join(sorted(by_id, key=natural_key)[:20])
        raise SystemExit(f"Requested case IDs not found: {', '.join(missing)}. Available preview: {available_preview}")
    return [by_id[case_id] for case_id in requested_ids]


def print_case_manifest(cases: list[GaitCase], models: list[str], dry_run: bool) -> None:
    by_dataset = Counter(case.patient_metadata.get("dataset_name", "") for case in cases)
    with_features = Counter(
        case.patient_metadata.get("dataset_name", "") for case in cases if case.abnormal_features
    )
    print("Batch manifest")
    print(f"  cases_total={len(cases)} models={len(models)} records={len(cases) * len(models)} dry_run={dry_run}")
    for dataset, count in sorted(by_dataset.items()):
        print(f"  dataset={dataset}: cases={count} with_features={with_features[dataset]}")
    print(f"  models={', '.join(models)}")


def subject_from_meta_name(name: str) -> str:
    parts = name.split("/")
    if len(parts) < 7:
        return name
    return parts[4]


def trial_sort_key(name: str) -> list[Any]:
    parts = name.split("/")
    trial = parts[5] if len(parts) >= 7 else name
    return natural_key(trial)


def natural_key(value: object) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def numeric_or_none(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
