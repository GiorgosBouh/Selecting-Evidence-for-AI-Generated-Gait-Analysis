from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gait_rag.data import GaitCase, GaitFeature, load_gait_cases
from gait_rag.explain import Claim, Evidence, Explanation
from gait_rag.metrics import evaluate_explanation
from gait_rag.graph import load_graph


DEFAULT_MODELS = ["mistral-small:24b", "qwen3:14b", "llama3.1:8b"]


DATASET_PATHOLOGY_FEATURES = {
    "pathology_above_knee_amputation",
    "pathology_acl",
    "pathology_cipn",
    "pathology_cva",
    "pathology_hoa",
    "pathology_koa",
    "pathology_pd",
    "pathology_ril",
}

PATHOLOGY_KEY_TO_FEATURE = {
    "ACL": "pathology_acl",
    "CIPN": "pathology_cipn",
    "CVA": "pathology_cva",
    "HOA": "pathology_hoa",
    "KOA": "pathology_koa",
    "PD": "pathology_pd",
    "RIL": "pathology_ril",
}

PATHOLOGY_TERM_SCOPES = [
    (re.compile(r"\bparkinson(?:ian)?(?: disease)?\b|\bpd\b", re.I), {"pathology_pd"}, "Parkinson disease"),
    (re.compile(r"\bcerebrovascular accident\b|\bcva\b|\bstroke\b|\bhemiplegi(?:a|c)\b", re.I), {"pathology_cva"}, "CVA/stroke"),
    (
        re.compile(r"\bchemotherapy-induced peripheral neuropathy\b|\btoxic peripheral neuropathy\b|\bperipheral neuropathy\b|\bcipn\b|\bneuropath(?:y|ic)\b", re.I),
        {"pathology_cipn"},
        "CIPN/toxic peripheral neuropathy",
    ),
    (
        re.compile(r"\bradiation-induced leukoencephalopathy\b|\bleukoencephalopathy\b|\bril\b", re.I),
        {"pathology_ril"},
        "radiation-induced leukoencephalopathy",
    ),
    (re.compile(r"\banterior cruciate ligament\b|\bacl\b", re.I), {"pathology_acl"}, "ACL injury"),
    (re.compile(r"\bknee osteoarthritis\b|\bknee oa\b|\bosteoarthritic knee\b", re.I), {"pathology_koa"}, "knee osteoarthritis"),
    (re.compile(r"\bhip osteoarthritis\b|\bhip oa\b|\barthritic hip\b", re.I), {"pathology_hoa"}, "hip osteoarthritis"),
    (re.compile(r"\bosteoarthritis\b|\boa\b", re.I), {"pathology_koa", "pathology_hoa"}, "osteoarthritis"),
    (re.compile(r"\bantalgic gait\b|\bantalgic\b", re.I), {"pathology_acl", "pathology_hoa", "pathology_koa"}, "antalgic gait"),
    (re.compile(r"\bataxi(?:a|c)\b|\bcerebellar\b", re.I), set(), "ataxia/cerebellar gait"),
    (re.compile(r"\bfunctional gait disorder\b", re.I), set(), "functional gait disorder"),
    (re.compile(r"\bvestibular\b", re.I), set(), "vestibular disorder"),
    (re.compile(r"\bmultiple sclerosis\b", re.I), set(), "multiple sclerosis"),
    (re.compile(r"\bhuntington(?:'s)?(?: disease| chorea)?\b", re.I), set(), "Huntington disease"),
    (re.compile(r"\bcerebral palsy\b|\bspastic diplegia\b|\bdiplegia\b", re.I), set(), "cerebral palsy/spastic diplegia"),
    (re.compile(r"\babove[- ]knee amput(?:ee|ation)\b|\bamput(?:ee|ation)\b|\bprosthe(?:sis|tic)\b", re.I), {"pathology_above_knee_amputation"}, "amputation/prosthetic gait"),
    (re.compile(r"\bdiabetic\b|\bdiabetes\b", re.I), set(), "diabetic gait/foot"),
    (re.compile(r"\bmyopathy\b|\bmyopathic\b|\bmuscular dystrophy\b", re.I), set(), "myopathy/muscular dystrophy"),
    (re.compile(r"\bspinal cord injury\b|\bmyelodysplasia\b", re.I), set(), "spinal cord injury/myelodysplasia"),
    (re.compile(r"\balzheimer(?:'s)?\b|\bdementia\b", re.I), set(), "Alzheimer/dementia"),
]


@dataclass(frozen=True)
class TextContext:
    context_id: str
    text: str
    citation: str
    supports: list[str]


@dataclass(frozen=True)
class TreeNode:
    node_id: str
    title: str
    summary: str
    text: str
    citation: str
    supports: list[str]
    children: tuple["TreeNode", ...]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 3-LLM full-dataset pilot comparison.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--health-case", default="PA277")
    parser.add_argument("--wearable-meta", default="dataset/data/neuro/PD/PD_6/PD_6_1/PD_6_1_meta.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/three_llm_full_dataset_pilot.json"))
    args = parser.parse_args()

    graph = load_graph(Path("resources/biomechanical_graph.json"))
    corpus = load_contexts(Path("resources/full_dataset_literature_corpus.json"))
    tree = load_tree(Path("resources/full_dataset_document_tree.json"))

    cases = [
        load_health_case(args.health_case),
        load_wearable_case(args.wearable_meta),
    ]

    results: list[dict[str, Any]] = []
    for model in args.models:
        model_start = time.time()
        model_result: dict[str, Any] = {"model": model, "cases": []}
        for case in cases:
            features = [feature.name for feature in case.abnormal_features]
            embedding_context = retrieve_embedding_contexts(features, corpus, limit=4)
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
                    item.context_id: Evidence(
                        evidence_id=f"context:{item.context_id}",
                        kind="text_chunk",
                        text=f"{item.text} | source: {item.citation}",
                    )
                    for item in embedding_context
                },
                context_supports={item.context_id: set(item.supports) for item in embedding_context},
                context_prefix="context",
                context_key="source_id",
            )

            tree_search = select_tree_nodes(model, case, tree)
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
                    node.node_id: Evidence(
                        evidence_id=f"node:{node.node_id}",
                        kind="document_tree_node",
                        text=f"{node.title}: {node.text} | source: {node.citation}",
                    )
                    for node in selected_nodes
                },
                context_supports={node.node_id: set(node.supports) for node in selected_nodes},
                context_prefix="node",
                context_key="source_id",
            )

            model_result["cases"].append(
                {
                    "case_id": case.case_id,
                    "dataset": case.patient_metadata.get("dataset_name", ""),
                    "feature_count": len(features),
                    "features": [
                        {
                            "name": feature.name,
                            "value": feature.value,
                            "evidence": feature.evidence,
                        }
                        for feature in case.abnormal_features
                    ],
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
        model_result["elapsed_sec"] = round(time.time() - model_start, 2)
        results.append(model_result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} for {len(args.models)} model(s) and {len(cases)} case(s).")


def load_health_case(case_id: str) -> GaitCase:
    cases = load_gait_cases(Path("prepared_data/health_gait/gait_parameters.csv"))
    by_id = {case.case_id: case for case in cases}
    case = by_id[case_id]
    patient_meta = read_csv_by_id(Path("prepared_data/health_gait/patients_measures.csv")).get(case_id, {})
    estimated = read_csv_by_id(Path("prepared_data/health_gait/gait_parameters_estimation.csv")).get(case_id, {})
    return GaitCase(
        case_id=case.case_id,
        measurements=case.measurements,
        abnormal_features=case.abnormal_features,
        estimated_measurements={key: float(value) for key, value in estimated.items() if key != "ID" and is_float(value)},
        patient_metadata={"dataset_name": "Health&Gait", **patient_meta},
        asset_counts={},
    )


def load_wearable_case(meta_name: str) -> GaitCase:
    zip_path = Path("Data_2_Clinical_Gait_Signals_with_Wearable_Sensors_from_Healthy_Neurological/dataset.zip")
    with zipfile.ZipFile(zip_path) as zf:
        meta = json.loads(zf.read(meta_name).decode("utf-8"))
    left_events = meta.get("leftGaitEvents") or []
    right_events = meta.get("rightGaitEvents") or []
    vga = float(meta.get("visualGaitAssessment") or 0.0)
    score_value = meta.get("evaluationScoreValue")
    features = [
        GaitFeature(
            "wearable_imu_trial",
            1.0,
            "present",
            1.0,
            "protocol",
            f"protocol={meta.get('protocol')}, freq={meta.get('freq')}, sensor={meta.get('sensor')}",
        ),
        GaitFeature(
            "u_turn_available",
            1.0 if meta.get("uturnBoundaries") else 0.0,
            "present",
            1.0,
            "uturnBoundaries",
            f"uturnBoundaries={meta.get('uturnBoundaries')}",
        ),
        GaitFeature(
            "gait_events_available",
            float(len(left_events) + len(right_events)),
            "present",
            1.0,
            "left/right gait events",
            f"left_events={len(left_events)}, right_events={len(right_events)}",
        ),
    ]
    if meta.get("group") == "neuro":
        features.append(GaitFeature("neurological_cohort", 1.0, "present", 1.0, "group", f"group={meta.get('group')}"))
    if meta.get("pathologyKey") == "PD":
        features.append(GaitFeature("pathology_pd", 1.0, "present", 1.0, "pathologyKey", f"pathology={meta.get('pathology')}"))
    if vga >= 1:
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
    if meta.get("evaluationScoreName") and is_float(score_value):
        features.append(
            GaitFeature(
                "clinical_score_updrs_iii",
                float(score_value),
                "present",
                0.0,
                "evaluationScoreValue",
                f"{meta.get('evaluationScoreName')}={score_value}",
            )
        )
    return GaitCase(
        case_id=f"{meta.get('subject')}_{meta.get('trial')}",
        measurements={
            "age": float(meta.get("age") or 0.0),
            "BMI": float(meta.get("BMI") or 0.0),
            "visualGaitAssessment": vga,
            "leftGaitEvents": float(len(left_events)),
            "rightGaitEvents": float(len(right_events)),
        },
        abnormal_features=features,
        patient_metadata={"dataset_name": "Clinical Gait Signals", **{k: v for k, v in meta.items() if k not in {"leftGaitEvents", "rightGaitEvents"}}},
        asset_counts={},
    )


def read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["ID"]: row for row in csv.DictReader(handle)}


def is_float(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def load_contexts(path: Path) -> list[TextContext]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        TextContext(
            context_id=item["chunk_id"],
            text=item["text"],
            citation=item["citation"],
            supports=list(item["supports"]),
        )
        for item in data
    ]


def load_tree(path: Path) -> list[TreeNode]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [tree_node(item) for item in data]


def tree_node(item: dict[str, Any]) -> TreeNode:
    return TreeNode(
        node_id=item["node_id"],
        title=item["title"],
        summary=item.get("summary", ""),
        text=item.get("text", ""),
        citation=f"{item.get('source_id', 'unknown')} ({item.get('locator', 'unknown locator')})",
        supports=list(item.get("supports", [])),
        children=tuple(tree_node(child) for child in item.get("children", [])),
    )


def flatten_tree(nodes: list[TreeNode] | tuple[TreeNode, ...]) -> list[TreeNode]:
    flat: list[TreeNode] = []
    for node in nodes:
        flat.append(node)
        flat.extend(flatten_tree(node.children))
    return flat


def retrieve_embedding_contexts(features: list[str], corpus: list[TextContext], limit: int) -> list[TextContext]:
    def score(item: TextContext) -> tuple[int, int]:
        matched = len(set(features) & set(item.supports))
        token_score = sum(1 for feature in features if feature.replace("_", " ") in item.text.lower())
        return matched, token_score

    return [item for item in sorted(corpus, key=score, reverse=True) if score(item)[0] > 0][:limit]


def select_tree_nodes(model: str, case: GaitCase, tree: list[TreeNode]) -> dict[str, Any]:
    prompt = f"""
You are given one gait case and a document tree. Select the most relevant non-root nodes for answering the case.

Use exact feature-tag coverage. A node is relevant only if its feature_tags contain at least one case feature ID exactly.
Select enough non-root nodes to cover every case feature when exact matching nodes are available.
Do not select a node only because the title uses related words such as cadence, speed, stance, or swing.
For Data_2 biomechanical cases, prefer nodes whose summaries/text explicitly mention Data_2, event-derived timing, healthy-reference z-scores, approximate speed/step/stride features, or pathology-aware interpretation.

Case ID: {case.case_id}
Case evidence:
{format_case_evidence_for_prompt(case)}

Document tree without full text:
{json.dumps([tree_prompt(node) for node in tree], indent=2)}

Return only JSON:
{{
  "thinking": "brief reason for exact feature-tag node coverage",
  "node_list": ["node_id"]
}}
""".strip()
    return call_ollama_json(model, prompt)


def tree_prompt(node: TreeNode) -> dict[str, Any]:
    data: dict[str, Any] = {
        "source_id": node.node_id,
        "title": node.title,
        "summary": node.summary,
        "feature_tags": node.supports,
    }
    if node.children:
        data["children"] = [tree_prompt(child) for child in node.children]
    return data


def format_case_evidence_for_prompt(case: GaitCase) -> str:
    packet = case.patient_metadata.get("case_feature_packet")
    if packet:
        return str(packet)
    return json.dumps(
        [
            {"name": feature.name, "value": feature.value, "evidence": feature.evidence}
            for feature in case.abnormal_features
        ],
        indent=2,
    )


def answer_from_context(
    model: str,
    case: GaitCase,
    contexts: list[dict[str, Any]],
    context_key: str,
) -> dict[str, Any]:
    scope_note = pathology_scope_note(case)
    prompt = f"""
Answer based only on the case evidence and retrieved context.

Case ID: {case.case_id}
Case evidence:
{format_case_evidence_for_prompt(case)}

Retrieved context:
{json.dumps(contexts, indent=2)}

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
{scope_note}

Return only JSON:
{{
  "summary": "short paragraph",
  "claims": [
    {{
      "text": "claim grounded in the context",
      "features": ["feature_id"],
      "{context_key}": "supporting id",
      "citation": "citation string from context"
    }}
  ]
}}
""".strip()
    return call_ollama_json(model, prompt)


FEATURE_VALUE_LABELS = {
    "wearable_imu_trial": "trial",
    "u_turn_available": "U-turn",
    "gait_events_available": "gait events",
    "healthy_cohort": "cohort",
    "neurological_cohort": "cohort",
    "orthopedic_cohort": "cohort",
    "visual_gait_impairment": "visual gait assessment",
    "clinical_score_available": "clinical score",
    "reduced_approximate_walking_speed": "approximate walking speed",
    "reduced_approximate_step_length": "approximate step length",
    "reduced_approximate_stride_length": "approximate stride length",
    "reduced_cadence": "cadence",
    "increased_cadence": "cadence",
    "prolonged_stride_time": "stride time",
    "increased_stride_time_variability": "stride-time variability",
    "increased_stride_time_asymmetry": "stride-time asymmetry",
    "prolonged_swing_time": "swing time",
    "increased_swing_time_asymmetry": "swing-time asymmetry",
    "prolonged_stance_time": "stance time",
    "increased_stance_time_asymmetry": "stance-time asymmetry",
    "increased_double_support_percentage": "double-support percentage",
    "prolonged_uturn_duration": "U-turn duration",
    "gaitex_motion_capture_trial": "GAITEX trial provenance",
    "gaitex_speed_protocol": "GAITEX speed protocol",
    "gaitex_multi_speed_progression": "GAITEX speed progression",
    "gaitex_repeated_task_segments": "GAITEX repeated task segments",
    "prosthetic_ankle_reduced_push_off_moment": "prosthetic ankle push-off moment",
    "prosthetic_ankle_rom_asymmetry": "prosthetic ankle ROM asymmetry",
    "prosthetic_knee_reduced_rom": "prosthetic knee ROM",
    "prosthetic_hip_compensation_rom": "prosthetic hip ROM compensation",
    "prosthetic_knee_moment_asymmetry": "prosthetic knee moment asymmetry",
    "prosthetic_hip_moment_asymmetry": "prosthetic hip moment asymmetry",
}


def feature_value_label(feature_id: str) -> str:
    if feature_id.startswith("pathology_"):
        return "pathology"
    return FEATURE_VALUE_LABELS.get(feature_id, feature_id)


def claim_with_case_values(claim_text: str, feature_ids: list[str], case_features: dict[str, GaitFeature]) -> str:
    """Append exact case values so review claims cannot hide key measurements."""
    values = []
    lowered_claim = claim_text.lower()
    for feature_id in feature_ids:
        feature = case_features.get(feature_id)
        if feature is None:
            continue
        evidence = feature.evidence.strip()
        if not evidence:
            continue
        if evidence.lower() in lowered_claim:
            continue
        values.append(f"{feature_value_label(feature_id)}: {evidence}")
    if not values:
        return claim_text
    suffix = "; ".join(values)
    if claim_text.endswith("."):
        claim_text = claim_text[:-1]
    return f"{claim_text} (case values: {suffix})."


def build_explanation(
    pipeline: str,
    case: GaitCase,
    answer: dict[str, Any],
    evidence_contexts: dict[str, Evidence],
    context_supports: dict[str, set[str]],
    context_prefix: str,
    context_key: str,
) -> Explanation:
    case_features = {feature.name: feature for feature in case.abnormal_features}
    evidence: list[Evidence] = list(evidence_contexts.values())
    claims: list[Claim] = []
    mentioned: set[str] = set()
    unsupported: list[str] = []
    for raw_claim in answer.get("claims", []):
        if not isinstance(raw_claim, dict):
            continue
        claim_text = str(raw_claim.get("text", "")).strip()
        scope_violation = pathology_scope_violation(claim_text, case)
        if scope_violation:
            unsupported.append(f"{claim_text} [scope violation: {scope_violation}]")
            continue
        context_id = str(raw_claim.get(context_key, ""))
        supported_by_context = context_supports.get(context_id, set())
        feature_ids = [
            feature
            for feature in raw_claim.get("features", [])
            if feature in case_features and feature in supported_by_context
        ]
        if context_id not in evidence_contexts or not feature_ids:
            unsupported.append(claim_text)
            continue
        for feature_id in feature_ids:
            mentioned.add(feature_id)
            evidence.append(
                Evidence(
                    evidence_id=f"case:{feature_id}",
                    kind="case_feature",
                    text=case_features[feature_id].evidence,
                )
            )
        enriched_claim_text = claim_with_case_values(claim_text, feature_ids, case_features)
        claims.append(
            Claim(
                text=enriched_claim_text,
                feature=feature_ids[0],
                evidence_ids=[*(f"case:{feature_id}" for feature_id in feature_ids), f"{context_prefix}:{context_id}"],
                source_citations=[str(raw_claim.get("citation", ""))],
            )
        )
    return Explanation(
        pipeline=pipeline,  # type: ignore[arg-type]
        case_id=case.case_id,
        mentioned_features=sorted(mentioned),
        claims=claims,
        supporting_evidence=dedupe_evidence(evidence),
        unsupported_claims=[item for item in unsupported if item],
        summary=str(answer.get("summary", "")),
    )


def case_scope_features(case: GaitCase) -> set[str]:
    features = {feature.name for feature in case.abnormal_features}
    pathology_key = str(case.patient_metadata.get("pathologyKey") or "")
    mapped = PATHOLOGY_KEY_TO_FEATURE.get(pathology_key)
    if mapped:
        features.add(mapped)
    if str(case.patient_metadata.get("dataset_name") or "") == "Above-Knee Amputee Gait":
        features.add("pathology_above_knee_amputation")
    return features


def pathology_scope_note(case: GaitCase) -> str:
    features = case_scope_features(case)
    dataset_pathologies = sorted(features & DATASET_PATHOLOGY_FEATURES)
    if dataset_pathologies:
        readable = ", ".join(feature.replace("pathology_", "").upper() for feature in dataset_pathologies)
        return (
            "Pathology scope guard: diagnostic/pathology labels are limited to the case evidence and retrieved context. "
            f"For this case, the allowed dataset pathology label(s) are: {readable}. "
            "Do not mention other diseases, syndromes, or pathology labels, even if they are known from general gait literature."
        )
    return (
        "Pathology scope guard: this case has no disease/pathology feature in the case evidence. "
        "Describe only measured gait patterns and do not introduce disease, syndrome, or diagnostic labels."
    )


def pathology_scope_violation(claim_text: str, case: GaitCase) -> str | None:
    features = case_scope_features(case)
    for pattern, allowed_features, label in PATHOLOGY_TERM_SCOPES:
        if not pattern.search(claim_text):
            continue
        if allowed_features and features & allowed_features:
            continue
        return label
    return None


def dedupe_evidence(items: list[Evidence]) -> list[Evidence]:
    deduped: dict[str, Evidence] = {}
    for item in items:
        deduped[item.evidence_id] = item
    return list(deduped.values())


def call_ollama_json(model: str, prompt: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = json.loads(response.read().decode("utf-8"))["response"]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from {model}, got {type(parsed)!r}")
    return parsed


if __name__ == "__main__":
    main()
