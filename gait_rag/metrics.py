from __future__ import annotations

from dataclasses import dataclass

from .data import GaitCase
from .explain import Explanation
from .graph import GraphRelation


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    pipeline: str
    feature_coverage: float
    faithfulness: float
    hallucination_rate: float
    citation_accuracy: float
    graph_path_validity: float | None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "case_id": self.case_id,
            "pipeline": self.pipeline,
            "feature_coverage": self.feature_coverage,
            "faithfulness": self.faithfulness,
            "hallucination_rate": self.hallucination_rate,
            "citation_accuracy": self.citation_accuracy,
            "graph_path_validity": self.graph_path_validity,
        }


def evaluate_explanation(
    case: GaitCase,
    explanation: Explanation,
    graph: list[GraphRelation],
) -> EvaluationResult:
    case_features = {feature.name for feature in case.abnormal_features}
    mentioned = set(explanation.mentioned_features)
    supported_claims = [
        claim
        for claim in explanation.claims
        if claim.feature in case_features and all(_has_evidence(evidence_id, explanation) for evidence_id in claim.evidence_ids)
    ]
    total_claims = len(explanation.claims) + len(explanation.unsupported_claims)
    cited_claims = [
        claim
        for claim in explanation.claims
        if claim.evidence_ids and all(_has_evidence(evidence_id, explanation) for evidence_id in claim.evidence_ids)
    ]
    valid_paths = [
        claim
        for claim in explanation.claims
        if claim.graph_path is not None and _path_exists(claim.graph_path, graph)
    ]
    graph_path_claims = [claim for claim in explanation.claims if claim.graph_path is not None]
    return EvaluationResult(
        case_id=case.case_id,
        pipeline=explanation.pipeline,
        feature_coverage=_ratio(len(mentioned & case_features), len(case_features)),
        faithfulness=_ratio(len(supported_claims), total_claims),
        hallucination_rate=_ratio(len(explanation.unsupported_claims), total_claims),
        citation_accuracy=_ratio(len(cited_claims), len(explanation.claims)),
        graph_path_validity=(
            _ratio(len(valid_paths), len(graph_path_claims)) if explanation.pipeline == "graph_guided_rag" else None
        ),
    )


def _has_evidence(evidence_id: str, explanation: Explanation) -> bool:
    return any(evidence.evidence_id == evidence_id for evidence in explanation.supporting_evidence)


def _path_exists(path: str, graph: list[GraphRelation]) -> bool:
    parts = path.split(" -> ")
    if len(parts) < 3 or (len(parts) - 1) % 2 != 0:
        return False
    graph_edges = {(relation.subject, relation.predicate, relation.object) for relation in graph}
    for index in range(0, len(parts) - 2, 2):
        edge = (parts[index], parts[index + 1], parts[index + 2])
        if edge not in graph_edges:
            return False
    return True


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)
