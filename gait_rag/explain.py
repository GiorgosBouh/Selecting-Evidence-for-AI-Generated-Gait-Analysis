from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .corpus import RetrievedChunk
from .data import GaitCase
from .graph import GraphPath
from .tree import RetrievedTreeNode


PipelineName = Literal["embedding_rag", "graph_guided_rag", "vectorless_tree_rag"]


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    kind: str
    text: str


@dataclass(frozen=True)
class Claim:
    text: str
    feature: str
    evidence_ids: list[str]
    graph_path: str | None = None
    source_citations: list[str] | None = None


@dataclass(frozen=True)
class Explanation:
    pipeline: PipelineName
    case_id: str
    mentioned_features: list[str]
    claims: list[Claim]
    supporting_evidence: list[Evidence]
    unsupported_claims: list[str]
    summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "pipeline": self.pipeline,
            "case_id": self.case_id,
            "mentioned_features": self.mentioned_features,
            "claims": [
                {
                    "text": claim.text,
                    "feature": claim.feature,
                    "evidence_ids": claim.evidence_ids,
                    "graph_path": claim.graph_path,
                    "source_citations": claim.source_citations or [],
                }
                for claim in self.claims
            ],
            "supporting_evidence": [
                {
                    "evidence_id": evidence.evidence_id,
                    "kind": evidence.kind,
                    "text": evidence.text,
                }
                for evidence in self.supporting_evidence
            ],
            "unsupported_claims": self.unsupported_claims,
            "summary": self.summary,
        }


def explain_with_chunks(case: GaitCase, retrieved: list[RetrievedChunk]) -> Explanation:
    evidence: list[Evidence] = []
    claims: list[Claim] = []
    for feature in case.abnormal_features:
        matched_items = [item for item in retrieved if feature.name in item.chunk.supports]
        matched = [f"chunk:{item.chunk.chunk_id}" for item in matched_items]
        if not matched:
            continue
        for item in matched_items:
            evidence.append(
                Evidence(
                    evidence_id=f"chunk:{item.chunk.chunk_id}",
                    kind="text_chunk",
                    text=item.chunk.text,
                )
            )
        claims.append(
            Claim(
                text=_feature_claim(feature.name),
                feature=feature.name,
                evidence_ids=[f"case:{feature.name}", *matched],
            )
        )
        evidence.append(
            Evidence(
                evidence_id=f"case:{feature.name}",
                kind="case_feature",
                text=feature.evidence,
            )
        )
    mentioned = [claim.feature for claim in claims]
    summary = _summary(case.case_id, mentioned, "retrieved literature chunks")
    return Explanation(
        pipeline="embedding_rag",
        case_id=case.case_id,
        mentioned_features=mentioned,
        claims=claims,
        supporting_evidence=_dedupe_evidence(evidence),
        unsupported_claims=[],
        summary=summary,
    )


def explain_with_graph(case: GaitCase, paths_by_feature: dict[str, list[GraphPath]]) -> Explanation:
    evidence: list[Evidence] = []
    claims: list[Claim] = []
    for feature in case.abnormal_features:
        paths = paths_by_feature.get(feature.name, [])
        if not paths:
            continue
        evidence.append(
            Evidence(
                evidence_id=f"case:{feature.name}",
                kind="case_feature",
                text=feature.evidence,
            )
        )
        for index, path in enumerate(paths, start=1):
            evidence_id = f"graph:{feature.name}:{index}"
            evidence.append(
                Evidence(
                    evidence_id=evidence_id,
                    kind="graph_path",
                    text=f"{path.path} | sources: {'; '.join(path.citations)}",
                )
            )
            claims.append(
                Claim(
                    text=_graph_claim(feature.name, path),
                    feature=feature.name,
                    evidence_ids=[f"case:{feature.name}", evidence_id],
                    graph_path=path.path,
                    source_citations=path.citations,
                )
            )
    mentioned = sorted({claim.feature for claim in claims})
    summary = _summary(case.case_id, mentioned, "valid graph paths")
    return Explanation(
        pipeline="graph_guided_rag",
        case_id=case.case_id,
        mentioned_features=mentioned,
        claims=claims,
        supporting_evidence=_dedupe_evidence(evidence),
        unsupported_claims=[],
        summary=summary,
    )


def explain_with_tree(case: GaitCase, retrieved: list[RetrievedTreeNode]) -> Explanation:
    evidence: list[Evidence] = []
    claims: list[Claim] = []
    feature_by_name = {feature.name: feature for feature in case.abnormal_features}
    for item in retrieved:
        matched_features = [feature for feature in item.matched_features if feature in feature_by_name]
        if not matched_features:
            continue
        for feature_name in matched_features:
            feature = feature_by_name[feature_name]
            evidence.append(
                Evidence(
                    evidence_id=f"case:{feature.name}",
                    kind="case_feature",
                    text=feature.evidence,
                )
            )
        tree_evidence_id = f"tree:{item.node.node_id}"
        evidence.append(
            Evidence(
                evidence_id=tree_evidence_id,
                kind="document_tree_node",
                text=f"{item.node.title}: {item.node.text} | source: {item.node.citation}",
            )
        )
        claims.append(
            Claim(
                text=item.node.claim_template or _tree_claim(matched_features, item.node.title),
                feature=matched_features[0],
                evidence_ids=[*(f"case:{feature}" for feature in matched_features), tree_evidence_id],
                source_citations=[item.node.citation],
            )
        )
    mentioned = sorted({feature for item in retrieved for feature in item.matched_features if feature in feature_by_name})
    summary = _summary(case.case_id, mentioned, "reasoned document-tree sections")
    return Explanation(
        pipeline="vectorless_tree_rag",
        case_id=case.case_id,
        mentioned_features=mentioned,
        claims=claims,
        supporting_evidence=_dedupe_evidence(evidence),
        unsupported_claims=[],
        summary=summary,
    )


def _feature_claim(feature: str) -> str:
    label = feature.replace("_", " ")
    return f"The case includes {label}, and the retrieved context supports discussing this gait abnormality."


def _graph_claim(feature: str, path: GraphPath) -> str:
    label = feature.replace("_", " ")
    relation = path.leaf_relation
    target = relation.object.replace("_", " ")
    predicate = relation.predicate.replace("_", " ")
    return f"The case includes {label}; the graph path supports that it {predicate} {target}."


def _tree_claim(features: list[str], title: str) -> str:
    readable = ", ".join(feature.replace("_", " ") for feature in features)
    return f"The document section '{title}' supports interpreting these case findings together: {readable}."


def _summary(case_id: str, features: list[str], context_name: str) -> str:
    if not features:
        return f"{case_id} has no supported abnormal gait claims under the available {context_name}."
    readable = ", ".join(feature.replace("_", " ") for feature in features)
    return f"{case_id} explanation is grounded in {context_name} for: {readable}."


def _dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
    deduped: dict[str, Evidence] = {}
    for item in evidence:
        deduped[item.evidence_id] = item
    return list(deduped.values())
