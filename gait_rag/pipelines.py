from __future__ import annotations

from dataclasses import dataclass

from .corpus import CorpusChunk, retrieve_chunks
from .data import GaitCase
from .explain import Explanation, explain_with_chunks, explain_with_graph, explain_with_tree
from .graph import GraphRelation, retrieve_graph_paths
from .metrics import EvaluationResult, evaluate_explanation
from .tree import DocumentNode, retrieve_tree_nodes


@dataclass(frozen=True)
class PipelineRun:
    explanation: Explanation
    evaluation: EvaluationResult


def run_embedding_rag(case: GaitCase, corpus: list[CorpusChunk], graph: list[GraphRelation]) -> PipelineRun:
    query_terms = [feature.name for feature in case.abnormal_features]
    retrieved = retrieve_chunks(query_terms, corpus)
    explanation = explain_with_chunks(case, retrieved)
    return PipelineRun(explanation=explanation, evaluation=evaluate_explanation(case, explanation, graph))


def run_graph_guided_rag(case: GaitCase, graph: list[GraphRelation]) -> PipelineRun:
    features = [feature.name for feature in case.abnormal_features]
    paths = retrieve_graph_paths(features, graph)
    explanation = explain_with_graph(case, paths)
    return PipelineRun(explanation=explanation, evaluation=evaluate_explanation(case, explanation, graph))


def run_vectorless_tree_rag(
    case: GaitCase,
    document_tree: list[DocumentNode],
    graph: list[GraphRelation],
) -> PipelineRun:
    features = [feature.name for feature in case.abnormal_features]
    nodes = retrieve_tree_nodes(features, document_tree)
    explanation = explain_with_tree(case, nodes)
    return PipelineRun(explanation=explanation, evaluation=evaluate_explanation(case, explanation, graph))
