from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import load_corpus
from .data import load_dataset_cases, load_gait_cases
from .graph import load_graph
from .pipelines import run_embedding_rag, run_graph_guided_rag, run_vectorless_tree_rag
from .tree import load_document_tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gait RAG prototype pipelines.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset_samples"))
    parser.add_argument("--cases", type=Path, default=None, help="Optional measured gait-parameters CSV override.")
    parser.add_argument("--corpus", type=Path, default=Path("resources/literature_corpus.json"))
    parser.add_argument("--graph", type=Path, default=Path("resources/biomechanical_graph.json"))
    parser.add_argument("--doc-tree", type=Path, default=Path("resources/document_tree.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/prototype_results.json"))
    args = parser.parse_args()

    cases = load_gait_cases(args.cases) if args.cases is not None else load_dataset_cases(args.dataset_root)
    corpus = load_corpus(args.corpus)
    graph = load_graph(args.graph)
    document_tree = load_document_tree(args.doc_tree)

    results = []
    for case in cases:
        embedding_run = run_embedding_rag(case, corpus, graph)
        graph_run = run_graph_guided_rag(case, graph)
        tree_run = run_vectorless_tree_rag(case, document_tree, graph)
        results.append(
            {
                "case_id": case.case_id,
                "dataset_sources": {
                    "measured_gait_parameters": str(args.cases or args.dataset_root / "gait_parameters.csv"),
                    "estimated_gait_parameters": str(args.dataset_root / "gait_parameters_estimation.csv"),
                    "patient_measures": str(args.dataset_root / "patients_measures.csv"),
                    "raw_asset_root": str(args.dataset_root),
                },
                "vectorless_sources": {
                    "source_registry": "resources/vectorless_source_registry.json",
                    "biomechanical_graph": str(args.graph),
                    "document_tree": str(args.doc_tree),
                    "primary_book": "gait_analysis_reviews/Jacquelin_Perry_MD_Gait_Analysis_Normal.pdf",
                },
                "measurements": case.measurements,
                "estimated_measurements": case.estimated_measurements,
                "patient_metadata": case.patient_metadata,
                "dataset_assets": case.asset_counts,
                "abnormal_features": [
                    {
                        "name": feature.name,
                        "value": feature.value,
                        "source_column": feature.source_column,
                        "evidence": feature.evidence,
                    }
                    for feature in case.abnormal_features
                ],
                "embedding_rag": {
                    "explanation": embedding_run.explanation.to_dict(),
                    "evaluation": embedding_run.evaluation.to_dict(),
                },
                "graph_guided_rag": {
                    "explanation": graph_run.explanation.to_dict(),
                    "evaluation": graph_run.evaluation.to_dict(),
                },
                "vectorless_tree_rag": {
                    "document_tree": str(args.doc_tree),
                    "explanation": tree_run.explanation.to_dict(),
                    "evaluation": tree_run.evaluation.to_dict(),
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} for {len(results)} case(s).")


if __name__ == "__main__":
    main()
