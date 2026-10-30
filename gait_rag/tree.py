from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class DocumentNode:
    node_id: str
    title: str
    summary: str
    text: str
    source_id: str
    locator: str | None
    supports: list[str]
    claim_template: str | None
    children: tuple["DocumentNode", ...]

    @property
    def citation(self) -> str:
        if self.locator:
            return f"{self.source_id} ({self.locator})"
        return self.source_id


@dataclass(frozen=True)
class RetrievedTreeNode:
    node: DocumentNode
    matched_features: list[str]


def load_document_tree(path: Path) -> list[DocumentNode]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [_node_from_dict(item) for item in data]


def retrieve_tree_nodes(
    features: list[str],
    tree: list[DocumentNode],
    limit: int = 5,
) -> list[RetrievedTreeNode]:
    """Retrieve structured document nodes without embeddings.

    This is a local deterministic stand-in for reasoning over a PageIndex-style
    tree: features are matched to explicit node metadata, then the most specific
    nodes with the broadest case coverage are selected.
    """
    feature_order = {feature: index for index, feature in enumerate(features)}
    candidates: list[RetrievedTreeNode] = []
    for node in _iter_nodes(tree):
        supported = set(node.supports)
        matched = [feature for feature in features if feature in supported]
        if not matched or not node.text:
            continue
        candidates.append(RetrievedTreeNode(node=node, matched_features=matched))
    return sorted(
        candidates,
        key=lambda item: (
            -len(item.matched_features),
            min(feature_order[feature] for feature in item.matched_features),
            item.node.node_id,
        ),
    )[:limit]


def _node_from_dict(item: dict[str, object]) -> DocumentNode:
    return DocumentNode(
        node_id=str(item["node_id"]),
        title=str(item["title"]),
        summary=str(item.get("summary", "")),
        text=str(item.get("text", "")),
        source_id=str(item.get("source_id", "unknown_source")),
        locator=str(item["locator"]) if item.get("locator") else None,
        supports=[str(feature) for feature in item.get("supports", [])],
        claim_template=str(item["claim_template"]) if item.get("claim_template") else None,
        children=tuple(_node_from_dict(child) for child in item.get("children", [])),
    )


def _iter_nodes(nodes: list[DocumentNode] | tuple[DocumentNode, ...]) -> Iterator[DocumentNode]:
    for node in nodes:
        yield node
        yield from _iter_nodes(node.children)
