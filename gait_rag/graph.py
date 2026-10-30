from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class GraphRelation:
    subject: str
    predicate: str
    object: str
    source: str
    source_id: str
    locator: str | None = None

    @property
    def path(self) -> str:
        return f"{self.subject} -> {self.predicate} -> {self.object}"

    @property
    def citation(self) -> str:
        if self.locator:
            return f"{self.source_id} ({self.locator})"
        return self.source_id


@dataclass(frozen=True)
class GraphPath:
    relations: tuple[GraphRelation, ...]

    @property
    def path(self) -> str:
        if not self.relations:
            return ""
        parts = [self.relations[0].subject]
        for relation in self.relations:
            parts.extend([relation.predicate, relation.object])
        return " -> ".join(parts)

    @property
    def leaf_relation(self) -> GraphRelation:
        return self.relations[-1]

    @property
    def citations(self) -> list[str]:
        seen: dict[str, None] = {}
        for relation in self.relations:
            seen[relation.citation] = None
        return list(seen.keys())


def load_graph(path: Path) -> list[GraphRelation]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        GraphRelation(
            subject=item["subject"],
            predicate=item["predicate"],
            object=item["object"],
            source=item.get("source", item.get("source_id", "manual_graph")),
            source_id=item.get("source_id", item.get("source", "manual_graph")),
            locator=item.get("locator"),
        )
        for item in data
    ]


def retrieve_graph_paths(features: list[str], graph: list[GraphRelation], max_depth: int = 2) -> dict[str, list[GraphPath]]:
    paths: dict[str, list[GraphPath]] = {}
    for feature in features:
        feature_paths: list[GraphPath] = []
        frontier: list[tuple[str, tuple[GraphRelation, ...]]] = [(feature, ())]
        for _depth in range(max_depth):
            next_frontier: list[tuple[str, tuple[GraphRelation, ...]]] = []
            for node, chain in frontier:
                for relation in graph:
                    if relation.subject != node:
                        continue
                    next_chain = (*chain, relation)
                    feature_paths.append(GraphPath(next_chain))
                    next_frontier.append((relation.object, next_chain))
            frontier = next_frontier
        if feature_paths:
            paths[feature] = feature_paths
    return paths
