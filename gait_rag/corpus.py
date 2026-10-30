from __future__ import annotations

from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
import re


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    text: str
    supports: list[str]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: CorpusChunk
    score: float


def load_corpus(path: Path) -> list[CorpusChunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        CorpusChunk(
            chunk_id=item["chunk_id"],
            text=item["text"],
            supports=list(item.get("supports", [])),
        )
        for item in data
    ]


def retrieve_chunks(query_terms: list[str], corpus: list[CorpusChunk], limit: int = 4) -> list[RetrievedChunk]:
    query_vector = _term_counts(" ".join(query_terms))
    scored: list[RetrievedChunk] = []
    for chunk in corpus:
        chunk_vector = _term_counts(" ".join([chunk.text, *chunk.supports]))
        score = _cosine(query_vector, chunk_vector)
        if score > 0:
            scored.append(RetrievedChunk(chunk=chunk, score=score))
    return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]


def _term_counts(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in TOKEN_RE.findall(text.lower().replace("_", " ")):
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(weight * right.get(term, 0.0) for term, weight in left.items())
    left_norm = sqrt(sum(weight * weight for weight in left.values()))
    right_norm = sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)

