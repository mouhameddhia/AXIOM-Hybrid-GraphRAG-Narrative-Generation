"""
Shared contracts for every retrieval modality.

The most important decision in this file is that *all* retrievers take the same
:class:`RetrievalQuery` and return the same :class:`RetrievalResult`. The
previous design gave the graph retriever a different signature (``start_id``
instead of query text), which forced the pipeline to special-case it and made
"fuse an arbitrary number of streams" impossible to honour. With one uniform
protocol the pipeline holds a list of retrievers it knows nothing about, and a
fourth modality can be added without touching fusion, reranking, or the
pipeline itself.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Anchor",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievalStream",
    "Retriever",
]

#: Shared read-only empty metadata: avoids allocating a dict per result.
_EMPTY_METADATA: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Anchor:
    """
    A query token resolved to a concrete node in the graph.

    Produced by :class:`src.retrieval.anchors.AnchorResolver` and consumed by
    the graph retriever. ``matched_property`` is carried so an explanation can
    state which identifier in the question grounded the traversal.
    """

    mention: str
    node_id: str
    record_id: str
    record_type: str
    matched_property: str
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """
    One retrieval request, in the form every modality can consume.

    Lexical and dense retrievers read ``text``; the graph retriever reads
    ``anchors``. A modality ignores the fields it does not need, which is what
    keeps the retriever protocol uniform.
    """

    text: str
    anchors: tuple[Anchor, ...] = ()
    top_k: int | None = None
    language: str | None = None

    @property
    def anchor_node_ids(self) -> tuple[str, ...]:
        """Node ids of all resolved anchors, in resolution order."""
        return tuple(anchor.node_id for anchor in self.anchors)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """
    A single piece of evidence returned by a retriever.

    ``score`` is retriever-local and deliberately *not* comparable across
    modalities: a BM25 score, a cosine similarity, and a hop-decay weight live
    on unrelated scales. Only rank order is meaningful downstream, which is
    precisely why fusion is rank-based.

    ``metadata`` carries the per-stage audit trail (contributing ranks, hop
    distance, traversal path, pre-rerank position). It is what turns a ranked
    list into something a reviewer can interrogate.
    """

    record_id: str
    record_type: str
    text: str
    score: float
    source: str
    metadata: Mapping[str, Any] = field(default=_EMPTY_METADATA)

    def with_score(
        self,
        score: float,
        *,
        source: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RetrievalResult":
        """
        Return a copy carrying a new score, optionally re-labelled.

        Used by fusion and reranking, which re-score existing evidence rather
        than creating new evidence. Copying instead of mutating keeps every
        stage's output independently inspectable in the trace.
        """
        merged: Mapping[str, Any]
        if metadata is None:
            merged = self.metadata
        else:
            merged = MappingProxyType({**self.metadata, **metadata})
        return RetrievalResult(
            record_id=self.record_id,
            record_type=self.record_type,
            text=self.text,
            score=score,
            source=source if source is not None else self.source,
            metadata=merged,
        )


@dataclass(frozen=True, slots=True)
class RetrievalStream:
    """
    One modality's ranked output, tagged for fusion.

    Binding the name and weight to the results -- rather than relying on
    positional order, as the previous ``list[list[...]]`` signature did -- is
    what lets the fuser accept an arbitrary number of streams and weight each
    one from configuration.
    """

    name: str
    results: tuple[RetrievalResult, ...]
    weight: float = 1.0

    @classmethod
    def of(
        cls, name: str, results: Sequence[RetrievalResult], weight: float = 1.0
    ) -> "RetrievalStream":
        """Build a stream from any sequence of results."""
        return cls(name=name, results=tuple(results), weight=weight)

    def __len__(self) -> int:
        return len(self.results)


@runtime_checkable
class Retriever(Protocol):
    """
    The uniform interface implemented by every modality.

    Implementations must be side-effect free and must return results already
    sorted best-first, since fusion consumes position as the relevance signal.
    """

    @property
    def name(self) -> str:
        """Stream name used for fusion weighting and trace labelling."""

    def search(self, query: RetrievalQuery, top_k: int) -> list[RetrievalResult]:
        """Return at most ``top_k`` results for ``query``, best-first."""
