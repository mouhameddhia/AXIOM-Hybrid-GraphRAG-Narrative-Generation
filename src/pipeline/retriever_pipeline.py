"""
The hybrid retrieval pipeline.

              question
                 |
         AnchorResolver          (grounds identifier mentions in the graph)
                 |
    +------------+------------+
    |            |            |
  BM25        Dense         Graph          ... and any further modality
    |            |            |
    +------------+------------+
                 |
       ReciprocalRankFuser               (rank-based, weighted, k-smoothed)
                 |
      CrossEncoderReranker               (optional, thresholded)
                 |
          ranked evidence + trace

The pipeline holds a ``Sequence[Retriever]`` and knows nothing about which
modalities are in it. Adding a fourth stream means appending one object and
optionally setting a weight -- no change here, in fusion, or in reranking.

Every stage's output is preserved in :class:`RetrievalOutcome`. For a
governance-grade system the intermediate ranks are not debug output; they are
the record of how a piece of evidence came to be cited.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from src.config.settings import RetrievalConfig
from src.retrieval.anchors import AnchorResolver
from src.retrieval.base import (
    Anchor,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStream,
    Retriever,
)
from src.retrieval.fusion import ReciprocalRankFuser
from src.retrieval.reranker import CrossEncoderReranker

__all__ = ["HybridRetrievalPipeline", "RetrievalOutcome"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    """
    The full, inspectable result of one retrieval request.

    Carries each stage's ranked output rather than only the final list, so an
    evaluation harness can measure recall at the fusion boundary separately
    from precision after reranking, and a reviewer can see which modality
    actually surfaced a cited record.
    """

    query: RetrievalQuery
    streams: tuple[RetrievalStream, ...]
    fused: tuple[RetrievalResult, ...]
    reranked: tuple[RetrievalResult, ...]
    timings_ms: Mapping[str, float] = field(default_factory=dict)

    @property
    def final(self) -> tuple[RetrievalResult, ...]:
        """Reranked results when reranking ran, otherwise the fused list."""
        return self.reranked or self.fused

    @property
    def anchors(self) -> tuple[Anchor, ...]:
        """Anchors that grounded the graph modality for this query."""
        return self.query.anchors

    def stream(self, name: str) -> RetrievalStream | None:
        """Return one modality's raw output by name."""
        return next((s for s in self.streams if s.name == name), None)

    @property
    def record_ids(self) -> tuple[str, ...]:
        """Citation handles of the final ranked evidence, best-first."""
        return tuple(result.record_id for result in self.final)


class HybridRetrievalPipeline:
    """
    Orchestrates anchor resolution, N retrieval modalities, fusion, reranking.

    Usage::

        pipeline = HybridRetrievalPipeline(
            retrievers=[bm25, dense, graph],
            fuser=ReciprocalRankFuser(settings.fusion),
            config=settings.retrieval,
            anchor_resolver=resolver,
            reranker=reranker,
        )
        outcome = pipeline.run("What is the list price of SKU0001?")
    """

    __slots__ = ("_retrievers", "_fuser", "_reranker", "_resolver", "_config")

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        fuser: ReciprocalRankFuser,
        config: RetrievalConfig,
        *,
        anchor_resolver: AnchorResolver | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("At least one retriever is required")
        names = [retriever.name for retriever in retrievers]
        if len(set(names)) != len(names):
            raise ValueError(f"Retriever names must be unique, got {names}")

        self._retrievers = tuple(retrievers)
        self._fuser = fuser
        self._config = config
        self._resolver = anchor_resolver
        self._reranker = reranker

    @property
    def stream_names(self) -> tuple[str, ...]:
        """Names of the registered modalities, in execution order."""
        return tuple(retriever.name for retriever in self._retrievers)

    def run(self, question: str, *, top_k: int | None = None) -> RetrievalOutcome:
        """
        Execute the full pipeline for one question.

        Args:
            question: the natural-language query.
            top_k: size of the final ranked list; defaults to
                ``DEFAULT_TOP_K``.

        Returns:
            A :class:`RetrievalOutcome` holding every stage's output.

        Notes:
            A modality that raises is logged and contributes an empty stream
            rather than failing the request. Partial evidence from two working
            modalities is materially more useful than an error, and the
            degradation is recorded in ``timings_ms`` and the logs rather than
            being hidden.
        """
        top_k = top_k if top_k is not None else self._config.default_top_k
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        timings: dict[str, float] = {}
        started = time.perf_counter()

        anchors = self._resolve_anchors(question, timings)
        query = RetrievalQuery(text=question, anchors=anchors, top_k=top_k)

        # Each modality retrieves deeper than the final answer needs, giving
        # fusion room to promote records that several streams agree on but none
        # ranked inside its own top-k.
        fan_out = self._config.candidate_top_k(top_k)
        streams = self._run_retrievers(query, fan_out, timings)

        fuse_started = time.perf_counter()
        fused = self._fuser.fuse(streams)
        timings["fusion"] = _elapsed_ms(fuse_started)

        reranked = self._run_reranker(question, fused, top_k, timings)
        if not reranked:
            fused = tuple(fused[:top_k])
        else:
            fused = tuple(fused)

        timings["total"] = _elapsed_ms(started)
        logger.info(
            "Retrieved %d evidence records for %r in %.1f ms (%s)",
            len(reranked or fused),
            question,
            timings["total"],
            ", ".join(f"{name}={len(stream)}" for name, stream in
                      ((s.name, s) for s in streams)),
        )
        return RetrievalOutcome(
            query=query,
            streams=tuple(streams),
            fused=fused,
            reranked=tuple(reranked),
            timings_ms=MappingProxyType(timings),
        )

    # -- stages -------------------------------------------------------------

    def _resolve_anchors(
        self, question: str, timings: dict[str, float]
    ) -> tuple[Anchor, ...]:
        """Ground identifier mentions, tolerating resolver failure."""
        if self._resolver is None:
            return ()
        started = time.perf_counter()
        try:
            anchors = self._resolver.resolve(question)
        except Exception as exc:  # noqa: BLE001 - never fail retrieval on this
            logger.error("Anchor resolution failed: %s", exc)
            anchors = ()
        timings["anchors"] = _elapsed_ms(started)
        if anchors:
            logger.debug(
                "Resolved anchors: %s",
                ", ".join(f"{a.mention}->{a.record_id}" for a in anchors),
            )
        return anchors

    def _run_retrievers(
        self, query: RetrievalQuery, fan_out: int, timings: dict[str, float]
    ) -> list[RetrievalStream]:
        """Run every modality, isolating failures to the stream that caused them."""
        streams: list[RetrievalStream] = []
        for retriever in self._retrievers:
            started = time.perf_counter()
            try:
                results = retriever.search(query, fan_out)
            except Exception as exc:  # noqa: BLE001 - degrade, do not fail
                logger.error("Retriever %r failed: %s", retriever.name, exc)
                results = []
            timings[retriever.name] = _elapsed_ms(started)
            streams.append(RetrievalStream.of(retriever.name, results))
        return streams

    def _run_reranker(
        self,
        question: str,
        fused: Sequence[RetrievalResult],
        top_k: int,
        timings: dict[str, float],
    ) -> list[RetrievalResult]:
        """Rerank the fused head, or return ``[]`` when reranking is disabled."""
        if self._reranker is None or not fused:
            return []
        started = time.perf_counter()
        reranked = self._reranker.rerank(question, fused, top_k=top_k)
        timings["rerank"] = _elapsed_ms(started)
        return reranked


def _elapsed_ms(started: float) -> float:
    """Milliseconds elapsed since ``started``, rounded for readable traces."""
    return round((time.perf_counter() - started) * 1000.0, 2)
