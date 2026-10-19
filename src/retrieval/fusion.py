"""
Reciprocal Rank Fusion over an arbitrary number of retrieval streams.

Why rank-based fusion at all
----------------------------
The three modalities produce scores on incompatible scales: BM25 is unbounded
and corpus-relative, cosine similarity sits in ``[-1, 1]``, hop-decay proximity
in ``(0, 1]``. Normalising them onto a common scale would require a calibration
set per modality and would still drift as the corpus changes. RRF sidesteps the
problem entirely by discarding magnitudes and consuming only position:

    score(record) = sum over streams of  weight(stream) / (k + rank)

The ``k`` constant flattens the head of the curve. With ``k = 60``, the gap
between rank 1 and rank 2 is small, so a record that *several* modalities rank
respectably beats one that a single modality ranks first. That is the desired
behaviour for a governance-grade system: cross-modal agreement is the strongest
available evidence signal.

What changed
------------
The previous implementation hard-coded ``_SOURCE_ORDER = {"dense": 0,
"bm25": 1, "graph": 2}``, which capped the design at exactly three named
modalities and gave every stream equal, unconfigurable influence. Streams are
now self-describing :class:`RetrievalStream` values, so the fuser accepts any
number of them and weights each from configuration.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

from src.config.settings import FusionConfig
from src.retrieval.base import RetrievalResult, RetrievalStream

__all__ = ["ReciprocalRankFuser", "reciprocal_rank_fusion"]

logger = logging.getLogger(__name__)

#: Separator used to render the contributing stream names on a fused result.
SOURCE_SEPARATOR = "+"


class ReciprocalRankFuser:
    """
    Fuses any number of ranked streams into one ranked list.

    Stateless and reusable; a single instance serves every query.

    Usage::

        fuser = ReciprocalRankFuser(settings.fusion)
        fused = fuser.fuse([
            RetrievalStream.of("bm25", bm25_results),
            RetrievalStream.of("dense", dense_results),
            RetrievalStream.of("graph", graph_results),
        ])
    """

    __slots__ = ("_config",)

    def __init__(self, config: FusionConfig) -> None:
        self._config = config

    def fuse(
        self, streams: Sequence[RetrievalStream], *, top_k: int | None = None
    ) -> list[RetrievalResult]:
        """
        Combine ``streams`` into a single ranked list.

        Args:
            streams: one per modality, each already sorted best-first. Empty
                streams are harmless and contribute nothing.
            top_k: truncation applied after fusion; defaults to the configured
                ``max_candidates``.

        Returns:
            Results sorted by fused score, de-duplicated on ``record_id``. Each
            result's ``metadata`` records the rank it held in every contributing
            stream and that stream's weighted contribution, so any position in
            the output can be reconstructed and defended.
        """
        limit = top_k if top_k is not None else self._config.max_candidates
        if limit < 1:
            return []

        totals: dict[str, float] = {}
        exemplars: dict[str, RetrievalResult] = {}
        contributions: dict[str, dict[str, int]] = {}
        weighted: dict[str, dict[str, float]] = {}
        # Stream registration order, used only to render `source` deterministically.
        order: dict[str, int] = {}

        for position, stream in enumerate(streams):
            order.setdefault(stream.name, position)
            # An explicit per-stream weight overrides configuration; otherwise
            # the configured weight (or the default) applies.
            weight = (
                stream.weight
                if stream.weight != 1.0
                else self._config.weight_for(stream.name)
            )
            if weight <= 0.0:
                logger.debug("Stream %r has zero weight; skipped", stream.name)
                continue

            for rank, result in enumerate(stream.results, start=1):
                increment = weight / (self._config.rrf_k + rank)
                record_id = result.record_id
                totals[record_id] = totals.get(record_id, 0.0) + increment
                contributions.setdefault(record_id, {})[stream.name] = rank
                weighted.setdefault(record_id, {})[stream.name] = increment
                # Keep the first exemplar seen: all modalities serialise a given
                # record identically, so text and type already agree.
                exemplars.setdefault(record_id, result)

        if not totals:
            return []

        ranked_ids = sorted(totals, key=lambda rid: (-totals[rid], rid))

        fused: list[RetrievalResult] = []
        for record_id in ranked_ids[:limit]:
            exemplar = exemplars[record_id]
            stream_names = sorted(
                contributions[record_id], key=lambda name: order.get(name, len(order))
            )
            fused.append(
                exemplar.with_score(
                    totals[record_id],
                    source=SOURCE_SEPARATOR.join(stream_names),
                    metadata={
                        "rrf_score": totals[record_id],
                        "rrf_k": self._config.rrf_k,
                        "stream_ranks": dict(contributions[record_id]),
                        "stream_contributions": dict(weighted[record_id]),
                        "stream_agreement": len(stream_names),
                    },
                )
            )

        logger.debug(
            "Fused %d stream(s) into %d candidates (%d unique records)",
            len(streams),
            len(fused),
            len(totals),
        )
        return fused


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievalResult]],
    config: FusionConfig | None = None,
    *,
    rrf_k: int | None = None,
) -> list[RetrievalResult]:
    """
    Functional wrapper over :class:`ReciprocalRankFuser`.

    Convenience for callers holding bare result lists rather than named
    streams; stream names are recovered from each result's ``source``. Prefer
    the class directly in the pipeline, where names and weights are explicit.
    """
    if config is None:
        from src.config.settings import get_settings

        config = get_settings().fusion
    if rrf_k is not None:
        from dataclasses import replace

        config = replace(config, rrf_k=rrf_k)

    streams = [
        RetrievalStream.of(_stream_name(results, position), results)
        for position, results in enumerate(result_lists)
    ]
    return ReciprocalRankFuser(config).fuse(streams)


def _stream_name(results: Sequence[RetrievalResult], position: int) -> str:
    """Infer a stream name from its results, falling back to its position."""
    return results[0].source if results else f"stream_{position}"
