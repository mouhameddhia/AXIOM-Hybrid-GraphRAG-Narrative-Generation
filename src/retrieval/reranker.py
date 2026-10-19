"""
Cross-encoder reranking.

RRF is cheap but blind: it knows where a record landed in each stream, never
how relevant it actually is. A cross-encoder reads the query and the candidate
*jointly* through a transformer, producing a genuine relevance judgement -- far
more accurate than any single retriever, and far too slow to run over a whole
corpus. Hence the standard shape: retrieve broadly, fuse, then rerank only the
fused head.

Score calibration
-----------------
``ms-marco-MiniLM`` emits an unbounded logit, roughly ``-11..+11``. The previous
implementation surfaced that raw value, which made any ``min_score`` threshold
meaningless -- there is no stable cut-point on an uncalibrated logit scale.
Applying a sigmoid maps it to ``(0, 1)``, where a threshold is a defensible,
auditable relevance floor. That floor is also what lets the system return
*nothing*, which is the retrieval-side precondition for an honest abstention.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from src.config.settings import RerankerConfig
from src.retrieval.base import RetrievalResult

__all__ = ["CrossEncoderModel", "CrossEncoderReranker", "SentenceTransformerCrossEncoder"]

logger = logging.getLogger(__name__)


@runtime_checkable
class CrossEncoderModel(Protocol):
    """Minimal contract: score a batch of (query, document) pairs."""

    def predict_pairs(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        """Return one relevance score per pair, in order."""


class SentenceTransformerCrossEncoder:
    """
    Production cross-encoder backed by ``sentence-transformers``.

    Loads lazily so that constructing a pipeline never triggers a model
    download; the cost is paid on the first rerank, or explicitly via
    :meth:`warmup`.
    """

    __slots__ = ("_config", "_model")

    def __init__(self, config: RerankerConfig) -> None:
        self._config = config
        self._model = None

    def warmup(self) -> None:
        """Load the model ahead of the first request."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder %s", self._config.model_name)
        self._model = CrossEncoder(self._config.model_name)

    def predict_pairs(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        self._ensure_loaded()
        scores = self._model.predict(
            pairs, batch_size=self._config.batch_size, show_progress_bar=False
        )
        return [float(score) for score in scores]


class CrossEncoderReranker:
    """
    Re-scores fused candidates by joint query-document relevance.

    Usage::

        reranker = CrossEncoderReranker(settings.reranker, model)
        final = reranker.rerank(query_text, fused, top_k=10)
    """

    __slots__ = ("_config", "_model", "_transform")

    def __init__(
        self,
        config: RerankerConfig,
        model: CrossEncoderModel,
    ) -> None:
        self._config = config
        self._model = model
        self._transform: Callable[[float], float] = (
            _sigmoid if config.score_transform == RerankerConfig.SIGMOID else _identity
        )

    def rerank(
        self,
        query_text: str,
        results: Sequence[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Re-score and re-order ``results`` against ``query_text``.

        Args:
            query_text: the original natural-language question.
            results: fused candidates. Only the configured ``top_n`` head is
                scored; the remainder is discarded rather than appended, since
                a candidate the cross-encoder never saw has no comparable score.
            top_k: how many results to return; defaults to everything scored.

        Returns:
            Results sorted by calibrated relevance, best-first, with scores
            below ``min_score`` removed -- except that ``min_results`` entries
            are always retained, so a threshold set too high degrades ranking
            quality rather than silently emptying the answer.

        Raises:
            Nothing when ``fail_open`` is set: a model failure logs and returns
            the fused order untouched. Retrieval has already succeeded at that
            point, and serving well-ordered-but-unreranked evidence is strictly
            better than failing the request.
        """
        if not results:
            return []

        candidates = list(results[: self._config.top_n])
        pairs = [(query_text, candidate.text) for candidate in candidates]

        try:
            raw_scores = self._model.predict_pairs(pairs)
        except Exception as exc:  # noqa: BLE001 - model backends raise broadly
            if not self._config.fail_open:
                raise
            logger.error(
                "Cross-encoder failed (%s); falling back to fused order", exc
            )
            return self._fallback(candidates, top_k, reason=str(exc))

        if len(raw_scores) != len(candidates):
            message = (
                f"Cross-encoder returned {len(raw_scores)} scores "
                f"for {len(candidates)} candidates"
            )
            if not self._config.fail_open:
                raise ValueError(message)
            logger.error("%s; falling back to fused order", message)
            return self._fallback(candidates, top_k, reason=message)

        scored = [
            (
                self._transform(float(raw)),
                float(raw),
                position,
                candidate,
            )
            for position, (candidate, raw) in enumerate(zip(candidates, raw_scores))
        ]
        # Best first; ties fall back to the fused position, so reranking never
        # reorders candidates it considers equally relevant.
        scored.sort(key=lambda entry: (-entry[0], entry[2]))

        kept = self._apply_threshold(scored)
        limit = top_k if top_k is not None else len(kept)

        return [
            candidate.with_score(
                calibrated,
                source=f"{candidate.source}|reranked",
                metadata={
                    "rerank_score": calibrated,
                    "rerank_raw_score": raw,
                    "rerank_transform": self._config.score_transform,
                    "rerank_model": self._config.model_name,
                    # Position before reranking: makes the stage's effect on
                    # any individual record directly measurable.
                    "pre_rerank_position": position + 1,
                    "fused_score": candidate.score,
                },
            )
            for calibrated, raw, position, candidate in kept[:limit]
        ]

    def _apply_threshold(
        self, scored: Sequence[tuple[float, float, int, RetrievalResult]]
    ) -> list[tuple[float, float, int, RetrievalResult]]:
        """Drop sub-threshold candidates while honouring ``min_results``."""
        above = [entry for entry in scored if entry[0] >= self._config.min_score]
        if len(above) >= self._config.min_results:
            if len(above) < len(scored):
                logger.debug(
                    "Rerank threshold %.3f dropped %d of %d candidates",
                    self._config.min_score,
                    len(scored) - len(above),
                    len(scored),
                )
            return above
        # Everything scored below the floor. Retain the configured minimum so
        # the caller can still decide; with min_results=0 this returns [] and
        # the generation layer abstains.
        logger.info(
            "All %d candidates scored below the rerank floor %.3f; "
            "retaining %d for the caller to judge",
            len(scored),
            self._config.min_score,
            self._config.min_results,
        )
        return list(scored[: self._config.min_results])

    def _fallback(
        self,
        candidates: Sequence[RetrievalResult],
        top_k: int | None,
        *,
        reason: str,
    ) -> list[RetrievalResult]:
        """Return fused order, flagged so the trace records the degradation."""
        limit = top_k if top_k is not None else len(candidates)
        return [
            candidate.with_score(
                candidate.score,
                metadata={"rerank_skipped": True, "rerank_error": reason},
            )
            for candidate in candidates[:limit]
        ]


def _sigmoid(value: float) -> float:
    """Numerically stable logistic transform onto ``(0, 1)``."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _identity(value: float) -> float:
    """Pass the raw model score through unchanged."""
    return value
