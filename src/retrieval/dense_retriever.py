"""
Dense (semantic) retrieval modality.

Catches evidence that shares meaning but not vocabulary with the query: the
question asks about "price reduction", the record says ``invoice_discount``.
BM25 scores that pair at zero.

The encoder is injected behind :class:`TextEncoder`, not constructed inline.
That keeps a 90 MB transformer download out of the unit-test path while leaving
the production path identical -- the tests exercise the same ranking, caching,
and truncation logic the real encoder runs through.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

from src.config.settings import EmbeddingConfig
from src.retrieval.base import RetrievalQuery, RetrievalResult
from src.retrieval.corpus import Corpus

__all__ = ["DenseRetriever", "SentenceTransformerEncoder", "TextEncoder"]

logger = logging.getLogger(__name__)


@runtime_checkable
class TextEncoder(Protocol):
    """Minimal encoder contract: a batch of strings in, a 2-D float array out."""

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` array of embeddings."""


class SentenceTransformerEncoder:
    """
    Production encoder backed by ``sentence-transformers``.

    The model loads lazily on first encode, so importing the module -- or
    constructing a pipeline that is never queried -- costs nothing.
    """

    __slots__ = ("_config", "_model")

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Imported here rather than at module scope: the import alone pulls in
        # torch, which is expensive and unnecessary for the JSON-only path.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s", self._config.model_name)
        self._model = SentenceTransformer(
            self._config.model_name, device=self._config.device
        )

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        return self._model.encode(
            texts,
            batch_size=self._config.batch_size,
            normalize_embeddings=self._config.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


class DenseRetriever:
    """
    Embedding retrieval over the corpus, backed by an in-memory matrix.

    An exhaustive matrix product is the correct choice at AXIOM's scale: a few
    thousand records times 384 dimensions is well under a millisecond, and it
    is exact, whereas an ANN index trades recall for a speedup that is not yet
    needed. The interface is unchanged by swapping in FAISS or a Neo4j vector
    index later; only :meth:`_similarities` would change.
    """

    __slots__ = ("_corpus", "_encoder", "_config", "_matrix", "_name")

    def __init__(
        self,
        corpus: Corpus,
        config: EmbeddingConfig,
        encoder: TextEncoder,
        *,
        name: str = "dense",
    ) -> None:
        self._corpus = corpus
        self._config = config
        self._encoder = encoder
        self._name = name
        self._matrix: np.ndarray | None = None

    @property
    def name(self) -> str:
        return self._name

    def warmup(self) -> None:
        """Build the embedding matrix ahead of the first query."""
        self._ensure_indexed()

    def _ensure_indexed(self) -> None:
        """Embed the corpus once and cache the result."""
        if self._matrix is not None:
            return
        if len(self._corpus) == 0:
            self._matrix = np.zeros((0, self._config.embedding_dim), dtype=np.float32)
            logger.warning("Dense index built over an empty corpus")
            return

        texts = [record.text for record in self._corpus]
        matrix = np.asarray(self._encoder.encode_batch(texts), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError(
                f"Encoder returned shape {matrix.shape}, expected "
                f"({len(texts)}, dim)"
            )
        self._matrix = self._normalise(matrix)
        logger.info(
            "Dense index built: %d vectors of dimension %d",
            self._matrix.shape[0],
            self._matrix.shape[1],
        )

    def _normalise(self, matrix: np.ndarray) -> np.ndarray:
        """
        L2-normalise rows so the dot product equals cosine similarity.

        Applied even when the encoder claims to normalise already: the cost is
        negligible and it removes any dependence on that guarantee holding.
        """
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def _similarities(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query against every indexed record."""
        return self._matrix @ query_vector

    def search(self, query: RetrievalQuery, top_k: int) -> list[RetrievalResult]:
        """
        Return the ``top_k`` most semantically similar records, best-first.

        Non-positive similarities are dropped for the same reason BM25 drops
        zero scores: an orthogonal or opposed vector is not evidence, and
        listing it would let it accrue RRF weight by position alone.
        """
        if top_k < 1 or not query.text.strip():
            return []
        self._ensure_indexed()
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []

        encoded = np.asarray(self._encoder.encode_batch([query.text]), dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] < 1:
            raise ValueError(f"Encoder returned shape {encoded.shape} for a single query")
        query_vector = self._normalise(encoded)[0]

        similarities = self._similarities(query_vector)
        # argpartition finds the top-k without fully sorting the corpus; the
        # subsequent sort only orders those k. O(n + k log k) rather than
        # O(n log n), which matters once the corpus reaches thousands of rows.
        width = min(top_k, similarities.shape[0])
        candidates = np.argpartition(-similarities, width - 1)[:width]
        ordered = sorted(
            candidates.tolist(),
            key=lambda index: (-float(similarities[index]), self._corpus[index].record_id),
        )

        results: list[RetrievalResult] = []
        for index in ordered:
            score = float(similarities[index])
            if score <= 0.0:
                break
            record = self._corpus[index]
            results.append(
                RetrievalResult(
                    record_id=record.record_id,
                    record_type=record.record_type,
                    text=record.text,
                    score=score,
                    source=self._name,
                    metadata={"similarity": score},
                )
            )
        return results
