"""
Lexical (BM25) retrieval modality.

BM25 is the modality that carries exact identifiers. In pricing questions the
decisive token is almost always a literal -- ``SKU0001``, ``CUST0015``,
``OPP00003``, ``162.5`` -- and an embedding blurs precisely those tokens
together. Dense retrieval will happily rank ``SKU0002`` next to ``SKU0001``;
BM25 will not.

That makes tokenisation the load-bearing detail here. The previous
implementation used ``text.lower().split()``, which leaves punctuation attached
(``SKU0001,`` never matches ``SKU0001``) and cannot segment Arabic. The
configured Unicode token pattern fixes both.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Final

from rank_bm25 import BM25Okapi

from src.config.settings import LexicalConfig
from src.retrieval.base import RetrievalQuery, RetrievalResult
from src.retrieval.corpus import Corpus

__all__ = ["BM25Retriever", "Tokenizer"]

logger = logging.getLogger(__name__)

#: Arabic diacritics (tashkeel) and the tatweel elongation mark. Stripping them
#: makes an Arabic query match an undecorated Arabic document, which matters for
#: the bilingual requirement.
_ARABIC_DIACRITICS: Final[re.Pattern[str]] = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")


class Tokenizer:
    """
    Configurable, Unicode-aware tokenizer shared by indexing and querying.

    Using one instance for both sides guarantees the query is segmented exactly
    as the corpus was; a mismatch silently destroys recall.
    """

    __slots__ = ("_pattern", "_min_length", "_stopwords")

    def __init__(self, config: LexicalConfig) -> None:
        try:
            self._pattern = re.compile(config.token_pattern, re.UNICODE)
        except re.error as exc:
            raise ValueError(
                f"BM25_TOKEN_PATTERN is not a valid regular expression: {exc}"
            ) from exc
        self._min_length = config.min_token_length
        self._stopwords = config.stopwords

    def __call__(self, text: str) -> list[str]:
        """
        Normalise and split ``text`` into comparable tokens.

        Stopwords are dropped from documents and queries alike -- filtering only
        one side would leave the index and the query segmented differently,
        which silently destroys recall.
        """
        # NFKC folds Arabic presentation forms and full-width Latin to their
        # canonical codepoints, so visually identical text compares equal.
        normalised = unicodedata.normalize("NFKC", text)
        normalised = _ARABIC_DIACRITICS.sub("", normalised)
        return [
            token
            for token in (
                raw.lower() for raw in self._pattern.findall(normalised)
            )
            if len(token) >= self._min_length and token not in self._stopwords
        ]


class BM25Retriever:
    """
    Okapi BM25 over the corpus, with parameters supplied by configuration.

    Usage::

        retriever = BM25Retriever(corpus, settings.lexical)
        results = retriever.search(RetrievalQuery(text="list price SKU0001"), top_k=10)
    """

    __slots__ = ("_corpus", "_tokenizer", "_config", "_index", "_name")

    def __init__(
        self,
        corpus: Corpus,
        config: LexicalConfig,
        *,
        name: str = "bm25",
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._corpus = corpus
        self._config = config
        self._tokenizer = tokenizer or Tokenizer(config)
        self._name = name
        self._index = self._build_index()

    @property
    def name(self) -> str:
        return self._name

    def _build_index(self) -> BM25Okapi | None:
        """Tokenise the corpus and build the BM25 index once, at construction."""
        if len(self._corpus) == 0:
            logger.warning("BM25 index built over an empty corpus")
            return None
        tokenized = [self._tokenizer(record.text) for record in self._corpus]
        index = BM25Okapi(
            tokenized,
            k1=self._config.k1,
            b=self._config.b,
            epsilon=self._config.epsilon,
        )
        logger.debug("BM25 index built over %d documents", len(tokenized))
        return index

    def search(self, query: RetrievalQuery, top_k: int) -> list[RetrievalResult]:
        """
        Return the ``top_k`` highest-scoring records for ``query``, best-first.

        Zero-scoring documents are dropped. Passing them on would hand fusion a
        ranked position for a document BM25 considers entirely irrelevant,
        letting noise accumulate RRF weight purely by being listed.
        """
        if self._index is None or top_k < 1:
            return []

        tokens = self._tokenizer(query.text)
        if not tokens:
            logger.debug("Query %r produced no lexical tokens", query.text)
            return []

        scores = self._index.get_scores(tokens)
        # Rank by score descending, breaking ties on record_id for determinism.
        ordered = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], self._corpus[index].record_id),
        )

        results: list[RetrievalResult] = []
        for index in ordered[:top_k]:
            score = float(scores[index])
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
                    metadata={"matched_tokens": tuple(tokens)},
                )
            )
        return results
