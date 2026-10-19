"""
Resolution of query mentions to concrete graph nodes.

Graph traversal needs a starting point. The previous implementation obtained
one by regex-matching the query and then running a Cypher query with a
hand-written ``OR`` chain over ten named identifier properties -- which meant
every new identifier property silently failed to resolve until someone
remembered to extend the chain.

Here the index is built *from the graph*: every identifier-shaped and
name-shaped property value present in the data becomes a lookup key. A new
property, label, or identifier convention is picked up automatically, because
nothing in this module names one.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from types import MappingProxyType

from src.config.settings import CorpusConfig, GraphTraversalConfig
from src.graph.model import KnowledgeGraph
from src.retrieval.base import Anchor
from src.retrieval.corpus import Corpus, RecordSerializer

__all__ = ["AnchorResolver"]

logger = logging.getLogger(__name__)


class AnchorResolver:
    """
    Maps identifier-like mentions in a question to nodes in the graph.

    Resolution is exact and case-insensitive against real property values, so a
    mention either grounds in the data or is discarded. Fuzzy matching is
    deliberately omitted: a wrongly grounded anchor sends traversal into an
    unrelated neighbourhood and produces evidence that *looks* authoritative
    while being about the wrong customer -- the most damaging failure mode a
    governance-grade system can have.
    """

    __slots__ = ("_graph", "_corpus", "_config", "_pattern", "_index")

    def __init__(
        self,
        graph: KnowledgeGraph,
        corpus: Corpus,
        serializer: RecordSerializer,
        graph_config: GraphTraversalConfig,
        corpus_config: CorpusConfig,
    ) -> None:
        self._graph = graph
        self._corpus = corpus
        self._config = graph_config
        try:
            self._pattern = re.compile(graph_config.anchor_pattern, re.UNICODE)
        except re.error as exc:
            raise ValueError(
                f"GRAPH_ANCHOR_PATTERN is not a valid regular expression: {exc}"
            ) from exc
        self._index = self._build_index(corpus_config)

    def _build_index(
        self, corpus_config: CorpusConfig
    ) -> Mapping[str, tuple[tuple[str, str], ...]]:
        """
        Build ``normalised value -> ((node_id, property_name), ...)``.

        Only identifier-shaped and name-shaped properties are indexed. Indexing
        every value would let an incidental token such as ``direct`` or
        ``standard`` anchor a traversal, flooding the graph stream with records
        that merely share a categorical attribute.
        """
        suffix = corpus_config.id_key_suffix
        name_suffixes = corpus_config.name_key_suffixes
        excluded = corpus_config.excluded_property_keys

        index: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
        for node in self._graph:
            for key, value in node.properties.items():
                if key in excluded or value is None or value == "":
                    continue
                if not (
                    key.endswith(suffix) or any(key.endswith(s) for s in name_suffixes)
                ):
                    continue
                index[_normalise(str(value))].append((node.node_id, key))

        logger.debug("Anchor index built over %d distinct values", len(index))
        return MappingProxyType(
            {value: tuple(entries) for value, entries in index.items()}
        )

    def resolve(self, query_text: str) -> tuple[Anchor, ...]:
        """
        Extract identifier-like mentions from ``query_text`` and ground them.

        Returns:
            Anchors in order of first mention, de-duplicated by node, capped at
            ``GRAPH_MAX_ANCHORS``. An empty tuple means the question named no
            resolvable entity -- a normal outcome for a purely conceptual
            question, which the graph modality then sits out.
        """
        anchors: list[Anchor] = []
        seen_nodes: set[str] = set()

        for match in self._pattern.finditer(query_text):
            mention = match.group(0)
            entries = self._index.get(_normalise(mention))
            if not entries:
                continue
            for node_id, matched_property in entries:
                if node_id in seen_nodes:
                    continue
                node = self._graph.node(node_id)
                if node is None:  # pragma: no cover - index is built from graph
                    continue
                record = self._corpus.by_node_id(node_id)
                if record is None:
                    # Filtered out by the corpus label allow/deny lists; it is
                    # not retrievable, so it must not anchor traversal either.
                    continue
                seen_nodes.add(node_id)
                anchors.append(
                    Anchor(
                        mention=mention,
                        node_id=node_id,
                        record_id=record.record_id,
                        record_type=record.record_type,
                        matched_property=matched_property,
                    )
                )
                if len(anchors) >= self._config.max_anchors:
                    return tuple(anchors)
        if not anchors:
            logger.debug("No anchors resolved for query %r", query_text)
        return tuple(anchors)

    @property
    def indexed_value_count(self) -> int:
        """Number of distinct values that can ground a traversal."""
        return len(self._index)


def _normalise(value: str) -> str:
    """Case-fold and Unicode-normalise a value for exact lookup."""
    return unicodedata.normalize("NFKC", value).strip().casefold()
