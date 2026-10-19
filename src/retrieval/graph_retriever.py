"""
Graph-traversal retrieval modality.

This is the modality that finds evidence sharing *no vocabulary* with the
question. Asked about ``CUST0015``, it surfaces the competitor's quoted price
on that customer's opportunity -- a record containing neither the customer id
nor any word from the question, reachable only through the relationship
structure.

The correction that matters here is scoring. The previous implementation
assigned every traversal hit ``score=1.0`` and relied on a Cypher ``LIMIT``
with no ``ORDER BY``. Since RRF is purely rank-based, that fed *arbitrary*
ranks into fusion: the first record the database happened to return received
the full rank-1 RRF weight. Proximity now supplies the ordering -- a record one
hop from the anchor outranks one three hops away -- so the graph stream
contributes a meaningful signal.
"""
from __future__ import annotations

import logging

from src.config.settings import GraphTraversalConfig
from src.graph.model import KnowledgeGraph, Neighbor
from src.retrieval.base import Anchor, RetrievalQuery, RetrievalResult
from src.retrieval.corpus import Corpus

__all__ = ["ANCHOR_PROXIMITY", "GraphRetriever"]

logger = logging.getLogger(__name__)

#: Proximity assigned to an anchor record (hop 0). Only its order relative to
#: the one-hop value of 1.0 matters, because fusion consumes rank, not score.
ANCHOR_PROXIMITY: float = 2.0


class GraphRetriever:
    """
    Retrieves evidence by expanding outward from anchors resolved in the query.

    Works against the in-memory :class:`KnowledgeGraph`, so it behaves
    identically whether the graph was loaded from JSON or projected from Neo4j.
    Results are looked up in the same :class:`Corpus` the other modalities
    index, which guarantees a record reached by traversal carries the exact
    ``record_id`` it would carry if reached lexically -- the precondition for
    RRF recognising cross-modal agreement.
    """

    __slots__ = ("_graph", "_corpus", "_config", "_name")

    def __init__(
        self,
        graph: KnowledgeGraph,
        corpus: Corpus,
        config: GraphTraversalConfig,
        *,
        name: str = "graph",
    ) -> None:
        self._graph = graph
        self._corpus = corpus
        self._config = config
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def search(self, query: RetrievalQuery, top_k: int) -> list[RetrievalResult]:
        """
        Expand from every anchor and return the closest connected records.

        Args:
            query: carries the resolved anchors; a query with none yields ``[]``
                so the pipeline simply fuses the remaining modalities.
            top_k: maximum records to return.

        Returns:
            Records ordered by proximity, best-first. When several anchors reach
            the same record, the shortest distance wins and the record's score
            is boosted, because being connected to *two* named entities in the
            question is stronger evidence than being connected to one.
        """
        if top_k < 1 or not query.anchors:
            return []

        # node_id -> (best neighbour, anchors that reached it)
        reached: dict[str, tuple[Neighbor, list[Anchor]]] = {}

        # The anchors themselves are evidence at hop 0. Omitting them was a
        # defect: when a question names AG001 and that grounds to
        # AuthorityPeriod:AG001, the record the question is *about* was absent
        # from the graph stream, while the agent's 40 quotes -- reached in one
        # hop and also matched lexically -- collected RRF's cross-modal
        # agreement bonus and outranked it. Returning the anchor at hop 0 puts
        # the named record where it belongs, and lets the other modalities
        # break ties between several anchors.
        for anchor in query.anchors:
            node = self._graph.node(anchor.node_id)
            if node is not None:
                reached[anchor.node_id] = (
                    Neighbor(node=node, hops=0, path=()),
                    [anchor],
                )

        for anchor in query.anchors:
            neighbours = self._graph.traverse(
                anchor.node_id,
                max_hops=self._config.max_hops,
                max_neighbors=self._config.max_neighbors,
                relationship_types=self._config.relationship_types,
                directed=self._config.directed,
            )
            for neighbour in neighbours:
                node_id = neighbour.node.node_id
                existing = reached.get(node_id)
                if existing is None:
                    reached[node_id] = (neighbour, [anchor])
                else:
                    best, anchors = existing
                    anchors.append(anchor)
                    if neighbour.hops < best.hops:
                        reached[node_id] = (neighbour, anchors)

        scored: list[tuple[float, str, Neighbor, list[Anchor]]] = []
        for node_id, (neighbour, anchors) in reached.items():
            record = self._corpus.by_node_id(node_id)
            if record is None:
                # Excluded from the corpus by label filtering, so it is not
                # citable evidence and must not be returned.
                continue
            scored.append((self._proximity_score(neighbour.hops, len(anchors)), record.record_id, neighbour, anchors))

        # Highest proximity first; ties broken on record_id so the ranks handed
        # to RRF are reproducible run to run.
        scored.sort(key=lambda entry: (-entry[0], entry[1]))

        results: list[RetrievalResult] = []
        for score, record_id, neighbour, anchors in scored[:top_k]:
            record = self._corpus.by_record_id(record_id)
            if record is None:  # pragma: no cover - resolved immediately above
                continue
            results.append(
                RetrievalResult(
                    record_id=record.record_id,
                    record_type=record.record_type,
                    text=record.text,
                    score=score,
                    source=self._name,
                    metadata={
                        "hops": neighbour.hops,
                        # The relationship chain walked to reach this record --
                        # the provenance a grounded explanation cites.
                        "traversal_path": neighbour.path,
                        "anchor_record_ids": tuple(a.record_id for a in anchors),
                        "anchor_count": len(anchors),
                    },
                )
            )
        return results

    def _proximity_score(self, hops: int, anchor_count: int) -> float:
        """
        Convert traversal distance into a rankable relevance score.

        Hop 0 is the anchor itself -- the record the question actually named --
        and scores strictly above anything reached by traversal. Beyond that,
        ``hop_decay ** (hops - 1)`` gives 1.0 at one hop and decays
        geometrically with distance.

        Only the resulting *order* reaches fusion, since RRF is rank-based, so
        ``ANCHOR_PROXIMITY`` needs only to exceed the one-hop value of 1.0; its
        exact magnitude carries no further meaning.

        The multi-anchor term is deliberately sub-linear: a record connected to
        two named entities should outrank an equidistant record connected to
        one, but must not leapfrog a record several hops closer.
        """
        proximity = (
            ANCHOR_PROXIMITY
            if hops <= 0
            else self._config.hop_decay ** (hops - 1)
        )
        return proximity * (1.0 + 0.5 * (anchor_count - 1))
