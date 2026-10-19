"""
Composition root for the hybrid retrieval stack.

Every module in ``src/retrieval`` receives its collaborators through its
constructor and builds nothing itself. This is the one place where the object
graph is assembled, which is what makes each component independently testable:
a test constructs exactly the pieces it needs with fakes, while production
calls :func:`build_retrieval_stack` and gets the fully wired system.

The factory is also the only place that knows whether the graph came from JSON
or Neo4j. Nothing downstream can tell the difference.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.config.settings import Settings, get_settings
from src.graph.model import KnowledgeGraph
from src.graph.sources import GraphSource, build_graph_source
from src.pipeline.retriever_pipeline import HybridRetrievalPipeline
from src.retrieval.anchors import AnchorResolver
from src.retrieval.base import Retriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.corpus import (
    Corpus,
    LabelIdentityPolicy,
    RecordSerializer,
    build_corpus,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
    SentenceTransformerEncoder,
    TextEncoder,
)
from src.retrieval.fusion import ReciprocalRankFuser
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.reranker import (
    CrossEncoderModel,
    CrossEncoderReranker,
    SentenceTransformerCrossEncoder,
)

__all__ = ["RetrievalStack", "build_retrieval_stack"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalStack:
    """
    The assembled system, with its parts exposed for inspection and evaluation.

    The evaluation harness needs the corpus (to build a gold set against real
    record ids) and the graph (to derive ground truth from structure), so they
    are returned alongside the pipeline rather than hidden inside it.
    """

    settings: Settings
    graph: KnowledgeGraph
    corpus: Corpus
    serializer: RecordSerializer
    anchor_resolver: AnchorResolver
    retrievers: tuple[Retriever, ...]
    pipeline: HybridRetrievalPipeline

    def describe(self) -> str:
        """One-line summary of what was loaded, for logs and demo headers."""
        stats = self.graph.stats
        return (
            f"{stats.node_count} nodes / {stats.edge_count} relationships -> "
            f"{len(self.corpus)} retrievable records "
            f"({len(stats.label_counts)} labels, "
            f"{len(stats.relationship_counts)} relationship types); "
            f"streams: {', '.join(self.pipeline.stream_names)}"
        )


def build_retrieval_stack(
    settings: Settings | None = None,
    *,
    graph_source: GraphSource | None = None,
    neo4j_client: Any | None = None,
    encoder: TextEncoder | None = None,
    cross_encoder: CrossEncoderModel | None = None,
    enable_dense: bool = True,
    enable_graph: bool = True,
) -> RetrievalStack:
    """
    Build the full retrieval stack from configuration.

    Args:
        settings: application configuration; defaults to :func:`get_settings`.
        graph_source: overrides the configured source. Supplying one lets a
            test drive the whole stack from a fixture graph.
        neo4j_client: required only when ``CORPUS_SOURCE=neo4j``.
        encoder: overrides the sentence-transformers encoder. Injecting a stub
            keeps a model download out of the unit-test path.
        cross_encoder: overrides the cross-encoder model, for the same reason.
        enable_dense: when False, omits the semantic modality. Useful for a
            fully offline run, since fusion tolerates any stream count.
        enable_graph: when False, omits the traversal modality.

    Returns:
        A :class:`RetrievalStack` whose pipeline is ready to serve queries.

    Raises:
        GraphSourceError: the configured source could not be read.
        ValueError: the configuration produced no usable retriever.
    """
    settings = settings or get_settings()

    source = graph_source or build_graph_source(settings, client=neo4j_client)
    logger.info("Loading knowledge graph from %s", source.name)
    graph = source.load()

    # Identity is decided per label, from the whole graph, before any record
    # is built -- so every node of a given label yields the same handle shape.
    policy = LabelIdentityPolicy.learn(graph, settings.corpus)
    serializer = RecordSerializer(settings.corpus, policy)
    corpus = build_corpus(graph, serializer)
    if len(corpus) == 0:
        raise ValueError(
            "The retrieval corpus is empty; check CORPUS_INCLUDE_LABELS / "
            "CORPUS_EXCLUDE_LABELS against the labels present in the graph"
        )

    anchor_resolver = AnchorResolver(
        graph=graph,
        corpus=corpus,
        serializer=serializer,
        graph_config=settings.graph,
        corpus_config=settings.corpus,
    )

    retrievers: list[Retriever] = [BM25Retriever(corpus, settings.lexical)]
    if enable_dense:
        retrievers.append(
            DenseRetriever(
                corpus,
                settings.embedding,
                encoder or SentenceTransformerEncoder(settings.embedding),
            )
        )
    if enable_graph:
        retrievers.append(GraphRetriever(graph, corpus, settings.graph))

    reranker = _build_reranker(settings, cross_encoder)

    pipeline = HybridRetrievalPipeline(
        retrievers=retrievers,
        fuser=ReciprocalRankFuser(settings.fusion),
        config=settings.retrieval,
        anchor_resolver=anchor_resolver,
        reranker=reranker,
    )

    stack = RetrievalStack(
        settings=settings,
        graph=graph,
        corpus=corpus,
        serializer=serializer,
        anchor_resolver=anchor_resolver,
        retrievers=tuple(retrievers),
        pipeline=pipeline,
    )
    logger.info("Retrieval stack ready: %s", stack.describe())
    return stack


def _build_reranker(
    settings: Settings, cross_encoder: CrossEncoderModel | None
) -> CrossEncoderReranker | None:
    """Construct the reranker, honouring the ``RERANK_ENABLED`` switch."""
    if not settings.reranker.enabled and cross_encoder is None:
        logger.info("Cross-encoder reranking disabled by configuration")
        return None
    model = cross_encoder or SentenceTransformerCrossEncoder(settings.reranker)
    return CrossEncoderReranker(settings.reranker, model)

