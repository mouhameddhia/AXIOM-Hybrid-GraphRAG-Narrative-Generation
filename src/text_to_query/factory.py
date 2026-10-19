"""
Composition root for the text-to-query engine.

Mirrors :func:`src.pipeline.factory.build_retrieval_stack`: collaborators are
injected, nothing is constructed implicitly, and this is the one place that
wires the concrete production pieces together.

A Neo4j client is optional here, not required: the engine always answers by
executing plans against the in-memory graph, and only additionally gains the
ability to answer with ``backend="neo4j"`` when one is supplied.
"""
from __future__ import annotations

import logging

from src.config.settings import PROJECT_ROOT, Settings, get_settings
from src.graph.model import KnowledgeGraph
from src.graph.neo4j_client import Neo4jClient
from src.retrieval.corpus import LabelIdentityPolicy, RecordSerializer
from src.text_to_query.engine import TextToQueryEngine
from src.text_to_query.examples import CypherExampleBank
from src.text_to_query.llm_client import LLMClient, OllamaClient
from src.text_to_query.schema import build_schema_from_graph

__all__ = ["build_text_to_query_engine"]

logger = logging.getLogger(__name__)

_DEFAULT_EXAMPLES_PATH = "data/cypher_examples.json"


def build_text_to_query_engine(
    graph: KnowledgeGraph,
    settings: Settings | None = None,
    *,
    llm_client: LLMClient | None = None,
    serializer: RecordSerializer | None = None,
    neo4j_client: Neo4jClient | None = None,
) -> TextToQueryEngine:
    """
    Build a fully wired :class:`TextToQueryEngine`.

    Args:
        graph: the knowledge graph. The schema is derived from it and plans
            execute against it -- the same graph object the retrieval stack
            uses, so identifier derivation agrees between the two components.
        settings: application configuration; defaults to :func:`get_settings`.
        llm_client: overrides the production Ollama client. Injecting a fake
            keeps a network call and a running model out of the test path.
        serializer: overrides the record serializer. Defaults to one built
            with a label-identity policy learned from ``graph``, so citations
            match retrieval's exactly.
        neo4j_client: enables ``engine.ask(..., backend="neo4j")``. Omit it
            (the default) and the engine still answers every question, by
            executing plans against ``graph`` in memory.

    Returns:
        A ready-to-use :class:`TextToQueryEngine`.
    """
    settings = settings or get_settings()

    resolved_serializer = serializer or RecordSerializer(
        settings.corpus, LabelIdentityPolicy.learn(graph, settings.corpus)
    )
    schema = build_schema_from_graph(graph)
    examples = CypherExampleBank.load(PROJECT_ROOT / _DEFAULT_EXAMPLES_PATH, schema)

    engine = TextToQueryEngine(
        graph,
        serializer=resolved_serializer,
        llm_client=llm_client or OllamaClient(settings.llm),
        examples=examples,
        config=settings.text_to_cypher,
        base_temperature=settings.llm.temperature,
        neo4j_client=neo4j_client,
    )
    logger.info(
        "Text-to-query engine ready: %d labels, %d relationship types, %d examples",
        len(schema.label_names),
        len(schema.relationship_names),
        len(examples),
    )
    return engine
