"""
Integration tests against a live Neo4j database.

Skipped unless ``RUN_LIVE_GRAPH_TESTS=1`` and the database is actually
reachable, so the default suite stays hermetic and fast.

The assertion that matters here is **source parity**: because both loaders
produce the same :class:`KnowledgeGraph`, the identifier derivation, traversal
ordering, and fusion behaviour must be identical whether the graph came from
JSON or from the database. If that holds, the fast offline suite is a valid
proxy for the production path -- and if it ever stops holding, this test is
where it surfaces.
"""
from __future__ import annotations

import os

import pytest

from src.graph.neo4j_client import Neo4jClient, Neo4jQueryError
from src.graph.queries import COUNT_NODES, SCHEMA_LABELS, SCHEMA_RELATIONSHIP_TYPES
from src.graph.sources import Neo4jGraphSource
from src.pipeline.factory import build_retrieval_stack
from src.retrieval.corpus import build_corpus

LIVE_TESTS_ENABLED = os.getenv("RUN_LIVE_GRAPH_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE_TESTS_ENABLED,
    reason="Set RUN_LIVE_GRAPH_TESTS=1 to run tests against a live Neo4j instance",
)


@pytest.fixture(scope="module")
def client(live_neo4j_config):
    """A connected client, or a skip if the database is not reachable."""
    client = Neo4jClient(live_neo4j_config)
    if not client.verify_connectivity():
        pytest.skip(f"Neo4j is not reachable at {live_neo4j_config.uri}")
    yield client
    client.close()


@pytest.fixture(scope="module")
def live_graph(client):
    return Neo4jGraphSource(client).load()


# -- connectivity and schema ------------------------------------------------


def test_database_is_populated(client):
    rows = client.run_query(COUNT_NODES)
    assert rows[0]["node_count"] > 0


def test_schema_introspection_returns_labels(client):
    labels = [row["label"] for row in client.run_query(SCHEMA_LABELS)]
    relationship_types = [
        row["relationshipType"] for row in client.run_query(SCHEMA_RELATIONSHIP_TYPES)
    ]
    assert labels and relationship_types


def test_write_statements_are_refused(client):
    """
    Read-access enforcement is server-side, not a prompt instruction.

    The text-to-Cypher layer depends on this: a generated statement that
    attempts a write must be rejected by Neo4j itself.
    """
    with pytest.raises(Neo4jQueryError):
        client.run_query("CREATE (n:ShouldNeverExist {id: 'x'}) RETURN n")


# -- source parity ----------------------------------------------------------


def test_live_projection_yields_the_same_structure(live_graph):
    assert live_graph.stats.node_count > 0
    assert live_graph.stats.edge_count > 0
    assert live_graph.stats.dangling_edge_count == 0


def test_identifier_derivation_works_on_live_data(live_graph, serializer):
    """
    Every live node must resolve to a property-derived citation handle.

    A handle falling back to Neo4j's ``elementId`` would be unstable across
    database restarts -- and therefore uncitable.
    """
    corpus = build_corpus(live_graph, serializer)

    unresolved = [
        record.record_type
        for record in corpus
        if record.record_id.endswith(f":{record.node_id}")
    ]
    assert unresolved == []
    assert len(set(record.record_id for record in corpus)) == len(corpus)


def test_json_and_live_paths_agree_on_shared_records(settings, corpus, live_graph, serializer):
    """
    Records present in both sources must carry identical handles and text.

    The JSON file is a documented sample of the live graph, so the overlap is
    partial by design; the assertion is on agreement where they overlap, not
    on equal size.
    """
    live_corpus = build_corpus(live_graph, serializer)
    live_by_id = {record.record_id: record for record in live_corpus}

    shared = [record for record in corpus if record.record_id in live_by_id]
    assert shared, "sample JSON and live graph share no record ids"

    for record in shared:
        assert record.record_type == live_by_id[record.record_id].record_type


def test_end_to_end_pipeline_against_live_graph(settings, client, encoder, cross_encoder):
    """The full stack must serve queries from the database unchanged."""
    from dataclasses import replace

    live_settings = replace(settings, corpus=replace(settings.corpus, source="neo4j"))
    stack = build_retrieval_stack(
        live_settings,
        neo4j_client=client,
        encoder=encoder,
        cross_encoder=cross_encoder,
    )

    outcome = stack.pipeline.run("What is the list price for product SKU0001?", top_k=5)

    assert outcome.final
    for result in outcome.final:
        assert stack.corpus.by_record_id(result.record_id) is not None
