"""
Tests for live schema derivation and pruning.

The property under test throughout: the schema is whatever the graph says it
is. No assertion here hardcodes an AXIOM label -- each one either checks a
structural rule, or checks against a value read from the graph fixture.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.graph.model import GraphEdge, GraphNode, KnowledgeGraph
from src.text_to_query.schema import build_schema_from_graph, prune_schema


@pytest.fixture
def schema(graph):
    return build_schema_from_graph(graph)


# -- derivation -------------------------------------------------------------


def test_every_label_in_the_graph_appears(graph, schema):
    assert schema.label_names == frozenset(graph.stats.label_counts)


def test_every_relationship_type_appears(graph, schema):
    assert schema.relationship_names == frozenset(graph.stats.relationship_counts)


def test_properties_are_the_union_across_nodes():
    """A property only some records carry must still be visible to the model."""
    graph = KnowledgeGraph(
        nodes=[
            GraphNode("a", ("Thing",), {"thing_id": "T1"}),
            GraphNode("b", ("Thing",), {"thing_id": "T2", "optional_note": "x"}),
        ]
    )

    schema = build_schema_from_graph(graph)

    assert schema.node_labels["Thing"] == ("optional_note", "thing_id")


def test_relationship_records_every_endpoint_pair_observed():
    """One relationship type may legitimately join more than one label pair."""
    graph = KnowledgeGraph(
        nodes=[
            GraphNode("a", ("A",), {"a_id": "1"}),
            GraphNode("b", ("B",), {"b_id": "1"}),
            GraphNode("c", ("C",), {"c_id": "1"}),
        ],
        edges=[GraphEdge("LINKS", "a", "b"), GraphEdge("LINKS", "c", "b")],
    )

    schema = build_schema_from_graph(graph)

    assert schema.relationship_types["LINKS"] == frozenset({("A", "B"), ("C", "B")})


def test_render_is_readable_and_complete(schema):
    rendered = schema.render()
    for label in schema.label_names:
        assert f"({label} " in rendered or f"({label})" in rendered
    for rel_type in schema.relationship_names:
        assert f"[:{rel_type}]" in rendered


# -- pruning ----------------------------------------------------------------


def test_pruning_keeps_the_label_the_question_names(graph, schema, settings):
    label = next(iter(schema.label_names))
    pruned = prune_schema(f"Tell me about {label}", schema, settings.text_to_cypher)

    assert label in pruned.node_labels


def test_pruning_drops_irrelevant_labels(schema, settings):
    """A question about one narrow topic must not pull in the whole schema."""
    pruned = prune_schema(
        "What is the list price of a product?", schema, settings.text_to_cypher
    )

    assert len(pruned.node_labels) < len(schema.node_labels)


def test_pruning_survives_plural_singular_mismatch(graph, schema, settings):
    """
    "opportunities" in the question must match "opportunity" in the schema.

    Without stemming, DealOutcome -- whose only link to the question is the
    property ``opportunity_id`` -- was excluded entirely from a question
    asking how many opportunities were lost, so the model could not have
    written a correct query even in principle.
    """
    pruned = prune_schema(
        "How many opportunities did we lose?", schema, settings.text_to_cypher
    )

    matching = [
        label
        for label, properties in schema.node_labels.items()
        if any("opportunity" in prop for prop in properties)
    ]
    assert matching, "graph fixture has no opportunity-bearing label to test with"
    assert any(label in pruned.node_labels for label in matching)


def test_pruning_falls_back_to_the_full_schema_when_nothing_matches(schema, settings):
    """An empty schema makes a correct query impossible; over-including does not."""
    pruned = prune_schema("zzzz qqqq xxxx", schema, settings.text_to_cypher)

    assert pruned.node_labels == schema.node_labels


def test_relationship_endpoints_are_always_included(schema, settings):
    """A relationship the model can see but cannot anchor on is useless."""
    pruned = prune_schema(
        "Which product does the order line refer to?", schema, settings.text_to_cypher
    )

    for pairs in pruned.relationship_types.values():
        for source, target in pairs:
            assert source in pruned.node_labels
            assert target in pruned.node_labels


def test_max_schema_labels_bounds_the_directly_matched_set(schema, settings):
    """
    The cap applies to labels matched on their own merit.

    Relationship endpoints are pulled in afterwards regardless, by design, so
    the final count can exceed the cap -- what must not happen is the cap
    being ignored entirely.
    """
    tight = replace(settings.text_to_cypher, max_schema_labels=1)
    loose = replace(settings.text_to_cypher, max_schema_labels=20)

    narrow = prune_schema("price product order customer quote", schema, tight)
    wide = prune_schema("price product order customer quote", schema, loose)

    assert len(narrow.node_labels) <= len(wide.node_labels)


def test_pruning_is_deterministic(schema, settings):
    question = "What did the customer pay for the product?"
    runs = {
        tuple(sorted(prune_schema(question, schema, settings.text_to_cypher).node_labels))
        for _ in range(3)
    }
    assert len(runs) == 1
