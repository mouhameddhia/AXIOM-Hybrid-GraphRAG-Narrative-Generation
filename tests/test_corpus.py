"""
Tests for corpus derivation.

These cover the replacement for the hard-coded schema maps. The critical
property is that identifier derivation works *without* being told the AXIOM
schema -- so every assertion here is either about a structural rule or is
computed from the graph rather than transcribed from it.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.graph.model import GraphNode, KnowledgeGraph
from src.retrieval.corpus import LabelIdentityPolicy, RecordSerializer, build_corpus


def test_every_node_yields_a_record(graph, corpus):
    """No node may be silently dropped from the retrievable corpus."""
    assert len(corpus) == graph.stats.node_count


def test_record_ids_are_unique(corpus):
    """Citation handles must be unique, or a citation is ambiguous."""
    assert len(set(corpus.record_ids)) == len(corpus)


def test_record_ids_are_label_prefixed(corpus, graph):
    """Each handle names its label, so a citation is self-describing."""
    for record in corpus:
        assert record.record_id.startswith(f"{record.record_type}:")
        assert graph.node(record.node_id) is not None


def test_identity_key_prefers_the_most_label_specific_identifier(graph, serializer):
    """
    ``OrderLine`` must select ``order_line_id`` over ``order_id``.

    This is the case the previous hard-coded ``coalesce`` ladder got right only
    because someone happened to order it correctly by hand.
    """
    node = next(n for n in graph if n.primary_label == "OrderLine")
    assert "order_id" in node.properties and "order_line_id" in node.properties
    assert serializer.identity_key(node) == "order_line_id"


def test_identity_falls_through_to_a_naming_property(graph, serializer):
    """A label with no ``*_id`` property still resolves (``City.city_name``)."""
    node = next(n for n in graph if n.primary_label == "City")
    assert not any(key.endswith("_id") for key in node.properties)
    assert serializer.identity_key(node) is not None
    assert serializer.record_id(node).startswith("City:")


def test_identity_derivation_needs_no_schema_knowledge(graph, serializer):
    """
    Every node resolves to a property-derived handle, not a node-id fallback.

    A handle ending in the raw node id means derivation failed for that label.
    """
    unresolved = [
        node.primary_label
        for node in graph
        if serializer.record_id(node).endswith(f":{node.node_id}")
    ]
    assert unresolved == []


def test_serializer_generalises_to_an_unseen_schema(settings):
    """
    A label the AXIOM graph has never contained must still resolve.

    This is the regression guard: with the old hard-coded maps, an unknown
    label produced an unstable ``elementId`` handle.
    """
    serializer = RecordSerializer(settings.corpus)
    node = GraphNode(
        node_id="n1",
        labels=("FreightSurcharge",),
        properties={"surcharge_id": "FS001", "amount": 12.5, "carrier_id": "C9"},
    )

    assert serializer.identity_key(node) == "surcharge_id"
    assert serializer.record_id(node) == "FreightSurcharge:FS001"


def test_colliding_handles_are_disambiguated(settings):
    """Two nodes sharing a business identifier must not share a handle."""
    serializer = RecordSerializer(settings.corpus)
    graph = KnowledgeGraph(
        nodes=[
            GraphNode("p1", ("PricePeriod",), {"price_id": "P1", "valid_from": "2025-01-01"}),
            GraphNode("p2", ("PricePeriod",), {"price_id": "P1", "valid_from": "2026-01-01"}),
        ]
    )

    corpus = build_corpus(graph, serializer)

    assert len(set(corpus.record_ids)) == 2
    assert "PricePeriod:P1" in corpus.record_ids
    assert any("#p2" in record_id for record_id in corpus.record_ids)


def test_text_leads_with_the_identifying_property(graph, serializer):
    """The identifier must survive truncation and score at the document front."""
    for node in graph:
        key = serializer.identity_key(node)
        if key is None:
            continue
        text = serializer.to_text(node)
        assert text.startswith(f"{node.primary_label} {key} ")


def test_floats_are_rendered_without_precision_artefacts(settings):
    """``162.5`` must index as ``162.5`` so an exact lexical match can hit."""
    serializer = RecordSerializer(settings.corpus)
    node = GraphNode("n", ("Item",), {"item_id": "I1", "price": 162.5, "qty": 100.0})

    text = serializer.to_text(node)

    assert "price 162.5" in text
    assert "qty 100 " in text or text.endswith("qty 100")


def test_excluded_properties_are_omitted(settings):
    """Internal keys and vectors must never reach the lexical index."""
    corpus_config = replace(
        settings.corpus, excluded_property_keys=frozenset({"_eid", "secret"})
    )
    serializer = RecordSerializer(corpus_config)
    node = GraphNode("n", ("Item",), {"item_id": "I1", "secret": "classified"})

    assert "classified" not in serializer.to_text(node)


def test_label_filters_are_applied(settings, graph):
    """Allow/deny lists replace the dead hard-coded ``NOT n:Label`` clause."""
    excluded = "Quote"
    serializer = RecordSerializer(
        replace(settings.corpus, exclude_labels=frozenset({excluded}))
    )

    corpus = build_corpus(graph, serializer)

    assert all(record.record_type != excluded for record in corpus)
    assert len(corpus) < graph.stats.node_count


def test_text_is_truncated_to_the_configured_limit(settings):
    serializer = RecordSerializer(replace(settings.corpus, max_text_chars=48))
    node = GraphNode("n", ("Item",), {"item_id": "I1", "note": "word " * 100})

    assert len(serializer.to_text(node)) <= 48


@pytest.mark.parametrize("lookup", ["by_record_id", "by_node_id"])
def test_corpus_lookups_resolve(corpus, lookup):
    record = corpus[0]
    key = record.record_id if lookup == "by_record_id" else record.node_id
    assert getattr(corpus, lookup)(key) is record


# -- regressions found on the full 5,794-node PoC export --------------------


def test_foreign_key_does_not_win_identity(settings):
    """
    A node must never be named after an entity it merely *references*.

    On the full export an ``Opportunity`` carrying both ``opportunity_id`` and
    ``revises_opportunity_id`` was handed the handle of the opportunity it
    revises -- a citation pointing at the wrong record. The tie-break now
    penalises tokens absent from the label (``revises``).
    """
    serializer = RecordSerializer(settings.corpus)
    node = GraphNode(
        node_id="o1",
        labels=("Opportunity",),
        properties={"opportunity_id": "OPP00003", "revises_opportunity_id": "OPP00002"},
    )

    assert serializer.identity_key(node) == "opportunity_id"
    assert serializer.record_id(node) == "Opportunity:OPP00003"


def test_label_policy_makes_handles_consistent_within_a_label(settings):
    """
    Optional properties must not give one label two citation schemes.

    Without a label-wide policy, the node lacking ``thing_id`` falls back to
    ``other_id`` and the label ends up with two incompatible handle shapes.
    """
    graph = KnowledgeGraph(
        nodes=[
            GraphNode("a", ("Thing",), {"thing_id": "T1", "other_id": "O1"}),
            GraphNode("b", ("Thing",), {"thing_id": "T2", "other_id": "O2"}),
            GraphNode("c", ("Thing",), {"other_id": "O3"}),  # sparse node
        ]
    )
    policy = LabelIdentityPolicy.learn(graph, settings.corpus)
    serializer = RecordSerializer(settings.corpus, policy)

    assert policy.key_for("Thing") == "thing_id"
    assert serializer.record_id(graph.node("a")) == "Thing:T1"
    assert serializer.record_id(graph.node("b")) == "Thing:T2"
    # The sparse node cannot use the policy key, so the per-node ladder
    # applies -- still stable and still citable, never a raw node id.
    assert serializer.record_id(graph.node("c")) == "Thing:O3"


def test_label_policy_is_decided_by_majority(settings):
    """A minority of irregular nodes must not redefine a label's scheme."""
    graph = KnowledgeGraph(
        nodes=[GraphNode(f"n{i}", ("Thing",), {"thing_id": f"T{i}"}) for i in range(5)]
        + [GraphNode("odd", ("Thing",), {"other_id": "X1"})]
    )

    assert LabelIdentityPolicy.learn(graph, settings.corpus).key_for("Thing") == "thing_id"


def test_label_policy_is_deterministic(graph, settings):
    """Learning the policy twice over the same graph must agree exactly."""
    first = LabelIdentityPolicy.learn(graph, settings.corpus).as_dict()
    second = LabelIdentityPolicy.learn(graph, settings.corpus).as_dict()

    assert dict(first) == dict(second)
