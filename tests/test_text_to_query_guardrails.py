"""
Tests for the Cypher guardrails.

Two distinct concerns, tested separately because they fail differently:

* read-only validation is a **safety** control -- a write must never execute;
* schema-reference validation is a **correctness** control -- Cypher does not
  error on an unknown label, it silently returns nothing, which is far more
  dangerous than an error because it looks like a valid empty answer.
"""
from __future__ import annotations

import pytest

from src.text_to_query.guardrails import (
    GuardrailViolation,
    is_read_only,
    referenced_labels,
    referenced_relationship_types,
    validate_read_only,
    validate_schema_references,
)

SAFE = "MATCH (a:AuthorityPeriod)-[:FOR_AGENT]->(s:SalesAgent) RETURN a, s LIMIT 10"


# -- read-only safety -------------------------------------------------------


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) DETACH DELETE n",
        "CREATE (n:Thing {id: 1})",
        "MATCH (n) SET n.price = 0",
        "MERGE (n:Thing {id: 1})",
        "MATCH (n) REMOVE n.price",
        "DROP INDEX my_index",
        "LOAD CSV FROM 'file:///x.csv' AS row RETURN row",
        "MATCH (n) FOREACH (x IN [1] | SET n.v = x)",
    ],
)
def test_write_clauses_are_rejected(cypher):
    assert is_read_only(cypher) is False


def test_read_only_query_passes():
    validate_read_only(SAFE)


def test_write_keyword_inside_a_string_literal_is_not_a_violation():
    """A property value mentioning DELETE must not trip the scanner."""
    validate_read_only("MATCH (n) WHERE n.notes = 'please DELETE later' RETURN n")


def test_write_keyword_inside_a_comment_is_not_a_violation():
    validate_read_only("MATCH (n) RETURN n // remember to DELETE this someday")


def test_empty_query_is_rejected():
    with pytest.raises(GuardrailViolation, match="empty"):
        validate_read_only("   ")


def test_dangerous_procedure_is_rejected():
    assert is_read_only("CALL apoc.create.node(['X'], {}) YIELD node RETURN node") is False


def test_violation_message_names_the_offending_clause():
    """The self-heal loop feeds this text back, so it must be specific."""
    with pytest.raises(GuardrailViolation, match="DELETE"):
        validate_read_only("MATCH (n) DETACH DELETE n")


# -- reference extraction ---------------------------------------------------


def test_extracts_labels_and_relationship_types():
    assert referenced_labels(SAFE) == frozenset({"AuthorityPeriod", "SalesAgent"})
    assert referenced_relationship_types(SAFE) == frozenset({"FOR_AGENT"})


def test_extracts_multi_label_nodes():
    assert referenced_labels("MATCH (n:A:B) RETURN n") == frozenset({"A", "B"})


def test_extracts_relationship_alternation():
    found = referenced_relationship_types("MATCH (a)-[:REL_A|REL_B]->(b) RETURN a")
    assert found == frozenset({"REL_A", "REL_B"})


def test_variable_length_path_isolates_the_type_name():
    found = referenced_relationship_types("MATCH (a)-[:FOR_AGENT*1..3]->(b) RETURN a")
    assert found == frozenset({"FOR_AGENT"})


def test_property_maps_are_not_mistaken_for_labels():
    """A map literal must not be read as a label reference."""
    cypher = "MATCH (a:AuthorityPeriod {agent_id: 'AG001'}) RETURN a.max_discount_pct"
    assert referenced_labels(cypher) == frozenset({"AuthorityPeriod"})


def test_anonymous_nodes_contribute_no_labels():
    assert referenced_labels("MATCH (a)-[:R]->(b) RETURN a") == frozenset()


# -- schema-reference correctness -------------------------------------------


def test_known_references_pass():
    validate_schema_references(SAFE, {"AuthorityPeriod", "SalesAgent"}, {"FOR_AGENT"})


def test_unknown_label_is_rejected():
    """
    The case that motivates this guardrail existing at all: Neo4j would
    accept this and return zero rows, indistinguishable from a real answer.
    """
    with pytest.raises(GuardrailViolation, match="Opprtunity"):
        validate_schema_references(
            "MATCH (n:Opprtunity) RETURN count(n)", {"Opportunity"}, set()
        )


def test_unknown_relationship_type_is_rejected():
    with pytest.raises(GuardrailViolation, match="FOR_AGNET"):
        validate_schema_references(
            "MATCH (a:A)-[:FOR_AGNET]->(b:B) RETURN a", {"A", "B"}, {"FOR_AGENT"}
        )


def test_label_check_runs_before_relationship_check():
    """Report the first problem found, so feedback stays focused on one fix."""
    with pytest.raises(GuardrailViolation, match="label"):
        validate_schema_references(
            "MATCH (n:Bad)-[:ALSO_BAD]->(m:Worse) RETURN n", set(), set()
        )


def test_real_schema_accepts_a_real_query(graph):
    """Against the actual graph, a genuinely valid query must pass."""
    from src.text_to_query.schema import build_schema_from_graph

    schema = build_schema_from_graph(graph)
    label = next(iter(schema.label_names))

    validate_schema_references(
        f"MATCH (n:{label}) RETURN count(n) AS total",
        schema.label_names,
        schema.relationship_names,
    )
