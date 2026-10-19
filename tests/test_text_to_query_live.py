"""
Integration tests: text-to-query against a live Neo4j database.

Skipped unless ``RUN_LIVE_GRAPH_TESTS=1`` and the database is actually
reachable, so the default suite stays hermetic and fast -- matching
``test_live_graph_validation.py``'s convention exactly.

The LLM is still a :class:`ScriptedLLM` here, not a real model. That is
deliberate, not a shortcut: a real model's *wording* is not deterministic
enough for a pytest assertion, but the property under test -- does the
compiled Cypher, executed against a real database, agree with the in-memory
executor on the same plan? -- has nothing to do with wording. Real-model
behaviour against this database is exercised live via
``evaluate_text_to_cypher.py --compare``, not baked into the automated suite.

This file exists because two real bugs were found the moment a genuinely live
database became available, neither visible from the in-memory path alone:

1. **Boolean filters silently matched nothing.** ``is_tender = 'true'``
   against a real Neo4j boolean returns 0, not an error -- a wrong answer
   indistinguishable from a right one. See ``test_boolean_filter_matches_the_
   real_boolean_type``.
2. **Aggregate + evidence produced a different number per node.** Naming the
   matched node in the same ``RETURN`` as an aggregate makes Cypher group by
   that node, turning a total into a per-record value. See
   ``test_aggregate_with_evidence_is_not_grouped_by_node``.

Both are now regression-tested here, against the real database, not just
asserted fixed.
"""
from __future__ import annotations

import os

import pytest

from src.graph.neo4j_client import Neo4jClient
from src.retrieval.corpus import LabelIdentityPolicy, RecordSerializer
from src.text_to_query.engine import TextToQueryEngine
from src.text_to_query.examples import CypherExampleBank
from src.text_to_query.llm_client import LLMRequestError

LIVE_TESTS_ENABLED = os.getenv("RUN_LIVE_GRAPH_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE_TESTS_ENABLED,
    reason="Set RUN_LIVE_GRAPH_TESTS=1 to run tests against a live Neo4j instance",
)


class ScriptedLLM:
    """Returns pre-written plans in order. No network, no model."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class UnreachableLLM:
    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        raise LLMRequestError("simulated backend outage")


@pytest.fixture(scope="module")
def client(live_neo4j_config):
    """A connected client, or a skip if the database is not reachable."""
    client = Neo4jClient(live_neo4j_config)
    if not client.verify_connectivity():
        pytest.skip(f"Neo4j is not reachable at {live_neo4j_config.uri}")
    yield client
    client.close()


@pytest.fixture
def ttq_serializer(graph, settings):
    return RecordSerializer(
        settings.corpus, LabelIdentityPolicy.learn(graph, settings.corpus)
    )


@pytest.fixture
def empty_examples(tmp_path):
    """An engine needs an example bank; content is irrelevant to these tests."""
    import json

    path = tmp_path / "examples.json"
    path.write_text(
        json.dumps({"examples": [{"id": "x", "question": "q", "plan": {"label": "Order"}}]}),
        encoding="utf-8",
    )
    return CypherExampleBank.load(path)


@pytest.fixture
def make_engine(graph, ttq_serializer, empty_examples, settings, client):
    """An engine wired to both the in-memory graph and the live database."""

    def _make(llm, config=None):
        return TextToQueryEngine(
            graph,
            serializer=ttq_serializer,
            llm_client=llm,
            examples=empty_examples,
            config=config or settings.text_to_cypher,
            neo4j_client=client,
        )

    return _make


def plan_json(**fields) -> str:
    import json

    return json.dumps(fields)


# -- source parity: json in-memory vs the real database ---------------------


class TestBackendAgreement:
    """
    The core claim: the same plan, executed two ways, gives the same answer.

    Each case here is one the onboarding gold set already scores; running
    them through both backends is what actually proves the compiled Cypher is
    correct, rather than merely that it looks plausible.
    """

    def test_pure_count(self, make_engine):
        plan = plan_json(
            label="DealOutcome",
            filters=[{"property": "result", "op": "=", "value": "lost_to_competitor"}],
            aggregate="count",
        )
        engine = make_engine(ScriptedLLM(plan, plan))

        json_result = engine.ask("How many were lost?", backend="json")
        neo4j_result = engine.ask("How many were lost?", backend="neo4j")

        assert json_result.success and neo4j_result.success
        assert json_result.value == neo4j_result.value
        assert neo4j_result.backend == "neo4j"
        assert json_result.backend == "json"

    def test_average_over_multiple_matches(self, make_engine):
        plan = plan_json(
            label="Quote",
            filters=[{"property": "opportunity_id", "op": "=", "value": "OPP00060"}],
            aggregate="avg",
            aggregate_property="unit_price",
        )
        engine = make_engine(ScriptedLLM(plan, plan))

        json_result = engine.ask("avg?", backend="json")
        neo4j_result = engine.ask("avg?", backend="neo4j")

        assert json_result.success and neo4j_result.success
        assert json_result.value == pytest.approx(neo4j_result.value, abs=0.01)

    def test_fact_with_evidence(self, make_engine):
        plan = plan_json(
            label="AuthorityPeriod",
            filters=[{"property": "agent_id", "op": "=", "value": "AG011"}],
            aggregate="max",
            aggregate_property="max_discount_pct",
            return_records=True,
        )
        engine = make_engine(ScriptedLLM(plan, plan))

        json_result = engine.ask("max discount?", backend="json")
        neo4j_result = engine.ask("max discount?", backend="neo4j")

        assert json_result.value == neo4j_result.value == 15
        assert json_result.citations == neo4j_result.citations
        assert neo4j_result.citations == ("AuthorityPeriod:AG011",)


# -- the two bugs this live database exposed, now regression-tested ---------


def test_boolean_filter_matches_the_real_boolean_type(make_engine, graph):
    """
    Regression test for a wrong answer that looked like a right one.

    ``is_tender = 'true'`` (the exact string form a real model produced)
    against a real Neo4j boolean matched zero rows -- no error, just an
    empty-looking correct-shaped answer. Confirmed live: this returned 0
    before the fix in ``plan.py``'s ``_filter_to_cypher``, and the expected
    value below is computed from the graph, not memorised, so this stays
    correct if the data changes.
    """
    expected = sum(
        1
        for node in graph
        if node.primary_label == "Opportunity"
        and str(node.properties.get("is_tender")).strip().lower() == "true"
    )
    assert expected > 0, "fixture has no is_tender=true Opportunity to test with"

    plan = plan_json(
        label="Opportunity",
        filters=[{"property": "is_tender", "op": "=", "value": "true"}],
        aggregate="count",
    )
    engine = make_engine(ScriptedLLM(plan))

    result = engine.ask("how many tenders?", backend="neo4j")

    assert result.success
    assert result.value == expected
    assert "toLower" in result.cypher, "the boolean-coercion compilation should be present"


def test_aggregate_with_evidence_is_not_grouped_by_node(make_engine, graph):
    """
    Regression test for a compilation bug: ``RETURN avg(n.x) AS avg, n AS
    evidence`` makes Cypher group by ``n``, turning a genuine average into a
    separate near-meaningless "average" per matched row -- one row per
    record instead of one row for the whole answer.

    Confirmed live before the fix: the query executed without error and
    returned a plausible-looking number, silently wrong. This asserts
    exactly one row comes back, with the correct true average.
    """
    prices = [
        node.properties["unit_price"]
        for node in graph
        if node.primary_label == "Quote" and node.properties.get("opportunity_id") == "OPP00060"
    ]
    assert len(prices) > 1, "fixture needs a multi-quote opportunity to expose the grouping bug"
    expected_avg = sum(prices) / len(prices)

    plan = plan_json(
        label="Quote",
        filters=[{"property": "opportunity_id", "op": "=", "value": "OPP00060"}],
        aggregate="avg",
        aggregate_property="unit_price",
        return_records=True,
    )
    engine = make_engine(ScriptedLLM(plan))

    result = engine.ask("average price?", backend="neo4j")

    assert result.success
    assert result.value == pytest.approx(expected_avg, abs=0.01)
    # The grouping bug's signature: one row per matched Quote instead of one
    # combined answer. A single citation proves the WITH-based aggregation
    # ran over the whole set rather than fanning out per node.
    assert len(result.citations) == 1
    assert "WITH" in result.cypher, "the group-by-safe compilation should be present"


# -- resilience ---------------------------------------------------------


def test_unreachable_llm_still_fails_cleanly_with_a_real_database_configured(make_engine):
    """A dead model backend must fail the same way regardless of which
    execution backend was requested -- no query was ever produced."""
    result = make_engine(UnreachableLLM()).ask("anything?", backend="neo4j")

    assert not result.success
    assert result.attempts == 1
    assert "simulated backend outage" in (result.error or "")


def test_neo4j_backend_without_a_client_raises_a_clear_error(
    graph, ttq_serializer, empty_examples, settings
):
    """Requesting backend='neo4j' on an engine built without one must fail
    fast and clearly, not with a confusing AttributeError deep in _result."""
    engine = TextToQueryEngine(
        graph,
        serializer=ttq_serializer,
        llm_client=ScriptedLLM(),
        examples=empty_examples,
        config=settings.text_to_cypher,
        neo4j_client=None,
    )

    with pytest.raises(ValueError, match="needs a Neo4j client"):
        engine.ask("anything?", backend="neo4j")
