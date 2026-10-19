"""
Integration: retrieval + text-to-query + narrative over one Track A payload.

The three components were built separately and are tested separately. What
this file tests is the *seams* between them, which is where the failures that
matter actually live:

**One citation scheme.** A record handle produced by retrieval, returned by
text-to-query, and cited in a narrative must be the same string, or a reader
cannot follow a sentence back to a record. All three derive handles from the
same ``RecordSerializer``, and that is asserted here rather than assumed.

**Closure survives the join.** Folding graph-computed context into a fact pack
must not open a hole in the rule that the model states no number of its own.

**Shape mismatches are caught.** Closure guarantees a number came from an
attested source; it does not guarantee that source answered the question that
was asked. Running the real chain surfaced exactly that: asked for a product's
list price, the text-to-query engine returned ``count(n)`` -- the number 1,
correctly computed by the graph, attributed to a real record, and useless as a
price. The regression for it is :meth:`TestShapeGuard.test_a_count_is_not_a_price`.
"""
from __future__ import annotations

import pytest

from src.config.settings import NarrativeConfig
from src.narrative.contract import PriceRecommendation
from src.narrative.evidence import EvidenceIndex
from src.narrative.facts import Fact, build_facts
from src.narrative.generator import NarrativeGenerator
from src.narrative.graph_facts import fact_from_answer
from src.retrieval.corpus import LabelIdentityPolicy, RecordSerializer
from src.text_to_query.engine import QueryResult
from src.text_to_query.evidence import EvidenceRecord
from src.text_to_query.plan import Filter, QueryPlan
from tests.test_narrative_contract import SAMPLE_PAYLOADS
from tests.test_narrative_generator import ScriptedLLM


@pytest.fixture(scope="module")
def serializer(graph, settings):
    """The one serializer all three components share."""
    return RecordSerializer(
        settings.corpus, LabelIdentityPolicy.learn(graph, settings.corpus)
    )


@pytest.fixture(scope="module")
def index(graph, serializer):
    return EvidenceIndex.build(graph, serializer)


@pytest.fixture
def config():
    return NarrativeConfig(
        enabled=True, max_retries=1, max_words=120,
        absolute_tolerance=0.01, relative_tolerance=0.005,
        min_citation_coverage=1.0, retry_temperature_step=0.2,
    )


def answer(
    *,
    value,
    label="ProductCostPeriod",
    aggregate="min",
    aggregate_property="unit_cost",
    citation="ProductCostPeriod:SKU0026",
    success=True,
):
    """A text-to-query result of the shape the D3 engine really returns."""
    plan = QueryPlan(
        label=label,
        filters=(Filter(property="product_id", op="=", value="SKU0026"),),
        aggregate=aggregate,
        aggregate_property=aggregate_property,
        return_records=bool(citation),
    )
    evidence = (
        (EvidenceRecord(record_id=citation, record_type=label, text="..."),)
        if citation
        else ()
    )
    return QueryResult(
        question="What is the unit cost of product SKU0026?",
        success=success,
        plan=plan,
        cypher=plan.to_cypher(),
        answer={"value": value} if value is not None else {},
        evidence=evidence,
        attempts=1,
    )


class TestShapeGuard:
    """
    An attested number can still be the wrong number. These guard the join.
    """

    def test_a_count_is_not_a_price(self):
        """
        Regression for a real failure in the end-to-end run: asked for a
        product's list price, the engine returned count(n) over
        SellableProduct. The value 1 was correctly computed by the graph and
        attributed to a real record -- and would have entered the narrative's
        allowed set as a price.
        """
        counted = answer(
            value=1, label="SellableProduct", aggregate="count",
            aggregate_property=None, citation="SellableProduct:SKU0026",
        )

        assert fact_from_answer(counted, key="k", label_en="list price") is None

    def test_an_aggregate_over_a_property_is_accepted(self):
        fact = fact_from_answer(answer(value=255.75), key="k", label_en="unit cost")

        assert fact is not None
        assert fact.fact.value == 255.75

    def test_a_count_is_accepted_when_a_count_was_asked_for(self):
        counted = answer(
            value=3, aggregate="count", aggregate_property=None,
            citation="Quote:OPP00060",
        )

        assert fact_from_answer(counted, key="k", label_en="quotes", kind="count")

    def test_a_failed_query_contributes_nothing(self):
        assert fact_from_answer(answer(value=1.0, success=False), key="k", label_en="x") is None

    def test_an_answer_with_no_value_contributes_nothing(self):
        assert fact_from_answer(answer(value=None), key="k", label_en="x") is None

    def test_a_boolean_is_not_read_as_a_quantity(self):
        assert fact_from_answer(answer(value=True), key="k", label_en="x") is None

    def test_a_pure_aggregate_is_marked_as_query_backed(self):
        """
        count(n) over eighty products is not *about* any one of them, so it
        cites the query rather than pretending to cite a record.
        """
        fact = fact_from_answer(
            answer(value=80, aggregate="count", aggregate_property=None, citation=""),
            key="graph.products", label_en="products", kind="count",
        )

        assert fact.fact.citation == "GraphQuery:graph.products"
        assert not fact.is_record_backed


class TestOneCitationScheme:
    """
    The property that makes this a pipeline rather than three programs.
    """

    def test_track_a_handles_resolve_in_the_shared_index(self, index):
        recommendation = PriceRecommendation.load(SAMPLE_PAYLOADS, "REC-0001")

        assert index.missing(recommendation.evidence_record_ids) == ()

    def test_a_text_to_query_citation_resolves_in_the_same_index(self, index):
        """
        The handle D3 returns must be the handle the narrative can cite and
        a reader can look up -- the same string, from the same serializer.
        """
        fact = fact_from_answer(answer(value=255.75), key="k", label_en="unit cost")

        assert index.resolve(fact.fact.citation) is not None

    def test_retrieval_and_narrative_agree_on_a_handle(self, index, serializer, graph):
        node = next(n for n in graph if n.primary_label == "ProductCostPeriod")

        assert serializer.record_id(node) in index


class TestChainedFactPack:
    def test_graph_context_joins_track_a_figures(self, index):
        recommendation = PriceRecommendation.load(SAMPLE_PAYLOADS, "REC-0001")
        pack = build_facts(recommendation)
        fact = fact_from_answer(answer(value=255.75), key="graph.cost", label_en="unit cost")

        extended = pack.extended([fact.fact])

        assert 255.75 in extended.allowed_values
        assert recommendation.recommended_price in extended.allowed_values
        assert fact.fact.citation in extended.citations

    def test_an_extension_cannot_redefine_an_engine_figure(self, index):
        """
        A graph fact must never silently overwrite a number Track A sent.
        """
        recommendation = PriceRecommendation.load(SAMPLE_PAYLOADS, "REC-0001")
        pack = build_facts(recommendation)
        impostor = Fact("recommended_price", 999.99, "money", "x", "recommended price")

        extended = pack.extended([impostor])

        assert 999.99 not in extended.allowed_values
        assert recommendation.recommended_price in extended.allowed_values

    def test_every_citation_in_the_joined_pack_resolves_or_is_a_query(self, index):
        recommendation = PriceRecommendation.load(SAMPLE_PAYLOADS, "REC-0001")
        fact = fact_from_answer(answer(value=255.75), key="graph.cost", label_en="unit cost")
        pack = build_facts(recommendation).extended([fact.fact])

        unresolvable = [
            handle
            for handle in pack.citations
            if not handle.startswith(("GraphQuery:", "trackA-"))
            and handle not in index
        ]

        assert unresolvable == []


class TestEndToEnd:
    def test_a_narrative_over_the_joined_pack_verifies_and_cites(self, index, config):
        """
        The whole chain, with the model scripted: Track A's figures plus a
        graph-computed one, explained, verified, and cited with handles that
        resolve in the shared index.
        """
        recommendation = PriceRecommendation.load(SAMPLE_PAYLOADS, "REC-0001")
        fact = fact_from_answer(answer(value=255.75), key="graph.cost", label_en="unit cost")
        pack = build_facts(recommendation).extended([fact.fact])

        narrative = NarrativeGenerator(
            ScriptedLLM(
                "The pocket price is 89,489.98 SAR [PriceComponent:PC0001166] "
                "against a unit cost of 255.75 SAR [ProductCostPeriod:SKU0026]."
            ),
            config,
        ).explain(pack, evidence=index)

        assert narrative.success
        assert set(narrative.citations) <= set(pack.citations)
        for handle in narrative.citations:
            assert index.resolve(handle) is not None

    def test_a_number_from_neither_source_is_still_refused(self, index, config):
        """
        Joining two attested sources must not weaken closure: a figure from
        neither Track A nor the graph is rejected exactly as before.
        """
        recommendation = PriceRecommendation.load(SAMPLE_PAYLOADS, "REC-0001")
        fact = fact_from_answer(answer(value=255.75), key="graph.cost", label_en="unit cost")
        pack = build_facts(recommendation).extended([fact.fact])

        narrative = NarrativeGenerator(
            ScriptedLLM("The margin is 7,890.16 SAR [ProductCostPeriod:SKU0026]."),
            config,
        ).explain(pack, evidence=index)

        assert narrative.abstained
        assert "7,890.16" in narrative.verification.summary()
