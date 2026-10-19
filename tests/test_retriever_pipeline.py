"""
End-to-end tests for the hybrid retrieval pipeline.

Runs the real stack over the real knowledge graph with deterministic stand-ins
for the two transformers, so the whole path -- anchor resolution, three
modalities, fusion, reranking -- is exercised without a network call.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.pipeline.factory import build_retrieval_stack
from src.pipeline.retriever_pipeline import HybridRetrievalPipeline
from src.retrieval.base import RetrievalQuery, RetrievalResult, RetrievalStream
from src.retrieval.fusion import ReciprocalRankFuser


@pytest.fixture
def stack(settings, graph_path, encoder, cross_encoder):
    """The full stack, wired from configuration with injected fake models."""
    return build_retrieval_stack(settings, encoder=encoder, cross_encoder=cross_encoder)


# -- assembly ---------------------------------------------------------------


def test_stack_assembles_three_modalities(stack):
    assert stack.pipeline.stream_names == ("bm25", "dense", "graph")
    assert len(stack.corpus) == stack.graph.stats.node_count


def test_stack_runs_without_a_database(stack):
    """The whole point of the JSON source: no Neo4j required."""
    outcome = stack.pipeline.run("What is the list price for SKU0001?", top_k=5)
    assert outcome.final


def test_dense_modality_is_optional(settings, cross_encoder):
    """Fusion tolerates any stream count, so a modality can be dropped."""
    stack = build_retrieval_stack(settings, cross_encoder=cross_encoder, enable_dense=False)

    assert stack.pipeline.stream_names == ("bm25", "graph")
    assert stack.pipeline.run("list price SKU0001", top_k=5).final


# -- retrieval quality ------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected_record_id"),
    [
        ("What is the list price for product SKU0001?", "ListPricePeriod:SKU0001"),
        ("Which competitor quoted against OPP00003?", "CompetitorPresence:OPP00003"),
        ("What discount was applied to order line OL000001?", "PriceComponent:PC0000001"),
    ],
)
def test_expected_record_is_retrieved(stack, question, expected_record_id):
    """
    The record that actually answers the question must appear in the top-k.

    This is an end-to-end wiring check, not a quality benchmark: it runs on a
    deliberately weak stand-in encoder, so it proves the pipeline connects
    anchors, modalities, fusion and reranking correctly -- it does not measure
    how good retrieval is. Quality is measured by ``evaluate_retrieval.py``
    against the gold set with the real models.

    Expectations are asserted to exist in the corpus first, so a schema change
    fails as a clear fixture error rather than as a misleading recall drop.
    """
    assert stack.corpus.by_record_id(expected_record_id) is not None

    outcome = stack.pipeline.run(question, top_k=10)

    assert expected_record_id in outcome.record_ids


def test_anchors_are_resolved_from_the_question(stack):
    outcome = stack.pipeline.run("Details for customer CUST0015", top_k=5)

    assert outcome.anchors
    assert all(anchor.mention.upper() == "CUST0015" for anchor in outcome.anchors)


def test_question_without_identifiers_still_retrieves(stack):
    """
    A purely conceptual question yields no anchors, so the graph modality sits
    out -- and the remaining two must still answer.
    """
    outcome = stack.pipeline.run("pricing discount policy", top_k=5)

    assert outcome.anchors == ()
    assert outcome.stream("graph").results == ()
    assert outcome.final


def test_cross_modal_agreement_is_recorded(stack):
    """Records found by several modalities must be marked as such."""
    outcome = stack.pipeline.run("list price for product SKU0001", top_k=10)

    agreements = [
        result.metadata.get("stream_agreement", 1) for result in outcome.fused
    ]
    assert max(agreements) > 1


# -- trace and contract -----------------------------------------------------


def test_every_stage_is_preserved_in_the_trace(stack):
    """Intermediate ranks are the audit record, not debug output."""
    outcome = stack.pipeline.run("list price SKU0001", top_k=5)

    assert len(outcome.streams) == 3
    assert outcome.fused
    assert outcome.reranked
    assert {"anchors", "bm25", "dense", "graph", "fusion", "rerank", "total"} <= set(
        outcome.timings_ms
    )


def test_final_results_are_citable_corpus_records(stack):
    """Every returned handle must resolve, or a citation cannot be verified."""
    outcome = stack.pipeline.run("competitor price for OPP00003", top_k=5)

    for result in outcome.final:
        assert stack.corpus.by_record_id(result.record_id) is not None


def test_top_k_is_respected(stack):
    assert len(stack.pipeline.run("price", top_k=3).final) <= 3


def test_pipeline_is_deterministic(stack):
    runs = {
        tuple(stack.pipeline.run("list price SKU0001", top_k=5).record_ids)
        for _ in range(3)
    }
    assert len(runs) == 1


def test_streams_fan_out_wider_than_top_k(stack, settings):
    """
    Each modality retrieves deeper than the answer needs.

    This is what gives fusion room to promote a record that several streams
    rank moderately but none ranks in its own top-k.
    """
    top_k = 2
    outcome = stack.pipeline.run("price product order customer", top_k=top_k)

    widest = max(len(stream) for stream in outcome.streams)
    assert widest > top_k
    assert len(outcome.final) <= top_k


# -- resilience -------------------------------------------------------------


def test_a_failing_modality_does_not_fail_the_request(stack, settings):
    """Partial evidence beats an error; the failure is logged, not hidden."""

    class BrokenRetriever:
        name = "broken"

        def search(self, query: RetrievalQuery, top_k: int):
            raise RuntimeError("modality offline")

    pipeline = HybridRetrievalPipeline(
        retrievers=[*stack.retrievers, BrokenRetriever()],
        fuser=ReciprocalRankFuser(settings.fusion),
        config=settings.retrieval,
        anchor_resolver=stack.anchor_resolver,
    )

    outcome = pipeline.run("list price SKU0001", top_k=5)

    assert outcome.final
    assert outcome.stream("broken").results == ()


def test_duplicate_retriever_names_are_rejected(stack, settings):
    """Two streams sharing a name would silently merge in fusion weighting."""
    with pytest.raises(ValueError, match="unique"):
        HybridRetrievalPipeline(
            retrievers=[stack.retrievers[0], stack.retrievers[0]],
            fuser=ReciprocalRankFuser(settings.fusion),
            config=settings.retrieval,
        )


def test_empty_retriever_list_is_rejected(settings):
    with pytest.raises(ValueError, match="At least one retriever"):
        HybridRetrievalPipeline(
            retrievers=[],
            fuser=ReciprocalRankFuser(settings.fusion),
            config=settings.retrieval,
        )


def test_invalid_top_k_is_rejected(stack):
    with pytest.raises(ValueError, match="top_k"):
        stack.pipeline.run("anything", top_k=0)


def test_reranking_can_be_disabled(settings, encoder):
    """Without a reranker the pipeline returns fused order, not an error."""
    disabled = replace(settings, reranker=replace(settings.reranker, enabled=False))
    stack = build_retrieval_stack(disabled, encoder=encoder)

    outcome = stack.pipeline.run("list price SKU0001", top_k=5)

    assert outcome.reranked == ()
    assert outcome.final == outcome.fused
    assert len(outcome.final) <= 5


def test_a_fourth_modality_needs_no_pipeline_change(stack, settings):
    """
    The extensibility claim, asserted rather than asserted-in-prose.

    A new stream is registered and weighted purely through configuration.
    """

    class ConstantRetriever:
        name = "synthetic"

        def search(self, query: RetrievalQuery, top_k: int):
            return [
                RetrievalResult(
                    record_id=stack.corpus[0].record_id,
                    record_type=stack.corpus[0].record_type,
                    text=stack.corpus[0].text,
                    score=1.0,
                    source=self.name,
                )
            ]

    weighted = replace(
        settings.fusion, weights={"synthetic": 10.0, "bm25": 0.1, "dense": 0.1, "graph": 0.1}
    )
    pipeline = HybridRetrievalPipeline(
        retrievers=[*stack.retrievers, ConstantRetriever()],
        fuser=ReciprocalRankFuser(weighted),
        config=settings.retrieval,
        anchor_resolver=stack.anchor_resolver,
    )

    outcome = pipeline.run("list price SKU0001", top_k=5)

    assert "synthetic" in pipeline.stream_names
    assert outcome.fused[0].record_id == stack.corpus[0].record_id
