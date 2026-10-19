"""
Tests for cross-encoder reranking.

Focus is on the two behaviours that make reranking safe to deploy: score
calibration (so a configured threshold means something) and fail-open
degradation (so a model outage cannot take retrieval down).
"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from src.config.settings import RerankerConfig, Settings
from src.retrieval.reranker import CrossEncoderReranker

from tests.conftest import ExplodingCrossEncoder, KeywordCrossEncoder


@pytest.fixture
def config(settings: Settings) -> RerankerConfig:
    return settings.reranker


@pytest.fixture
def reranker(config, cross_encoder):
    return CrossEncoderReranker(config, cross_encoder)


def test_reranking_reorders_by_relevance(reranker, make_result):
    """The point of the stage: a relevant record must climb."""
    candidates = [
        make_result("irrelevant", text="weather forecast sunny breeze"),
        make_result("relevant", text="list price discount margin for product"),
    ]

    reranked = reranker.rerank("list price discount", candidates)

    assert reranked[0].record_id == "relevant"


def test_sigmoid_calibrates_scores_into_zero_one(reranker, make_result):
    """
    A raw MiniLM logit is unbounded; a threshold on it is meaningless.

    Calibration is what makes ``RERANK_MIN_SCORE`` an auditable relevance floor.
    """
    reranked = reranker.rerank(
        "price", [make_result("a", text="price"), make_result("b", text="unrelated")]
    )

    for result in reranked:
        assert 0.0 < result.score < 1.0
        assert result.metadata["rerank_transform"] == "sigmoid"
        # The uncalibrated value is retained for audit.
        assert result.score == pytest.approx(
            1.0 / (1.0 + math.exp(-result.metadata["rerank_raw_score"]))
        )


def test_identity_transform_passes_raw_scores_through(config, cross_encoder, make_result):
    reranker = CrossEncoderReranker(
        replace(config, score_transform=RerankerConfig.IDENTITY), cross_encoder
    )

    reranked = reranker.rerank("price", [make_result("a", text="price")])

    assert reranked[0].score == reranked[0].metadata["rerank_raw_score"]


def test_threshold_drops_weak_candidates(config, cross_encoder, make_result):
    reranker = CrossEncoderReranker(
        replace(config, min_score=0.9, min_results=0), cross_encoder
    )

    reranked = reranker.rerank(
        "list price discount margin",
        [
            make_result("strong", text="list price discount margin"),
            make_result("weak", text="entirely unrelated content here"),
        ],
    )

    assert [result.record_id for result in reranked] == ["strong"]


def test_threshold_can_empty_the_result(config, cross_encoder, make_result):
    """
    ``min_results=0`` lets retrieval return nothing.

    That is the precondition for an honest abstention: the generator must be
    able to see that no evidence cleared the bar.
    """
    reranker = CrossEncoderReranker(
        replace(config, min_score=0.99, min_results=0), cross_encoder
    )

    assert reranker.rerank("query terms", [make_result("x", text="nothing alike")]) == []


def test_min_results_guarantees_a_non_empty_answer(config, cross_encoder, make_result):
    """A threshold set too high must degrade ranking, not silently empty it."""
    reranker = CrossEncoderReranker(
        replace(config, min_score=0.99, min_results=1), cross_encoder
    )

    reranked = reranker.rerank("query terms", [make_result("x", text="nothing alike")])

    assert len(reranked) == 1


def test_only_the_configured_head_is_scored(config, make_result):
    """The cross-encoder must never see the whole corpus."""

    class CountingEncoder(KeywordCrossEncoder):
        def __init__(self):
            super().__init__()
            self.pair_count = 0

        def predict_pairs(self, pairs):
            self.pair_count += len(pairs)
            return super().predict_pairs(pairs)

    model = CountingEncoder()
    reranker = CrossEncoderReranker(replace(config, top_n=3), model)

    reranker.rerank("query", [make_result(f"r{index}") for index in range(20)])

    assert model.pair_count == 3


def test_fail_open_degrades_to_fused_order(config, make_result):
    """A model outage must not fail a request that retrieval already satisfied."""
    reranker = CrossEncoderReranker(
        replace(config, fail_open=True), ExplodingCrossEncoder()
    )
    candidates = [make_result("a", score=0.9), make_result("b", score=0.5)]

    reranked = reranker.rerank("query", candidates)

    assert [result.record_id for result in reranked] == ["a", "b"]
    assert reranked[0].metadata["rerank_skipped"] is True
    assert "unavailable" in reranked[0].metadata["rerank_error"]


def test_fail_closed_propagates_the_error(config, make_result):
    reranker = CrossEncoderReranker(
        replace(config, fail_open=False), ExplodingCrossEncoder()
    )

    with pytest.raises(RuntimeError, match="unavailable"):
        reranker.rerank("query", [make_result("a")])


def test_mismatched_score_count_is_caught(config, make_result):
    """A backend returning the wrong number of scores must not misalign ranks."""

    class ShortEncoder:
        def predict_pairs(self, pairs):
            return [1.0]

    reranker = CrossEncoderReranker(replace(config, fail_open=False), ShortEncoder())

    with pytest.raises(ValueError, match="returned 1 scores"):
        reranker.rerank("query", [make_result("a"), make_result("b")])


def test_empty_input_returns_empty(reranker):
    assert reranker.rerank("query", []) == []


def test_metadata_records_the_rank_movement(reranker, make_result):
    """The stage's effect on each record must be measurable after the fact."""
    reranked = reranker.rerank(
        "price discount",
        [make_result("first", text="unrelated"), make_result("second", text="price discount")],
    )

    promoted = next(r for r in reranked if r.record_id == "second")
    assert promoted.metadata["pre_rerank_position"] == 2
    assert "fused_score" in promoted.metadata
    assert promoted.source.endswith("|reranked")


def test_ties_preserve_fused_order(reranker, make_result):
    """Equally relevant candidates must not be reshuffled."""
    candidates = [make_result("a", text="same"), make_result("b", text="same")]

    reranked = reranker.rerank("query", candidates)

    assert [result.record_id for result in reranked] == ["a", "b"]
