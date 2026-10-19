"""
Tests for Reciprocal Rank Fusion.

The properties asserted here are the ones the architecture depends on:
arbitrary stream count, configuration-driven weighting, cross-stream agreement
beating single-stream depth, and reproducible tie-breaking.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.config.settings import FusionConfig, Settings
from src.retrieval.base import RetrievalStream
from src.retrieval.fusion import ReciprocalRankFuser, reciprocal_rank_fusion


@pytest.fixture
def fusion_config(settings: Settings) -> FusionConfig:
    return settings.fusion


@pytest.fixture
def fuser(fusion_config: FusionConfig) -> ReciprocalRankFuser:
    return ReciprocalRankFuser(fusion_config)


def test_agreement_across_streams_beats_depth_in_one(fuser, make_result):
    """A record two streams rank second should beat one a single stream ranks first."""
    stream_a = RetrievalStream.of("a", [make_result("solo"), make_result("shared")])
    stream_b = RetrievalStream.of("b", [make_result("other"), make_result("shared")])

    fused = fuser.fuse([stream_a, stream_b])

    assert fused[0].record_id == "shared"
    assert fused[0].source == "a+b"
    assert fused[0].metadata["stream_agreement"] == 2
    assert fused[0].metadata["stream_ranks"] == {"a": 2, "b": 2}


@pytest.mark.parametrize("stream_count", [1, 2, 3, 5, 8])
def test_accepts_arbitrary_stream_count(fuser, make_result, stream_count):
    """The fuser is not limited to the three named modalities."""
    streams = [
        RetrievalStream.of(
            f"stream_{index}", [make_result("common"), make_result(f"record_{index}")]
        )
        for index in range(stream_count)
    ]

    fused = fuser.fuse(streams)

    assert fused[0].record_id == "common"
    assert fused[0].metadata["stream_agreement"] == stream_count
    assert len(fused) == stream_count + 1


def test_single_stream_preserves_order(fuser, make_result):
    results = [make_result("a"), make_result("b"), make_result("c")]

    fused = fuser.fuse([RetrievalStream.of("only", results)])

    assert [result.record_id for result in fused] == ["a", "b", "c"]


def test_empty_streams_return_empty(fuser):
    assert fuser.fuse([]) == []
    assert fuser.fuse([RetrievalStream.of("a", []), RetrievalStream.of("b", [])]) == []


def test_weights_shift_the_ranking(fusion_config, make_result):
    """A configured weight must be able to outrank positional advantage."""
    streams = [
        RetrievalStream.of("weak", [make_result("from_weak")]),
        RetrievalStream.of("strong", [make_result("from_strong")]),
    ]

    unweighted = ReciprocalRankFuser(fusion_config).fuse(streams)
    # Equal rank, equal weight -> tie broken alphabetically on record_id.
    assert [result.record_id for result in unweighted] == ["from_strong", "from_weak"]

    weighted = ReciprocalRankFuser(
        replace(fusion_config, weights={"weak": 5.0, "strong": 1.0})
    ).fuse(streams)
    assert weighted[0].record_id == "from_weak"


def test_zero_weight_excludes_a_stream(fusion_config, make_result):
    """A stream weighted to zero must contribute nothing at all."""
    fuser = ReciprocalRankFuser(replace(fusion_config, weights={"muted": 0.0}))

    fused = fuser.fuse(
        [
            RetrievalStream.of("muted", [make_result("muted_record")]),
            RetrievalStream.of("live", [make_result("live_record")]),
        ]
    )

    assert [result.record_id for result in fused] == ["live_record"]


def test_rrf_k_controls_head_flattening(fusion_config, make_result):
    """
    A small k sharpens the head; a large k flattens it.

    ``deep_in_both`` sits at rank 5 in two streams, ``top_of_one`` at rank 1 in
    one. With k=1 the rank-1 advantage wins (1/2 > 2/6); with k=60 the curve is
    flat enough that agreement wins (2/65 > 1/61). This is the knob that decides
    whether cross-modal consensus or single-modality confidence dominates.
    """
    # Filler is stream-local: shared filler would itself accumulate agreement
    # weight and confound what this test is isolating.
    def filler(stream: str) -> list:
        return [make_result(f"filler_{stream}_{index}") for index in range(3)]

    streams = [
        RetrievalStream.of(
            "a", [make_result("top_of_one"), *filler("a"), make_result("deep_in_both")]
        ),
        RetrievalStream.of(
            "b", [make_result("other"), *filler("b"), make_result("deep_in_both")]
        ),
    ]

    def position(fused, record_id: str) -> int:
        return [result.record_id for result in fused].index(record_id)

    sharp = ReciprocalRankFuser(replace(fusion_config, rrf_k=1)).fuse(streams)
    flat = ReciprocalRankFuser(replace(fusion_config, rrf_k=60)).fuse(streams)

    # Relative order is the claim; absolute position also depends on the other
    # rank-1 records, which are not what this test is isolating.
    assert position(sharp, "top_of_one") < position(sharp, "deep_in_both")
    assert position(flat, "deep_in_both") < position(flat, "top_of_one")


def test_score_matches_the_rrf_formula(fusion_config, make_result):
    """The fused score must be exactly the weighted reciprocal-rank sum."""
    fuser = ReciprocalRankFuser(fusion_config)
    k = fusion_config.rrf_k

    fused = fuser.fuse(
        [
            RetrievalStream.of("a", [make_result("x"), make_result("target")]),
            RetrievalStream.of("b", [make_result("target")]),
        ]
    )

    target = next(result for result in fused if result.record_id == "target")
    assert target.score == pytest.approx(1.0 / (k + 2) + 1.0 / (k + 1))


def test_max_candidates_truncates(fusion_config, make_result):
    fuser = ReciprocalRankFuser(replace(fusion_config, max_candidates=3))

    fused = fuser.fuse(
        [RetrievalStream.of("a", [make_result(f"r{index}") for index in range(10)])]
    )

    assert len(fused) == 3


def test_tie_breaking_is_deterministic(fuser, make_result):
    """Identical scores must resolve identically on every run."""
    streams = [
        RetrievalStream.of("a", [make_result("zebra")]),
        RetrievalStream.of("b", [make_result("alpha")]),
    ]

    orderings = {tuple(r.record_id for r in fuser.fuse(streams)) for _ in range(5)}

    assert orderings == {("alpha", "zebra")}


def test_functional_wrapper_infers_stream_names(fusion_config, make_result):
    """The list-of-lists convenience API still works for ad-hoc callers."""
    fused = reciprocal_rank_fusion(
        [
            [make_result("shared", source="dense"), make_result("only_dense", source="dense")],
            [make_result("shared", source="bm25")],
        ],
        fusion_config,
    )

    assert fused[0].record_id == "shared"
    assert fused[0].source == "dense+bm25"
