"""
Tests for resolving Track A's citations against the graph.

This runs against the real PoC export. The property that matters is that a
handle Track A sends either resolves to a record a reader could look up, or
is reported as missing -- a citation that silently points at nothing is worse
than no citation, because it looks checkable.
"""
from __future__ import annotations

import pytest

from src.narrative.contract import PriceRecommendation
from src.narrative.evidence import EvidenceIndex
from src.retrieval.corpus import LabelIdentityPolicy, RecordSerializer
from tests.test_narrative_contract import SAMPLE_PAYLOADS


@pytest.fixture(scope="module")
def serializer(graph, settings):
    return RecordSerializer(
        settings.corpus, LabelIdentityPolicy.learn(graph, settings.corpus)
    )


@pytest.fixture(scope="module")
def index(graph, serializer):
    return EvidenceIndex.build(graph, serializer)


class TestResolution:
    def test_the_index_covers_the_graph(self, index, graph):
        assert len(index) > 0
        assert len(index) <= len(graph)

    def test_a_real_handle_resolves_to_its_record(self, index):
        record = index.resolve("OrderLine:OL000248")

        assert record is not None
        assert record.record_type == "OrderLine"
        assert "OL000248" in record.text

    def test_an_unknown_handle_resolves_to_nothing(self, index):
        assert index.resolve("OrderLine:OL999999") is None

    def test_membership_matches_resolution(self, index):
        assert "OrderLine:OL000248" in index
        assert "OrderLine:OL999999" not in index

    def test_resolve_all_keeps_the_order_given(self, index):
        handles = ["TradeCustomer:CUST0009", "OrderLine:OL000248"]

        resolved = index.resolve_all(handles)

        assert [r.record_id for r in resolved] == handles

    def test_resolve_all_drops_what_it_cannot_find(self, index):
        resolved = index.resolve_all(["OrderLine:OL000248", "OrderLine:OL999999"])

        assert len(resolved) == 1

    def test_missing_reports_exactly_the_unresolvable_handles(self, index):
        missing = index.missing(["OrderLine:OL000248", "Nope:XXX", "Also:YYY"])

        assert missing == ("Nope:XXX", "Also:YYY")

    def test_a_preview_is_bounded(self, index):
        record = index.resolve("OrderLine:OL000248")

        assert len(record.preview(width=40)) <= 40


class TestAgainstTheShippedPayloads:
    def test_every_citation_in_the_sample_payloads_resolves(self, index):
        """
        The provisional payloads carry invented figures but real citations.
        If this breaks, the sample file has drifted from the graph and the
        demo would cite records a reviewer cannot look up.
        """
        for rec_id in ("REC-0001", "REC-0002"):
            rec = PriceRecommendation.load(SAMPLE_PAYLOADS, rec_id)
            handles = list(rec.evidence_record_ids) + list(rec.waterfall.record_ids)

            assert index.missing(handles) == (), f"{rec_id} cites missing records"

    def test_citation_handles_match_the_retrieval_scheme(self, index, serializer, graph):
        """
        One citation scheme across retrieval, text-to-query and narrative is
        what lets a reader follow a sentence back to one exact record.
        """
        node = next(n for n in graph if n.primary_label == "OrderLine")

        assert serializer.record_id(node) in index
