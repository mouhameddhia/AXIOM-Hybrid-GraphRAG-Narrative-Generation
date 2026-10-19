"""
Tests for the fact pack.

The pack is the closed set a narrative may draw on, so the property under
test throughout is *closure*: every number the pack publishes traces to a
field Track A sent, and no number appears that Track A did not send.
"""
from __future__ import annotations

from src.narrative.contract import PriceRecommendation
from src.narrative.facts import build_facts
from tests.test_narrative_contract import valid_payload, valid_waterfall


def pack_for(**overrides):
    return build_facts(PriceRecommendation.from_dict(valid_payload(**overrides)))


class TestRecommendationFacts:
    def test_engine_figures_are_published_unchanged(self):
        pack = pack_for(recommended_price=378.0)

        assert 378.0 in pack.allowed_values

    def test_ratios_become_percentages_exactly_once(self):
        """One quantity, one numeric form -- every extra form is one a model
        could reach by accident and the verifier would then have to admit."""
        pack = pack_for(confidence=0.78)

        assert 78.0 in pack.allowed_values
        assert 0.78 not in pack.allowed_values

    def test_engine_figures_are_attributed_to_the_engine(self):
        pack = pack_for(engine_version="trackA-test-1")
        engine_facts = [f for f in pack.facts if f.key == "recommended_price"]

        assert engine_facts[0].citation == "trackA-test-1"

    def test_the_evidence_handles_become_citations(self):
        pack = pack_for(evidence_record_ids=["TradeCustomer:CUST0009"])

        assert "TradeCustomer:CUST0009" in pack.citations


class TestWaterfallFacts:
    def test_track_a_subtotals_are_published_verbatim(self):
        pack = pack_for(waterfall=valid_waterfall())

        assert 850.00 in pack.allowed_values, "invoice price"
        assert 810.00 in pack.allowed_values, "pocket price"
        assert 1000.00 in pack.allowed_values, "list price"

    def test_deductions_are_published_as_magnitudes(self):
        """Publishing both -100.0 and 100.0 would widen the admitted set for
        no benefit, since the narrative states sizes, not signed deltas."""
        pack = pack_for(waterfall=valid_waterfall())

        assert all(value >= 0 for value in pack.allowed_values)

    def test_step_labels_name_the_stage_they_belong_to(self):
        """
        Without this the model files an off-invoice rebate under the invoice
        price: every number traceable, the attribution wrong.
        """
        pack = pack_for(waterfall=valid_waterfall())
        labels = [f.label_en for f in pack.facts if f.key.startswith("step.")]

        assert any("(shown on the invoice)" in label for label in labels)
        assert any("NOT shown on the invoice" in label for label in labels)

    def test_step_citations_come_from_the_payload(self):
        pack = pack_for(waterfall=valid_waterfall())

        assert "PriceComponent:PC0000002" in pack.citations
        assert "PriceComponent:PC0000001" in pack.citations

    def test_a_pack_without_a_waterfall_publishes_only_engine_figures(self):
        pack = pack_for()

        assert not any(f.key.startswith("step.") for f in pack.facts)
        assert pack.kind == "price_recommendation"

    def test_a_pack_with_a_waterfall_is_marked_as_such(self):
        assert pack_for(waterfall=valid_waterfall()).kind == "recommendation_and_waterfall"

    def test_no_number_appears_that_track_a_did_not_send(self):
        """
        The closure property, asserted directly: every published value must
        be findable somewhere in the original payload.
        """
        payload = valid_payload(waterfall=valid_waterfall())
        pack = build_facts(PriceRecommendation.from_dict(payload))

        sent = {
            payload["recommended_price"],
            payload["price_range"]["low"],
            payload["price_range"]["high"],
            round(payload["confidence"] * 100, 1),
            payload["expected_margin_pct"],
            round(payload["acceptance_probability"] * 100, 1),
            payload["waterfall"]["list_price"],
            payload["waterfall"]["invoice_price"],
            payload["waterfall"]["pocket_price"],
            abs(payload["waterfall"]["total_deductions"]),
            payload["waterfall"]["erosion_pct"],
            float(len(payload["waterfall"]["steps"])),
            *(abs(s["amount"]) for s in payload["waterfall"]["steps"]),
        }

        assert set(pack.allowed_values) <= sent


class TestRendering:
    def test_the_fact_table_shows_values_and_citations(self):
        rendered = pack_for(waterfall=valid_waterfall()).render("en")

        assert "1,000.00 SAR" in rendered
        assert "[PriceComponent:PC0000001]" in rendered

    def test_percentages_render_with_one_decimal(self):
        assert "78.0%" in pack_for(confidence=0.78).render("en")

    def test_arabic_labels_are_kept_for_the_deferred_path(self):
        """Arabic is not this phase's focus, but the labels stay beside the
        English ones so re-enabling it cannot introduce a second set of
        numbers."""
        assert "سعر القائمة" in pack_for(waterfall=valid_waterfall()).render("ar")
