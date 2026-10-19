"""
Tests for the Track A result contract.

This is a boundary between two people's code, and the contract is still
provisional, so the tests here are mostly about *rejection*: a payload that
drifts must fail loudly at the boundary rather than flow into a narrative and
surface as a confident sentence about a missing number.
"""
from __future__ import annotations

import json

import pytest

from src.config.settings import PROJECT_ROOT
from src.narrative.contract import ContractError, PriceRange, PriceRecommendation

SAMPLE_PAYLOADS = PROJECT_ROOT / "data" / "track_a_sample_payloads.json"


def valid_payload(**overrides):
    payload = {
        "recommendation_id": "REC-TEST",
        "product_id": "SKU0026",
        "customer_id": "CUST0009",
        "currency_code": "SAR",
        "recommended_price": 378.0,
        "price_range": {"low": 362.0, "high": 395.0},
        "confidence": 0.78,
        "expected_margin_pct": 32.3,
        "acceptance_probability": 0.64,
        "evidence_record_ids": ["SellableProduct:SKU0026"],
    }
    payload.update(overrides)
    return payload


class TestParsing:
    def test_a_well_formed_payload_round_trips(self):
        rec = PriceRecommendation.from_dict(valid_payload())

        assert rec.recommended_price == 378.0
        assert rec.price_range.low == 362.0
        assert rec.evidence_record_ids == ("SellableProduct:SKU0026",)

    def test_nothing_is_recomputed_on_the_way_in(self):
        """
        The contract reads figures; it never derives them.

        A payload whose range does not straddle its recommended price is
        still parsed as given -- correcting it here would mean this layer had
        an opinion about Track A's numbers, which is exactly the boundary the
        track's rule draws.
        """
        rec = PriceRecommendation.from_dict(
            valid_payload(recommended_price=500.0, price_range={"low": 10.0, "high": 20.0})
        )

        assert rec.recommended_price == 500.0
        assert (rec.price_range.low, rec.price_range.high) == (10.0, 20.0)

    def test_engine_version_defaults_when_absent(self):
        assert PriceRecommendation.from_dict(valid_payload()).engine_version == "unspecified"


class TestRejection:
    def test_payload_without_evidence_is_refused(self):
        """A recommendation citing nothing cannot be explained with citations."""
        with pytest.raises(ContractError, match="evidence_record_ids"):
            PriceRecommendation.from_dict(valid_payload(evidence_record_ids=[]))

    def test_evidence_must_not_be_a_bare_string(self):
        with pytest.raises(ContractError, match="list of citation handles"):
            PriceRecommendation.from_dict(
                valid_payload(evidence_record_ids="SellableProduct:SKU0026")
            )

    @pytest.mark.parametrize("field", ["recommendation_id", "product_id", "currency_code"])
    def test_missing_string_fields_are_named_in_the_error(self, field):
        payload = valid_payload()
        del payload[field]

        with pytest.raises(ContractError, match=field):
            PriceRecommendation.from_dict(payload)

    def test_a_probability_outside_zero_to_one_is_refused(self):
        with pytest.raises(ContractError, match="probability in \\[0, 1\\]"):
            PriceRecommendation.from_dict(valid_payload(confidence=1.4))

    def test_a_boolean_is_not_accepted_as_a_number(self):
        """bool subclasses int; True must never be read as the price 1."""
        with pytest.raises(ContractError, match="recommended_price"):
            PriceRecommendation.from_dict(valid_payload(recommended_price=True))

    def test_an_inverted_price_range_is_refused(self):
        with pytest.raises(ContractError, match="above"):
            PriceRange(low=400.0, high=100.0)

    def test_missing_price_range_is_refused(self):
        payload = valid_payload()
        del payload["price_range"]

        with pytest.raises(ContractError, match="price_range"):
            PriceRecommendation.from_dict(payload)


class TestLoading:
    def test_the_shipped_sample_payloads_are_valid(self):
        """The provisional stand-in must itself satisfy the contract."""
        rec = PriceRecommendation.load(SAMPLE_PAYLOADS, "REC-0001")

        assert rec.product_id == "SKU0026"
        assert rec.evidence_record_ids

    def test_loading_without_an_id_takes_the_first(self):
        assert PriceRecommendation.load(SAMPLE_PAYLOADS).recommendation_id == "REC-0001"

    def test_an_unknown_id_lists_what_is_available(self):
        with pytest.raises(ContractError, match="REC-0001"):
            PriceRecommendation.load(SAMPLE_PAYLOADS, "REC-NOPE")

    def test_a_missing_file_is_a_contract_error(self, tmp_path):
        with pytest.raises(ContractError, match="not found"):
            PriceRecommendation.load(tmp_path / "absent.json")

    def test_malformed_json_names_the_line(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(ContractError, match="not valid JSON"):
            PriceRecommendation.load(path)

    def test_a_single_payload_file_loads_without_a_collection_wrapper(self, tmp_path):
        path = tmp_path / "one.json"
        path.write_text(json.dumps(valid_payload()), encoding="utf-8")

        assert PriceRecommendation.load(path).recommendation_id == "REC-TEST"


def valid_waterfall(**overrides):
    """A walk that adds up: 1000 - 100 - 50 (on) - 40 (off) = 810."""
    waterfall = {
        "list_price": 1000.00,
        "list_price_record_id": "PriceComponent:PC0000001",
        "steps": [
            {"name": "invoice_discount", "amount": -100.00, "on_invoice": True,
             "record_id": "PriceComponent:PC0000002"},
            {"name": "freight", "amount": -50.00, "on_invoice": True,
             "record_id": "PriceComponent:PC0000003"},
            {"name": "volume_rebate", "amount": -40.00, "on_invoice": False,
             "record_id": "PriceComponent:PC0000004"},
        ],
        "invoice_price": 850.00,
        "pocket_price": 810.00,
        "total_deductions": -190.00,
        "erosion_pct": 19.0,
    }
    waterfall.update(overrides)
    return waterfall


class TestWaterfallFromTrackA:
    """
    The waterfall arrives *computed*. This layer reads it and checks it
    against itself; it never derives a figure for the narrative to quote.
    """

    def test_a_consistent_waterfall_parses(self):
        rec = PriceRecommendation.from_dict(valid_payload(waterfall=valid_waterfall()))

        assert rec.waterfall is not None
        assert rec.waterfall.pocket_price == 810.00
        assert len(rec.waterfall.steps) == 3

    def test_the_walk_is_read_not_recomputed(self):
        """
        Track A's stated subtotals are what the narrative gets, even when a
        recomputation would round differently. This layer does not own them.
        """
        rec = PriceRecommendation.from_dict(valid_payload(waterfall=valid_waterfall()))

        assert rec.waterfall.invoice_price == 850.00
        assert rec.waterfall.total_deductions == -190.00

    def test_steps_are_partitioned_by_stage(self):
        waterfall = PriceRecommendation.from_dict(
            valid_payload(waterfall=valid_waterfall())
        ).waterfall

        assert len(waterfall.on_invoice_steps) == 2
        assert len(waterfall.off_invoice_steps) == 1

    def test_a_waterfall_that_does_not_add_up_is_refused(self):
        """
        An inconsistent payload is refused, not repaired: correcting it would
        mean this layer had taken ownership of a pricing figure.
        """
        with pytest.raises(ContractError, match="does not add up"):
            PriceRecommendation.from_dict(
                valid_payload(waterfall=valid_waterfall(pocket_price=999.00))
            )

    def test_an_inconsistent_invoice_price_is_named_precisely(self):
        with pytest.raises(ContractError, match="invoice_price"):
            PriceRecommendation.from_dict(
                valid_payload(waterfall=valid_waterfall(invoice_price=700.00))
            )

    def test_an_inconsistent_total_is_caught(self):
        with pytest.raises(ContractError, match="total_deductions"):
            PriceRecommendation.from_dict(
                valid_payload(waterfall=valid_waterfall(total_deductions=-1.00))
            )

    def test_a_waterfall_with_no_steps_is_refused(self):
        with pytest.raises(ContractError, match="non-empty 'steps'"):
            PriceRecommendation.from_dict(valid_payload(waterfall=valid_waterfall(steps=[])))

    def test_on_invoice_accepts_the_string_form(self):
        """
        The JSON export writes "true" where Neo4j stores a real boolean;
        a payload assembled from either source may carry either form.
        """
        waterfall = valid_waterfall()
        for step in waterfall["steps"]:
            step["on_invoice"] = "true" if step["on_invoice"] else "false"

        parsed = PriceRecommendation.from_dict(
            valid_payload(waterfall=waterfall)
        ).waterfall

        assert len(parsed.on_invoice_steps) == 2

    def test_a_non_boolean_on_invoice_is_refused(self):
        waterfall = valid_waterfall()
        waterfall["steps"][0]["on_invoice"] = "yes"

        with pytest.raises(ContractError, match="boolean"):
            PriceRecommendation.from_dict(valid_payload(waterfall=waterfall))

    def test_record_ids_cover_the_whole_walk(self):
        waterfall = PriceRecommendation.from_dict(
            valid_payload(waterfall=valid_waterfall())
        ).waterfall

        assert len(waterfall.record_ids) == 4
        assert waterfall.record_ids[0] == "PriceComponent:PC0000001"

    def test_a_payload_without_a_waterfall_is_still_valid(self):
        """Not every recommendation has an existing line to walk."""
        assert PriceRecommendation.from_dict(valid_payload()).waterfall is None

    def test_the_shipped_sample_waterfall_is_consistent(self):
        """The provisional stand-in must satisfy its own contract."""
        rec = PriceRecommendation.load(SAMPLE_PAYLOADS, "REC-0001")

        assert rec.waterfall is not None
        assert rec.waterfall.consistency_error() is None
