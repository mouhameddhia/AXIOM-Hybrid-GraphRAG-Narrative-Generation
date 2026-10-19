"""
Tests for the verification control and right-to-left handling.

The verifier is the mechanism that turns "the model should not invent
numbers" into something enforced rather than requested, so these tests are
deliberately adversarial: they check that a fabricated figure is caught, and
equally that the things which merely *look* like figures -- record ids,
dates, Arabic-Indic digits, bidirectional control characters -- do not
produce false violations. A check that cries wolf gets loosened, and a
loosened check is how a fabricated number reaches a reader.
"""
from __future__ import annotations

import pytest

from src.narrative.facts import Fact, FactPack
from src.narrative.rtl import (
    LRI,
    PDI,
    RLM,
    apply_rtl,
    contains_rtl,
    direction_for,
    isolate_ltr_runs,
    strip_bidi_controls,
)
from src.narrative.verifier import verify


@pytest.fixture
def pack():
    """A small closed fact set: list price, one deduction, a percentage."""
    return FactPack(
        subject="OL000248",
        kind="price_waterfall",
        currency_code="SAR",
        facts=(
            Fact("list_price", 103125.00, "money", "PriceComponent:PC0001166", "list price", "سعر القائمة"),
            Fact("pocket_price", 89489.98, "money", "", "pocket price", "السعر الصافي"),
            Fact("erosion_pct", 13.2, "percent", "", "erosion", "نسبة التآكل"),
            Fact("steps", 4.0, "count", "", "deduction steps", "خطوات الخصم"),
        ),
        citations=("PriceComponent:PC0001166", "PriceComponent:PC0001167"),
        engine_version="trackA-provisional-0.1",
    )


class TestNumbers:
    def test_a_narrative_using_only_published_figures_passes(self, pack):
        text = (
            "The list price is 103,125.00 SAR [PriceComponent:PC0001166] and the "
            "pocket price is 89,489.98 SAR."
        )

        assert verify(text, pack).ok

    def test_a_fabricated_number_is_caught(self, pack):
        """The failure this whole module exists for."""
        text = "The list price is 103,125.00 SAR [PriceComponent:PC0001166], a margin of 7,890.16."

        report = verify(text, pack)

        assert not report.ok
        assert any(v.kind == "unsupported-number" for v in report.violations)
        assert "7,890.16" in report.summary()

    def test_arithmetic_the_model_performed_itself_is_caught(self, pack):
        """
        103,125.00 - 89,489.98 = 13,635.02 is *correct* arithmetic, and still
        a violation: nobody published it, so nobody can attest to it. The
        rule is not "no wrong numbers", it is "no uncomputed numbers".
        """
        text = "Erosion is 13,635.02 SAR [PriceComponent:PC0001166]."

        assert not verify(text, pack).ok

    def test_rounding_for_readability_is_accepted(self, pack):
        text = "Erosion reached 13.2% [PriceComponent:PC0001166]."

        assert verify(text, pack).ok

    def test_rounding_beyond_tolerance_is_not(self, pack):
        text = "Erosion reached 15% [PriceComponent:PC0001166]."

        assert not verify(text, pack).ok

    def test_a_figure_written_without_separators_still_matches(self, pack):
        text = "The list price is 103125.00 SAR [PriceComponent:PC0001166]."

        assert verify(text, pack).ok

    def test_record_ids_are_not_read_as_numbers(self, pack):
        """
        PC0001166 contains digits. Reading it as 1166 would fail every
        narrative that cites anything.
        """
        text = "Line OL000248 rests on PC0001166 [PriceComponent:PC0001166]."

        report = verify(text, pack)

        assert report.ok
        assert report.numbers_found == ()

    def test_dates_and_years_are_not_read_as_numbers(self, pack):
        text = "Posted on 2026-10-02, later than 2025 [PriceComponent:PC0001166]."

        assert verify(text, pack).ok


class TestArabic:
    def test_arabic_indic_digits_are_verified_identically(self, pack):
        """
        Arabic faithfulness is tested explicitly, not assumed: if these were
        not normalised, the Arabic path would enforce nothing at all.
        """
        text = "سعر القائمة ١٠٣٬١٢٥٫٠٠ ريال [PriceComponent:PC0001166]."

        assert verify(text, pack).ok

    def test_a_fabricated_number_in_arabic_indic_digits_is_still_caught(self, pack):
        text = "الهامش ٧٬٨٩٠٫١٦ ريال [PriceComponent:PC0001166]."

        report = verify(text, pack)

        assert not report.ok
        assert any(v.kind == "unsupported-number" for v in report.violations)

    def test_bidi_controls_do_not_hide_a_fabricated_number(self, pack):
        """
        An isolate between digits would split one number into two and let a
        fabrication through, so controls are stripped before extraction.
        """
        text = apply_rtl("الهامش 7,890.16 ريال [PriceComponent:PC0001166].")

        assert not verify(text, pack).ok

    def test_a_verified_arabic_narrative_survives_rtl_processing(self, pack):
        text = apply_rtl("سعر القائمة 103,125.00 ريال [PriceComponent:PC0001166].")

        assert verify(text, pack).ok


class TestCitations:
    def test_a_citation_outside_the_evidence_set_is_caught(self, pack):
        text = "The list price is 103,125.00 SAR [PriceComponent:PC9999999]."

        report = verify(text, pack)

        assert not report.ok
        assert any(v.kind == "unknown-citation" for v in report.violations)

    def test_a_narrative_citing_nothing_is_caught(self, pack):
        text = "The list price is 103,125.00 SAR."

        report = verify(text, pack)

        assert not report.ok
        assert any(v.kind == "missing-citation" for v in report.violations)

    def test_the_engine_version_is_a_valid_attribution(self, pack):
        """Engine figures have no single graph record behind them."""
        text = "The pocket price is 89,489.98 SAR [trackA-provisional-0.1]."

        assert verify(text, pack).ok

    def test_citations_are_reported_in_first_seen_order(self, pack):
        text = (
            "A 103,125.00 SAR [PriceComponent:PC0001167] and "
            "B [PriceComponent:PC0001166]."
        )

        assert verify(text, pack).citations_found == (
            "PriceComponent:PC0001167",
            "PriceComponent:PC0001166",
        )


class TestRightToLeft:
    def test_latin_runs_inside_arabic_are_isolated(self):
        out = isolate_ltr_runs("الخط OL000248 يبدأ")

        assert f"{LRI}OL000248{PDI}" in out

    def test_a_bracketed_citation_is_isolated_as_one_unit(self):
        """
        Isolating only the inside would let the brackets reorder away from
        their contents in an RTL paragraph.
        """
        out = isolate_ltr_runs("القيمة [PriceComponent:PC0001166] هنا")

        assert f"{LRI}[PriceComponent:PC0001166]{PDI}" in out

    def test_arabic_gets_an_explicit_rtl_base_direction(self):
        assert apply_rtl("سعر القائمة").startswith(RLM)

    def test_english_is_left_untouched(self):
        text = "The list price is 103,125.00 SAR [PriceComponent:PC0001166]."

        assert apply_rtl(text) == text

    def test_applying_twice_does_not_double_wrap(self):
        once = apply_rtl("الخط OL000248")

        assert apply_rtl(once) == once

    def test_stripping_controls_recovers_the_readable_text(self):
        original = "الخط OL000248 بسعر 103,125.00"

        assert strip_bidi_controls(apply_rtl(original)) == original

    def test_direction_is_derived_from_the_language(self):
        assert direction_for("ar") == "rtl"
        assert direction_for("en") == "ltr"

    def test_rtl_detection_finds_arabic_script(self):
        assert contains_rtl("سعر")
        assert not contains_rtl("price 103,125.00")


class TestCitationCoverage:
    """
    "Every narrative claim cites a specific record" (acceptance criteria).

    Before this check existed, a narrative could state twelve figures across
    four sentences and satisfy the verifier with a single citation at the end
    -- every number traceable in principle, and no way for a reader to tell
    which record supported which figure.
    """

    def test_a_narrative_citing_every_claim_passes(self, pack):
        text = (
            "The list price is 103,125.00 SAR [PriceComponent:PC0001166]. "
            "The pocket price is 89,489.98 SAR [PriceComponent:PC0001167]."
        )

        report = verify(text, pack)

        assert report.ok
        assert report.citation_coverage == 1.0
        assert (report.claims_cited, report.claims_total) == (2, 2)

    def test_a_sentence_stating_a_figure_without_a_citation_is_caught(self, pack):
        text = (
            "The list price is 103,125.00 SAR [PriceComponent:PC0001166]. "
            "That is 13.2% erosion."
        )

        report = verify(text, pack)

        assert not report.ok
        assert any(v.kind == "uncited-claim" for v in report.violations)
        assert report.citation_coverage == 0.5

    def test_one_trailing_citation_does_not_cover_a_paragraph(self, pack):
        """The exact pattern the check was added to reject."""
        text = (
            "The list price is 103,125.00 SAR. "
            "The pocket price is 89,489.98 SAR. "
            "Erosion reached 13.2% [PriceComponent:PC0001166]."
        )

        report = verify(text, pack)

        assert not report.ok
        assert report.citation_coverage < 1.0

    def test_a_sentence_with_no_figure_needs_no_citation(self, pack):
        """
        Prose that asserts nothing quantitative is not a claim, so requiring
        a citation on it would push the model to attach meaningless brackets.
        """
        text = (
            "This order line was reviewed by the pricing desk. "
            "The list price is 103,125.00 SAR [PriceComponent:PC0001166]."
        )

        report = verify(text, pack)

        assert report.ok
        assert report.claims_total == 1

    def test_a_record_id_alone_does_not_make_a_sentence_a_claim(self, pack):
        """OL000248 is full of digits but asserts no quantity."""
        text = "Line OL000248 was quoted in 2026. The list price is 103,125.00 SAR [PriceComponent:PC0001166]."

        report = verify(text, pack)

        assert report.claims_total == 1
        assert report.ok

    def test_coverage_can_be_relaxed_by_configuration(self, pack):
        text = (
            "The list price is 103,125.00 SAR [PriceComponent:PC0001166]. "
            "That is 13.2% erosion."
        )

        assert verify(text, pack, min_citation_coverage=0.5).ok
        assert not verify(text, pack, min_citation_coverage=1.0).ok

    def test_a_narrative_with_no_claims_scores_full_coverage(self, pack):
        """
        Scoring an abstention message zero would punish correct behaviour --
        it has no unsupported statements because it has no statements.
        """
        report = verify("No explanation [PriceComponent:PC0001166].", pack)

        assert report.claims_total == 0
        assert report.citation_coverage == 1.0
