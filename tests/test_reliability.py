"""
Tests for the reliability layer (D5).

Every gate in this layer can only ever *withhold* an answer that D4 would
have served, so the tests are written from that direction: each one asserts
either that a specific defect is caught, or that healthy output is left
alone. A safeguard that fires on correct narratives is worse than no
safeguard, because it gets switched off.

The language model is scripted throughout. Disagreement between samples and
misattribution within one cannot be provoked on demand from a real model,
and a k=3 loop against Ollama would make the suite minutes long.
"""
from __future__ import annotations

import pytest

from src.config.settings import ConfigError, NarrativeConfig, ReliabilityConfig
from src.narrative.contract import PriceRecommendation
from src.narrative.evidence import EvidenceIndex, ResolvedRecord
from src.narrative.facts import Fact, FactPack, build_facts
from src.narrative.generator import NarrativeGenerator
from src.reliability.attribution import check_attribution
from src.reliability.consistency import assess_consistency
from src.reliability.corrective import apply_corrective_retrieval, assess_evidence
from src.reliability.narrator import ReliableNarrator
from tests.test_narrative_contract import SAMPLE_PAYLOADS

EVAL_PAYLOADS = SAMPLE_PAYLOADS.parent / "track_a_payloads_eval.json"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def pack():
    """A small pack with the roles the attribution checker knows."""
    return FactPack(
        subject="OL000248",
        kind="recommendation_and_waterfall",
        currency_code="SAR",
        facts=(
            Fact("list_price", 103125.00, "money", "PriceComponent:PC0001166", "list price"),
            Fact("invoice_price", 95350.07, "money", "", "invoice price"),
            Fact("pocket_price", 89489.98, "money", "", "pocket price actually kept"),
            Fact("step.volume_rebate", 1534.35, "money", "PriceComponent:PC0001168",
                 "volume rebate (NOT shown on the invoice)"),
        ),
        citations=("PriceComponent:PC0001166", "PriceComponent:PC0001168"),
        engine_version="trackA-provisional-0.1",
    )


@pytest.fixture
def narrative_config():
    return NarrativeConfig(
        enabled=True, max_retries=0, max_words=120,
        absolute_tolerance=0.01, relative_tolerance=0.005,
        min_citation_coverage=1.0, retry_temperature_step=0.2,
    )


@pytest.fixture
def reliability_config():
    return ReliabilityConfig(
        enabled=True, samples=3, min_verified=2, min_agreement=0.6,
        sample_temperature=0.7, enforce_attribution=True, corrective_enabled=True,
        min_support_ratio=0.75, corrective_top_k=6, corrective_max_added=4,
    )


class CyclingLLM:
    """Returns each scripted response in turn, then repeats the last."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.temperatures: list[float | None] = []

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        self.temperatures.append(temperature)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


GOOD = (
    "The list price is 103,125.00 SAR [PriceComponent:PC0001166]. "
    "A volume rebate of 1,534.35 SAR applies [PriceComponent:PC0001168]."
)
GOOD_BROADER = (
    "The list price is 103,125.00 SAR [PriceComponent:PC0001166]. "
    "A volume rebate of 1,534.35 SAR applies [PriceComponent:PC0001168]. "
    "The invoice price is 95,350.07 SAR [PriceComponent:PC0001166]."
)
MISATTRIBUTED = (
    "The list price is 103,125.00 SAR [PriceComponent:PC0001166]. "
    "The pocket price is 95,350.07 SAR [PriceComponent:PC0001168]."
)
FABRICATED = "The margin is 7,890.16 SAR [PriceComponent:PC0001166]."


def generate(llm, narrative_config, pack, count):
    generator = NarrativeGenerator(llm, narrative_config, base_temperature=0.7)
    return [generator.explain(pack, language="en") for _ in range(count)]


# --------------------------------------------------------------------------
# hallucination checks: attribution
# --------------------------------------------------------------------------
class TestAttributionGate:
    def test_a_role_stated_with_another_roles_value_is_caught(self, pack):
        """
        The residual served-wrong risk after D4: 95,350.07 is a real figure
        from the pack with a resolvable citation, and it is not the pocket
        price. Verification passes this; attribution must not.
        """
        report = check_attribution(MISATTRIBUTED, pack)

        assert not report.ok
        assert report.misattributions[0].role_key == "pocket_price"
        assert report.misattributions[0].expected == 89489.98

    def test_healthy_prose_is_left_alone(self, pack):
        assert check_attribution(GOOD, pack).ok
        assert check_attribution(GOOD_BROADER, pack).ok

    def test_a_referential_mention_asserts_nothing(self, pack):
        """"below the list price" names it without claiming a value for it."""
        text = (
            "The pocket price is 89,489.98 SAR [PriceComponent:PC0001166], "
            "13,635.02 SAR below the list price [PriceComponent:PC0001168]."
        )

        assert check_attribution(text, pack).ok

    def test_an_identifier_is_not_read_as_a_value(self, pack):
        """SKU0057 ends in digits; reading it as 57 was a real false positive."""
        text = "The list price for SKU0057 is 103,125.00 SAR [PriceComponent:PC0001166]."

        assert check_attribution(text, pack).ok

    def test_a_value_stated_before_its_noun_is_accepted(self, pack):
        """English allows "1,534.35 SAR for a volume rebate"."""
        text = "A deduction of 1,534.35 SAR for a volume rebate [PriceComponent:PC0001168]."

        assert check_attribution(text, pack).ok

    @pytest.mark.parametrize(
        "text",
        [
            "Total erosion from the list price to the pocket price is "
            "1,534.35 SAR [PriceComponent:PC0001168].",
            "The walk runs from the list price to the invoice price "
            "[PriceComponent:PC0001166].",
            "Deductions of 1,534.35 SAR bring it down to the pocket price "
            "[PriceComponent:PC0001168].",
        ],
    )
    def test_a_role_named_as_a_range_endpoint_asserts_nothing(self, pack, text):
        """
        Regression for a live false positive that inflated the measured
        served-wrong rate. "from the list price to the pocket price" names
        both endpoints; "from the" was treated as referential and "to the"
        was not, so the pocket price read as an assertion, the sentence held
        no pocket figure, and a correct narrative was reported as wrong.
        """
        assert check_attribution(text, pack).ok

    @pytest.mark.parametrize(
        "text",
        [
            "Total erosion is 13,635.02 SAR, with the final pocket price "
            "being 1,534.35 SAR lower than the invoice price "
            "[PriceComponent:PC0001168].",
            "The pocket price is 1,534.35 SAR below the invoice price "
            "[PriceComponent:PC0001168].",
        ],
    )
    def test_a_comparative_states_a_difference_not_a_value(self, pack, text):
        """
        Regression for a live false positive. "the pocket price being
        1,534.35 SAR lower than the invoice price" is correct English and a
        true statement; the figure is the gap, not the pocket price.
        """
        assert check_attribution(text, pack).ok

    def test_no_pattern_contains_a_literal_control_character(self):
        """
        A heredoc once turned an intended word boundary into a literal
        backspace (0x08), which silently disabled a whole exemption -- the
        pattern still looked right in the file and matched nothing.
        """
        import re as _re

        from src.reliability import attribution as module

        for name in dir(module):
            value = getattr(module, name)
            if isinstance(value, _re.Pattern):
                assert "" not in value.pattern, f"{name} carries a backspace"

    def test_a_sentence_with_no_figures_is_not_inspected(self, pack):
        assert check_attribution("The pocket price was reviewed.", pack).ok

    @pytest.mark.parametrize(
        "text",
        [
            "The list price is reduced to 95,350.07 SAR [PriceComponent:PC0001166].",
            "The list price is brought down to 89,489.98 SAR [PriceComponent:PC0001166].",
            "The discount reduces the list price to 95,350.07 SAR [PriceComponent:PC0001166].",
            "The list price is lowered to 95,350.07 SAR [PriceComponent:PC0001166].",
        ],
    )
    def test_a_transformed_role_is_a_source_not_a_claim(self, pack, text):
        """
        A waterfall narrative's whole subject is one figure becoming another.
        "The list price is reduced to 95,350.07" states the *result*, and
        flagging it made the checker fire on correct prose.
        """
        assert check_attribution(text, pack).ok

    def test_a_bare_assertion_with_the_wrong_value_still_fires(self, pack):
        """
        The distinction that matters: no transformation verb, so this really
        does claim the list price is 95,350.07.
        """
        text = "The list price is 95,350.07 SAR [PriceComponent:PC0001166]."

        report = check_attribution(text, pack)

        assert not report.ok
        assert report.misattributions[0].role_key == "list_price"


# --------------------------------------------------------------------------
# self-consistency
# --------------------------------------------------------------------------
class TestSelfConsistency:
    def test_agreeing_samples_are_served(self, narrative_config, pack):
        samples = generate(CyclingLLM(GOOD), narrative_config, pack, 3)

        report = assess_consistency(samples, pack)

        assert not report.abstained
        assert report.agreement == 1.0
        assert report.consensus is not None

    def test_breadth_is_not_disagreement(self, narrative_config, pack):
        """
        Measured behaviour: real samples differ in how much they say, with
        identical citations and no contradiction. Treating that as conflict
        would abstain on healthy output every time.
        """
        samples = generate(
            CyclingLLM(GOOD, GOOD_BROADER, GOOD), narrative_config, pack, 3
        )

        report = assess_consistency(samples, pack)

        assert not report.abstained
        assert 0.6 <= report.agreement < 1.0

    def test_a_misattributed_sample_cannot_vote(self, narrative_config, pack):
        samples = generate(
            CyclingLLM(GOOD, MISATTRIBUTED, GOOD), narrative_config, pack, 3
        )

        report = assess_consistency(samples, pack)

        assert len(report.usable) == 2
        assert not report.abstained, "two clean samples still make quorum"
        assert "misattributed" in report.samples[1].rejection

    def test_losing_quorum_abstains(self, narrative_config, pack):
        """Two of three samples defective: the model is unstable here."""
        samples = generate(
            CyclingLLM(GOOD, MISATTRIBUTED, FABRICATED), narrative_config, pack, 3
        )

        report = assess_consistency(samples, pack)

        assert report.abstained
        assert report.consensus is None
        assert "passed verification" in report.reason

    def test_disagreement_below_the_floor_abstains(self, narrative_config, pack):
        """Samples describing disjoint figures drive the core to empty."""
        a = "The list price is 103,125.00 SAR [PriceComponent:PC0001166]."
        b = "A volume rebate of 1,534.35 SAR applies [PriceComponent:PC0001168]."
        samples = generate(CyclingLLM(a, b), narrative_config, pack, 2)

        report = assess_consistency(samples, pack, min_verified=2, min_agreement=0.6)

        assert report.abstained
        assert report.agreement == 0.0
        assert "disagree" in report.reason

    def test_the_consensus_covers_the_core(self, narrative_config, pack):
        samples = generate(
            CyclingLLM(GOOD_BROADER, GOOD, GOOD), narrative_config, pack, 3
        )

        report = assess_consistency(samples, pack)

        assert report.consensus is not None
        assert report.core <= report.consensus.claims

    def test_a_lone_usable_sample_agrees_with_itself(self, narrative_config, pack):
        """
        With nothing to compare against, self-consistency has no opinion and
        verification plus attribution are what stand behind the answer.
        """
        samples = generate(CyclingLLM(GOOD), narrative_config, pack, 1)

        report = assess_consistency(samples, pack, min_verified=1)

        assert not report.abstained
        assert report.agreement == 1.0

    def test_an_empty_batch_is_a_programming_error(self, pack):
        with pytest.raises(ValueError, match="at least one sample"):
            assess_consistency([], pack)


# --------------------------------------------------------------------------
# corrective retrieval
# --------------------------------------------------------------------------
class TestEvidenceAssessment:
    def test_a_pack_whose_headline_figures_lack_records_is_weak(self, pack):
        """
        invoice_price and pocket_price arrive from Track A as subtotals with
        no row behind them -- half this pack's figures cite nothing.
        """
        assessment = assess_evidence(pack, minimum_support=0.75)

        assert assessment.facts_with_citation == 2
        assert assessment.facts_total == 4
        assert assessment.citation_density == 0.5
        assert assessment.weak

    def test_a_well_backed_pack_is_not_weak(self):
        strong = FactPack(
            "s", "k", "SAR",
            (Fact("list_price", 1.0, "money", "PriceComponent:PC1", "list price"),),
            ("PriceComponent:PC1",),
        )

        assert not assess_evidence(strong, minimum_support=0.75).weak

    def test_the_engine_version_is_not_reported_as_unresolvable(self, pack):
        """It is an attribution, not a graph record, so it never resolves."""
        index = EvidenceIndex(
            {
                handle: ResolvedRecord(handle, "PriceComponent", "...")
                for handle in pack.citations
            }
        )

        assert assess_evidence(pack, index).unresolvable == ()


class TestCorrectiveRetrieval:
    @pytest.fixture
    def index(self):
        handles = [
            "PriceComponent:PC0001166",
            "PriceComponent:PC0001168",
            "OrderLine:OL000248",
            "ListPricePeriod:SKU0026",
        ]
        return EvidenceIndex(
            {h: ResolvedRecord(h, h.split(":")[0], "...") for h in handles}
        )

    @pytest.fixture
    def recommendation(self):
        return PriceRecommendation.load(SAMPLE_PAYLOADS, "REC-0001")

    def test_resolvable_records_are_admitted(self, pack, recommendation, index):
        finder = lambda q, k: ["OrderLine:OL000248", "ListPricePeriod:SKU0026"]

        outcome = apply_corrective_retrieval(pack, recommendation, finder, index)

        assert outcome.improved
        assert "OrderLine:OL000248" in outcome.pack.citations
        assert outcome.after.citations_total > outcome.before.citations_total

    def test_no_figure_is_ever_added(self, pack, recommendation, index):
        """
        The closed fact set is what stops the model inventing numbers.
        Corrective retrieval widens what may be *cited*, never what may be
        *stated*.
        """
        finder = lambda q, k: ["OrderLine:OL000248"]

        outcome = apply_corrective_retrieval(pack, recommendation, finder, index)

        assert outcome.pack.allowed_values == pack.allowed_values
        assert outcome.pack.facts == pack.facts

    def test_unresolvable_candidates_are_discarded(self, pack, recommendation, index):
        finder = lambda q, k: ["Ghost:NOPE", "OrderLine:OL000248"]

        outcome = apply_corrective_retrieval(pack, recommendation, finder, index)

        assert outcome.added == ("OrderLine:OL000248",)

    def test_records_already_cited_are_not_duplicated(self, pack, recommendation, index):
        finder = lambda q, k: ["PriceComponent:PC0001166"]

        outcome = apply_corrective_retrieval(pack, recommendation, finder, index)

        assert not outcome.improved
        assert outcome.pack is pack

    def test_the_cap_is_respected(self, pack, recommendation, index):
        finder = lambda q, k: ["OrderLine:OL000248", "ListPricePeriod:SKU0026"]

        outcome = apply_corrective_retrieval(
            pack, recommendation, finder, index, max_added=1
        )

        assert len(outcome.added) == 1

    def test_the_query_ladder_runs_narrow_before_broad(
        self, pack, recommendation, index
    ):
        seen: list[str] = []

        def finder(query, k):
            seen.append(query)
            return []

        apply_corrective_retrieval(pack, recommendation, finder, index)

        assert len(seen) > 1
        assert recommendation.order_line_id in seen[0], "narrow query leads"
        assert recommendation.order_line_id not in seen[-1], "broad query trails"

    def test_off_subject_records_are_refused(self, pack, recommendation, index):
        """
        Retrieval for order line OL000248 once admitted OL000085 and
        OL000183 -- other deals entirely. Under a rule that every claim must
        be cited, that invites a citation which resolves cleanly and points
        at the wrong line: worse than no citation, because it survives
        inspection.
        """
        extra = EvidenceIndex(
            {
                h: ResolvedRecord(h, h.split(":")[0], "...")
                for h in (
                    "OrderLine:OL000248",
                    "OrderLine:OL000085",
                    "ListPricePeriod:SKU0026",
                )
            }
        )
        finder = lambda q, k: ["OrderLine:OL000085", "OrderLine:OL000248"]

        outcome = apply_corrective_retrieval(pack, recommendation, finder, extra)

        assert outcome.added == ("OrderLine:OL000248",)
        assert "OrderLine:OL000085" in outcome.rejected

    def test_a_record_naming_the_product_is_on_subject(
        self, pack, recommendation, index
    ):
        extra = EvidenceIndex(
            {
                h: ResolvedRecord(h, h.split(":")[0], "...")
                for h in ("ListPricePeriod:SKU0026", "ProductCostPeriod:SKU0026")
            }
        )
        finder = lambda q, k: ["ListPricePeriod:SKU0026", "ProductCostPeriod:SKU0026"]

        outcome = apply_corrective_retrieval(pack, recommendation, finder, extra)

        assert set(outcome.added) == {
            "ListPricePeriod:SKU0026",
            "ProductCostPeriod:SKU0026",
        }

    def test_the_trigger_metric_actually_moves(self, pack, recommendation, index):
        """
        An earlier version triggered on citation density, which corrective
        retrieval cannot change -- so it fired on every explanation and
        improved nothing. Support ratio is the measure the remedy moves.
        """
        finder = lambda q, k: ["OrderLine:OL000248"]

        outcome = apply_corrective_retrieval(pack, recommendation, finder, index)

        assert outcome.after.support_ratio > outcome.before.support_ratio
        assert outcome.after.citation_density == outcome.before.citation_density

    def test_a_failing_retriever_does_not_break_generation(
        self, pack, recommendation, index
    ):
        """Corrective retrieval is best-effort; its failure must not raise."""

        def finder(query, k):
            raise RuntimeError("index unavailable")

        outcome = apply_corrective_retrieval(pack, recommendation, finder, index)

        assert not outcome.improved


# --------------------------------------------------------------------------
# the assembled layer
# --------------------------------------------------------------------------
class TestReliableNarrator:
    @pytest.fixture
    def index(self):
        handles = [
            "PriceComponent:PC0001166",
            "PriceComponent:PC0001168",
            "OrderLine:OL000248",
        ]
        return EvidenceIndex(
            {h: ResolvedRecord(h, h.split(":")[0], "...") for h in handles}
        )

    @pytest.fixture
    def recommendation(self):
        return PriceRecommendation.load(SAMPLE_PAYLOADS, "REC-0001")

    def narrator(self, llm, reliability_config, narrative_config, index, finder=None):
        generator = NarrativeGenerator(
            llm, narrative_config, base_temperature=reliability_config.sample_temperature
        )
        return ReliableNarrator(
            generator, reliability_config, narrative_config,
            evidence=index, finder=finder,
        )

    def test_k_samples_are_generated(
        self, pack, recommendation, reliability_config, narrative_config, index
    ):
        llm = CyclingLLM(GOOD)

        self.narrator(llm, reliability_config, narrative_config, index).explain(
            recommendation, pack=pack
        )

        assert llm.calls == reliability_config.samples

    def test_sampling_happens_above_zero_temperature(
        self, pack, recommendation, reliability_config, narrative_config, index
    ):
        """At temperature 0 the k samples are identical and measure nothing."""
        llm = CyclingLLM(GOOD)

        self.narrator(llm, reliability_config, narrative_config, index).explain(
            recommendation, pack=pack
        )

        assert all(t and t > 0 for t in llm.temperatures)

    def test_an_agreed_explanation_is_served(
        self, pack, recommendation, reliability_config, narrative_config, index
    ):
        result = self.narrator(
            CyclingLLM(GOOD), reliability_config, narrative_config, index
        ).explain(recommendation, pack=pack)

        assert result.served
        assert result.text
        assert result.agreement == 1.0

    def test_a_majority_of_defective_samples_withholds_the_answer(
        self, pack, recommendation, reliability_config, narrative_config, index
    ):
        result = self.narrator(
            CyclingLLM(GOOD, MISATTRIBUTED, FABRICATED),
            reliability_config, narrative_config, index,
        ).explain(recommendation, pack=pack)

        assert not result.served
        assert result.narrative is None
        assert result.reason

    def test_weak_evidence_triggers_corrective_retrieval(
        self, pack, recommendation, reliability_config, narrative_config, index
    ):
        calls: list[str] = []

        def finder(query, k):
            calls.append(query)
            return ["OrderLine:OL000248"]

        result = self.narrator(
            CyclingLLM(GOOD), reliability_config, narrative_config, index, finder
        ).explain(recommendation, pack=pack)

        assert result.assessment.weak
        assert result.correction is not None
        assert result.correction.added == ("OrderLine:OL000248",)
        assert calls

    def test_without_a_finder_weakness_is_reported_not_repaired(
        self, pack, recommendation, reliability_config, narrative_config, index
    ):
        result = self.narrator(
            CyclingLLM(GOOD), reliability_config, narrative_config, index
        ).explain(recommendation, pack=pack)

        assert result.assessment.weak
        assert result.correction is None
        assert result.served, "weak evidence alone does not withhold the answer"

    def test_strong_evidence_skips_correction(
        self, recommendation, reliability_config, narrative_config, index
    ):
        strong = FactPack(
            "OL000248", "recommendation_and_waterfall", "SAR",
            (Fact("list_price", 103125.00, "money", "PriceComponent:PC0001166",
                  "list price"),),
            ("PriceComponent:PC0001166",),
        )
        called = []

        result = self.narrator(
            CyclingLLM("The list price is 103,125.00 SAR [PriceComponent:PC0001166]."),
            reliability_config, narrative_config, index,
            lambda q, k: called.append(q) or [],
        ).explain(recommendation, pack=strong)

        assert not result.assessment.weak
        assert called == []
        assert result.served

    def test_the_report_explains_the_decision(
        self, pack, recommendation, reliability_config, narrative_config, index
    ):
        result = self.narrator(
            CyclingLLM(GOOD, MISATTRIBUTED, FABRICATED),
            reliability_config, narrative_config, index,
        ).explain(recommendation, pack=pack)

        described = result.describe()

        assert "ABSTAINED" in described
        assert "evidence:" in described


class TestConfiguration:
    def test_an_unreachable_quorum_is_refused_at_construction(self):
        """
        min_verified > samples would abstain on every explanation forever,
        which is a configuration error, not a runtime outcome.
        """
        with pytest.raises(ConfigError, match="never"):
            ReliabilityConfig(
                enabled=True, samples=2, min_verified=3, min_agreement=0.6,
                sample_temperature=0.7, enforce_attribution=True, corrective_enabled=True,
                min_support_ratio=0.75, corrective_top_k=6,
                corrective_max_added=4,
            )

    def test_the_shipped_defaults_match_the_plan(self):
        from src.config.settings import get_settings

        config = get_settings().reliability

        assert config.samples == 3, "the plan specifies k=3"
        assert config.sample_temperature > 0.0
        assert config.min_verified <= config.samples


class TestAttributionAtScale:
    """
    The D5 claim rests on this checker, so it is exercised across every
    payload rather than on a handful of hand-written strings: a correct
    narrative for each, and every pairwise swap of the three walk levels.
    """

    ROLES = (
        ("list price", "list_price"),
        ("invoice price", "invoice_price"),
        ("pocket price", "pocket_price"),
    )

    def _cases(self):
        import json

        doc = json.loads(EVAL_PAYLOADS.read_text(encoding="utf-8"))
        for entry in doc["recommendations"]:
            rec = PriceRecommendation.load(
                EVAL_PAYLOADS, entry["recommendation_id"]
            )
            pack = build_facts(rec)
            yield entry["recommendation_id"], pack, {
                f.key: f.value for f in pack.facts
            }, pack.citations[0]

    def test_no_false_positive_on_any_correct_narrative(self):
        for rec_id, pack, values, cite in self._cases():
            text = " ".join(
                f"The {phrase} is {values[key]:,.2f} SAR [{cite}]."
                for phrase, key in self.ROLES
            )

            assert check_attribution(text, pack).ok, f"false positive on {rec_id}"

    def test_every_pairwise_swap_of_the_walk_levels_is_caught(self):
        checked = 0
        for rec_id, pack, values, cite in self._cases():
            for i, (_, key_i) in enumerate(self.ROLES):
                for j, (_, key_j) in enumerate(self.ROLES):
                    if i == j or values[key_i] == values[key_j]:
                        continue
                    text = " ".join(
                        f"The {phrase} is "
                        f"{values[key_j] if k == i else values[key]:,.2f} SAR [{cite}]."
                        for k, (phrase, key) in enumerate(self.ROLES)
                    )

                    assert not check_attribution(text, pack).ok, (
                        f"missed {key_i} <- {key_j} in {rec_id}"
                    )
                    checked += 1

        assert checked >= 100, "the sweep should cover every payload"


class TestAgainstRealPayloads:
    """The eval set is real data; the layer must handle all of it."""

    def test_every_eval_payload_builds_an_assessable_pack(self):
        import json

        doc = json.loads(EVAL_PAYLOADS.read_text(encoding="utf-8"))
        for entry in doc["recommendations"]:
            rec = PriceRecommendation.load(
                EVAL_PAYLOADS, entry["recommendation_id"]
            )
            assessment = assess_evidence(build_facts(rec), minimum_support=0.75)

            assert assessment.facts_total > 0
            assert 0.0 <= assessment.citation_density <= 1.0
