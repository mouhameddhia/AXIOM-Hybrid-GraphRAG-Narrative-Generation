"""
Tests for the narrative generator's control flow.

The language model is scripted here. That is not a way of avoiding the real
model -- it is the only way to test the behaviour that matters, because the
questions being asked are "what does the system do when the model fabricates
a number?" and "what does it do when the backend is down?", and neither can
be provoked on demand from a real model.

Abstention is asserted as a *successful* outcome throughout: the deliverable
requires the system to withhold rather than guess, so a test that accepted a
best-effort narrative on unverifiable input would be encoding the wrong rule.
"""
from __future__ import annotations

import pytest

from src.config.settings import NarrativeConfig
from src.narrative.evidence import EvidenceIndex, ResolvedRecord
from src.narrative.facts import Fact, FactPack
from src.narrative.generator import NarrativeGenerator
from src.narrative.rtl import RLM
from src.text_to_query.llm_client import LLMRequestError

GOOD = "The list price is 103,125.00 SAR [PriceComponent:PC0001166]."
GOOD_AR = "سعر القائمة 103,125.00 ريال [PriceComponent:PC0001166]."
FABRICATED = "The margin is 7,890.16 SAR [PriceComponent:PC0001166]."
UNCITED = "The list price is 103,125.00 SAR."


class ScriptedLLM:
    """Returns pre-written narratives in order. No network, no model."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.temperatures: list[float | None] = []

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        self.prompts.append(prompt)
        self.temperatures.append(temperature)
        return self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]


class UnreachableLLM:
    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        raise LLMRequestError("simulated backend outage")


@pytest.fixture
def pack():
    return FactPack(
        subject="OL000248",
        kind="price_waterfall",
        currency_code="SAR",
        facts=(
            Fact("list_price", 103125.00, "money", "PriceComponent:PC0001166", "list price", "سعر القائمة"),
        ),
        citations=("PriceComponent:PC0001166",),
    )


@pytest.fixture
def config():
    return NarrativeConfig(
        enabled=True,
        max_retries=1,
        max_words=120,
        absolute_tolerance=0.01,
        relative_tolerance=0.005,
        min_citation_coverage=1.0,
        retry_temperature_step=0.2,
    )


class TestSuccess:
    def test_a_verified_narrative_is_returned(self, pack, config):
        narrative = NarrativeGenerator(ScriptedLLM(GOOD), config).explain(pack)

        assert narrative.success
        assert not narrative.abstained
        assert narrative.attempts == 1
        assert narrative.citations == ("PriceComponent:PC0001166",)

    def test_arabic_output_carries_rtl_processing(self, pack, config):
        narrative = NarrativeGenerator(ScriptedLLM(GOOD_AR), config).explain(
            pack, language="ar"
        )

        assert narrative.success
        assert narrative.direction == "rtl"
        assert narrative.text.startswith(RLM)
        assert narrative.plain_text == GOOD_AR

    def test_the_fact_table_reaches_the_prompt(self, pack, config):
        llm = ScriptedLLM(GOOD)
        NarrativeGenerator(llm, config).explain(pack)

        assert "103,125.00 SAR" in llm.prompts[0]
        assert "PriceComponent:PC0001166" in llm.prompts[0]

    def test_the_arabic_prompt_asks_for_arabic(self, pack, config):
        llm = ScriptedLLM(GOOD_AR)
        NarrativeGenerator(llm, config).explain(pack, language="ar")

        assert "Arabic" in llm.prompts[0]
        assert "سعر القائمة" in llm.prompts[0]

    def test_code_fences_are_stripped_before_verification(self, pack, config):
        narrative = NarrativeGenerator(ScriptedLLM(f"```\n{GOOD}\n```"), config).explain(pack)

        assert narrative.success
        assert narrative.text == GOOD

    def test_html_rendering_carries_direction_and_language(self, pack, config):
        narrative = NarrativeGenerator(ScriptedLLM(GOOD_AR), config).explain(
            pack, language="ar"
        )

        assert 'dir="rtl"' in narrative.as_html()
        assert 'lang="ar"' in narrative.as_html()


class TestSelfCorrection:
    def test_a_fabricated_number_triggers_a_corrective_attempt(self, pack, config):
        llm = ScriptedLLM(FABRICATED, GOOD)

        narrative = NarrativeGenerator(llm, config).explain(pack)

        assert narrative.success
        assert narrative.attempts == 2

    def test_the_exact_violation_is_fed_back(self, pack, config):
        llm = ScriptedLLM(FABRICATED, GOOD)

        NarrativeGenerator(llm, config).explain(pack)

        assert "7,890.16" in llm.prompts[1]
        assert FABRICATED in llm.prompts[1]

    def test_retries_escalate_the_sampling_temperature(self, pack, config):
        """
        At a fixed temperature a model re-reads the corrected prompt and
        regenerates the same rejected text -- the same trap the text-to-query
        self-heal loop hit.
        """
        llm = ScriptedLLM(FABRICATED, GOOD)

        NarrativeGenerator(llm, config).explain(pack)

        assert llm.temperatures[1] > llm.temperatures[0]


class TestAbstention:
    def test_an_unverifiable_narrative_is_withheld(self, pack, config):
        llm = ScriptedLLM(FABRICATED, FABRICATED)

        narrative = NarrativeGenerator(llm, config).explain(pack)

        assert not narrative.success
        assert narrative.abstained
        assert narrative.text == ""
        assert "could not be verified" in narrative.reason

    def test_an_abstention_keeps_the_report_that_caused_it(self, pack, config):
        narrative = NarrativeGenerator(
            ScriptedLLM(FABRICATED, FABRICATED), config
        ).explain(pack)

        assert narrative.verification is not None
        assert "7,890.16" in narrative.verification.summary()

    def test_an_uncited_narrative_is_withheld(self, pack, config):
        narrative = NarrativeGenerator(ScriptedLLM(UNCITED, UNCITED), config).explain(pack)

        assert narrative.abstained

    def test_a_pack_with_nothing_to_cite_abstains_before_calling_the_model(self, config):
        empty = FactPack(
            subject="nothing", kind="price_waterfall", currency_code="SAR",
            facts=(Fact("x", 1.0, "count", "", "x", "x"),), citations=(),
        )
        llm = ScriptedLLM(GOOD)

        narrative = NarrativeGenerator(llm, config).explain(empty)

        assert narrative.abstained
        assert narrative.attempts == 0
        assert llm.prompts == [], "the model must not be called with no evidence"
        assert "no records to cite" in narrative.reason

    def test_a_pack_with_no_figures_abstains(self, config):
        empty = FactPack(
            subject="nothing", kind="price_waterfall", currency_code="SAR",
            facts=(), citations=("PriceComponent:PC0001166",),
        )

        narrative = NarrativeGenerator(ScriptedLLM(GOOD), config).explain(empty)

        assert narrative.abstained
        assert "no figures" in narrative.reason

    def test_an_unreachable_model_abstains_rather_than_raising(self, pack, config):
        narrative = NarrativeGenerator(UnreachableLLM(), config).explain(pack)

        assert narrative.abstained
        assert "unreachable" in narrative.reason
        assert "simulated backend outage" in narrative.reason


class TestEvidenceResolution:
    """
    A citation must point at a record that exists. One that does not is worse
    than an uncited claim, because it looks verifiable.
    """

    def test_a_citation_that_resolves_lets_generation_proceed(self, pack, config):
        index = EvidenceIndex(
            {
                "PriceComponent:PC0001166": ResolvedRecord(
                    "PriceComponent:PC0001166", "PriceComponent", "amount 103125"
                )
            }
        )

        narrative = NarrativeGenerator(ScriptedLLM(GOOD), config).explain(
            pack, evidence=index
        )

        assert narrative.success

    def test_an_unresolvable_citation_abstains_before_the_model_is_called(
        self, pack, config
    ):
        llm = ScriptedLLM(GOOD)

        narrative = NarrativeGenerator(llm, config).explain(
            pack, evidence=EvidenceIndex({})
        )

        assert narrative.abstained
        assert narrative.attempts == 0
        assert llm.prompts == [], "the model must not run on unresolvable evidence"
        assert "do not exist in the graph" in narrative.reason
        assert "PriceComponent:PC0001166" in narrative.reason

    def test_omitting_the_index_skips_the_check(self, pack, config):
        """Resolution is opt-in: a caller with no graph loaded still works."""
        assert NarrativeGenerator(ScriptedLLM(GOOD), config).explain(pack).success


class TestGuards:
    def test_an_unsupported_language_is_refused(self, pack, config):
        with pytest.raises(ValueError, match="language must be one of"):
            NarrativeGenerator(ScriptedLLM(GOOD), config).explain(pack, language="fr")

    def test_zero_retries_means_one_attempt(self, pack, config):
        from dataclasses import replace

        llm = ScriptedLLM(FABRICATED, GOOD)

        narrative = NarrativeGenerator(llm, replace(config, max_retries=0)).explain(pack)

        assert narrative.abstained
        assert len(llm.prompts) == 1
