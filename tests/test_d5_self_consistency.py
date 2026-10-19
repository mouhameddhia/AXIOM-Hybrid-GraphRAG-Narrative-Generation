"""
Self-consistency (k=3) with abstain-on-disagreement -- for review with the mentor.

Skipped unless ``RUN_LIVE_NARRATIVE_DEMO=1`` -- run it explicitly:

    RUN_LIVE_NARRATIVE_DEMO=1 pytest tests/test_d5_self_consistency.py -v -s

``-s`` is required to see the printed samples and agreement breakdown.

What this demonstrates
-----------------------
Two scenarios, both using the real production function
``assess_consistency()`` from ``src/reliability/consistency.py``:

1. **Agreement on healthy samples** (live model). Three narratives are
   generated independently from the same fact pack, at temperature 0.7 (not
   0 -- see the note below). Each sample is shown with how many figures it
   states; then the consensus-core agreement score is computed; then the
   serve/abstain decision that score produces.

2. **Abstention on real disagreement** (scripted, deterministic). Three
   samples are constructed so two of them state genuinely *different*
   figures, not just different amounts of detail -- the one case that must
   trigger abstention. This is scripted rather than waited-for because a
   real model disagreeing with itself is rare and not reproducible on
   demand; the mechanism itself is the real, unmodified production code.

Why temperature matters here
------------------------------
Narrative generation (D4) defaults to temperature 0 for reproducibility.
Self-consistency needs the opposite: at temperature 0, three samples of one
prompt come back byte-identical, which "agrees" perfectly and proves
nothing. This file samples at ``settings.reliability.sample_temperature``
(0.7 by default), the same value the production ``ReliableNarrator`` uses.

Why agreement is not simple set-equality
------------------------------------------
Measuring real samples first is what shaped this design: three samples of
one healthy fact pack came back with 7, 9, and 11 distinct figures --
identical citations, zero contradictions, just different levels of detail.
A naive "do the sets match exactly" check would abstain on every healthy
run. Agreement is instead the *consensus core* (figures at least 2 of 3
samples state) as a fraction of the *union* (every figure any sample
states) -- extra, uncontested detail lowers this gently; real disagreement
drives it toward zero. That is exactly what scenario 1 vs scenario 2 below
makes visible side by side.
"""
from __future__ import annotations

import os

import pytest

from src.config.settings import PROJECT_ROOT, get_settings
from src.narrative.contract import PriceRecommendation
from src.narrative.facts import build_facts
from src.narrative.generator import NarrativeGenerator
from src.reliability.consistency import assess_consistency
from src.text_to_query.llm_client import OllamaClient

LIVE_DEMO_ENABLED = os.getenv("RUN_LIVE_NARRATIVE_DEMO") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE_DEMO_ENABLED,
    reason="Set RUN_LIVE_NARRATIVE_DEMO=1 to run against a live Ollama model",
)

PAYLOADS = PROJECT_ROOT / "data" / "track_a_payloads_eval.json"
RULE = "=" * 78


def heading(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def show_samples(samples) -> None:
    """Sample 1, its figure count -- sample 2, its figure count -- sample 3, ..."""
    for i, sample in enumerate(samples, 1):
        figures = len(sample.verification.numbers_found) if sample.verification else 0
        print(f"\nSAMPLE {i} ({figures} figure(s) stated):")
        print(f"  {sample.text}")


def show_decision(report) -> None:
    print(f"\n{'-' * 78}")
    print(f"consensus core   : {len(report.core)} figure(s)")
    print(f"union (any sample): {len(report.union)} figure(s)")
    print(f"agreement        : {report.agreement:.2f}  "
          f"(floor: {get_settings().reliability.min_agreement:.2f})")
    print(f"DECISION         : {'ABSTAINED' if report.abstained else 'SERVED'}")
    if report.abstained:
        print(f"reason           : {report.reason}")


class TestSelfConsistencyOnHealthyEvidence:
    """Scenario 1: three independent samples of the same healthy fact pack --
    must agree and must be served."""

    def test_three_samples_agree_and_are_served(self):
        heading("SELF-CONSISTENCY (k=3) -- healthy evidence, live model")

        settings = get_settings()
        recommendation = PriceRecommendation.load(PAYLOADS, "REC-0005")
        pack = build_facts(recommendation)

        generator = NarrativeGenerator(
            OllamaClient(settings.llm), settings.narrative,
            base_temperature=settings.reliability.sample_temperature,
        )
        samples = [generator.explain(pack, language="en") for _ in range(3)]
        show_samples(samples)

        report = assess_consistency(
            samples, pack,
            min_verified=settings.reliability.min_verified,
            min_agreement=settings.reliability.min_agreement,
        )
        show_decision(report)

        assert report.k == 3
        assert not report.abstained, f"healthy samples should agree, got: {report.reason}"
        assert report.agreement >= settings.reliability.min_agreement


class TestSelfConsistencyOnDisagreement:
    """Scenario 2: samples that state genuinely different figures -- must
    abstain rather than pick one arbitrarily. Scripted, not live, because a
    real model disagreeing with itself on demand is not reproducible; the
    consistency-checking code itself is the real, unmodified production
    function."""

    def test_abstains_when_samples_genuinely_disagree(self):
        heading("SELF-CONSISTENCY (k=3) -- genuine disagreement, scripted samples")

        settings = get_settings()
        recommendation = PriceRecommendation.load(PAYLOADS, "REC-0016")
        pack = build_facts(recommendation)
        citation = pack.citations[0]

        class ScriptedLLM:
            def __init__(self, *responses):
                self.responses, self.i = list(responses), 0

            def complete(self, prompt, *, temperature=None):
                response = self.responses[min(self.i, len(self.responses) - 1)]
                self.i += 1
                return response

        # Two samples describe genuinely different figures -- the opposite
        # of TestSelfConsistencyOnHealthyEvidence, where samples differ in
        # breadth, not in which figures they state.
        scripted = ScriptedLLM(
            f"The list price is {recommendation.waterfall.list_price:,.2f} SAR [{citation}].",
            f"The pocket price is {recommendation.waterfall.pocket_price:,.2f} SAR [{citation}].",
            f"The list price is {recommendation.waterfall.list_price:,.2f} SAR [{citation}].",
        )
        generator = NarrativeGenerator(scripted, settings.narrative, base_temperature=0.7)
        samples = [generator.explain(pack, language="en") for _ in range(3)]
        show_samples(samples)

        report = assess_consistency(
            samples, pack,
            min_verified=settings.reliability.min_verified,
            min_agreement=settings.reliability.min_agreement,
        )
        show_decision(report)

        assert report.k == 3
        assert report.abstained, "samples stating different figures must not be served"
