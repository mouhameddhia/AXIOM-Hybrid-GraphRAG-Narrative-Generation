"""
Live narrative generation (D4) -- for review with the mentor.

Skipped unless ``RUN_LIVE_NARRATIVE_DEMO=1`` and a local Ollama model is
actually reachable -- run it explicitly:

    RUN_LIVE_NARRATIVE_DEMO=1 pytest tests/test_narrative_demo_generation.py -v -s

``-s`` is required to see the printed narratives -- without it pytest
captures stdout and shows only pass/fail.

What this file demonstrates
----------------------------
D4 (grounded narrative generation) across three genuinely different cases: a
simple 2-step high-confidence recommendation, a complex 4-step high-erosion
one, and a low-confidence one. The point of using three different shapes is
to prove the narrative adapts to the fact pack it is given rather than being
a template with numbers swapped in.

This is the real model (``qwen2.5:7b`` by default), not the scripted
stand-ins used in ``test_narrative_generator.py`` -- that file verifies the
generation *mechanism* hermetically and in under a second; this file shows
what it actually produces.

For the D5 reliability layer, see ``test_narrative_demo_reliability.py``.
For the full D2+D3+D4 integration chain, see
``test_narrative_demo_integration.py``. To run everything together:

    RUN_LIVE_NARRATIVE_DEMO=1 pytest tests/test_narrative_demo_generation.py tests/test_narrative_demo_reliability.py tests/test_narrative_demo_integration.py -v -s
"""
from __future__ import annotations

import os

import pytest

from src.config.settings import PROJECT_ROOT, get_settings
from src.narrative.contract import PriceRecommendation
from src.narrative.evidence import EvidenceIndex
from src.narrative.facts import build_facts
from src.narrative.factory import build_narrative_generator, build_serializer
from src.graph.sources import build_graph_source

LIVE_DEMO_ENABLED = os.getenv("RUN_LIVE_NARRATIVE_DEMO") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE_DEMO_ENABLED,
    reason="Set RUN_LIVE_NARRATIVE_DEMO=1 to run against a live Ollama model",
)

PAYLOADS = PROJECT_ROOT / "data" / "track_a_payloads_eval.json"
RULE = "=" * 78


def heading(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def index(settings):
    graph = build_graph_source(settings).load()
    return EvidenceIndex.build(graph, build_serializer(graph, settings))


class TestLiveNarrativeGeneration:
    """D4: does the narrative adapt to the shape of the fact pack, or is it
    a template? Three cases, chosen to differ on every axis that matters."""

    @pytest.mark.parametrize(
        "recommendation_id,label",
        [
            ("REC-0001", "simple: 2-step waterfall, high confidence (0.78)"),
            ("REC-0009", "complex: 4-step waterfall, high erosion (18.4%)"),
            ("REC-0012", "low confidence (0.41)"),
        ],
    )
    def test_generates_and_verifies(self, settings, index, recommendation_id, label):
        heading(f"NARRATIVE GENERATION -- {recommendation_id} ({label})")

        recommendation = PriceRecommendation.load(PAYLOADS, recommendation_id)
        pack = build_facts(recommendation)
        generator = build_narrative_generator(settings)

        narrative = generator.explain(pack, language="en", evidence=index)

        print(f"subject   : {pack.subject}")
        print(f"facts     : {len(pack.facts)} figures, {len(pack.citations)} citable records")
        print(f"\n{narrative.text}\n")
        if narrative.verification is not None:
            print(f"verified on attempt {narrative.attempts}: "
                  f"{len(narrative.verification.numbers_found)} number(s), "
                  f"citation coverage {narrative.verification.citation_coverage:.2f}")

        assert narrative.success, f"abstained: {narrative.reason}"
        assert narrative.verification.citation_coverage == 1.0
        for citation in narrative.citations:
            # trackA-provisional-* is the attribution for Track A's own
            # engine-computed figures (confidence, recommended price) -- it
            # is a valid citation, but it is not a graph record and was
            # never meant to resolve as one.
            if citation.startswith("trackA-"):
                continue
            assert index.resolve(citation) is not None, (
                f"narrative cited {citation}, which does not resolve in the graph"
            )
