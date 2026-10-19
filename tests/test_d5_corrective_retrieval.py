"""
Corrective-retrieval fallback -- for review with the mentor.

Skipped unless ``RUN_LIVE_NARRATIVE_DEMO=1`` -- run it explicitly:

    RUN_LIVE_NARRATIVE_DEMO=1 pytest tests/test_d5_corrective_retrieval.py -v -s

``-s`` is required to see the printed before/after evidence comparison.

What this demonstrates
-----------------------
When a fact pack's citable evidence is judged thin, the system re-queries
the real hybrid retrieval stack for more support *before* generating --
"re-retrieve... before generating," from the onboarding plan.

Two scenarios:

1. **Degraded evidence recovers.** A real payload has its citation handles
   deliberately stripped (simulating a case where Track A sent numbers but
   few supporting records). Corrective retrieval runs against the real
   graph and must recover enough on-subject records to raise the support
   ratio.

2. **Off-subject records are refused.** The first version of this mechanism
   admitted whatever retrieval ranked highest, with no regard for whether it
   was actually about the right deal -- for order line OL000001 it once
   cited *other, unrelated* order lines. This scenario proves that defect is
   fixed: candidates that don't name this recommendation's own product,
   customer, or order line are rejected, even when retrieval ranks them
   highly.

Why two separate metrics (support ratio vs. citation density)
-----------------------------------------------------------------
An earlier version triggered on *citation density* -- the share of figures
with their own record -- and fired on every single payload, because the gap
is structural: subtotals like ``pocket_price`` arrive from Track A with no
graph row behind them at all, by design. No amount of retrieval can fix
that. The real trigger is *support ratio* (citable records per figure),
which retrieval genuinely can move. This file asserts on support ratio
specifically, and prints both so the distinction is visible.
"""
from __future__ import annotations

import json
import os

import pytest

from src.config.settings import PROJECT_ROOT, get_settings
from src.narrative.contract import PriceRecommendation
from src.narrative.evidence import EvidenceIndex
from src.narrative.facts import build_facts
from src.narrative.factory import build_serializer
from src.pipeline.factory import build_retrieval_stack
from src.reliability.corrective import apply_corrective_retrieval, assess_evidence
from src.reliability.factory import retrieval_finder

LIVE_DEMO_ENABLED = os.getenv("RUN_LIVE_NARRATIVE_DEMO") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE_DEMO_ENABLED,
    reason="Set RUN_LIVE_NARRATIVE_DEMO=1 to run against the live retrieval stack",
)

PAYLOADS = PROJECT_ROOT / "data" / "track_a_payloads_eval.json"
RULE = "=" * 78


def heading(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def stack(settings):
    return build_retrieval_stack(settings, enable_dense=False)


@pytest.fixture(scope="module")
def index(stack, settings):
    return EvidenceIndex.build(stack.graph, build_serializer(stack.graph, settings))


@pytest.fixture
def degraded_recommendation():
    """REC-0005 with its citation handles stripped out."""
    entries = json.loads(PAYLOADS.read_text(encoding="utf-8"))["recommendations"]
    entry = next(e for e in entries if e["recommendation_id"] == "REC-0005")
    degraded = json.loads(json.dumps(entry))
    degraded["evidence_record_ids"] = [f"OrderLine:{degraded['order_line_id']}"]
    degraded["waterfall"]["list_price_record_id"] = ""
    for step in degraded["waterfall"]["steps"]:
        step["record_id"] = ""
    return PriceRecommendation.from_dict(degraded)


class TestCorrectiveRetrievalRecoversEvidence:
    def test_degraded_evidence_recovers(self, stack, index, settings, degraded_recommendation):
        heading("CORRECTIVE RETRIEVAL -- degraded evidence recovers on-subject records")

        pack = build_facts(degraded_recommendation)
        before = assess_evidence(pack, index, minimum_support=settings.reliability.min_support_ratio)
        print(f"BEFORE: {before.describe()}")
        assert before.weak, "the fixture must actually be degraded, or this proves nothing"

        outcome = apply_corrective_retrieval(
            pack, degraded_recommendation, retrieval_finder(stack.pipeline), index,
            top_k=8, max_added=6, minimum_support=settings.reliability.min_support_ratio,
        )
        print(f"\n{outcome.describe()}")
        print(f"AFTER:  {outcome.after.describe()}")
        print(f"citable now: {list(outcome.pack.citations)}")

        assert outcome.improved
        assert outcome.after.support_ratio > before.support_ratio


class TestCorrectiveRetrievalRefusesOffSubject:
    def test_off_subject_candidates_are_rejected(self, stack, index, settings, degraded_recommendation):
        heading("CORRECTIVE RETRIEVAL -- off-subject records are refused")

        pack = build_facts(degraded_recommendation)
        outcome = apply_corrective_retrieval(
            pack, degraded_recommendation, retrieval_finder(stack.pipeline), index,
            top_k=8, max_added=6, minimum_support=settings.reliability.min_support_ratio,
        )

        print(f"admitted: {list(outcome.added)}")
        print(f"citable now: {list(outcome.pack.citations)}")

        for handle in outcome.added:
            on_subject = (
                degraded_recommendation.product_id in handle
                or degraded_recommendation.customer_id in handle
                or degraded_recommendation.order_line_id in handle
            )
            print(f"  {handle}: on-subject={on_subject}")
            assert on_subject, (
                f"{handle} was admitted but names none of this recommendation's own subject"
            )
