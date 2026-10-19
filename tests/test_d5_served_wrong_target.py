"""
Measured move toward the <5% served-wrong target -- for review with the mentor.

Skipped unless ``RUN_LIVE_NARRATIVE_DEMO=1`` -- run it explicitly:

    RUN_LIVE_NARRATIVE_DEMO=1 pytest tests/test_d5_served_wrong_target.py -v -s

``-s`` is required to see the per-case breakdown and summary metrics.

What this demonstrates
-----------------------
The onboarding plan's acceptance criterion directly: faithfulness on the
gold set, and a measured served-wrong rate against the <5% target. This runs
the *real* evaluation harness (``src/evaluation/narrative_harness.py``) and
the *real* frozen gold set (``data/narrative_gold_set.json``) -- nothing
here is reimplemented or simplified. The only thing narrowed is how many of
the 24 gold cases run: a fixed, representative subset of 5, so this stays a
2-3 minute test rather than the ~20-30 minutes the full set takes at k=3.
The full 24-case measurement is ``evaluate_narratives.py`` (see
``docs/MENTOR_DEMO.md``, Step 9) -- that command, not this file, is the
official number.

Why the denominator is "served," not "every case"
-----------------------------------------------------
Served-wrong rate is deliberately computed over cases the system actually
answered, not over every case attempted. Abstaining is the designed outcome
when a check fails -- counting an abstention as "wrong" would penalise the
exact behaviour the reliability layer exists to produce. ``weak_citation``
(correct content, but citing too few distinct records) is also tracked
separately from ``served_wrong`` (false content) for the same reason: an
earlier version folded them together and it inflated the measured error
rate for a defect that was never actually a false number.
"""
from __future__ import annotations

import os
from dataclasses import replace

import pytest

from src.config.settings import PROJECT_ROOT, get_settings
from src.evaluation.narrative_harness import SERVED_WRONG_TARGET, NarrativeGoldSet, evaluate_narratives
from src.narrative.evidence import EvidenceIndex
from src.narrative.factory import build_serializer
from src.pipeline.factory import build_retrieval_stack
from src.reliability.factory import build_reliable_narrator, retrieval_finder

LIVE_DEMO_ENABLED = os.getenv("RUN_LIVE_NARRATIVE_DEMO") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE_DEMO_ENABLED,
    reason="Set RUN_LIVE_NARRATIVE_DEMO=1 to run against a live Ollama model",
)

GOLD_SET = PROJECT_ROOT / "data" / "narrative_gold_set.json"
RULE = "=" * 78

#: A fixed, representative subset: the first 5 cases of the real frozen gold
#: set, spanning different waterfall shapes (2/3/4-step) since the set is
#: built that way deliberately.
DEMO_SUBSET_SIZE = 5


def heading(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


class TestServedWrongTarget:
    def test_meets_target_on_a_representative_subset(self):
        heading(
            f"D5 MEASUREMENT -- faithfulness + served-wrong rate "
            f"({DEMO_SUBSET_SIZE} of {24} gold cases, k=3)"
        )

        settings = get_settings()
        full_gold_set = NarrativeGoldSet.load(GOLD_SET)
        demo_gold_set = replace(full_gold_set, cases=full_gold_set.cases[:DEMO_SUBSET_SIZE])

        stack = build_retrieval_stack(settings, enable_dense=False)
        index = EvidenceIndex.build(stack.graph, build_serializer(stack.graph, settings))
        narrator = build_reliable_narrator(
            stack.graph, settings, finder=retrieval_finder(stack.pipeline), evidence=index,
        )

        report = evaluate_narratives(
            demo_gold_set, narrator, label="D5 demo subset",
            config_summary={
                "model": settings.llm.model_name,
                "samples": settings.reliability.samples,
                "min_agreement": settings.reliability.min_agreement,
            },
        )

        print(f"\ngold set   : {GOLD_SET.name} ({report.cases} of 24 cases)")
        for row in report.results:
            print(f"  {row.describe()}")

        print(f"\n{'-' * 78}")
        print(report.summary())

        assert report.trustworthy, f"gold-set drift detected: {report.gold_drift}"
        assert report.service_rate > 0.0, "nothing was served at all -- check Ollama is running"
        assert report.meets_served_wrong_target, (
            f"served-wrong rate {report.served_wrong_rate:.1%} exceeds the "
            f"{SERVED_WRONG_TARGET:.0%} target"
        )
