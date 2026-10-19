"""
Hallucination checks (attribution gate) -- for review with the mentor.

Skipped unless ``RUN_LIVE_NARRATIVE_DEMO=1`` -- run it explicitly:

    RUN_LIVE_NARRATIVE_DEMO=1 pytest tests/test_d5_hallucination_checks.py -v -s

(Nothing here actually needs a live model -- the flag is kept for
consistency with the rest of the D5 demo suite, so one on/off switch
controls all of it.)

What a "hallucination" means in this system
----------------------------------------------
Not a number that doesn't exist -- D4's verifier already catches that. This
catches a number that is completely real, correctly cited, and attached to
the *wrong label*: a sentence says "the pocket price is 95,350.07," and
95,350.07 genuinely is a figure from the fact pack with a valid citation --
it is simply the *invoice* price, not the pocket price. Verification cannot
see this, because nothing about the number or the citation is wrong. This
module (``src/reliability/attribution.py``) checks whether the role being
named and the value being stated actually match.

Why four cases, not one
--------------------------
Building this checker took five rounds of fixing real false positives, each
found by running it against actual generated narratives rather than by
inspection: referential mentions ("13.2% *of the* list price"), transformed
values ("the list price *is reduced to* X"), range endpoints ("*from the*
list price *to the* pocket price"), and comparatives ("*44.33 lower than*
the invoice price") all had to be explicitly exempted, or correct prose got
flagged. The four cases below are chosen to show both halves at once: a
genuine misattribution gets caught, and prose that merely *mentions* a
figure without asserting it does not.
"""
from __future__ import annotations

import os

import pytest

from src.config.settings import PROJECT_ROOT, get_settings
from src.narrative.contract import PriceRecommendation
from src.narrative.facts import build_facts
from src.reliability.attribution import check_attribution

LIVE_DEMO_ENABLED = os.getenv("RUN_LIVE_NARRATIVE_DEMO") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE_DEMO_ENABLED,
    reason="Set RUN_LIVE_NARRATIVE_DEMO=1 (kept for consistency with the D5 demo suite)",
)

PAYLOADS = PROJECT_ROOT / "data" / "track_a_payloads_eval.json"
RULE = "=" * 78


def heading(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def check_and_show(label: str, text: str, pack, *, expect_flag: bool) -> None:
    report = check_attribution(text, pack)
    verdict = "FLAGGED" if not report.ok else "clean"
    print(f"\n{label}:")
    print(f"  {text}")
    print(f"  -> {verdict}")
    for m in report.misattributions:
        print(f"     role={m.role_key}  expected={m.expected:,.2f}  "
              f"sentence states={[f'{v:,.2f}' for v in m.stated]}")
    if expect_flag:
        assert not report.ok, f"should have been flagged: {text}"
    else:
        assert report.ok, f"should have been clean: {text}"


@pytest.fixture(scope="module")
def pack():
    recommendation = PriceRecommendation.load(PAYLOADS, "REC-0009")
    return build_facts(recommendation), recommendation


class TestHallucinationCaught:
    """A real, correctly-cited figure attached to the wrong noun."""

    def test_pocket_price_given_the_invoice_price_value(self, pack):
        heading("HALLUCINATION CHECK -- misattributed figure")
        fact_pack, recommendation = pack
        w = recommendation.waterfall

        correct = (
            f"The list price is {w.list_price:,.2f} SAR [{w.list_price_record_id}]. "
            f"The pocket price is {w.pocket_price:,.2f} SAR [{w.list_price_record_id}]."
        )
        corrupted = (
            f"The list price is {w.list_price:,.2f} SAR [{w.list_price_record_id}]. "
            f"The pocket price is {w.invoice_price:,.2f} SAR [{w.list_price_record_id}]."
        )

        check_and_show("CORRECT", correct, fact_pack, expect_flag=False)
        check_and_show("CORRUPTED (pocket price given the invoice price's value)",
                        corrupted, fact_pack, expect_flag=True)


class TestReferentialMentionsAreNotFalsePositives:
    """Prose that *names* a figure without asserting a value for it must
    stay clean -- this is what four of the five fix rounds were about."""

    def test_referential_and_comparative_phrasings_are_not_flagged(self, pack):
        heading("HALLUCINATION CHECK -- referential mentions are not false positives")
        fact_pack, recommendation = pack
        w = recommendation.waterfall
        cite = w.list_price_record_id

        cases = [
            (f"Erosion reached {w.erosion_pct}% of the list price [{cite}].",
             "percentage OF the list price -- names it, asserts nothing"),
            (f"The pocket price is {w.pocket_price:,.2f} SAR, "
             f"{abs(w.total_deductions):,.2f} SAR below the list price [{cite}].",
             "comparative -- states a DIFFERENCE, not the list price's value"),
        ]
        for text, description in cases:
            print(f"\n({description})")
            check_and_show("PROSE", text, fact_pack, expect_flag=False)
