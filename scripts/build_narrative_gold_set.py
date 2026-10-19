"""
Build the narrative gold set (D6).

The onboarding plan asks for "a small curated bilingual gold set of pricing
questions with known-correct, graph-grounded answers; freeze versions for
comparability; run it as a regression check". This script writes that file.
Arabic is out of scope for the current phase, so the set is English-only and
says so in its own metadata rather than leaving a reader to infer it.

Why generate it instead of hand-writing it
------------------------------------------
Every expected value here is copied from a Track A payload, and a hand-typed
copy of sixteen figures across twelve cases is a transcription-error surface
with no upside. What matters for the evaluation to be meaningful is that the
gold values reach the harness **without passing through the code under test**:
the harness reads them from this frozen JSON, never from
``build_facts()``. If the fact pack and the gold set ever disagree, the
harness reports it rather than silently agreeing with itself.

That independence is the whole point. A faithfulness score computed against
the same fact pack the generator was given is close to tautological -- it can
only catch a broken verifier. Checking a served narrative against figures
recorded separately is what makes the served-wrong rate a real measurement.

What each case carries
----------------------
``expected_values``
    Ground truth, keyed by fact key, copied from the payload.
``required_facts``
    The figures an explanation of this recommendation must actually state to
    be *relevant*. A narrative that quotes only the list price is faithful
    and useless; answer relevance is what separates the two, and it needs a
    stated expectation per case.
``min_citations``
    A floor on distinct cited records, scaled to the size of the walk.

    python scripts/build_narrative_gold_set.py
    python scripts/build_narrative_gold_set.py --out data/narrative_gold_set.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import PROJECT_ROOT  # noqa: E402
from src.narrative.contract import PriceRecommendation  # noqa: E402

DEFAULT_PAYLOADS = PROJECT_ROOT / "data" / "track_a_payloads_eval.json"
DEFAULT_OUT = PROJECT_ROOT / "data" / "narrative_gold_set.json"

DESCRIPTION = (
    "Narrative evaluation gold set for D6. English only -- Arabic is deferred "
    "for this phase. Every expected value is copied from the Track A payload "
    "named in 'payloads' and is read by the harness directly from this file, "
    "never recomputed through the narrative code under test. Frozen for "
    "comparability: regenerating it after a payload change is a deliberate act."
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    document = json.loads(args.payloads.read_text(encoding="utf-8"))
    entries = document.get("recommendations", [])
    if not entries:
        print(f"No recommendations in {args.payloads}", file=sys.stderr)
        return 1

    relative = _relative_to_root(args.payloads)
    cases = []
    for index, entry in enumerate(entries, start=1):
        recommendation = PriceRecommendation.load(
            args.payloads, entry["recommendation_id"]
        )
        cases.append(_build_case(recommendation, index, relative))

    payload = {
        "description": DESCRIPTION,
        "language": "en",
        "version": "1.0",
        "cases": cases,
    }
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(cases)} case(s) to {args.out}")
    return 0


def _build_case(
    recommendation: PriceRecommendation, index: int, payloads: str
) -> dict:
    rec = recommendation
    expected: dict[str, float] = {
        "recommended_price": rec.recommended_price,
        "price_range_low": rec.price_range.low,
        "price_range_high": rec.price_range.high,
        "confidence": round(rec.confidence * 100, 1),
        "expected_margin_pct": rec.expected_margin_pct,
        "acceptance_probability": round(rec.acceptance_probability * 100, 1),
    }
    # An explanation with no walk can only be about the recommendation itself.
    required = ["recommended_price"]

    if rec.waterfall is not None:
        walk = rec.waterfall
        expected.update(
            {
                "list_price": walk.list_price,
                "invoice_price": walk.invoice_price,
                "pocket_price": walk.pocket_price,
                "total_deductions": round(abs(walk.total_deductions), 2),
                "erosion_pct": walk.erosion_pct,
                "deduction_count": float(len(walk.steps)),
            }
        )
        for step in walk.steps:
            expected[f"step.{step.name}"] = step.magnitude
        # The three levels of the walk are what a waterfall explanation is
        # *for*: naming the list price alone hides the leakage the pocket
        # price exposes, which is the whole reason the walk is explained.
        required += ["list_price", "invoice_price", "pocket_price"]

    citable = len(rec.evidence_record_ids) + (
        len(rec.waterfall.record_ids) if rec.waterfall else 0
    )
    return {
        "case_id": f"NAR-{index:04d}",
        "recommendation_id": rec.recommendation_id,
        "payloads": payloads,
        "subject": f"{rec.product_id}/{rec.customer_id}/{rec.order_line_id}",
        "required_facts": required,
        # Two records is the floor at which a citation is doing work: one
        # record repeated across every claim carries no information about
        # which figure rests on what.
        "min_citations": 2 if citable >= 2 else 1,
        "expected_values": {key: round(value, 2) for key, value in expected.items()},
    }


def _relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
