"""
Explain a Track A price recommendation, with citations (D4).

    python explain_price.py                        # first payload, English
    python explain_price.py --id REC-0002
    python explain_price.py --facts                # show the fact table first
    python explain_price.py --evidence             # show the resolved records
    python explain_price.py --list                 # list available payloads
    python explain_price.py --lang ar              # Arabic (deferred; English is the focus)

Every number printed below came from Track A's payload. This program computes
no pricing figure: it reads the payload, confirms the records it cites exist
in the graph, shows the model only that closed set of numbers, and then checks
the narrative it gets back. A narrative that fails the check is withheld.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from src.config.settings import PROJECT_ROOT, get_settings
from src.graph.sources import GraphSourceError, build_graph_source
from src.narrative.contract import ContractError, PriceRecommendation
from src.narrative.evidence import EvidenceIndex
from src.narrative.facts import build_facts
from src.narrative.factory import build_narrative_generator, build_serializer
from src.narrative.generator import Narrative
from src.utils.helpers import setup_logging

DEFAULT_PAYLOADS = PROJECT_ROOT / "data" / "track_a_sample_payloads.json"
WIDTH = 78
RULE = "=" * WIDTH


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--id", dest="recommendation_id", help="Which payload to explain")
    parser.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS)
    parser.add_argument("--lang", choices=("en", "ar"), default="en")
    parser.add_argument("--facts", action="store_true", help="Show the fact table")
    parser.add_argument(
        "--evidence", action="store_true", help="Show the resolved cited records"
    )
    parser.add_argument("--list", action="store_true", help="List payloads and exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(logging.INFO if args.verbose else logging.WARNING)
    settings = get_settings()

    if args.list:
        return _list_payloads(args.payloads)

    try:
        recommendation = PriceRecommendation.load(args.payloads, args.recommendation_id)
    except ContractError as exc:
        print(f"Track A payload refused: {exc}", file=sys.stderr)
        return 2

    try:
        graph = build_graph_source(settings).load()
    except GraphSourceError as exc:
        print(f"Could not load the graph: {exc}", file=sys.stderr)
        return 2

    serializer = build_serializer(graph, settings)
    evidence = EvidenceIndex.build(graph, serializer)
    pack = build_facts(recommendation)

    print(RULE)
    print(f"Explaining: {pack.subject}")
    print(f"Source    : {recommendation.recommendation_id} "
          f"from {Path(args.payloads).name} (engine {recommendation.engine_version})")
    print(RULE)

    if args.facts:
        print("\nFigures Track A sent. The narrative may state these and nothing else:")
        print(pack.render(args.lang))

    if args.evidence:
        print(f"\nCited records, resolved against the graph:")
        for handle in pack.citations:
            record = evidence.resolve(handle)
            if record is None:
                print(f"  {handle}  -- NOT FOUND IN GRAPH")
            else:
                print(f"  {handle}\n      {record.preview()}")

    generator = build_narrative_generator(settings)
    narrative = generator.explain(pack, language=args.lang, evidence=evidence)
    _print_narrative(narrative)
    return 0 if narrative.success else 1


def _list_payloads(path: Path) -> int:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read {path}: {exc}", file=sys.stderr)
        return 2

    entries = data.get("recommendations", []) if isinstance(data, dict) else []
    print(f"{len(entries)} payload(s) in {path.name}:\n")
    for entry in entries:
        waterfall = "with waterfall" if entry.get("waterfall") else "no waterfall"
        print(f"  {entry.get('recommendation_id'):<10} "
              f"{entry.get('product_id')} / {entry.get('customer_id'):<10} "
              f"line {entry.get('order_line_id', '-'):<10} {waterfall}")
    provenance = data.get("_provenance") if isinstance(data, dict) else None
    if provenance:
        print(f"\nProvenance: {provenance}")
    return 0


def _print_narrative(narrative: Narrative) -> None:
    print(f"\n{'-' * WIDTH}")

    if not narrative.success:
        print(f"Abstained after {narrative.attempts} attempt(s).")
        print(f"Reason: {narrative.reason}")
        if narrative.verification is not None:
            print("\nWhat the check found:")
            print(narrative.verification.summary())
        print(
            "\nNo explanation is returned. Withholding one is the designed "
            "outcome when it cannot be traced to the evidence."
        )
        return

    print(f"{narrative.text}\n")
    print(f"{'-' * WIDTH}")
    print(f"Verified on attempt {narrative.attempts}"
          f"{' (self-corrected)' if narrative.attempts > 1 else ''}")
    report = narrative.verification
    if report is not None:
        print(f"All {len(report.numbers_found)} number(s) stated trace to a figure "
              f"Track A sent.")
    print(f"Cited records ({len(narrative.citations)}):")
    for citation in narrative.citations:
        print(f"  {citation}")


if __name__ == "__main__":
    raise SystemExit(main())
