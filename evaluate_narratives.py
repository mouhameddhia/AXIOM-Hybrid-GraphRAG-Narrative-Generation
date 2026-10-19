"""
Evaluate grounded narratives against the frozen gold set (D6).

    python evaluate_narratives.py                     # full reliability layer
    python evaluate_narratives.py --no-reliability    # D4 baseline, k=1
    python evaluate_narratives.py --samples 5
    python evaluate_narratives.py --out docs/narrative_evaluation.json
    python evaluate_narratives.py --show              # print each narrative

Reports faithfulness, answer relevance, citation coverage, attribution and the
served-wrong rate over ``data/narrative_gold_set.json``. Every expected figure
comes from that frozen file rather than from the narrative code, so the
numbers measure the system instead of agreeing with it.

``--no-reliability`` is the comparison that gives the reliability layer a
number: it runs one sample with no quorum and no agreement test, which is the
D4 behaviour, and the difference between the two served-wrong rates is what
D5 bought.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from src.config.settings import PROJECT_ROOT, get_settings
from src.evaluation.narrative_harness import (
    NarrativeGoldSet,
    NarrativeGoldSetError,
    evaluate_narratives,
)
from src.graph.sources import GraphSourceError
from src.narrative.evidence import EvidenceIndex
from src.narrative.factory import build_serializer
from src.narrative.generator import NarrativeGenerator
from src.pipeline.factory import build_retrieval_stack
from src.reliability.factory import retrieval_finder
from src.reliability.narrator import ReliableNarrator
from src.text_to_query.llm_client import OllamaClient
from src.utils.helpers import setup_logging

DEFAULT_GOLD_SET = PROJECT_ROOT / "data" / "narrative_gold_set.json"
WIDTH = 78


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--gold-set", type=Path, default=DEFAULT_GOLD_SET)
    parser.add_argument("--samples", type=int, help="Override k")
    parser.add_argument(
        "--no-reliability", action="store_true",
        help="One sample, no quorum, no agreement test -- the D4 baseline",
    )
    parser.add_argument(
        "--no-corrective", action="store_true", help="Disable corrective retrieval"
    )
    parser.add_argument("--offline", action="store_true", help="Lexical + graph only")
    parser.add_argument("--out", type=Path, help="Write the report as JSON")
    parser.add_argument("--show", action="store_true", help="Print each narrative")
    parser.add_argument("--label", default="", help="Tag for the report header")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(logging.INFO if args.verbose else logging.WARNING)
    settings = get_settings()

    try:
        gold_set = NarrativeGoldSet.load(args.gold_set)
    except NarrativeGoldSetError as exc:
        print(f"Gold set unusable: {exc}", file=sys.stderr)
        return 2

    reliability = settings.reliability
    if args.no_reliability:
        reliability = replace(
            reliability, samples=1, min_verified=1, min_agreement=0.0,
            enforce_attribution=False,
        )
    if args.samples:
        reliability = replace(
            reliability,
            samples=args.samples,
            min_verified=min(reliability.min_verified, args.samples),
        )
    if args.no_corrective:
        reliability = replace(reliability, corrective_enabled=False)

    try:
        stack = build_retrieval_stack(settings, enable_dense=not args.offline)
    except (GraphSourceError, ValueError) as exc:
        print(f"Could not build the retrieval stack: {exc}", file=sys.stderr)
        return 2

    index = EvidenceIndex.build(stack.graph, build_serializer(stack.graph, settings))
    narrator = ReliableNarrator(
        NarrativeGenerator(
            OllamaClient(settings.llm),
            settings.narrative,
            base_temperature=reliability.sample_temperature,
        ),
        reliability,
        settings.narrative,
        evidence=index,
        finder=retrieval_finder(stack.pipeline) if reliability.corrective_enabled else None,
    )

    label = args.label or ("D4 baseline (k=1)" if args.no_reliability else "D5 reliability layer")
    print("=" * WIDTH)
    print(f"Evaluating: {label}")
    print(f"Gold set  : {gold_set.source.name} "
          f"({len(gold_set.cases)} cases, language {gold_set.language}, "
          f"version {gold_set.version or 'unversioned'})")
    print("=" * WIDTH)

    report = evaluate_narratives(
        gold_set,
        narrator,
        label=label,
        config_summary={
            "model": settings.llm.model_name,
            "samples": reliability.samples,
            "min_verified": reliability.min_verified,
            "min_agreement": reliability.min_agreement,
            "temperature": reliability.sample_temperature,
            "min_citation_coverage": settings.narrative.min_citation_coverage,
            "corrective": reliability.corrective_enabled,
        },
    )

    print()
    for row in report.results:
        print("  " + row.describe())
        if args.show and row.served:
            print(f"      {row.text}")

    print()
    print(report.summary())

    if not report.trustworthy:
        print("\nGold-set drift detected:", file=sys.stderr)
        for row in report.gold_drift:
            for difference in row.pack_gold_diff:
                print(f"  {row.case_id}: {difference}", file=sys.stderr)
        print(
            "\nRegenerate the gold set (scripts/build_narrative_gold_set.py) "
            "or fix the payloads before trusting these metrics.",
            file=sys.stderr,
        )
        return 2

    if args.out:
        args.out.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nWrote {args.out}")

    return 0 if report.meets_served_wrong_target else 1


if __name__ == "__main__":
    raise SystemExit(main())
