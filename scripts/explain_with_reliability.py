"""
Explain one pricing result with the full reliability layer.

    python scripts/explain_with_reliability.py                  # REC-0005
    python scripts/explain_with_reliability.py --id REC-0009
    python scripts/explain_with_reliability.py --list
    python scripts/explain_with_reliability.py --dense          # include the dense stream

The layer assesses the evidence, strengthens it by corrective retrieval when it
is weak, samples several explanations, and serves one only if enough samples
pass verification and the wrong-label check and agree. Otherwise it abstains
and prints the reason. Needs a running Ollama server with the configured model.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings  # noqa: E402
from src.narrative.contract import PriceRecommendation  # noqa: E402
from src.narrative.evidence import EvidenceIndex  # noqa: E402
from src.narrative.factory import build_serializer  # noqa: E402
from src.pipeline.factory import build_retrieval_stack  # noqa: E402
from src.reliability.factory import build_reliable_narrator, retrieval_finder  # noqa: E402
from src.utils.helpers import setup_logging  # noqa: E402

DEFAULT_PAYLOADS = PROJECT_ROOT / "data" / "track_a_payloads_eval.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--id", dest="recommendation_id", default="REC-0005")
    parser.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS)
    parser.add_argument("--list", action="store_true", help="List the available results and exit")
    parser.add_argument("--dense", action="store_true", help="Include the dense retrieval stream")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(logging.INFO if args.verbose else logging.WARNING)

    if args.list:
        entries = json.loads(args.payloads.read_text(encoding="utf-8"))["recommendations"]
        print(f"{len(entries)} result(s) in {args.payloads.name}:")
        for entry in entries:
            print(f"  {entry['recommendation_id']}  {entry['product_id']} / {entry['customer_id']}  line {entry['order_line_id']}")
        return 0

    settings = get_settings()
    stack = build_retrieval_stack(settings, enable_dense=args.dense)
    index = EvidenceIndex.build(stack.graph, build_serializer(stack.graph, settings))
    narrator = build_reliable_narrator(
        stack.graph, settings, finder=retrieval_finder(stack.pipeline), evidence=index
    )

    recommendation = PriceRecommendation.load(args.payloads, args.recommendation_id)
    result = narrator.explain(recommendation)

    print(f"Result: {args.recommendation_id}\n")
    print(result.describe())
    print("\n" + "-" * 78)
    print(result.text if result.served else f"ABSTAINED: {result.reason}")
    print("-" * 78)
    return 0 if result.served else 1


if __name__ == "__main__":
    raise SystemExit(main())
