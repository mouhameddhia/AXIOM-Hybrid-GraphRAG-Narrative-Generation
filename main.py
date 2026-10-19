"""
Entry point for the AXIOM hybrid retrieval pipeline.

Runs a single question through the full stack and prints the ranked, citable
evidence.

    python main.py "What is the list price of SKU0001?"
    python main.py --offline "..."      # no model downloads

The corpus is derived from the configured knowledge graph
(``CORPUS_SOURCE=json`` by default, ``neo4j`` for a live database). There is no
built-in sample corpus: evidence shown here is always real data from the
configured source, so a passing run means the real pipeline worked.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from src.config.settings import get_settings
from src.graph.sources import GraphSourceError
from src.pipeline.factory import build_retrieval_stack
from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_QUESTION = "What is the list price and unit cost for product SKU0001?"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "question", nargs="?", default=DEFAULT_QUESTION, help="Question to answer"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Lexical + graph only; skips embedding and cross-encoder downloads",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    try:
        stack = build_retrieval_stack(
            get_settings(), enable_dense=not args.offline
        )
    except (GraphSourceError, ValueError) as exc:
        logger.error("Could not build the retrieval stack: %s", exc)
        return 2

    outcome = stack.pipeline.run(args.question, top_k=args.top_k)

    print(f"\nQuestion: {args.question}")
    print(f"Corpus  : {stack.describe()}\n")

    if not outcome.final:
        print("No evidence met the configured relevance floor.")
        return 1

    print("Ranked evidence:")
    for rank, result in enumerate(outcome.final, start=1):
        ranks = result.metadata.get("stream_ranks", {})
        provenance = ", ".join(f"{name}#{pos}" for name, pos in sorted(ranks.items()))
        print(f"  {rank}. [{result.record_type}] {result.record_id}")
        print(f"     score={result.score:.4f}  via {provenance}")
        print(f"     {' '.join(result.text.split())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
