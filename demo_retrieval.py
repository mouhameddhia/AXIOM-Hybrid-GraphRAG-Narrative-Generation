"""
Live demonstration of the AXIOM hybrid retrieval pipeline.

Runs against the knowledge-graph JSON by default, so it works with no database,
no network, and no model download when invoked with ``--offline``.

    python demo_retrieval.py                  # full stack: BM25 + dense + graph + rerank
    python demo_retrieval.py --offline        # BM25 + graph only, no model downloads
    python demo_retrieval.py --source neo4j   # project from a live Neo4j instance
    python demo_retrieval.py --query "..."    # ask your own question

Evaluation note: the demo grades itself against ground truth *derived from the
graph* -- the set of records reachable from the entity the question names --
rather than against an answer key typed into this file. A hard-coded expected
answer would only prove that the corpus had not changed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from src.config.settings import CorpusConfig, Settings, get_settings
from src.graph.sources import GraphSourceError
from src.pipeline.factory import RetrievalStack, build_retrieval_stack
from src.pipeline.retriever_pipeline import RetrievalOutcome
from src.retrieval.base import RetrievalResult
from src.utils.helpers import setup_logging

#: Questions exercising each modality's distinctive strength. Identifiers are
#: resolved against the live corpus at runtime, so an unknown id is reported
#: rather than silently scoring zero.
DEMO_QUESTIONS: tuple[str, ...] = (
    "What is the list price and unit cost for product SKU0001?",
    "Which competitor quoted against opportunity OPP00003 and at what price?",
    "What discount components were applied to order ORD00001?",
    "What is the discount authority limit for agent AG001?",
)

SEPARATOR = "=" * 78


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Graph-derived quality signals for one question."""

    anchor_count: int
    anchored_hit_rank: int | None
    reciprocal_rank: float
    connected_precision: float
    multi_stream_share: float

    @property
    def passed(self) -> bool:
        """
        Whether retrieval grounded the question in the right neighbourhood.

        Requires the anchored record itself in the top-k *and* a majority of
        returned evidence to be graph-connected to it -- the two conditions
        under which a downstream explanation can be genuinely grounded.
        """
        return self.anchored_hit_rank is not None and self.connected_precision >= 0.5


def evaluate(stack: RetrievalStack, outcome: RetrievalOutcome) -> QueryMetrics:
    """
    Score one outcome against ground truth derived from graph structure.

    The relevant set is the anchored nodes plus everything within the
    configured traversal radius of them: records that are, by the graph's own
    account, about the entity the question named.
    """
    anchors = outcome.anchors
    if not anchors:
        return QueryMetrics(0, None, 0.0, 0.0, 0.0)

    anchor_record_ids = {anchor.record_id for anchor in anchors}
    relevant: set[str] = set(anchor_record_ids)
    for anchor in anchors:
        for neighbour in stack.graph.traverse(
            anchor.node_id,
            max_hops=stack.settings.graph.max_hops,
            max_neighbors=stack.settings.graph.max_neighbors,
        ):
            record = stack.corpus.by_node_id(neighbour.node.node_id)
            if record is not None:
                relevant.add(record.record_id)

    final = outcome.final
    hit_rank = next(
        (
            rank
            for rank, result in enumerate(final, start=1)
            if result.record_id in anchor_record_ids
        ),
        None,
    )
    connected = sum(1 for result in final if result.record_id in relevant)
    multi_stream = sum(
        1 for result in final if result.metadata.get("stream_agreement", 1) > 1
    )
    total = len(final) or 1
    return QueryMetrics(
        anchor_count=len(anchors),
        anchored_hit_rank=hit_rank,
        reciprocal_rank=1.0 / hit_rank if hit_rank else 0.0,
        connected_precision=connected / total,
        multi_stream_share=multi_stream / total,
    )


def print_stream(name: str, results: Sequence[RetrievalResult], limit: int) -> None:
    """Render one modality's raw ranked output."""
    print(f"\n  [{name}] {len(results)} result(s)")
    if not results:
        print("      (no results -- this modality did not contribute)")
        return
    for rank, result in enumerate(results[:limit], start=1):
        detail = ""
        if "hops" in result.metadata:
            path = " -> ".join(result.metadata.get("traversal_path", ()))
            detail = f"  [{result.metadata['hops']} hop(s): {path}]"
        print(f"      {rank}. {result.record_id:<34} {result.score:>9.4f}{detail}")


def print_final(results: Sequence[RetrievalResult]) -> None:
    """Render the fused/reranked evidence with its full provenance."""
    print("\n  FINAL RANKED EVIDENCE")
    if not results:
        print("      (empty -- every candidate fell below the rerank floor)")
        return
    for rank, result in enumerate(results, start=1):
        metadata = result.metadata
        ranks = metadata.get("stream_ranks", {})
        provenance = ", ".join(f"{name}#{position}" for name, position in sorted(ranks.items()))
        print(f"      {rank}. {result.record_id:<34} {result.score:>8.4f}  [{provenance}]")
        print(f"         {_preview(result.text)}")


def _preview(text: str, width: int = 120) -> str:
    """Single-line, width-limited preview of a record's indexed text."""
    flattened = " ".join(text.split())
    return flattened if len(flattened) <= width else flattened[: width - 3] + "..."


def build_stack(args: argparse.Namespace) -> RetrievalStack:
    """Assemble the stack according to the command-line options."""
    settings = get_settings()
    if args.source:
        settings = _with_source(settings, args.source)

    client = None
    if settings.corpus.source == CorpusConfig.NEO4J:
        from src.graph.neo4j_client import Neo4jClient

        client = Neo4jClient(settings.neo4j)
        if not client.verify_connectivity():
            raise GraphSourceError(
                f"Neo4j is not reachable at {settings.neo4j.uri}. "
                f"Re-run without --source neo4j to use the JSON graph."
            )

    # --offline keeps the run free of transformer downloads. Dropping the dense
    # modality and the reranker is a configuration choice, not a separate code
    # path: fusion accepts any number of streams.
    if args.offline:
        settings = _with_rerank_disabled(settings)

    return build_retrieval_stack(
        settings,
        neo4j_client=client,
        enable_dense=not args.offline,
    )


def _with_source(settings: Settings, source: str) -> Settings:
    """Return a copy of ``settings`` with the corpus source overridden."""
    from dataclasses import replace

    return replace(settings, corpus=replace(settings.corpus, source=source))


def _with_rerank_disabled(settings: Settings) -> Settings:
    """Return a copy of ``settings`` with cross-encoder reranking switched off."""
    from dataclasses import replace

    return replace(settings, reranker=replace(settings.reranker, enabled=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--offline",
        action="store_true",
        help="BM25 + graph only; no embedding or cross-encoder downloads",
    )
    parser.add_argument(
        "--source",
        choices=(CorpusConfig.JSON, CorpusConfig.NEO4J),
        help="Override the configured graph source",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Final result count")
    parser.add_argument(
        "--per-stream", type=int, default=4, help="Per-modality results to display"
    )
    parser.add_argument(
        "--query", action="append", help="Ask a specific question (repeatable)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    setup_logging(logging.DEBUG if args.verbose else logging.WARNING)

    try:
        stack = build_stack(args)
    except (GraphSourceError, ValueError) as exc:
        print(f"Could not build the retrieval stack: {exc}", file=sys.stderr)
        return 2

    questions = tuple(args.query) if args.query else DEMO_QUESTIONS

    print(SEPARATOR)
    print("AXIOM HYBRID RETRIEVAL")
    print(SEPARATOR)
    print(f"  {stack.describe()}")
    print(f"  anchor index: {stack.anchor_resolver.indexed_value_count} resolvable values")
    print(f"  fusion: RRF k={stack.settings.fusion.rrf_k}")
    reranker = stack.settings.reranker
    print(
        f"  rerank: {'enabled' if reranker.enabled and not args.offline else 'disabled'}"
        f" (transform={reranker.score_transform}, floor={reranker.min_score})"
    )

    passed = 0
    for number, question in enumerate(questions, start=1):
        print(f"\n{SEPARATOR}\nQ{number}: {question}\n{SEPARATOR}")
        outcome = stack.pipeline.run(question, top_k=args.top_k)

        if outcome.anchors:
            print("\n  ANCHORS (query mentions grounded in the graph)")
            for anchor in outcome.anchors:
                print(
                    f"      {anchor.mention} -> {anchor.record_id} "
                    f"(via {anchor.matched_property})"
                )
        else:
            print("\n  ANCHORS: none resolved -- graph modality sits this one out")

        for name in stack.pipeline.stream_names:
            stream = outcome.stream(name)
            if stream is not None:
                print_stream(name, stream.results, args.per_stream)
        print_final(outcome.final)

        metrics = evaluate(stack, outcome)
        print("\n  GRAPH-DERIVED METRICS")
        print(f"      anchors resolved      : {metrics.anchor_count}")
        print(f"      anchored record rank  : {metrics.anchored_hit_rank or 'not retrieved'}")
        print(f"      reciprocal rank       : {metrics.reciprocal_rank:.3f}")
        print(f"      connected precision   : {metrics.connected_precision:.3f}")
        print(f"      multi-stream agreement: {metrics.multi_stream_share:.3f}")
        print(f"      verdict               : {'PASS' if metrics.passed else 'REVIEW'}")
        passed += int(metrics.passed)

        latency = ", ".join(f"{k}={v:.1f}ms" for k, v in outcome.timings_ms.items())
        print(f"      latency               : {latency}")

    print(f"\n{SEPARATOR}")
    print(f"RESULT: {passed}/{len(questions)} questions grounded successfully")
    print(SEPARATOR)
    return 0 if passed == len(questions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
