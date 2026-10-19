"""
End-to-end: retrieval -> text-to-query -> grounded narrative (D2 + D3 + D4).

    python explain_end_to_end.py                     # REC-0001, full chain
    python explain_end_to_end.py --id REC-0002
    python explain_end_to_end.py --offline           # skip the dense retriever
    python explain_end_to_end.py --no-graph-facts    # Track A's numbers only

Runs the whole Track B pipeline over one Track A result object, printing each
stage so the chain is visible rather than implied:

    stage 1  Track A payload        the pricing numbers, validated at the boundary
    stage 2  hybrid retrieval       surrounding evidence for this product/customer
    stage 3  text-to-query          exact context facts, computed BY THE GRAPH
    stage 4  grounded narrative     one explanation, every number traced and cited

The seam that makes this a pipeline rather than three programs: all three
stages share one graph and one ``RecordSerializer``, so a record cited in the
narrative is the same handle retrieval ranked and text-to-query returned.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from src.config.settings import PROJECT_ROOT, get_settings
from src.graph.sources import GraphSourceError
from src.narrative.contract import ContractError, PriceRecommendation
from src.narrative.evidence import EvidenceIndex
from src.narrative.facts import build_facts
from src.narrative.factory import build_narrative_generator
from src.narrative.graph_facts import GraphFact, fact_from_answer
from src.pipeline.factory import build_retrieval_stack
from src.text_to_query.examples import CypherExampleError
from src.text_to_query.factory import build_text_to_query_engine
from src.utils.helpers import setup_logging

DEFAULT_PAYLOADS = PROJECT_ROOT / "data" / "track_a_sample_payloads.json"
WIDTH = 78
RULE = "=" * WIDTH


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--id", dest="recommendation_id")
    parser.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--offline", action="store_true", help="Lexical + graph retrieval only"
    )
    parser.add_argument(
        "--no-graph-facts", action="store_true",
        help="Skip stage 3; explain Track A's numbers alone",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(logging.INFO if args.verbose else logging.WARNING)
    settings = get_settings()

    # -- stage 1: the Track A result object --------------------------------
    try:
        recommendation = PriceRecommendation.load(args.payloads, args.recommendation_id)
    except ContractError as exc:
        print(f"Track A payload refused at the boundary: {exc}", file=sys.stderr)
        return 2

    try:
        stack = build_retrieval_stack(settings, enable_dense=not args.offline)
    except (GraphSourceError, ValueError) as exc:
        print(f"Could not build the retrieval stack: {exc}", file=sys.stderr)
        return 2

    print(RULE)
    print("STAGE 1 -- Track A result object (the numbers; Track B computes none)")
    print(RULE)
    print(f"  {recommendation.recommendation_id}: {recommendation.product_id} "
          f"to {recommendation.customer_id}")
    print(f"  recommended {recommendation.recommended_price:,.2f} "
          f"{recommendation.currency_code} "
          f"(range {recommendation.price_range.low:,.2f}-"
          f"{recommendation.price_range.high:,.2f})")
    print(f"  confidence {recommendation.confidence:.0%}, "
          f"margin {recommendation.expected_margin_pct}%, "
          f"acceptance {recommendation.acceptance_probability:.0%}")
    if recommendation.waterfall is not None:
        w = recommendation.waterfall
        print(f"  waterfall: list {w.list_price:,.2f} -> invoice {w.invoice_price:,.2f} "
              f"-> pocket {w.pocket_price:,.2f} ({len(w.steps)} deductions)")
    print(f"  cites {len(recommendation.evidence_record_ids)} record(s); "
          f"engine {recommendation.engine_version}")
    print(f"\n  corpus: {stack.describe()}")

    # -- stage 2: hybrid retrieval ----------------------------------------
    query = (
        f"{recommendation.product_id} {recommendation.customer_id} "
        f"price discount deal"
    )
    print(f"\n{RULE}")
    print("STAGE 2 -- Hybrid retrieval (D2): surrounding evidence")
    print(RULE)
    print(f"  query: {query}")
    outcome = stack.pipeline.run(query, top_k=args.top_k)
    if outcome.final:
        print(f"\n  top {len(outcome.final)} of the ranked evidence:")
        for rank, result in enumerate(outcome.final, start=1):
            ranks = result.metadata.get("stream_ranks", {})
            provenance = ", ".join(f"{n}#{p}" for n, p in sorted(ranks.items()))
            print(f"    {rank}. {result.record_id:<34} {result.score:.4f}  [{provenance}]")
    else:
        print("\n  no evidence met the relevance floor")

    evidence_index = EvidenceIndex.build(stack.graph, stack.serializer)
    missing = evidence_index.missing(recommendation.evidence_record_ids)
    print(f"\n  Track A's cited records resolved: "
          f"{len(recommendation.evidence_record_ids) - len(missing)}"
          f"/{len(recommendation.evidence_record_ids)}")
    for handle in recommendation.evidence_record_ids:
        record = evidence_index.resolve(handle)
        marker = "  " if record else "!!"
        detail = record.preview(60) if record else "NOT FOUND IN GRAPH"
        print(f"    {marker} {handle:<34} {detail}")

    # -- stage 3: text-to-query -------------------------------------------
    pack = build_facts(recommendation)
    graph_facts: list[GraphFact] = []

    if not args.no_graph_facts:
        print(f"\n{RULE}")
        print("STAGE 3 -- Text-to-query (D3): context computed BY THE GRAPH")
        print(RULE)
        try:
            engine = build_text_to_query_engine(stack.graph, settings)
        except (CypherExampleError, ValueError) as exc:
            print(f"  Could not build the engine: {exc}", file=sys.stderr)
            return 2

        for spec in _context_questions(recommendation):
            answer = engine.ask(spec["question"])
            fact = fact_from_answer(
                answer,
                key=spec["key"],
                label_en=spec["label"],
                kind=spec["kind"],
            )
            print(f"\n  Q: {spec['question']}")
            if fact is None:
                print(f"     no usable answer -- this fact is simply omitted"
                      f"{f' ({answer.error})' if answer.error else ''}")
                continue
            graph_facts.append(fact)
            print(f"     cypher: {fact.cypher}")
            print(f"     -> {fact.fact.formatted(pack.currency_code)}  "
                  f"[{fact.fact.citation}]"
                  f"{'' if fact.is_record_backed else '  (query-backed, not record-backed)'}")

        pack = pack.extended([gf.fact for gf in graph_facts])

    # -- stage 4: grounded narrative --------------------------------------
    print(f"\n{RULE}")
    print("STAGE 4 -- Grounded narrative (D4)")
    print(RULE)
    print(f"\n  The closed fact set ({len(pack.facts)} figures, "
          f"{len(pack.citations)} citable records):")
    print("\n".join(f"  {line}" for line in pack.render("en").splitlines()))

    generator = build_narrative_generator(settings)
    narrative = generator.explain(pack, language="en", evidence=evidence_index)

    print(f"\n{'-' * WIDTH}")
    if not narrative.success:
        print(f"ABSTAINED after {narrative.attempts} attempt(s).")
        print(f"Reason: {narrative.reason}")
        if narrative.verification is not None:
            print(narrative.verification.summary())
        print("\nNo explanation is returned. That is the designed outcome.")
        return 1

    print(narrative.text)
    print(f"{'-' * WIDTH}")
    report = narrative.verification
    print(f"\n  verified on attempt {narrative.attempts}"
          f"{' (self-corrected)' if narrative.attempts > 1 else ''}")
    if report is not None:
        print(f"  all {len(report.numbers_found)} number(s) stated trace to the fact set")
    print(f"  cited {len(narrative.citations)} record(s): "
          f"{', '.join(narrative.citations)}")

    print(f"\n{RULE}")
    print("CHAIN SUMMARY")
    print(RULE)
    print(f"  Track A supplied      : {len(recommendation.evidence_record_ids)} record ids, "
          f"{len(build_facts(recommendation).facts)} figures")
    print(f"  retrieval ranked      : {len(outcome.final)} evidence records")
    print(f"  the graph computed    : {len(graph_facts)} context fact(s) via text-to-query")
    print(f"  the model computed    : 0 numbers  <- the rule this track is built on")
    print(f"  narrative cited       : {len(narrative.citations)} record(s)")
    return 0


def _context_questions(recommendation: PriceRecommendation) -> list[dict[str, str]]:
    """
    Context worth asking the graph about this recommendation.

    Phrased as natural-language questions on purpose: this is the D3 engine
    doing its actual job, not a hand-written Cypher shortcut.
    """
    product = recommendation.product_id
    return [
        {
            "key": "graph.list_price",
            "question": f"What is the list price of product {product}?",
            "label": f"current list price of {product} per unit",
            "kind": "money",
        },
        {
            "key": "graph.unit_cost",
            "question": f"What is the unit cost of product {product}?",
            "label": f"unit cost of {product}",
            "kind": "money",
        },
    ]


if __name__ == "__main__":
    raise SystemExit(main())
