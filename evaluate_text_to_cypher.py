"""
Evaluate text-to-query against the starter gold set.

    python evaluate_text_to_cypher.py              # in-memory JSON graph, no database
    python evaluate_text_to_cypher.py --live       # live Neo4j database
    python evaluate_text_to_cypher.py --compare    # both backends, side by side
    python evaluate_text_to_cypher.py --cases      # per-case detail
    python evaluate_text_to_cypher.py --plans      # show the plan and Cypher per case
    python evaluate_text_to_cypher.py --json out.json

Measures the two things the gate asks for:

**Correctness** -- does the generated query return the value the graph
actually holds? Expected values were computed independently from the source
CSVs/JSON, so this compares against the data, not against the engine's own
previous output.

**Safety** -- every plan is validated against the live schema before it runs,
and a plan is structurally incapable of expressing a write.

Two execution backends answer the same questions the same way:

**json** (default) executes each plan directly against the in-memory graph
loaded from the JSON export. No database is required.

**neo4j** (``--live``) compiles the identical plan to Cypher and executes it
against a real, running Neo4j database -- the same database
``scripts/import_synthetic_data.py`` populated from the source CSVs.
``--compare`` runs the whole gold set through both and reports whether they
agree, which is the actual proof that the compiled Cypher is correct, not
merely that it looks plausible.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from src.config.settings import PROJECT_ROOT, get_settings
from src.evaluation.cypher_harness import (
    CypherEvaluationReport,
    CypherGoldSet,
    CypherGoldSetError,
    evaluate_cypher,
)
from src.graph.neo4j_client import Neo4jClient
from src.graph.sources import GraphSourceError, build_graph_source
from src.text_to_query.examples import CypherExampleError
from src.text_to_query.factory import build_text_to_query_engine
from src.utils.helpers import setup_logging

DEFAULT_GOLD_SET = PROJECT_ROOT / "data" / "cypher_gold_set.json"
WIDTH = 92
RULE = "=" * WIDTH


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--gold-set", type=Path, default=DEFAULT_GOLD_SET)
    parser.add_argument(
        "--live", action="store_true",
        help="Execute against a live Neo4j database instead of the in-memory JSON graph",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both backends over the same gold set and report whether they agree "
             "(needs a reachable Neo4j)",
    )
    parser.add_argument("--cases", action="store_true", help="Show the per-case table")
    parser.add_argument(
        "--plans", action="store_true", help="Show the plan and compiled Cypher per case"
    )
    parser.add_argument("--json", type=Path, help="Write machine-readable results here")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(logging.INFO if args.verbose else logging.WARNING)
    settings = get_settings()

    try:
        graph = build_graph_source(settings).load()
        gold_set = CypherGoldSet.load(args.gold_set)
    except (GraphSourceError, CypherGoldSetError) as exc:
        print(f"Could not start: {exc}", file=sys.stderr)
        return 2

    neo4j_client = None
    if args.live or args.compare:
        neo4j_client = Neo4jClient(settings.neo4j)
        if not neo4j_client.verify_connectivity():
            print(
                f"Neo4j is not reachable at {settings.neo4j.uri}.\n"
                f"Start the instance in Neo4j Desktop, or drop --live/--compare "
                f"to evaluate against the in-memory JSON graph only.",
                file=sys.stderr,
            )
            return 2

    try:
        engine = build_text_to_query_engine(graph, settings, neo4j_client=neo4j_client)
    except (CypherExampleError, ValueError) as exc:
        print(f"Could not build the engine: {exc}", file=sys.stderr)
        return 2

    print(RULE)
    print("TEXT-TO-QUERY GOLD SET")
    print(RULE)
    print(f"  graph    : {graph.stats.node_count} nodes / {graph.stats.edge_count} relationships")
    print(f"  gold set : {len(gold_set.cases)} cases from {gold_set.source.name}")
    print(f"  model    : {settings.llm.model_name}")
    print(f"  retries  : up to {settings.text_to_cypher.max_retries} per question")

    if args.compare:
        return _run_compare(engine, gold_set, args, settings)

    backend = "neo4j" if args.live else "json"
    print(f"  backend  : {backend}"
          + (f" ({settings.neo4j.uri})" if backend == "neo4j" else " (no database)"))

    report = evaluate_cypher(engine, gold_set, label=backend, backend=backend)
    _print_report(report, show_cases=args.cases, show_plans=args.plans)

    if args.json:
        args.json.write_text(
            json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nWrote machine-readable results to {args.json}")

    return 0 if report.accuracy == 1.0 else 1


def _run_compare(engine, gold_set, args, settings) -> int:
    """Run the gold set through both backends and report whether they agree."""
    print(f"  backend  : json AND neo4j ({settings.neo4j.uri})")

    json_report = evaluate_cypher(engine, gold_set, label="json", backend="json")
    neo4j_report = evaluate_cypher(engine, gold_set, label="neo4j", backend="neo4j")

    print(f"\n{RULE}")
    print("BACKEND AGREEMENT")
    print(RULE)
    print(f"\n  {'case':<32} {'json':>12} {'neo4j':>12}  agree?")
    print("  " + "-" * (WIDTH - 4))

    agree_count = 0
    by_id = {row.case.case_id: row for row in neo4j_report.results}
    for json_row in json_report.results:
        neo4j_row = by_id.get(json_row.case.case_id)
        json_val = json_row.result.value
        neo4j_val = neo4j_row.result.value if neo4j_row else None
        agree = _values_agree(json_val, neo4j_val)
        agree_count += agree
        print(
            f"  {json_row.case.case_id:<32} {str(json_val):>12} {str(neo4j_val):>12}  "
            f"{'yes' if agree else 'NO -- DISAGREE'}"
        )

    print(f"\n  {'':32} {'accuracy':>12} {'accuracy':>12}")
    print(f"  {'(vs. independently-computed expected value)':<32} "
          f"{json_report.accuracy:>11.1%} {neo4j_report.accuracy:>11.1%}")

    print(f"\n{RULE}")
    print("SUMMARY")
    print(RULE)
    print(f"\n  backends agree on the same value: {agree_count}/{len(json_report.results)}")
    print(f"  json  correct vs. expected      : {json_report.accuracy:.1%}")
    print(f"  neo4j correct vs. expected      : {neo4j_report.accuracy:.1%}")

    if args.cases:
        print("\n  -- json backend --")
        _print_report(json_report, show_cases=True, show_plans=args.plans)
        print("\n  -- neo4j backend --")
        _print_report(neo4j_report, show_cases=True, show_plans=args.plans)

    if args.json:
        combined = {"json": json_report.as_dict(), "neo4j": neo4j_report.as_dict()}
        args.json.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote machine-readable results to {args.json}")

    return 0 if agree_count == len(json_report.results) else 1


def _values_agree(a, b, tolerance: float = 0.01) -> bool:
    """Compare two answer values, tolerating int/float and None mismatches."""
    if a is None or b is None:
        return a == b
    try:
        return abs(float(a) - float(b)) <= tolerance
    except (TypeError, ValueError):
        return str(a) == str(b)


def _print_report(report: CypherEvaluationReport, *, show_cases: bool, show_plans: bool) -> None:
    """Render the correctness report."""
    print(f"\n{RULE}")
    print(f"RESULTS ({report.label})" if report.label else "RESULTS")
    print(RULE)
    print(f"\n  correct answers : {report.accuracy:.1%}")
    print(f"  answered at all : {report.answered:.1%}")
    print(f"  mean attempts   : {report.mean_attempts:.2f}")
    print(f"  self-healed     : {report.self_healed} case(s) recovered after a failure")

    if show_cases:
        print(f"\n  {'case':<32} {'expected':>12} {'got':>14} {'att':>4}  ok")
        print("  " + "-" * (WIDTH - 4))
        for row in report.results:
            got = (
                row.result.answer.get(row.matched_field)
                if row.matched_field
                else (row.result.value if row.result.answer else "-")
            )
            print(
                f"  {row.case.case_id:<32} {str(row.case.expect_value):>12} "
                f"{str(got):>14} {row.result.attempts:>4}  {'yes' if row.passed else 'NO'}"
            )

    if show_plans:
        print(f"\n{RULE}")
        print("GENERATED QUERIES")
        print(RULE)
        for row in report.results:
            print(f"\n  [{'ok ' if row.passed else 'NO '}] {row.case.case_id}")
            print(f"       q: {row.case.question}")
            if row.result.cypher:
                print(f"       cypher: {row.result.cypher}")
            if row.result.citations:
                print(f"       cites : {', '.join(row.result.citations)}")

    failures = report.failures()
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for row in failures:
            print(f"    - {row.case.case_id}: expected {row.case.expect_value}")
            print(f"      q: {row.case.question}")
            if row.result.cypher:
                print(f"      generated: {row.result.cypher}")
            if row.result.answer:
                print(f"      returned : {dict(row.result.answer)}")
            if row.result.error:
                print(f"      error    : {row.result.error}")


if __name__ == "__main__":
    raise SystemExit(main())
