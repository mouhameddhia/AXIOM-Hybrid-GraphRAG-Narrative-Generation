"""
Ask the graph an exact question. The graph computes the number, not the model.

    python ask_graph.py "How many opportunities were lost to a competitor?"
    python ask_graph.py --live "..."       # execute against a real Neo4j
    python ask_graph.py --plan "..."       # show the query plan as well
    python ask_graph.py --schema           # print the live schema and exit
    python ask_graph.py --dry-run "..."    # build and validate, do not execute

By default runs entirely against the configured JSON graph -- no database
required. Pass ``--live`` to compile the same plan to Cypher and execute it
against a real, running Neo4j instead. Either way, the model's only job is to
describe *what* to compute; the value comes from the graph itself, and comes
back with the record ids that support it.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from src.config.settings import get_settings
from src.graph.neo4j_client import Neo4jClient
from src.graph.sources import GraphSourceError, build_graph_source
from src.text_to_query.examples import CypherExampleError
from src.text_to_query.factory import build_text_to_query_engine
from src.text_to_query.schema import build_schema_from_graph, prune_schema
from src.utils.helpers import setup_logging

WIDTH = 78
RULE = "=" * WIDTH


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("question", nargs="?", help="The question to answer")
    parser.add_argument("--plan", action="store_true", help="Show the JSON query plan")
    parser.add_argument(
        "--live", action="store_true",
        help="Execute against a live Neo4j database instead of the in-memory JSON graph",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build and validate without executing"
    )
    parser.add_argument(
        "--schema", action="store_true", help="Print the live schema and exit"
    )
    parser.add_argument(
        "--show-prompt", action="store_true", help="Show the pruned schema sent to the model"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(logging.INFO if args.verbose else logging.WARNING)
    settings = get_settings()

    try:
        graph = build_graph_source(settings).load()
    except GraphSourceError as exc:
        print(f"Could not load the graph: {exc}", file=sys.stderr)
        return 2

    if args.schema:
        schema = build_schema_from_graph(graph)
        print(
            f"Live schema from {settings.corpus.json_path.name}: "
            f"{len(schema.label_names)} labels, "
            f"{len(schema.relationship_names)} relationship types\n"
        )
        print(schema.render())
        return 0

    if not args.question:
        parser.error("a question is required (or use --schema)")

    neo4j_client = None
    if args.live:
        neo4j_client = Neo4jClient(settings.neo4j)
        if not neo4j_client.verify_connectivity():
            print(
                f"Neo4j is not reachable at {settings.neo4j.uri}.\n"
                f"Start the instance in Neo4j Desktop, or drop --live to answer "
                f"from the in-memory JSON graph instead.",
                file=sys.stderr,
            )
            return 2

    try:
        engine = build_text_to_query_engine(graph, settings, neo4j_client=neo4j_client)
    except (CypherExampleError, ValueError) as exc:
        print(f"Could not build the engine: {exc}", file=sys.stderr)
        return 2

    if args.show_prompt:
        pruned = prune_schema(args.question, engine.schema, settings.text_to_cypher)
        print(RULE)
        print("PRUNED SCHEMA SENT TO THE MODEL")
        print(RULE)
        print(pruned.render())
        print(
            f"\n({len(pruned.node_labels)} of {len(engine.schema.node_labels)} "
            f"labels kept)\n"
        )

    backend = "neo4j" if args.live else "json"

    print(RULE)
    print(f"Q: {args.question}")
    print(f"backend: {backend}" + (f"  ({settings.neo4j.uri})" if backend == "neo4j" else ""))
    print(RULE)

    result = engine.ask(args.question, execute=not args.dry_run, backend=backend)

    if not result.success:
        print(f"\nNo answer after {result.attempts} attempt(s).")
        print(f"Reason: {result.error}")
        print(
            "\nThis is a refusal, not a guess: the plan could not be validated "
            "against the graph, so nothing was computed."
        )
        return 1

    if args.plan and result.plan is not None:
        print("\nQuery plan:")
        print("    " + json.dumps(_plan_as_dict(result.plan)))

    if backend == "neo4j":
        print("\nCypher executed against Neo4j:")
    else:
        print("\nEquivalent Cypher (compiled from the plan; runs unchanged against Neo4j):")
    print(f"    {result.cypher}")

    print(f"\nAttempts: {result.attempts}"
          f"{' (self-healed)' if result.attempts > 1 else ''}")

    if args.dry_run:
        print(
            "\nDry run: the plan above was validated against the live schema "
            "but not executed."
        )
        return 0

    print("\nAnswer (computed from the graph, not by the model):")
    if result.answer:
        for key, value in result.answer.items():
            print(f"    {key} = {value}")
    else:
        print("    (no rows matched)")

    print("\nCited records:")
    if result.evidence:
        for record in result.evidence:
            print(f"    {record.record_id}")
            print(f"        {_preview(record.text)}")
    else:
        print("    (none -- a pure aggregate has no individual record behind it)")
    return 0


def _plan_as_dict(plan) -> dict[str, object]:
    """Render the plan for display, omitting fields left at their default."""
    data: dict[str, object] = {"label": plan.label}
    if plan.filters:
        data["filters"] = [
            {"property": f.property, "op": f.op, "value": f.value} for f in plan.filters
        ]
    if plan.aggregate:
        data["aggregate"] = plan.aggregate
    if plan.aggregate_property:
        data["aggregate_property"] = plan.aggregate_property
    if plan.return_records:
        data["return_records"] = True
    if plan.limit:
        data["limit"] = plan.limit
    return data


def _preview(text: str, width: int = 70) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
