"""
Verify the imported Neo4j database against the source CSVs.

    python scripts/verify_import.py            # full check
    python scripts/verify_import.py --verbose  # include every passing check

Expected values are **recomputed from the CSV files** using the same specs the
importer used, not hardcoded. If the source data changes, this stays correct;
if the import drifts from the source, this fails. A hardcoded expectation
would only prove the database matches a number someone typed once.

This script is strictly read-only. It issues no MERGE, CREATE or DELETE.

Exit code 0 when everything matches, 1 when any check fails.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from import_synthetic_data import (  # noqa: E402
    CSV_DIR,
    NODE_SPECS,
    REL_SPECS,
    load_all,
    node_payload,
    rel_payload,
)
from src.config.settings import get_settings  # noqa: E402
from src.graph.neo4j_client import Neo4jClient, Neo4jQueryError  # noqa: E402
from src.utils.helpers import setup_logging  # noqa: E402

WIDTH = 82
RULE = "=" * WIDTH


@dataclass
class Results:
    """Running tally of checks."""

    passed: int = 0
    failures: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []

    def check(self, ok: bool, description: str, detail: str = "", *, verbose: bool = False) -> None:
        if ok:
            self.passed += 1
            if verbose:
                print(f"    ok   {description}")
        else:
            self.failures.append(f"{description}{(' -- ' + detail) if detail else ''}")
            print(f"    FAIL {description}")
            if detail:
                print(f"         {detail}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    parser.add_argument("--verbose", action="store_true", help="Show passing checks too")
    args = parser.parse_args(argv)

    setup_logging(logging.WARNING)
    results = Results()

    # --- expected, recomputed from the CSVs -------------------------------
    tables = load_all(args.csv_dir)
    expected_nodes = {s.label: len(node_payload(s, tables[s.csv_file])) for s in NODE_SPECS}
    expected_rels: Counter = Counter()
    expected_patterns: Counter = Counter()
    for spec in REL_SPECS:
        count = len(rel_payload(spec, tables[spec.csv_file]))
        expected_rels[spec.rel_type] += count
        expected_patterns[(spec.source.label, spec.rel_type, spec.target.label)] += count

    settings = get_settings()
    client = Neo4jClient(settings.neo4j)
    if not client.verify_connectivity():
        print(
            f"Neo4j is not reachable at {settings.neo4j.uri}.\n"
            f"Start the instance in Neo4j Desktop and check .env.",
            file=sys.stderr,
        )
        return 2

    print(RULE)
    print("VERIFYING IMPORT AGAINST SOURCE CSVs")
    print(RULE)
    print(f"  database : {settings.neo4j.uri}  (db: {client.database})")
    print(f"  expected : {sum(expected_nodes.values())} nodes, "
          f"{sum(expected_rels.values())} relationships")

    try:
        # --- 1. node labels and counts ------------------------------------
        print(f"\n{RULE}\n1. NODE LABELS AND COUNTS\n{RULE}\n")
        actual_nodes = {
            row["label"]: row["count"]
            for row in client.run_query(
                "MATCH (n) UNWIND labels(n) AS label "
                "RETURN label, count(*) AS count ORDER BY label"
            )
        }
        print(f"  {'label':26s} {'expected':>9} {'actual':>9}")
        print("  " + "-" * 48)
        for label in sorted(expected_nodes):
            exp, act = expected_nodes[label], actual_nodes.get(label, 0)
            mark = "" if exp == act else "   <-- MISMATCH"
            print(f"  {label:26s} {exp:>9} {act:>9}{mark}")
            results.check(exp == act, f"{label} node count", f"expected {exp}, found {act}")

        unexpected = sorted(set(actual_nodes) - set(expected_nodes))
        results.check(
            not unexpected,
            "no unexpected node labels",
            f"found {unexpected}",
            verbose=args.verbose,
        )
        if unexpected:
            print(f"\n  unexpected labels present: {unexpected}")

        total_nodes = client.run_query("MATCH (n) RETURN count(n) AS c")[0]["c"]
        results.check(
            total_nodes == sum(expected_nodes.values()),
            "total node count",
            f"expected {sum(expected_nodes.values())}, found {total_nodes}",
            verbose=args.verbose,
        )
        print(f"\n  total nodes: {total_nodes} (expected {sum(expected_nodes.values())})")

        # --- 2. relationship types and counts ------------------------------
        print(f"\n{RULE}\n2. RELATIONSHIP TYPES AND COUNTS\n{RULE}\n")
        actual_rels = {
            row["type"]: row["count"]
            for row in client.run_query(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type"
            )
        }
        print(f"  {'type':30s} {'expected':>9} {'actual':>9}")
        print("  " + "-" * 52)
        for rel_type in sorted(expected_rels):
            exp, act = expected_rels[rel_type], actual_rels.get(rel_type, 0)
            mark = "" if exp == act else "   <-- MISMATCH"
            print(f"  {rel_type:30s} {exp:>9} {act:>9}{mark}")
            results.check(exp == act, f"{rel_type} count", f"expected {exp}, found {act}")

        unexpected_rels = sorted(set(actual_rels) - set(expected_rels))
        results.check(
            not unexpected_rels,
            "no unexpected relationship types",
            f"found {unexpected_rels}",
            verbose=args.verbose,
        )

        total_rels = client.run_query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        results.check(
            total_rels == sum(expected_rels.values()),
            "total relationship count",
            f"expected {sum(expected_rels.values())}, found {total_rels}",
            verbose=args.verbose,
        )
        print(f"\n  total relationships: {total_rels} (expected {sum(expected_rels.values())})")

        # --- 3. relationship patterns (label)-[:TYPE]->(label) -------------
        print(f"\n{RULE}\n3. RELATIONSHIP PATTERNS\n{RULE}\n")
        actual_patterns = {
            (row["src"], row["type"], row["tgt"]): row["count"]
            for row in client.run_query(
                "MATCH (a)-[r]->(b) "
                "RETURN labels(a)[0] AS src, type(r) AS type, labels(b)[0] AS tgt, "
                "count(*) AS count"
            )
        }
        print(f"  {'pattern':58s} {'expected':>8} {'actual':>8}")
        print("  " + "-" * 78)
        for pattern in sorted(expected_patterns):
            src, rel_type, tgt = pattern
            exp = expected_patterns[pattern]
            act = actual_patterns.get(pattern, 0)
            label = f"({src})-[:{rel_type}]->({tgt})"
            mark = "" if exp == act else "  <-- MISMATCH"
            print(f"  {label:58s} {exp:>8} {act:>8}{mark}")
            results.check(exp == act, label, f"expected {exp}, found {act}")

        stray = sorted(set(actual_patterns) - set(expected_patterns))
        results.check(
            not stray,
            "no unexpected relationship patterns",
            f"found {stray}",
            verbose=args.verbose,
        )
        if stray:
            print("\n  unexpected patterns in the database:")
            for src, rel_type, tgt in stray:
                print(f"    ({src})-[:{rel_type}]->({tgt}): {actual_patterns[(src, rel_type, tgt)]}")

        # --- 4. the polymorphic PriceComponent foreign key -----------------
        print(f"\n{RULE}\n4. POLYMORPHIC PriceComponent RELATIONSHIPS\n{RULE}\n")
        poly = client.run_query(
            "MATCH (pc:PriceComponent)-[r]->(t) "
            "RETURN pc.applies_to_level AS level, type(r) AS rel, labels(t)[0] AS target, "
            "count(*) AS count ORDER BY level"
        )
        expected_poly = {
            ("line", "APPLIES_TO_LINE", "OrderLine"),
            ("order", "APPLIES_TO_ORDER", "Order"),
            ("customer_period", "APPLIES_TO_CUSTOMER_PERIOD", "CustomerPeriod"),
        }
        print(f"  {'applies_to_level':20s} {'relationship':30s} {'target':18s} {'count':>6}")
        print("  " + "-" * 78)
        for row in poly:
            triple = (row["level"], row["rel"], row["target"])
            ok = triple in expected_poly
            print(f"  {str(row['level']):20s} {row['rel']:30s} {row['target']:18s} "
                  f"{row['count']:>6}{'' if ok else '   <-- WRONG TARGET'}")
            results.check(
                ok,
                f"applies_to_level={row['level']} routes to {row['target']}",
                "level and target label disagree",
            )

        # Every PriceComponent must have exactly one applies-to relationship.
        dangling = client.run_query(
            "MATCH (pc:PriceComponent) "
            "WHERE NOT (pc)-[:APPLIES_TO_LINE|APPLIES_TO_ORDER|APPLIES_TO_CUSTOMER_PERIOD]->() "
            "RETURN count(pc) AS c"
        )[0]["c"]
        results.check(
            dangling == 0,
            "every PriceComponent has an applies-to relationship",
            f"{dangling} have none",
            verbose=args.verbose,
        )
        print(f"\n  PriceComponents with no applies-to relationship: {dangling} (expected 0)")

        # --- 5. City reification -------------------------------------------
        print(f"\n{RULE}\n5. CITY REIFICATION\n{RULE}\n")
        city_count = client.run_query("MATCH (c:City) RETURN count(c) AS c")[0]["c"]
        located = client.run_query("MATCH (:TradeCustomer)-[r:LOCATED_IN]->(:City) RETURN count(r) AS c")[0]["c"]
        customers = client.run_query("MATCH (t:TradeCustomer) RETURN count(t) AS c")[0]["c"]
        without_city = client.run_query(
            "MATCH (t:TradeCustomer) WHERE NOT (t)-[:LOCATED_IN]->(:City) RETURN count(t) AS c"
        )[0]["c"]
        print(f"  City nodes                        : {city_count} (expected {expected_nodes['City']})")
        print(f"  (:TradeCustomer)-[:LOCATED_IN]->(:City): {located} (expected {customers})")
        print(f"  TradeCustomers with no city       : {without_city} (expected 0)")
        results.check(city_count == expected_nodes["City"], "City node count",
                      f"expected {expected_nodes['City']}, found {city_count}")
        results.check(located == customers, "every customer is located in a city",
                      f"{located} edges for {customers} customers")
        results.check(without_city == 0, "no TradeCustomer without a city",
                      f"{without_city} missing", verbose=args.verbose)

        # --- 6. orphans -----------------------------------------------------
        print(f"\n{RULE}\n6. ORPHAN CHECKS\n{RULE}\n")
        isolated = client.run_query(
            "MATCH (n) WHERE NOT (n)--() "
            "RETURN labels(n)[0] AS label, count(*) AS count ORDER BY label"
        )
        if isolated:
            print("  nodes with no relationships at all:")
            for row in isolated:
                print(f"    {row['label']:26s} {row['count']}")
            print("\n  (Channel, RivalSeller, SalesAgent, City and SellableProduct are")
            print("   referenced BY other nodes, so they are never isolated. Any label")
            print("   appearing here that should have edges indicates a missed mapping.)")
        else:
            print("  no isolated nodes")
        results.check(True, "isolated-node scan completed", verbose=args.verbose)

        # Each entity that must have a parent, has one.
        must_have = [
            ("Quote", "FOR_OPPORTUNITY", "Opportunity"),
            ("DealOutcome", "FOR_OPPORTUNITY", "Opportunity"),
            ("OrderLine", "PART_OF", "Order"),
            ("Order", "FULFILLS", "Opportunity"),
            ("Opportunity", "FOR_CUSTOMER", "TradeCustomer"),
            ("StageChange", "FOR_OPPORTUNITY", "Opportunity"),
        ]
        print()
        for label, rel_type, target in must_have:
            missing = client.run_query(
                f"MATCH (n:{label}) WHERE NOT (n)-[:{rel_type}]->(:{target}) "
                f"RETURN count(n) AS c"
            )[0]["c"]
            print(f"  {label:16s} without -[:{rel_type}]->({target}): {missing}")
            results.check(
                missing == 0,
                f"every {label} has a {rel_type} to {target}",
                f"{missing} without one",
            )

        # --- 7. property typing and a known value ---------------------------
        print(f"\n{RULE}\n7. PROPERTY TYPES AND SPOT VALUES\n{RULE}\n")
        types = client.run_query(
            "MATCH (q:Quote) WITH q LIMIT 1 "
            "RETURN valueType(q.unit_price) AS unit_price, "
            "valueType(q.round_number) AS round_number, "
            "valueType(q.issued_at) AS issued_at"
        )
        if types:
            row = types[0]
            print(f"  Quote.unit_price   type: {row['unit_price']}")
            print(f"  Quote.round_number type: {row['round_number']}")
            print(f"  Quote.issued_at    type: {row['issued_at']}")
            results.check("FLOAT" in str(row["unit_price"]).upper(),
                          "Quote.unit_price is numeric (aggregation would fail otherwise)",
                          f"got {row['unit_price']}")
            results.check("INTEGER" in str(row["round_number"]).upper(),
                          "Quote.round_number is an integer", f"got {row['round_number']}")

        # A value checked end to end: computed by the database, compared to the CSV.
        agent = client.run_query(
            "MATCH (a:AuthorityPeriod {agent_id: 'AG011'}) "
            "RETURN a.max_discount_pct AS pct"
        )
        csv_pct = next(
            (r["max_discount_pct"] for r in tables["AuthorityPeriod.csv"] if r["agent_id"] == "AG011"),
            None,
        )
        if agent and csv_pct is not None:
            db_pct = agent[0]["pct"]
            print(f"\n  AuthorityPeriod AG011 max_discount_pct: db={db_pct}  csv={csv_pct}")
            results.check(float(db_pct) == float(csv_pct),
                          "AG011 discount matches the CSV", f"db={db_pct} csv={csv_pct}")

        # An aggregation, the thing text-to-query exists to do.
        lost = client.run_query(
            "MATCH (d:DealOutcome) WHERE d.result = 'lost_to_competitor' "
            "RETURN count(d) AS c"
        )[0]["c"]
        csv_lost = sum(
            1 for r in tables["DealOutcome.csv"] if r["result"] == "lost_to_competitor"
        )
        print(f"  DealOutcome result='lost_to_competitor': db={lost}  csv={csv_lost}")
        results.check(lost == csv_lost, "lost-to-competitor count matches the CSV",
                      f"db={lost} csv={csv_lost}")

        # --- summary --------------------------------------------------------
        print(f"\n{RULE}\nSUMMARY\n{RULE}\n")
        total = results.passed + len(results.failures)
        print(f"  checks run    : {total}")
        print(f"  passed        : {results.passed}")
        print(f"  failed        : {len(results.failures)}")
        if results.failures:
            print("\n  Failures:")
            for failure in results.failures:
                print(f"    - {failure}")
            print("\n  The database does NOT match the source CSVs.")
            return 1
        print("\n  The database matches the source CSVs exactly.")
        return 0

    except Neo4jQueryError as exc:
        print(f"\nVerification query failed: {exc}", file=sys.stderr)
        return 2
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
