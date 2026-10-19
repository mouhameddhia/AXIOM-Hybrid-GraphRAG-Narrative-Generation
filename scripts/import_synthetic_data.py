"""
Import the Syntetic_Data_PoC CSV export into Neo4j.

    python scripts/import_synthetic_data.py --dry-run     # validate only, no writes
    python scripts/import_synthetic_data.py               # import
    python scripts/import_synthetic_data.py --batch-size 1000

The mapping from CSV to graph is declared as data in NODE_SPECS and
REL_SPECS below, derived from an inspection of the actual files rather than
assumed. Everything else in this script is a generic executor over those
specs, so reviewing the mapping means reading two lists, not the whole file.

Safety
------
This script only ever runs MERGE, CREATE CONSTRAINT and CREATE INDEX. It
contains no DELETE, no DETACH DELETE, and no DROP: it cannot remove data that
is already in the database, and re-running it is safe. MERGE means a second
run matches what the first run created instead of duplicating it.

Design decisions (agreed before implementation)
-----------------------------------------------
* **City is reified.** The CSVs carry city as a flat string on TradeCustomer;
  the JSON export models it as its own node. City nodes are built from the
  distinct values so the Neo4j graph matches the JSON structurally.
* **Two redundant edges are deliberately NOT created**, matching the JSON:
  ``StageChange.changed_by`` stays a property, and ``Order.customer_id`` stays
  a property (the customer is reachable via FULFILLS -> Opportunity ->
  FOR_CUSTOMER).
* **CUST0008 is imported as-is.** Its CustomerClassPeriod row has
  ``valid_from`` after ``valid_to``; the row is loaded unchanged and the
  problem is reported in the summary rather than silently repaired.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Make `src` importable when run as `python scripts/import_synthetic_data.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings  # noqa: E402
from src.graph.neo4j_client import Neo4jClient, Neo4jQueryError, WriteCounters  # noqa: E402
from src.utils.helpers import setup_logging  # noqa: E402

logger = logging.getLogger("import")

CSV_DIR = PROJECT_ROOT / "data" / "Syntetic_Data_PoC" / "Syntetic_Data_PoC"

WIDTH = 78
RULE = "=" * WIDTH


# ---------------------------------------------------------------------------
# The mapping, as data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """How one CSV becomes one node label."""

    label: str
    csv_file: str
    #: Columns whose values together identify the node. A single column where
    #: the table has a natural primary key; several for period and event
    #: tables that have none.
    key_columns: tuple[str, ...]
    #: Columns to store as properties. Empty means every column in the file.
    property_columns: tuple[str, ...] = ()
    #: CSV column name -> graph property name, where they differ.
    rename: dict[str, str] = field(default_factory=dict)
    #: True for a label derived from distinct values of another table's column
    #: (City, built from TradeCustomer.city).
    distinct: bool = False

    def key_properties(self) -> tuple[str, ...]:
        """The property names the key columns land on after renaming."""
        return tuple(self.rename.get(col, col) for col in self.key_columns)


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One end of a relationship: which node, and which columns identify it."""

    label: str
    #: Property names on the target node.
    key_properties: tuple[str, ...]
    #: Columns in the current CSV row supplying those values, in the same order.
    key_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RelSpec:
    """How columns in one CSV become one relationship type."""

    rel_type: str
    csv_file: str
    source: Endpoint
    target: Endpoint
    #: Skip the row when any of these columns is empty -- a nullable foreign
    #: key means "no relationship", not "broken relationship".
    required_columns: tuple[str, ...] = ()
    #: Restricts the spec to a subset of rows. Used for the one polymorphic
    #: foreign key, where the target label depends on a sibling column.
    row_filter: Callable[[dict[str, str]], bool] | None = None
    note: str = ""


# Node labels. Key columns were verified unique over the real files before
# being chosen: a MERGE on a non-unique key would silently collapse distinct
# rows into one node and lose data.
NODE_SPECS: tuple[NodeSpec, ...] = (
    # --- natural single-column primary keys ---
    NodeSpec("Channel", "Channel.csv", ("channel_id",)),
    NodeSpec("Opportunity", "Opportunity.csv", ("opportunity_id",)),
    NodeSpec("Order", "Order.csv", ("order_id",)),
    NodeSpec("OrderLine", "OrderLine.csv", ("order_line_id",)),
    NodeSpec("PriceComponent", "PriceComponent.csv", ("component_id",)),
    NodeSpec("Quote", "Quote.csv", ("quote_id",)),
    NodeSpec("RivalSeller", "RivalSeller.csv", ("competitor_id",)),
    NodeSpec("SalesAgent", "SalesAgent.csv", ("agent_id",)),
    NodeSpec("SellableProduct", "SellableProduct.csv", ("product_id",)),
    NodeSpec("TradeCustomer", "TradeCustomer.csv", ("customer_id",)),
    NodeSpec("ProductFamily", "ProductFamily.csv", ("family",)),
    NodeSpec("ProductLine", "ProductLine.csv", ("line",)),
    # --- one row per parent entity, so the parent id identifies the record ---
    NodeSpec("DealOutcome", "DealOutcome.csv", ("opportunity_id",)),
    NodeSpec("CompetitorPresence", "CompetitorPresence.csv", ("opportunity_id",)),
    NodeSpec("AuthorityPeriod", "AuthorityPeriod.csv", ("agent_id", "valid_from")),
    NodeSpec("IncentivePeriod", "IncentivePeriod.csv", ("agent_id", "valid_from")),
    NodeSpec("CustomerClassPeriod", "CustomerClassPeriod.csv", ("customer_id", "valid_from")),
    NodeSpec("ListPricePeriod", "ListPricePeriod.csv", ("product_id", "valid_from")),
    NodeSpec("ProductCostPeriod", "ProductCostPeriod.csv", ("product_id", "valid_from")),
    NodeSpec("ProductStagePeriod", "ProductStagePeriod.csv", ("product_id", "valid_from")),
    # --- genuinely composite keys ---
    NodeSpec("CustomerPeriod", "CustomerPeriod.csv", ("customer_id", "period")),
    NodeSpec("ProductSubstitute", "ProductSubstitute.csv", ("product_id", "substitute_product_id")),
    NodeSpec(
        "StageChange",
        "StageChange.csv",
        ("opportunity_id", "from_stage", "to_stage", "occurred_at"),
    ),
    # --- reified from a flat column, to match the JSON graph ---
    NodeSpec(
        "City",
        "TradeCustomer.csv",
        ("city",),
        property_columns=("city",),
        rename={"city": "city_name"},
        distinct=True,
    ),
)


def _level_is(level: str) -> Callable[[dict[str, str]], bool]:
    """Row filter for the polymorphic PriceComponent foreign key."""
    return lambda row: row.get("applies_to_level") == level


REL_SPECS: tuple[RelSpec, ...] = (
    # --- period and event records pointing at the entity they describe ---
    RelSpec(
        "FOR_AGENT", "AuthorityPeriod.csv",
        Endpoint("AuthorityPeriod", ("agent_id", "valid_from"), ("agent_id", "valid_from")),
        Endpoint("SalesAgent", ("agent_id",), ("agent_id",)),
    ),
    RelSpec(
        "FOR_AGENT", "IncentivePeriod.csv",
        Endpoint("IncentivePeriod", ("agent_id", "valid_from"), ("agent_id", "valid_from")),
        Endpoint("SalesAgent", ("agent_id",), ("agent_id",)),
    ),
    RelSpec(
        "FOR_CUSTOMER", "CustomerClassPeriod.csv",
        Endpoint("CustomerClassPeriod", ("customer_id", "valid_from"), ("customer_id", "valid_from")),
        Endpoint("TradeCustomer", ("customer_id",), ("customer_id",)),
    ),
    RelSpec(
        "FOR_CUSTOMER", "CustomerPeriod.csv",
        Endpoint("CustomerPeriod", ("customer_id", "period"), ("customer_id", "period")),
        Endpoint("TradeCustomer", ("customer_id",), ("customer_id",)),
    ),
    RelSpec(
        "FOR_PRODUCT", "ListPricePeriod.csv",
        Endpoint("ListPricePeriod", ("product_id", "valid_from"), ("product_id", "valid_from")),
        Endpoint("SellableProduct", ("product_id",), ("product_id",)),
    ),
    RelSpec(
        "FOR_PRODUCT", "ProductCostPeriod.csv",
        Endpoint("ProductCostPeriod", ("product_id", "valid_from"), ("product_id", "valid_from")),
        Endpoint("SellableProduct", ("product_id",), ("product_id",)),
    ),
    RelSpec(
        "FOR_PRODUCT", "ProductStagePeriod.csv",
        Endpoint("ProductStagePeriod", ("product_id", "valid_from"), ("product_id", "valid_from")),
        Endpoint("SellableProduct", ("product_id",), ("product_id",)),
    ),
    # --- opportunity-centred records ---
    RelSpec(
        "FOR_OPPORTUNITY", "Quote.csv",
        Endpoint("Quote", ("quote_id",), ("quote_id",)),
        Endpoint("Opportunity", ("opportunity_id",), ("opportunity_id",)),
    ),
    RelSpec(
        "FOR_OPPORTUNITY", "DealOutcome.csv",
        Endpoint("DealOutcome", ("opportunity_id",), ("opportunity_id",)),
        Endpoint("Opportunity", ("opportunity_id",), ("opportunity_id",)),
    ),
    RelSpec(
        "FOR_OPPORTUNITY", "CompetitorPresence.csv",
        Endpoint("CompetitorPresence", ("opportunity_id",), ("opportunity_id",)),
        Endpoint("Opportunity", ("opportunity_id",), ("opportunity_id",)),
    ),
    RelSpec(
        "FOR_OPPORTUNITY", "StageChange.csv",
        Endpoint(
            "StageChange",
            ("opportunity_id", "from_stage", "to_stage", "occurred_at"),
            ("opportunity_id", "from_stage", "to_stage", "occurred_at"),
        ),
        Endpoint("Opportunity", ("opportunity_id",), ("opportunity_id",)),
    ),
    # --- the commercial chain ---
    RelSpec(
        "FOR_CUSTOMER", "Opportunity.csv",
        Endpoint("Opportunity", ("opportunity_id",), ("opportunity_id",)),
        Endpoint("TradeCustomer", ("customer_id",), ("customer_id",)),
    ),
    RelSpec(
        "SOLD_THROUGH", "Opportunity.csv",
        Endpoint("Opportunity", ("opportunity_id",), ("opportunity_id",)),
        Endpoint("Channel", ("channel_id",), ("channel_id",)),
    ),
    RelSpec(
        "REVISES", "Opportunity.csv",
        Endpoint("Opportunity", ("opportunity_id",), ("opportunity_id",)),
        Endpoint("Opportunity", ("opportunity_id",), ("revises_opportunity_id",)),
        required_columns=("revises_opportunity_id",),
        note="nullable -- only 2 of 500 opportunities revise another",
    ),
    RelSpec(
        "ISSUED_BY", "Quote.csv",
        Endpoint("Quote", ("quote_id",), ("quote_id",)),
        Endpoint("SalesAgent", ("agent_id",), ("issued_by_agent_id",)),
    ),
    RelSpec(
        "WON_BY", "DealOutcome.csv",
        Endpoint("DealOutcome", ("opportunity_id",), ("opportunity_id",)),
        Endpoint("Quote", ("quote_id",), ("winning_quote_id",)),
        required_columns=("winning_quote_id",),
        note="nullable -- only won deals name a winning quote",
    ),
    RelSpec(
        "OF_COMPETITOR", "CompetitorPresence.csv",
        Endpoint("CompetitorPresence", ("opportunity_id",), ("opportunity_id",)),
        Endpoint("RivalSeller", ("competitor_id",), ("competitor_id",)),
        required_columns=("competitor_id",),
        note="nullable -- 42 rows record a competitor whose identity is unknown",
    ),
    RelSpec(
        "FULFILLS", "Order.csv",
        Endpoint("Order", ("order_id",), ("order_id",)),
        Endpoint("Opportunity", ("opportunity_id",), ("opportunity_id",)),
    ),
    RelSpec(
        "PART_OF", "OrderLine.csv",
        Endpoint("OrderLine", ("order_line_id",), ("order_line_id",)),
        Endpoint("Order", ("order_id",), ("order_id",)),
    ),
    RelSpec(
        "FOR_PRODUCT", "OrderLine.csv",
        Endpoint("OrderLine", ("order_line_id",), ("order_line_id",)),
        Endpoint("SellableProduct", ("product_id",), ("product_id",)),
    ),
    # --- the product hierarchy. Both edges point parent -> child, which is the
    #     inverse of the foreign key: the child row names its parent. ---
    RelSpec(
        "CONTAINS", "ProductLine.csv",
        Endpoint("ProductFamily", ("family",), ("family",)),
        Endpoint("ProductLine", ("line",), ("line",)),
        note="inverse of the FK: ProductLine.family names the parent",
    ),
    RelSpec(
        "CONTAINS", "SellableProduct.csv",
        Endpoint("ProductLine", ("line",), ("product_line",)),
        Endpoint("SellableProduct", ("product_id",), ("product_id",)),
        note="inverse of the FK: SellableProduct.product_line names the parent",
    ),
    RelSpec(
        "FOR_PRODUCT", "ProductSubstitute.csv",
        Endpoint("ProductSubstitute", ("product_id", "substitute_product_id"),
                 ("product_id", "substitute_product_id")),
        Endpoint("SellableProduct", ("product_id",), ("product_id",)),
    ),
    RelSpec(
        "SUBSTITUTE_FOR", "ProductSubstitute.csv",
        Endpoint("ProductSubstitute", ("product_id", "substitute_product_id"),
                 ("product_id", "substitute_product_id")),
        Endpoint("SellableProduct", ("product_id",), ("substitute_product_id",)),
        note="the second edge from the same row, to the substitute product",
    ),
    # --- the polymorphic foreign key: applies_to_id points at three different
    #     labels depending on applies_to_level, so it needs three specs. ---
    RelSpec(
        "APPLIES_TO_LINE", "PriceComponent.csv",
        Endpoint("PriceComponent", ("component_id",), ("component_id",)),
        Endpoint("OrderLine", ("order_line_id",), ("applies_to_id",)),
        row_filter=_level_is("line"),
        note="applies_to_level == 'line'",
    ),
    RelSpec(
        "APPLIES_TO_ORDER", "PriceComponent.csv",
        Endpoint("PriceComponent", ("component_id",), ("component_id",)),
        Endpoint("Order", ("order_id",), ("applies_to_id",)),
        row_filter=_level_is("order"),
        note="applies_to_level == 'order'",
    ),
    RelSpec(
        "APPLIES_TO_CUSTOMER_PERIOD", "PriceComponent.csv",
        Endpoint("PriceComponent", ("component_id",), ("component_id",)),
        # applies_to_id here is a composite "CUST0034|2025-Q1"; the loader
        # splits it into the two derived columns used below.
        Endpoint("CustomerPeriod", ("customer_id", "period"),
                 ("_applies_customer_id", "_applies_period")),
        row_filter=_level_is("customer_period"),
        note="applies_to_level == 'customer_period'; composite key split on '|'",
    ),
    # --- reified city ---
    RelSpec(
        "LOCATED_IN", "TradeCustomer.csv",
        Endpoint("TradeCustomer", ("customer_id",), ("customer_id",)),
        Endpoint("City", ("city_name",), ("city",)),
        required_columns=("city",),
    ),
)


# ---------------------------------------------------------------------------
# Loading and type conversion
# ---------------------------------------------------------------------------

_INT = re.compile(r"-?\d+$")
_FLOAT = re.compile(r"-?\d+\.\d+$")


def convert(value: str) -> Any:
    """
    Convert a CSV string to the type it should have in the graph.

    Numbers must become real numbers: Cypher cannot ``avg()`` or ``sum()`` a
    string, and numeric aggregation is the main thing this graph exists to
    answer. Booleans become real booleans for the same reason.

    Dates deliberately stay ISO strings rather than becoming Neo4j date
    values: that matches the JSON export exactly, sorts and compares
    correctly, and avoids introducing a temporal type the downstream
    evidence-serialisation path has never seen.
    """
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    if _INT.fullmatch(text):
        return int(text)
    if _FLOAT.fullmatch(text):
        return float(text)
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    return text


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV, tolerating a UTF-8 BOM."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_all(csv_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Load every CSV referenced by the specs, and derive extra columns."""
    needed = {spec.csv_file for spec in NODE_SPECS} | {spec.csv_file for spec in REL_SPECS}
    tables = {name: read_csv(csv_dir / name) for name in sorted(needed)}

    # The one derived case: split the composite applies_to_id so the
    # relationship spec can reference its two parts as ordinary columns.
    for row in tables.get("PriceComponent.csv", []):
        row["_applies_customer_id"] = ""
        row["_applies_period"] = ""
        if row.get("applies_to_level") == "customer_period":
            raw = row.get("applies_to_id", "")
            if "|" in raw:
                customer_id, _, period = raw.partition("|")
                row["_applies_customer_id"] = customer_id
                row["_applies_period"] = period
    return tables


# ---------------------------------------------------------------------------
# Validation (runs before any write)
# ---------------------------------------------------------------------------


@dataclass
class Validation:
    """Everything worth reporting that was found before writing."""

    duplicate_keys: list[str] = field(default_factory=list)
    orphan_refs: list[str] = field(default_factory=list)
    data_quality: list[str] = field(default_factory=list)
    skipped_rows: Counter = field(default_factory=Counter)

    @property
    def blocking(self) -> bool:
        """Duplicates and orphans are blocking; data-quality notes are not."""
        return bool(self.duplicate_keys or self.orphan_refs)


def node_key_index(tables: dict[str, list[dict[str, str]]]) -> dict[str, set[tuple]]:
    """Build, per label, the set of key tuples that will exist as nodes."""
    index: dict[str, set[tuple]] = {}
    for spec in NODE_SPECS:
        rows = tables[spec.csv_file]
        keys = set()
        for row in rows:
            key = tuple(convert(row.get(col, "")) for col in spec.key_columns)
            if any(part is None for part in key):
                continue
            keys.add(key)
        index[spec.label] = keys
    return index


def validate(tables: dict[str, list[dict[str, str]]]) -> Validation:
    """Check keys and references before writing anything."""
    report = Validation()

    # 1. Node keys must be unique, or MERGE would collapse distinct rows.
    for spec in NODE_SPECS:
        rows = tables[spec.csv_file]
        keys = [tuple(row.get(col, "") for col in spec.key_columns) for row in rows]
        if spec.distinct:
            continue  # dedup is the point for a derived label
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        if duplicates:
            report.duplicate_keys.append(
                f"{spec.label}: {len(duplicates)} duplicate key(s) on "
                f"{spec.key_columns}, e.g. {duplicates[0]}"
            )

    # 2. Every relationship endpoint must resolve to a node that will exist.
    index = node_key_index(tables)
    for spec in REL_SPECS:
        rows = tables[spec.csv_file]
        for endpoint in (spec.source, spec.target):
            missing = set()
            for row in rows:
                if spec.row_filter and not spec.row_filter(row):
                    continue
                if any(not row.get(col, "").strip() for col in spec.required_columns):
                    continue
                key = tuple(convert(row.get(col, "")) for col in endpoint.key_columns)
                if any(part is None for part in key):
                    continue
                if key not in index[endpoint.label]:
                    missing.add(key)
            if missing:
                sample = sorted(str(k) for k in missing)[:5]
                report.orphan_refs.append(
                    f"{spec.csv_file} -[:{spec.rel_type}]-> {endpoint.label}: "
                    f"{len(missing)} unresolved key(s), e.g. {sample}"
                )

    # 3. Data quality: reversed validity windows are reported, not repaired.
    for spec in NODE_SPECS:
        rows = tables[spec.csv_file]
        for row in rows:
            start, end = row.get("valid_from", ""), row.get("valid_to", "")
            if start and end and end < start:
                identity = ", ".join(f"{c}={row.get(c)}" for c in spec.key_columns)
                report.data_quality.append(
                    f"{spec.label} ({identity}): valid_from={start} is AFTER "
                    f"valid_to={end} -- imported unchanged, not repaired"
                )

    # 4. Rows skipped because a nullable foreign key is empty.
    for spec in REL_SPECS:
        if not spec.required_columns:
            continue
        rows = tables[spec.csv_file]
        skipped = sum(
            1
            for row in rows
            if (not spec.row_filter or spec.row_filter(row))
            and any(not row.get(col, "").strip() for col in spec.required_columns)
        )
        if skipped:
            report.skipped_rows[f"{spec.rel_type} from {spec.csv_file}"] = skipped

    return report


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def constraint_statements() -> list[tuple[str, str]]:
    """
    Uniqueness constraints for single-key labels, indexes for composite ones.

    A composite uniqueness constraint is not available on every Neo4j edition,
    so composite-key labels get a plain index instead: it gives MERGE the
    lookup performance it needs, and this importer is single-threaded, so the
    concurrency hazard a constraint would guard against does not arise.
    """
    statements: list[tuple[str, str]] = []
    for spec in NODE_SPECS:
        props = spec.key_properties()
        safe = spec.label.lower()
        if len(props) == 1:
            statements.append((
                f"{spec.label}.{props[0]} unique",
                f"CREATE CONSTRAINT {safe}_key IF NOT EXISTS "
                f"FOR (n:{spec.label}) REQUIRE n.{props[0]} IS UNIQUE",
            ))
        else:
            cols = ", ".join(f"n.{p}" for p in props)
            statements.append((
                f"{spec.label}({', '.join(props)}) index",
                f"CREATE INDEX {safe}_key IF NOT EXISTS "
                f"FOR (n:{spec.label}) ON ({cols})",
            ))
    return statements


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def node_payload(spec: NodeSpec, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build the parameter rows for one node label."""
    if not rows:
        return []
    columns = spec.property_columns or tuple(rows[0].keys())
    payload: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    for row in rows:
        key = tuple(convert(row.get(col, "")) for col in spec.key_columns)
        if any(part is None for part in key):
            continue
        if spec.distinct:
            if key in seen:
                continue
            seen.add(key)

        properties: dict[str, Any] = {}
        for col in columns:
            if col.startswith("_"):  # derived helper column, not real data
                continue
            value = convert(row.get(col, ""))
            if value is not None:  # Neo4j stores no nulls; omit instead
                properties[spec.rename.get(col, col)] = value

        payload.append({
            "key": dict(zip(spec.key_properties(), key)),
            "props": properties,
        })
    return payload


def import_nodes(
    client: Neo4jClient, spec: NodeSpec, rows: list[dict[str, str]], batch_size: int
) -> tuple[WriteCounters, int]:
    """MERGE every node for one label. Returns counters and rows attempted."""
    payload = node_payload(spec, rows)
    key_match = ", ".join(f"{p}: row.key.{p}" for p in spec.key_properties())
    cypher = (
        f"UNWIND $rows AS row "
        f"MERGE (n:{spec.label} {{{key_match}}}) "
        f"SET n += row.props"
    )
    total = WriteCounters()
    for batch in batched(payload, batch_size):
        total = total + client.execute_write(cypher, {"rows": list(batch)})
    return total, len(payload)


def rel_payload(spec: RelSpec, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build the parameter rows for one relationship type."""
    payload: list[dict[str, Any]] = []
    for row in rows:
        if spec.row_filter and not spec.row_filter(row):
            continue
        if any(not row.get(col, "").strip() for col in spec.required_columns):
            continue
        source = tuple(convert(row.get(col, "")) for col in spec.source.key_columns)
        target = tuple(convert(row.get(col, "")) for col in spec.target.key_columns)
        if any(p is None for p in source) or any(p is None for p in target):
            continue
        payload.append({
            "src": dict(zip(spec.source.key_properties, source)),
            "tgt": dict(zip(spec.target.key_properties, target)),
        })
    return payload


def import_relationships(
    client: Neo4jClient, spec: RelSpec, rows: list[dict[str, str]], batch_size: int
) -> tuple[WriteCounters, int]:
    """MERGE every relationship for one spec. Returns counters and rows attempted."""
    payload = rel_payload(spec, rows)
    src_match = ", ".join(f"{p}: row.src.{p}" for p in spec.source.key_properties)
    tgt_match = ", ".join(f"{p}: row.tgt.{p}" for p in spec.target.key_properties)
    cypher = (
        f"UNWIND $rows AS row "
        f"MATCH (a:{spec.source.label} {{{src_match}}}) "
        f"MATCH (b:{spec.target.label} {{{tgt_match}}}) "
        f"MERGE (a)-[r:{spec.rel_type}]->(b)"
    )
    total = WriteCounters()
    for batch in batched(payload, batch_size):
        total = total + client.execute_write(cypher, {"rows": list(batch)})
    return total, len(payload)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_validation(report: Validation) -> None:
    print(f"\n{RULE}\nVALIDATION\n{RULE}")

    if report.duplicate_keys:
        print("\n  BLOCKING -- duplicate node keys (MERGE would collapse distinct rows):")
        for line in report.duplicate_keys:
            print(f"    {line}")
    else:
        print("\n  node keys      : OK, every key is unique within its label")

    if report.orphan_refs:
        print("\n  BLOCKING -- unresolved relationship endpoints:")
        for line in report.orphan_refs:
            print(f"    {line}")
    else:
        print("  references     : OK, every relationship endpoint resolves to a node")

    if report.data_quality:
        print(f"\n  DATA QUALITY -- {len(report.data_quality)} issue(s), imported unchanged:")
        for line in report.data_quality:
            print(f"    {line}")
    else:
        print("  data quality   : no issues found")

    if report.skipped_rows:
        print("\n  rows with an empty nullable foreign key (no relationship created):")
        for name, count in sorted(report.skipped_rows.items()):
            print(f"    {name}: {count}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and report only. Touches nothing in the database.",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(logging.INFO if args.verbose else logging.WARNING)

    print(RULE)
    print("AXIOM SYNTHETIC DATA -> NEO4J IMPORT")
    print(RULE)
    print(f"  source   : {args.csv_dir}")
    print(f"  node specs        : {len(NODE_SPECS)} labels")
    print(f"  relationship specs: {len(REL_SPECS)} (across "
          f"{len({s.rel_type for s in REL_SPECS})} distinct types)")

    try:
        tables = load_all(args.csv_dir)
    except FileNotFoundError as exc:
        print(f"\nCould not read the CSVs: {exc}", file=sys.stderr)
        return 2
    print(f"  CSV files loaded  : {len(tables)} "
          f"({sum(len(r) for r in tables.values())} rows total)")

    report = validate(tables)
    print_validation(report)

    if report.blocking:
        print("\nRefusing to import while blocking problems remain. Nothing was written.")
        return 1

    if args.dry_run:
        node_total = sum(len(node_payload(s, tables[s.csv_file])) for s in NODE_SPECS)
        rel_total = sum(len(rel_payload(s, tables[s.csv_file])) for s in REL_SPECS)
        print(f"\n{RULE}\nDRY RUN -- nothing was written\n{RULE}")
        print(f"  would MERGE {node_total} nodes and {rel_total} relationships")
        return 0

    settings = get_settings()
    client = Neo4jClient(settings.neo4j)
    if not client.verify_connectivity():
        print(
            f"\nNeo4j is not reachable at {settings.neo4j.uri}.\n"
            f"Check the instance is started in Neo4j Desktop and that\n"
            f"NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in .env are correct.",
            file=sys.stderr,
        )
        return 2
    print(f"\n  connected to      : {settings.neo4j.uri} (database: {client.database})")

    try:
        # --- constraints and indexes ---
        print(f"\n{RULE}\nCONSTRAINTS AND INDEXES\n{RULE}\n")
        schema_counters = WriteCounters()
        for description, statement in constraint_statements():
            schema_counters = schema_counters + client.execute_write(statement)
            print(f"  ensured {description}")
        print(f"\n  {schema_counters.constraints_added} constraint(s) and "
              f"{schema_counters.indexes_added} index(es) newly created")

        # --- nodes ---
        print(f"\n{RULE}\nNODES\n{RULE}\n")
        print(f"  {'label':24s} {'rows':>7} {'created':>9} {'matched':>9}")
        print("  " + "-" * 52)
        node_totals = WriteCounters()
        node_rows_total = 0
        for spec in NODE_SPECS:
            counters, attempted = import_nodes(
                client, spec, tables[spec.csv_file], args.batch_size
            )
            node_totals = node_totals + counters
            node_rows_total += attempted
            matched = attempted - counters.nodes_created
            print(f"  {spec.label:24s} {attempted:>7} {counters.nodes_created:>9} {matched:>9}")

        # --- relationships ---
        print(f"\n{RULE}\nRELATIONSHIPS\n{RULE}\n")
        print(f"  {'type':28s} {'from':24s} {'rows':>6} {'created':>8} {'matched':>8}")
        print("  " + "-" * 78)
        rel_totals = WriteCounters()
        rel_rows_total = 0
        per_type: Counter = Counter()
        for spec in REL_SPECS:
            counters, attempted = import_relationships(
                client, spec, tables[spec.csv_file], args.batch_size
            )
            rel_totals = rel_totals + counters
            rel_rows_total += attempted
            per_type[spec.rel_type] += attempted
            matched = attempted - counters.relationships_created
            print(f"  {spec.rel_type:28s} {spec.csv_file:24s} {attempted:>6} "
                  f"{counters.relationships_created:>8} {matched:>8}")

        # --- summary ---
        print(f"\n{RULE}\nSUMMARY\n{RULE}\n")
        print(f"  nodes        attempted={node_rows_total:6d}  "
              f"created={node_totals.nodes_created:6d}  "
              f"matched={node_rows_total - node_totals.nodes_created:6d}")
        print(f"  relationships attempted={rel_rows_total:6d}  "
              f"created={rel_totals.relationships_created:6d}  "
              f"matched={rel_rows_total - rel_totals.relationships_created:6d}")
        print(f"  properties set: {node_totals.properties_set + rel_totals.properties_set}")

        print("\n  relationships per type:")
        for rel_type, count in sorted(per_type.items()):
            print(f"    {rel_type:30s} {count}")

        skipped = sum(report.skipped_rows.values())
        print(f"\n  rows skipped (empty nullable FK): {skipped}")
        print(f"  data-quality issues reported     : {len(report.data_quality)}")
        for line in report.data_quality:
            print(f"    {line}")

        totals = client.run_query(
            "MATCH (n) RETURN count(n) AS nodes"
        )[0]["nodes"]
        rel_count = client.run_query(
            "MATCH ()-[r]->() RETURN count(r) AS rels"
        )[0]["rels"]
        print(f"\n  database now holds: {totals} nodes, {rel_count} relationships")
        print("\n  Verify with:  python scripts/verify_import.py")
        return 0

    except Neo4jQueryError as exc:
        print(f"\nImport failed: {exc}", file=sys.stderr)
        if exc.cypher:
            print(f"Statement: {exc.cypher[:300]}", file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
