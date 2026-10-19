"""
The query plan: a small structured query that runs on the JSON graph.

Why a plan rather than Cypher text
----------------------------------
Cypher only executes against Neo4j. The AXIOM graph we actually have is a JSON
export, so a component that emits Cypher text emits something it cannot run --
which makes it useless exactly when the database is unavailable, which is most
of the time on this project.

A plan is a declarative description of the query instead: which label, which
filters, which aggregate. It has three properties Cypher text does not:

* **It runs on the graph we have.** :func:`execute_plan` evaluates it directly
  against the in-memory :class:`KnowledgeGraph`, whatever its source.
* **It cannot be injected.** There is no query language to escape from -- a
  plan is data, validated field by field against the live schema. The
  read-only guarantee is structural: nothing here can express a write.
* **It still produces Cypher.** :meth:`QueryPlan.to_cypher` compiles the plan,
  so the generated query is visible for review and runs unchanged against
  Neo4j when one is available.

Scope
-----
A plan matches one label, filters on its properties, and optionally
aggregates. It deliberately does not traverse relationships: the AXIOM export
denormalises foreign keys onto node properties (``Quote.opportunity_id``,
``PriceComponent.applies_to_id``), so single-label filtering answers the
aggregation questions in scope without a join. Traversal is a clear extension
point, not a hidden limitation -- see :mod:`src.retrieval.graph_retriever` for
the component that does walk relationships.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.graph.model import GraphNode, KnowledgeGraph

__all__ = [
    "AGGREGATES",
    "Filter",
    "OPERATORS",
    "PlanError",
    "QueryPlan",
    "execute_plan",
]

logger = logging.getLogger(__name__)

#: Comparison operators a filter may use.
OPERATORS: frozenset[str] = frozenset(
    {"=", "!=", ">", ">=", "<", "<=", "contains", "starts_with", "exists"}
)

#: Aggregate functions a plan may request. ``count`` needs no property; the
#: rest operate on one numeric property.
AGGREGATES: frozenset[str] = frozenset({"count", "sum", "avg", "min", "max"})


class PlanError(ValueError):
    """Raised when a plan is malformed or references something that does not exist."""


@dataclass(frozen=True, slots=True)
class Filter:
    """One property comparison."""

    property: str
    op: str
    value: Any = None

    def matches(self, node: GraphNode) -> bool:
        """Whether ``node`` satisfies this filter."""
        actual = node.properties.get(self.property)

        if self.op == "exists":
            return actual is not None
        if actual is None:
            return False

        if self.op in {"=", "!="}:
            equal = _loosely_equal(actual, self.value)
            return equal if self.op == "=" else not equal

        if self.op in {"contains", "starts_with"}:
            text, needle = str(actual).casefold(), str(self.value).casefold()
            return needle in text if self.op == "contains" else text.startswith(needle)

        left, right = _as_number(actual), _as_number(self.value)
        if left is None or right is None:
            return False
        if self.op == ">":
            return left > right
        if self.op == ">=":
            return left >= right
        if self.op == "<":
            return left < right
        return left <= right  # "<="


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """A validated, executable description of one question."""

    label: str
    filters: tuple[Filter, ...] = ()
    aggregate: str | None = None
    aggregate_property: str | None = None
    return_records: bool = False
    limit: int | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QueryPlan":
        """
        Build a plan from the model's JSON, rejecting anything malformed.

        Raises:
            PlanError: naming the specific field at fault, so the self-healing
                loop has something concrete to correct.
        """
        if not isinstance(payload, Mapping):
            raise PlanError(f"Plan must be a JSON object, got {type(payload).__name__}")

        label = payload.get("label")
        if not isinstance(label, str) or not label:
            raise PlanError("Plan needs a 'label' naming the record type to match")

        aggregate = payload.get("aggregate")
        if aggregate is not None:
            if not isinstance(aggregate, str) or aggregate.lower() not in AGGREGATES:
                raise PlanError(
                    f"'aggregate' must be one of {sorted(AGGREGATES)}, got {aggregate!r}"
                )
            aggregate = aggregate.lower()

        aggregate_property = payload.get("aggregate_property")
        if aggregate and aggregate != "count" and not aggregate_property:
            raise PlanError(f"'{aggregate}' needs an 'aggregate_property' to operate on")

        limit = payload.get("limit")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise PlanError(f"'limit' must be a positive integer, got {limit!r}")

        return cls(
            label=label,
            filters=tuple(_parse_filter(entry, i) for i, entry in enumerate(payload.get("filters") or [])),
            aggregate=aggregate,
            aggregate_property=aggregate_property,
            return_records=bool(payload.get("return_records", False)),
            limit=limit,
        )

    def validate_against(self, schema: "GraphSchemaLike") -> None:
        """
        Check every name the plan uses actually exists in the graph.

        This is the correctness guardrail, and it replaces regex-scanning a
        query string: a plan cannot reference a label or property that does
        not exist without being caught here, before anything runs.

        Raises:
            PlanError: naming the unknown label or property.
        """
        known_labels = schema.label_names
        if self.label not in known_labels:
            close = _closest(self.label, known_labels)
            hint = f" Did you mean {close!r}?" if close else ""
            raise PlanError(f"Unknown label {self.label!r}.{hint}")

        known_properties = set(schema.node_labels[self.label])
        for filter_ in self.filters:
            if filter_.property not in known_properties:
                close = _closest(filter_.property, known_properties)
                hint = f" Did you mean {close!r}?" if close else ""
                raise PlanError(
                    f"{self.label} has no property {filter_.property!r}.{hint}"
                )
        if self.aggregate_property and self.aggregate_property not in known_properties:
            close = _closest(self.aggregate_property, known_properties)
            hint = f" Did you mean {close!r}?" if close else ""
            raise PlanError(
                f"{self.label} has no property {self.aggregate_property!r}.{hint}"
            )

        self._validate_filter_values(schema)

    def _validate_filter_values(self, schema: "GraphSchemaLike") -> None:
        """
        Reject an equality filter on a value the property never takes.

        The schema records the *complete* distinct value set for
        low-cardinality properties, so a filter such as
        ``customer_type = 'key account'`` -- when the six real values are
        retailer, wholesaler, direct, distributor, institution, government --
        provably matches nothing.

        Without this, that plan runs and returns ``0``: a silent wrong answer
        that looks like a legitimate finding. Turning it into an error that
        lists the real values gives the self-healing loop something to correct,
        and in practice sends the model looking for the property that *does*
        hold the concept it wanted.

        Only ``=`` is checked. For ``!=`` and the comparisons, a value outside
        the observed set is meaningful rather than mistaken.
        """
        samples = getattr(schema, "value_samples", {}).get(self.label, {})
        for filter_ in self.filters:
            if filter_.op != "=":
                continue
            observed = samples.get(filter_.property)
            if not observed:
                continue
            if any(_loosely_equal(value, filter_.value) for value in observed):
                continue

            elsewhere = _where_value_occurs(schema, filter_.value, exclude=self.label)
            hint = (
                f" That value does occur elsewhere: {'; '.join(elsewhere)}."
                if elsewhere
                else " If none of those is what you meant, the concept lives on "
                "a different label or property."
            )
            raise PlanError(
                f"{self.label}.{filter_.property} never has the value "
                f"{filter_.value!r}. Its only values are: "
                f"{', '.join(repr(v) for v in observed)}.{hint}"
            )

    def to_cypher(self) -> str:
        """
        Compile the plan to the equivalent Cypher.

        Used for display and for execution against a live Neo4j. Values are
        rendered as literals here because they came from a validated plan,
        not from free text -- there is no injection path into this string.

        ``aggregate`` combined with ``return_records`` needs a ``WITH``
        stage, not a single ``RETURN``. Naming the matched node alongside an
        aggregate in one ``RETURN`` -- ``RETURN count(n) AS total, n AS
        evidence`` -- makes Cypher implicitly group by ``n``, turning the
        aggregate into a *per-node* count instead of a total over the whole
        match: exactly the trap the prompt in ``engine.py`` warns the model
        away from. Computing the aggregate in a ``WITH`` first, with nothing
        else in scope to group by, then attaching one representative node
        afterwards, keeps the aggregate genuinely aggregate.
        """
        parts = [f"MATCH (n:{self.label})"]
        if self.filters:
            conditions = " AND ".join(_filter_to_cypher(f) for f in self.filters)
            parts.append(f"WHERE {conditions}")

        if not self.aggregate:
            parts.append("RETURN n")
        elif not self.return_records:
            parts.append(f"RETURN {_aggregate_expression(self)} AS {_aggregate_field_name(self)}")
        else:
            field = _aggregate_field_name(self)
            parts.append(
                f"WITH {_aggregate_expression(self)} AS {field}, "
                f"collect(n)[0] AS evidence"
            )
            parts.append(f"RETURN {field}, evidence")

        if self.limit:
            parts.append(f"LIMIT {self.limit}")
        return " ".join(parts)


class GraphSchemaLike:
    """Structural type for what :meth:`QueryPlan.validate_against` needs."""

    label_names: frozenset[str]
    node_labels: Mapping[str, Sequence[str]]
    value_samples: Mapping[str, Mapping[str, Sequence[str]]]


def execute_plan(
    plan: QueryPlan, graph: KnowledgeGraph
) -> list[dict[str, Any]]:
    """
    Run ``plan`` against the graph and return rows.

    Rows use the same shape the Neo4j path produces -- scalars for values,
    ``{"labels": [...], "properties": {...}}`` for records -- so evidence
    assembly and the result format are identical whichever source answered.

    The aggregate is computed here, over the graph's own data. Nothing in the
    pipeline asks a language model to do arithmetic.
    """
    matched = [
        node
        for node in graph
        if node.primary_label == plan.label
        and all(f.matches(node) for f in plan.filters)
    ]
    logger.debug("Plan matched %d %s record(s)", len(matched), plan.label)

    if plan.aggregate:
        value = _aggregate(plan, matched)
        row: dict[str, Any] = {_aggregate_field_name(plan): value}
        if plan.return_records and matched:
            # One representative record, so an aggregate can still be traced
            # to something concrete without returning the entire match set.
            row["evidence"] = _as_row_node(matched[0])
        return [row]

    records = matched[: plan.limit] if plan.limit else matched
    return [{"record": _as_row_node(node)} for node in records]


def _aggregate(plan: QueryPlan, matched: Sequence[GraphNode]) -> float | int | None:
    """Compute the requested aggregate over the matched records."""
    if plan.aggregate == "count":
        return len(matched)

    values = [
        number
        for node in matched
        if (number := _as_number(node.properties.get(plan.aggregate_property))) is not None
    ]
    if not values:
        return None
    if plan.aggregate == "sum":
        return sum(values)
    if plan.aggregate == "avg":
        return sum(values) / len(values)
    if plan.aggregate == "min":
        return min(values)
    return max(values)  # "max"


def _aggregate_field_name(plan: QueryPlan) -> str:
    """The answer field name, matching what ``to_cypher`` aliases it to."""
    if plan.aggregate == "count":
        return "count"
    return f"{plan.aggregate}_{plan.aggregate_property}"


def _aggregate_expression(plan: QueryPlan) -> str:
    """The Cypher aggregate function call for this plan, unaliased."""
    if plan.aggregate == "count":
        return "count(n)"
    return f"{plan.aggregate}(n.{plan.aggregate_property})"


def _as_row_node(node: GraphNode) -> dict[str, Any]:
    """Render a node in the same shape the Neo4j entity-aware path returns."""
    return {"labels": list(node.labels), "properties": dict(node.properties)}


def _parse_filter(entry: Any, index: int) -> Filter:
    """Parse and validate one filter entry."""
    if not isinstance(entry, Mapping):
        raise PlanError(f"Filter {index} must be an object")
    prop = entry.get("property")
    op = entry.get("op", "=")
    if not isinstance(prop, str) or not prop:
        raise PlanError(f"Filter {index} needs a 'property'")
    if not isinstance(op, str) or op not in OPERATORS:
        raise PlanError(
            f"Filter {index} has operator {op!r}; must be one of {sorted(OPERATORS)}"
        )
    if op != "exists" and "value" not in entry:
        raise PlanError(f"Filter {index} needs a 'value' for operator {op!r}")
    return Filter(property=prop, op=op, value=entry.get("value"))


def _loosely_equal(actual: Any, expected: Any) -> bool:
    """
    Compare two values, tolerating the type drift real exports contain.

    This export stores booleans as the strings ``'true'``/``'false'`` and some
    numbers as floats. A model that writes ``true`` or ``5`` should still match
    ``'true'`` or ``5.0`` -- rejecting those would fail on a data-encoding
    detail rather than on the question being wrong. Exact equality is tried
    first, so nothing is coerced unnecessarily.
    """
    if actual == expected:
        return True

    left, right = _as_number(actual), _as_number(expected)
    if left is not None and right is not None:
        return left == right

    return str(actual).strip().casefold() == str(expected).strip().casefold()


def _as_number(value: Any) -> float | None:
    """Coerce to a number, or None when the value is not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _filter_to_cypher(filter_: Filter) -> str:
    """
    Render one filter as a Cypher condition.

    Equality on a boolean-shaped value is coerced through ``toString()`` on
    both sides. This is not cosmetic: the JSON export and the CSV-derived
    Neo4j import store the same fact in two different shapes -- the JSON
    keeps a real ``true``/``false``, the CSVs (and therefore Neo4j, after
    :func:`scripts.import_synthetic_data.convert` typed them) also store a
    real boolean, but the *model* frequently writes the filter value as the
    quoted string ``"true"``. A plain ``=`` against a real Neo4j boolean then
    silently matches zero rows -- confirmed directly: ``is_tender = 'true'``
    returns 0 where ``is_tender = true`` returns the correct 55. That is a
    wrong answer that looks exactly like a right one, so it is coerced away
    here rather than left for someone to notice by accident.
    """
    if filter_.op == "exists":
        return f"n.{filter_.property} IS NOT NULL"
    if filter_.op == "contains":
        return f"toLower(toString(n.{filter_.property})) CONTAINS toLower({_literal(filter_.value)})"
    if filter_.op == "starts_with":
        return f"toLower(toString(n.{filter_.property})) STARTS WITH toLower({_literal(filter_.value)})"
    if filter_.op in ("=", "!=") and _is_boolean_like(filter_.value):
        operator = "<>" if filter_.op == "!=" else "="
        bool_text = "true" if _as_bool(filter_.value) else "false"
        return f"toLower(toString(n.{filter_.property})) {operator} '{bool_text}'"
    operator = "<>" if filter_.op == "!=" else filter_.op
    return f"n.{filter_.property} {operator} {_literal(filter_.value)}"


def _is_boolean_like(value: Any) -> bool:
    """Whether a plan value represents a boolean, as a real bool or a string."""
    if isinstance(value, bool):
        return True
    return isinstance(value, str) and value.strip().lower() in ("true", "false")


def _as_bool(value: Any) -> bool:
    """Extract the boolean a plan value represents, in either shape."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _literal(value: Any) -> str:
    """Render a validated plan value as a Cypher literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _where_value_occurs(
    schema: "GraphSchemaLike", wanted: Any, *, exclude: str, limit: int = 3
) -> list[str]:
    """
    Find where a rejected filter value actually occurs in the graph.

    Rejecting a plan tells the model it was wrong; this tells it where to look
    instead, which is the difference between a guardrail that blocks and one
    that repairs. Asked for ``customer_type = 'key account'``, it reports that
    ``CustomerClassPeriod.account_class`` takes the value ``'key'`` -- the
    label the question actually needed.

    Matching is loose in both directions (``'key'`` matches a request for
    ``'key account'`` and vice versa), because the model's wording rarely
    matches the stored token exactly -- that mismatch is the whole reason the
    plan was rejected.
    """
    needle = str(wanted).strip().casefold()
    if not needle:
        return []

    found: list[str] = []
    for label, properties in getattr(schema, "value_samples", {}).items():
        if label == exclude:
            continue
        for prop, values in properties.items():
            for value in values:
                candidate = str(value).strip().casefold()
                if candidate == needle or (
                    # Substring matching only for tokens long enough to be
                    # meaningful. Without the length floor, a one-character
                    # value like tier 'A' matches almost any phrase.
                    min(len(candidate), len(needle)) >= 3
                    and (candidate in needle or needle in candidate)
                ):
                    found.append(f"{label}.{prop} = {value!r}")
                    break
            if len(found) >= limit:
                return found
    return found


def _closest(name: str, candidates: "frozenset[str] | set[str]") -> str | None:
    """
    Suggest the nearest known name, to make a repair prompt actionable.

    A model that writes ``Opprtunity`` repairs far faster when told the real
    name than when told only that its guess was wrong.
    """
    import difflib

    matches = difflib.get_close_matches(name, sorted(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None
