"""
Guardrails for generated Cypher.

The text-to-query component lets an LLM write Cypher from natural language.
Before any generated query is executed, it passes through this module. Two
independent checks, for two different failure modes:

``validate_read_only``
    The query must never write. This is the safety guardrail -- a pricing
    evidence system must never let an LLM mutate the graph. It is also not the
    *only* enforcement: :meth:`src.graph.neo4j_client.Neo4jClient.run_query`
    executes through a read-only driver transaction, so even a write clause
    this regex somehow missed is refused server-side. Defense in depth.

``validate_schema_references``
    The query must reference only labels and relationship types that actually
    exist. This is a correctness guardrail, not a safety one: Cypher does not
    error on a typo'd label -- ``MATCH (n:Opprtunity)`` simply matches nothing
    and returns an empty result, which looks like a valid (if boring) answer
    rather than a failure. Left unchecked, that is a silent wrong-answer risk
    the self-healing loop would never see a reason to retry.
"""
import re
from collections.abc import Iterable

# Cypher clauses that mutate the graph. Generated queries must never
# contain any of these, regardless of the LLM's stated intent.
WRITE_CLAUSES = (
    "CREATE", "MERGE", "DELETE", "DETACH DELETE", "SET", "REMOVE",
    "DROP", "CALL {", "LOAD CSV", "FOREACH",
)

# Cypher/APOC procedures that can have write side-effects even when the
# surrounding query looks read-only.
DANGEROUS_PROCEDURES = (
    "apoc.create", "apoc.merge", "apoc.refactor", "db.index", "dbms.",
)

#: A node label reference: ``(n:Label)``, ``(:Label)``, or a multi-label node
#: ``(n:Label1:Label2)``. Anchored on the opening parenthesis so a property
#: key or map literal elsewhere in the query is never mistaken for a label.
_LABEL_PATTERN = re.compile(r"\(\s*\w*\s*((?::[A-Za-z_][A-Za-z0-9_]*)+)")

#: A relationship-type reference: ``[:REL]``, ``[r:REL]``, or an alternation
#: ``[:REL_A|REL_B]``. Anchored on the opening bracket for the same reason.
#: Excludes ``*`` so a variable-length path (``[:REL*1..3]``) still isolates
#: the type name correctly.
_RELATIONSHIP_PATTERN = re.compile(
    r"\[\s*\w*\s*:([A-Za-z_][A-Za-z0-9_]*(?:\|[A-Za-z_][A-Za-z0-9_]*)*)"
)


class GuardrailViolation(Exception):
    """Raised when a generated Cypher query fails a safety check."""


def _strip_string_literals_and_comments(cypher: str) -> str:
    """
    Remove string literals and comments before keyword scanning, so a write
    keyword mentioned inside a string (e.g. a text field) doesn't trigger a
    false positive.
    """
    without_comments = re.sub(r"//.*", "", cypher)
    without_strings = re.sub(r"'[^']*'|\"[^\"]*\"", "", without_comments)
    return without_strings


def validate_read_only(cypher: str) -> None:
    """
    Validate that a generated Cypher query is read-only.

    Args:
        cypher: the generated Cypher query string.

    Raises:
        GuardrailViolation: if the query contains a write clause or a
            dangerous procedure call. The message names which check failed,
            so the self-healing retry loop in engine.py can feed
            that back to the LLM.
    """
    cleaned = _strip_string_literals_and_comments(cypher).upper()

    for clause in WRITE_CLAUSES:
        # Word-boundary match so e.g. "SET" doesn't false-positive on a
        # property named "settings".
        if re.search(rf"\b{re.escape(clause)}\b", cleaned):
            raise GuardrailViolation(f"Query contains a write clause: {clause}")

    lowered = cypher.lower()
    for proc in DANGEROUS_PROCEDURES:
        if proc in lowered:
            raise GuardrailViolation(f"Query calls a disallowed procedure: {proc}")

    if not cleaned.strip():
        raise GuardrailViolation("Query is empty")


def is_read_only(cypher: str) -> bool:
    """Convenience boolean wrapper around validate_read_only()."""
    try:
        validate_read_only(cypher)
        return True
    except GuardrailViolation:
        return False


def referenced_labels(cypher: str) -> frozenset[str]:
    """Every node label the query pattern-matches on."""
    cleaned = _strip_string_literals_and_comments(cypher)
    labels: set[str] = set()
    for match in _LABEL_PATTERN.finditer(cleaned):
        labels.update(match.group(1).lstrip(":").split(":"))
    return frozenset(labels)


def referenced_relationship_types(cypher: str) -> frozenset[str]:
    """Every relationship type the query pattern-matches on."""
    cleaned = _strip_string_literals_and_comments(cypher)
    rel_types: set[str] = set()
    for match in _RELATIONSHIP_PATTERN.finditer(cleaned):
        rel_types.update(match.group(1).split("|"))
    return frozenset(rel_types)


def validate_schema_references(
    cypher: str, known_labels: Iterable[str], known_relationship_types: Iterable[str]
) -> None:
    """
    Validate that every label and relationship type the query names exists.

    Cypher does not raise an error for an unknown label or relationship type
    -- it simply matches nothing, which is indistinguishable from a correct
    query with a genuinely empty answer. This check turns that silent
    wrong-answer risk into an explicit, retryable error naming exactly which
    reference was invalid, so the self-healing loop has something concrete to
    fix rather than a query that merely "didn't find anything".

    Args:
        cypher: the generated Cypher query.
        known_labels: every label that exists in the current graph.
        known_relationship_types: every relationship type that exists.

    Raises:
        GuardrailViolation: naming the first unknown label or relationship
            type found, in that order.
    """
    known_label_set = frozenset(known_labels)
    known_rel_set = frozenset(known_relationship_types)

    unknown_labels = referenced_labels(cypher) - known_label_set
    if unknown_labels:
        raise GuardrailViolation(
            f"Query references unknown label(s): {', '.join(sorted(unknown_labels))}"
        )

    unknown_rels = referenced_relationship_types(cypher) - known_rel_set
    if unknown_rels:
        raise GuardrailViolation(
            f"Query references unknown relationship type(s): "
            f"{', '.join(sorted(unknown_rels))}"
        )

