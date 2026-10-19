"""
Turns raw query rows into citable evidence.

A number returned by text-to-Cypher is only trustworthy if it can be traced
back to the record(s) it came from. This module does that by reusing the
*exact same* identity logic retrieval already uses
(:class:`~src.retrieval.corpus.RecordSerializer`), so a node found by
text-to-Cypher cites identically to the same node found by search --
``AuthorityPeriod:AG001`` either way, never a second citation scheme.

A result row has two kinds of field: a scalar (the answer -- a count, an
average, a property value) and a node (``{"labels": [...], "properties":
{...}}`` -- the evidence). This module separates the two.

Both execution paths emit that same row shape, deliberately:
:func:`src.text_to_query.plan.execute_plan` when running against the JSON
graph, and
:meth:`~src.graph.neo4j_client.Neo4jClient.run_query_with_entities` when
running against Neo4j. Evidence assembly therefore needs no knowledge of
which one answered.
"""
from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.graph.model import GraphNode
from src.retrieval.corpus import DISAMBIGUATOR, RecordSerializer

__all__ = ["EvidenceRecord", "assemble_evidence", "split_row"]


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One citable record backing a text-to-Cypher answer."""

    record_id: str
    record_type: str
    text: str


def split_row(row: Mapping[str, object]) -> tuple[dict[str, object], list[Mapping[str, object]]]:
    """
    Separate one result row into its scalar fields and its node-shaped fields.

    A node-shaped field is a dict carrying exactly ``labels`` and
    ``properties`` -- the shape :meth:`Neo4jClient.run_query_with_entities`
    produces. Everything else is a scalar answer value.
    """
    scalars: dict[str, object] = {}
    nodes: list[Mapping[str, object]] = []
    for key, value in row.items():
        if _is_node_shaped(value):
            nodes.append(value)  # type: ignore[arg-type]
        else:
            scalars[key] = value
    return scalars, nodes


def _is_node_shaped(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and "labels" in value
        and "properties" in value
        and isinstance(value["properties"], Mapping)
    )


def assemble_evidence(
    rows: Sequence[Mapping[str, object]], serializer: RecordSerializer
) -> tuple[EvidenceRecord, ...]:
    """
    Build the citable evidence set for a whole result.

    Args:
        rows: the query result, as returned by
            :meth:`Neo4jClient.run_query_with_entities`.
        serializer: the same instance the retrieval corpus was built with, so
            identifier derivation is identical between the two paths.

    Returns:
        One :class:`EvidenceRecord` per distinct node found across every row,
        in first-seen order. The *same* node cited from two rows (one quote
        reached via two different joins) is de-duplicated to a single entry;
        two genuinely *different* nodes that happen to share an identity
        value (as some AXIOM period records legitimately do) are disambiguated
        by a suffix, never silently collapsed into one -- the same convention
        :func:`src.retrieval.corpus.build_corpus` uses.
    """
    # record_id -> a signature identifying which node currently holds it, so
    # a second node claiming the same handle can be told apart from the same
    # node seen again through a different join.
    holder_by_id: dict[str, tuple[str, ...]] = {}
    records: dict[str, EvidenceRecord] = {}

    for row in rows:
        _, node_fields = split_row(row)
        for node_data in node_fields:
            parsed = _parse_node(node_data)
            if parsed is None:
                continue
            node, signature = parsed
            record = serializer.to_record(node)

            existing_signature = holder_by_id.get(record.record_id)
            if existing_signature is None:
                holder_by_id[record.record_id] = signature
                records[record.record_id] = EvidenceRecord(
                    record.record_id, record.record_type, record.text
                )
            elif existing_signature != signature:
                # A different node genuinely collides on this handle.
                disambiguated_id = f"{record.record_id}{DISAMBIGUATOR}{node.node_id}"
                records.setdefault(
                    disambiguated_id,
                    EvidenceRecord(disambiguated_id, record.record_type, record.text),
                )
            # else: the same node, reached again -- already recorded.

    return tuple(records.values())


def _parse_node(node_data: Mapping[str, object]) -> tuple[GraphNode, tuple[str, ...]] | None:
    """Build a :class:`GraphNode` from one node-shaped field, with a content signature."""
    labels = node_data.get("labels")
    properties = node_data.get("properties")
    if not labels or not isinstance(properties, Mapping):
        return None

    properties = dict(properties)
    # A synthetic node id: text-to-Cypher evidence has no graph-internal id to
    # anchor to. Derived from content via crc32 (deterministic across runs,
    # unlike Python's randomised str hash) so the *same* node produces the
    # same id wherever it is returned from, and a *different* node with an
    # identical identity value still gets a distinct, reproducible suffix.
    signature = tuple(f"{key}={properties[key]!r}" for key in sorted(properties))
    node_id = f"ttc:{zlib.crc32('|'.join(signature).encode('utf-8')):08x}"
    return GraphNode(node_id=node_id, labels=tuple(labels), properties=properties), signature
