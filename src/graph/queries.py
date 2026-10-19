"""
Centralised Cypher for live-graph operations.

What was removed and why
------------------------
The previous version of this module carried two hard-coded structures:

* a ``coalesce(n.quote_id, n.order_line_id, n.order_id, ...)`` ladder of
  fourteen property names, used to synthesise a record id. Any label whose
  identifier was not in that list silently fell through to ``elementId``,
  producing an unstable, uncitable handle. Identifier derivation now happens in
  :class:`src.retrieval.corpus.RecordSerializer`, driven by the data.

* a ``WHERE NOT n:Transaction AND NOT n:Customer AND NOT n:Product ...`` filter
  naming five labels that do not exist in the AXIOM graph -- a dead clause that
  filtered nothing while appearing to enforce a policy. Label filtering is now
  configuration (``CORPUS_INCLUDE_LABELS`` / ``CORPUS_EXCLUDE_LABELS``).

What remains here names no label, relationship type, or property: these
statements ask the database what it contains.
"""
from __future__ import annotations

__all__ = [
    "COUNT_NODES",
    "SCHEMA_LABELS",
    "SCHEMA_NODE_PROPERTIES",
    "SCHEMA_RELATIONSHIP_TYPES",
    "SCHEMA_VISUALISATION",
]

#: Row count, used by the demo and health checks to confirm a populated graph.
COUNT_NODES = "MATCH (n) RETURN count(n) AS node_count"

#: Live schema introspection. These feed the dynamic schema-pruning step of the
#: text-to-Cypher engine, so that the prompt describes the database as it is
#: rather than as a static document claims it to be.
SCHEMA_LABELS = "CALL db.labels() YIELD label RETURN label ORDER BY label"

SCHEMA_RELATIONSHIP_TYPES = (
    "CALL db.relationshipTypes() YIELD relationshipType "
    "RETURN relationshipType ORDER BY relationshipType"
)

#: Property keys per label, sampled rather than exhaustively scanned so that
#: introspection stays cheap on a large graph.
SCHEMA_NODE_PROPERTIES = """
MATCH (n)
WITH labels(n)[0] AS label, keys(n) AS property_keys
LIMIT $sample_size
UNWIND property_keys AS property_key
RETURN label, collect(DISTINCT property_key) AS property_keys
ORDER BY label
"""

#: Full structural summary (label -> relationship -> label), the backbone of
#: the pruned schema shown to the model.
SCHEMA_VISUALISATION = "CALL db.schema.visualization()"
