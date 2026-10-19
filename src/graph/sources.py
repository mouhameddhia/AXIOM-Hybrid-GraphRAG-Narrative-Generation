"""
Pluggable loaders that produce a :class:`~src.graph.model.KnowledgeGraph`.

Two interchangeable implementations satisfy the same protocol:

``JsonGraphSource``
    Reads an exported knowledge-graph JSON document. This is the default,
    because it makes the entire retrieval stack runnable, testable, and
    demonstrable with no database dependency.

``Neo4jGraphSource``
    Projects a live Neo4j database into the same structure, using schema
    introspection rather than any hard-coded label or property name.

Because both return the identical type, retrieval behaviour is independent of
where the graph came from -- and an offline test asserts exactly the same
invariants the production path relies on.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.config.settings import CorpusConfig, Settings
from src.graph.model import GraphEdge, GraphNode, KnowledgeGraph

__all__ = [
    "GraphSource",
    "GraphSourceError",
    "JsonGraphSource",
    "Neo4jGraphSource",
    "build_graph_source",
]

logger = logging.getLogger(__name__)

#: Accepted spellings for relationship endpoints across export dialects.
_SOURCE_KEYS: tuple[str, ...] = ("source", "start", "from", "start_node", "source_id")
_TARGET_KEYS: tuple[str, ...] = ("target", "end", "to", "end_node", "target_id")
_TYPE_KEYS: tuple[str, ...] = ("type", "rel_type", "relationship", "label")
_NODE_ID_KEYS: tuple[str, ...] = ("id", "node_id", "_id", "elementId")


class GraphSourceError(RuntimeError):
    """Raised when a graph source cannot be read or is structurally invalid."""


@runtime_checkable
class GraphSource(Protocol):
    """Anything able to materialise a :class:`KnowledgeGraph`."""

    @property
    def name(self) -> str:
        """Short identifier for this source, recorded in the pipeline trace."""

    def load(self) -> KnowledgeGraph:
        """Read the underlying store and return an in-memory graph."""


class JsonGraphSource:
    """
    Loads a knowledge graph from a JSON export.

    The parser is deliberately permissive about dialect -- endpoint keys, node
    id keys, and the top-level container keys are all resolved by probing a set
    of accepted spellings (or overridden via :class:`CorpusConfig`) -- and
    deliberately strict about structure: a malformed entry is dropped with a
    warning and counted, never silently coerced into a bogus node.
    """

    def __init__(
        self,
        path: Path,
        *,
        nodes_key: str = "nodes",
        relationships_key: str = "relationships",
        encoding: str = "utf-8",
    ) -> None:
        self._path = Path(path)
        self._nodes_key = nodes_key
        self._relationships_key = relationships_key
        self._encoding = encoding

    @property
    def name(self) -> str:
        return f"json:{self._path.name}"

    @classmethod
    def from_config(cls, config: CorpusConfig) -> "JsonGraphSource":
        """Build a source from the corpus section of the application settings."""
        return cls(
            path=config.json_path,
            nodes_key=config.nodes_key,
            relationships_key=config.relationships_key,
        )

    def load(self) -> KnowledgeGraph:
        """Parse the JSON document into a :class:`KnowledgeGraph`."""
        payload = self._read_payload()
        raw_nodes = self._section(payload, self._nodes_key)
        raw_edges = self._section(payload, self._relationships_key, required=False)

        nodes = list(self._parse_nodes(raw_nodes))
        if not nodes:
            raise GraphSourceError(
                f"{self._path} contains no usable nodes under key "
                f"{self._nodes_key!r}"
            )
        edges = list(self._parse_edges(raw_edges))

        graph = KnowledgeGraph(nodes=nodes, edges=edges)
        logger.info(
            "Loaded %d nodes / %d relationships from %s",
            graph.stats.node_count,
            graph.stats.edge_count,
            self._path.name,
        )
        return graph

    # -- parsing helpers ----------------------------------------------------

    def _read_payload(self) -> Mapping[str, Any]:
        try:
            raw = self._path.read_text(encoding=self._encoding)
        except FileNotFoundError as exc:
            raise GraphSourceError(
                f"Knowledge-graph file not found: {self._path}"
            ) from exc
        except OSError as exc:
            raise GraphSourceError(f"Cannot read {self._path}: {exc}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GraphSourceError(
                f"{self._path} is not valid JSON (line {exc.lineno}): {exc.msg}"
            ) from exc

        if not isinstance(payload, Mapping):
            raise GraphSourceError(
                f"{self._path} must contain a JSON object, "
                f"got {type(payload).__name__}"
            )
        return payload

    def _section(
        self, payload: Mapping[str, Any], key: str, *, required: bool = True
    ) -> Sequence[Any]:
        section = payload.get(key)
        if section is None:
            if required:
                raise GraphSourceError(f"{self._path} is missing the {key!r} array")
            return ()
        if not isinstance(section, Sequence) or isinstance(section, (str, bytes)):
            raise GraphSourceError(
                f"{self._path}[{key!r}] must be an array, got {type(section).__name__}"
            )
        return section

    def _parse_nodes(self, raw_nodes: Sequence[Any]) -> Iterable[GraphNode]:
        skipped = 0
        for index, entry in enumerate(raw_nodes):
            if not isinstance(entry, Mapping):
                skipped += 1
                continue
            node_id = _first_present(entry, _NODE_ID_KEYS)
            if node_id is None:
                skipped += 1
                logger.warning("Node at index %d has no id key; skipped", index)
                continue
            properties = entry.get("properties", {})
            if not isinstance(properties, Mapping):
                properties = {}
            yield GraphNode(
                node_id=str(node_id),
                labels=_normalise_labels(entry.get("labels", entry.get("label"))),
                properties=dict(properties),
            )
        if skipped:
            logger.warning("Skipped %d malformed node entries in %s", skipped, self._path.name)

    def _parse_edges(self, raw_edges: Sequence[Any]) -> Iterable[GraphEdge]:
        skipped = 0
        for index, entry in enumerate(raw_edges):
            if not isinstance(entry, Mapping):
                skipped += 1
                continue
            source_id = _first_present(entry, _SOURCE_KEYS)
            target_id = _first_present(entry, _TARGET_KEYS)
            rel_type = _first_present(entry, _TYPE_KEYS)
            if source_id is None or target_id is None or rel_type is None:
                skipped += 1
                logger.warning(
                    "Relationship at index %d is missing source/target/type; skipped",
                    index,
                )
                continue
            yield GraphEdge(
                rel_type=str(rel_type),
                source_id=str(source_id),
                target_id=str(target_id),
            )
        if skipped:
            logger.warning(
                "Skipped %d malformed relationship entries in %s", skipped, self._path.name
            )


class Neo4jGraphSource:
    """
    Projects a live Neo4j database into a :class:`KnowledgeGraph`.

    The Cypher below names no label, relationship type, or property: it asks
    the database what exists. That is what keeps the production path free of
    the schema assumptions that a hand-written ``coalesce(...)`` identifier
    ladder would otherwise bake into the query.
    """

    #: ``elementId`` is the only node identifier guaranteed to exist and be
    #: unique, so it anchors the projection; business identifiers are derived
    #: later, from properties, by the corpus serializer.
    NODES_CYPHER = (
        "MATCH (n) "
        "RETURN elementId(n) AS node_id, labels(n) AS labels, "
        "properties(n) AS properties "
        "ORDER BY node_id"
    )
    RELATIONSHIPS_CYPHER = (
        "MATCH (a)-[r]->(b) "
        "RETURN type(r) AS rel_type, elementId(a) AS source_id, "
        "elementId(b) AS target_id "
        "ORDER BY rel_type, source_id, target_id"
    )

    def __init__(self, client: Any, *, node_limit: int | None = None) -> None:
        """
        Args:
            client: anything exposing ``run_query(cypher, parameters) -> list[dict]``
                (see :class:`src.graph.neo4j_client.Neo4jClient`). Injected rather
                than constructed so tests can substitute a fake.
            node_limit: optional cap for large graphs, applied server-side.
        """
        self._client = client
        self._node_limit = node_limit

    @property
    def name(self) -> str:
        return "neo4j"

    def load(self) -> KnowledgeGraph:
        """Read all nodes and relationships and return an in-memory graph."""
        nodes_cypher = self.NODES_CYPHER
        if self._node_limit is not None:
            nodes_cypher = f"{nodes_cypher} LIMIT $node_limit"

        try:
            node_rows = self._client.run_query(
                nodes_cypher,
                {"node_limit": self._node_limit} if self._node_limit else None,
            )
            edge_rows = self._client.run_query(self.RELATIONSHIPS_CYPHER, None)
        except Exception as exc:  # noqa: BLE001 - driver raises a wide family
            raise GraphSourceError(f"Neo4j projection failed: {exc}") from exc

        nodes = [
            GraphNode(
                node_id=str(row["node_id"]),
                labels=_normalise_labels(row.get("labels")),
                properties=dict(row.get("properties") or {}),
            )
            for row in node_rows
        ]
        edges = [
            GraphEdge(
                rel_type=str(row["rel_type"]),
                source_id=str(row["source_id"]),
                target_id=str(row["target_id"]),
            )
            for row in edge_rows
        ]
        graph = KnowledgeGraph(nodes=nodes, edges=edges)
        logger.info(
            "Projected %d nodes / %d relationships from Neo4j",
            graph.stats.node_count,
            graph.stats.edge_count,
        )
        return graph


def build_graph_source(settings: Settings, *, client: Any | None = None) -> GraphSource:
    """
    Select the graph source named by ``settings.corpus.source``.

    Args:
        settings: the application configuration.
        client: a Neo4j client, required only when the configured source is
            ``neo4j``. Passing it in keeps this factory free of connection
            handling and keeps the JSON path free of the driver entirely.
    """
    corpus = settings.corpus
    if corpus.source == CorpusConfig.JSON:
        return JsonGraphSource.from_config(corpus)
    if corpus.source == CorpusConfig.NEO4J:
        if client is None:
            raise GraphSourceError(
                "CORPUS_SOURCE=neo4j requires a Neo4j client to be supplied"
            )
        return Neo4jGraphSource(client)
    raise GraphSourceError(f"Unsupported corpus source: {corpus.source!r}")


def _first_present(entry: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    """Return the first non-empty value among ``keys``, else ``None``."""
    for key in keys:
        value = entry.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalise_labels(value: Any) -> tuple[str, ...]:
    """Coerce a label field (string, list, or missing) into an ordered tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if item)
    return (str(value),)
