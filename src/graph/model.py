"""
Source-agnostic in-memory model of the AXIOM knowledge graph.

The retrieval layer is written against this model rather than against Neo4j,
which is what allows a single code path to serve both deployment modes:

    axiom_pricing_module_Knowledge-graph_PoC.json --\\
                                                  >-- KnowledgeGraph --> retrieval
    live Neo4j database --------------------------/

Nothing here knows an AXIOM label, relationship type, or property name. The
graph is whatever the loaded source says it is, and traversal is driven purely
by the adjacency actually present in the data.
"""
from __future__ import annotations

import logging
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = ["GraphEdge", "GraphNode", "KnowledgeGraph", "Neighbor"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GraphNode:
    """A single graph node: a stable id, its labels, and its properties."""

    node_id: str
    labels: tuple[str, ...]
    properties: Mapping[str, Any]

    @property
    def primary_label(self) -> str:
        """
        The label used as the record type in citations.

        Multi-label nodes take their first label, matching the order the source
        reports. Unlabelled nodes fall back to a neutral marker so downstream
        formatting never has to special-case an empty type.
        """
        return self.labels[0] if self.labels else "Unlabelled"


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """A directed, typed relationship between two nodes."""

    rel_type: str
    source_id: str
    target_id: str


@dataclass(frozen=True, slots=True)
class Neighbor:
    """
    A node reached by traversal, with the provenance of how it was reached.

    ``hops`` is the shortest-path distance from the anchor and is what the graph
    retriever converts into a relevance score. ``path`` records the relationship
    types walked, so an explanation can state *why* a record is connected rather
    than merely asserting that it is.
    """

    node: GraphNode
    hops: int
    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GraphStats:
    """Load-time integrity counters, surfaced in the pipeline trace."""

    node_count: int
    edge_count: int
    dangling_edge_count: int
    duplicate_node_count: int
    label_counts: Mapping[str, int]
    relationship_counts: Mapping[str, int]


class KnowledgeGraph:
    """
    An immutable, indexed view over a set of nodes and edges.

    Construction validates referential integrity: edges pointing at unknown
    nodes are dropped and counted rather than raising, because a partial graph
    export is a normal occurrence and should degrade retrieval quality, not
    take the service down. The drop count is reported through :attr:`stats` so
    the condition stays visible instead of silent.
    """

    __slots__ = ("_nodes", "_edges", "_adjacency", "_stats")

    def __init__(
        self,
        nodes: Iterable[GraphNode],
        edges: Iterable[GraphEdge] = (),
    ) -> None:
        node_map: dict[str, GraphNode] = {}
        duplicates = 0
        for node in nodes:
            if node.node_id in node_map:
                duplicates += 1
                logger.warning(
                    "Duplicate node id %r; keeping the first occurrence", node.node_id
                )
                continue
            node_map[node.node_id] = node

        kept_edges: list[GraphEdge] = []
        dangling = 0
        # Undirected adjacency, stored once per direction so a single BFS can
        # honour either the directed or undirected traversal mode.
        adjacency: dict[str, list[tuple[str, str, bool]]] = {
            node_id: [] for node_id in node_map
        }
        for edge in edges:
            if edge.source_id not in node_map or edge.target_id not in node_map:
                dangling += 1
                logger.debug(
                    "Dropping edge %s: %r -> %r references an unknown node",
                    edge.rel_type,
                    edge.source_id,
                    edge.target_id,
                )
                continue
            kept_edges.append(edge)
            # (neighbour_id, rel_type, is_outgoing)
            adjacency[edge.source_id].append((edge.target_id, edge.rel_type, True))
            adjacency[edge.target_id].append((edge.source_id, edge.rel_type, False))

        # Sorting the adjacency lists makes traversal order deterministic, which
        # in turn makes the ranks fed into RRF reproducible across runs.
        for neighbours in adjacency.values():
            neighbours.sort(key=lambda entry: (entry[1], entry[0], not entry[2]))

        self._nodes: Mapping[str, GraphNode] = MappingProxyType(node_map)
        self._edges: tuple[GraphEdge, ...] = tuple(kept_edges)
        self._adjacency: Mapping[str, tuple[tuple[str, str, bool], ...]] = (
            MappingProxyType(
                {node_id: tuple(items) for node_id, items in adjacency.items()}
            )
        )
        self._stats = GraphStats(
            node_count=len(node_map),
            edge_count=len(kept_edges),
            dangling_edge_count=dangling,
            duplicate_node_count=duplicates,
            label_counts=MappingProxyType(
                dict(Counter(node.primary_label for node in node_map.values()))
            ),
            relationship_counts=MappingProxyType(
                dict(Counter(edge.rel_type for edge in kept_edges))
            ),
        )
        if dangling:
            logger.warning(
                "Dropped %d edge(s) referencing unknown nodes during graph load",
                dangling,
            )

    # -- container protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __iter__(self) -> Iterator[GraphNode]:
        """Iterate nodes in insertion order (stable across runs for a JSON source)."""
        return iter(self._nodes.values())

    # -- accessors ----------------------------------------------------------

    @property
    def nodes(self) -> Mapping[str, GraphNode]:
        """Read-only mapping of ``node_id -> GraphNode``."""
        return self._nodes

    @property
    def edges(self) -> Sequence[GraphEdge]:
        """All edges whose endpoints both resolved at load time."""
        return self._edges

    @property
    def stats(self) -> GraphStats:
        """Load-time integrity counters for this graph."""
        return self._stats

    def node(self, node_id: str) -> GraphNode | None:
        """Return the node with ``node_id``, or ``None`` when absent."""
        return self._nodes.get(node_id)

    def degree(self, node_id: str) -> int:
        """Number of incident edges (both directions) for ``node_id``."""
        return len(self._adjacency.get(node_id, ()))

    # -- traversal ----------------------------------------------------------

    def traverse(
        self,
        start_id: str,
        *,
        max_hops: int,
        max_neighbors: int,
        relationship_types: frozenset[str] = frozenset(),
        directed: bool = False,
    ) -> list[Neighbor]:
        """
        Breadth-first traversal from ``start_id``, returning shortest-path hits.

        Args:
            start_id: anchor node to expand from; an unknown id yields ``[]``.
            max_hops: maximum path length to expand, inclusive.
            max_neighbors: hard cap on returned neighbours, applied in BFS
                order so the closest records survive truncation.
            relationship_types: when non-empty, only these relationship types
                are traversed. Empty means every type in the graph.
            directed: when True, follow relationships only in their declared
                direction.

        Returns:
            Neighbours ordered by ``(hops, node_id)``. The anchor itself is
            excluded. Each node appears once, at its shortest distance, so a
            densely connected record cannot flood the result with duplicates.
        """
        if start_id not in self._nodes or max_hops < 1 or max_neighbors < 1:
            return []

        visited: set[str] = {start_id}
        found: list[Neighbor] = []
        queue: deque[tuple[str, int, tuple[str, ...]]] = deque([(start_id, 0, ())])

        while queue and len(found) < max_neighbors:
            current_id, hops, path = queue.popleft()
            if hops >= max_hops:
                continue
            for neighbour_id, rel_type, is_outgoing in self._adjacency[current_id]:
                if neighbour_id in visited:
                    continue
                if relationship_types and rel_type not in relationship_types:
                    continue
                if directed and not is_outgoing:
                    continue
                visited.add(neighbour_id)
                neighbour_path = (*path, rel_type)
                found.append(
                    Neighbor(
                        node=self._nodes[neighbour_id],
                        hops=hops + 1,
                        path=neighbour_path,
                    )
                )
                if len(found) >= max_neighbors:
                    break
                queue.append((neighbour_id, hops + 1, neighbour_path))

        # BFS already emits in non-decreasing hop order; the explicit sort makes
        # the tie-break on node_id part of the contract rather than an accident
        # of adjacency ordering.
        found.sort(key=lambda neighbour: (neighbour.hops, neighbour.node.node_id))
        return found
