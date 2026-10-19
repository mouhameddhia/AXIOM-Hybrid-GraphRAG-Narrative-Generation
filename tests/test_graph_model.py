"""
Tests for the in-memory graph model and its loaders.

Traversal determinism gets particular attention: because RRF consumes rank
position, a non-deterministic traversal order silently makes the whole
pipeline's output non-reproducible -- which was the defect in the previous
``LIMIT``-without-``ORDER BY`` Cypher.
"""
from __future__ import annotations

import json

import pytest

from src.graph.model import GraphEdge, GraphNode, KnowledgeGraph
from src.graph.sources import GraphSourceError, JsonGraphSource, Neo4jGraphSource


# -- loading ----------------------------------------------------------------


def test_loads_the_real_graph(graph):
    assert graph.stats.node_count > 0
    assert graph.stats.edge_count > 0
    assert graph.stats.dangling_edge_count == 0
    assert graph.stats.duplicate_node_count == 0


def test_missing_file_raises_a_clear_error(tmp_path):
    source = JsonGraphSource(tmp_path / "absent.json")
    with pytest.raises(GraphSourceError, match="not found"):
        source.load()


def test_invalid_json_raises_a_clear_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(GraphSourceError, match="not valid JSON"):
        JsonGraphSource(path).load()


def test_missing_nodes_key_raises(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"relationships": []}), encoding="utf-8")
    with pytest.raises(GraphSourceError, match="missing the 'nodes' array"):
        JsonGraphSource(path).load()


def test_alternate_endpoint_spellings_are_accepted(tmp_path):
    """Export dialects differ; the loader probes rather than assuming one."""
    path = tmp_path / "dialect.json"
    path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "a", "labels": ["A"], "properties": {"a_id": "A1"}},
                    {"id": "b", "labels": ["B"], "properties": {"b_id": "B1"}},
                ],
                "relationships": [{"rel_type": "LINKS", "start": "a", "end": "b"}],
            }
        ),
        encoding="utf-8",
    )

    graph = JsonGraphSource(path).load()

    assert graph.stats.edge_count == 1
    assert graph.traverse("a", max_hops=1, max_neighbors=5)[0].node.node_id == "b"


def test_dangling_edges_are_dropped_and_counted(tmp_path):
    """A partial export degrades retrieval; it must not crash the loader."""
    path = tmp_path / "partial.json"
    path.write_text(
        json.dumps(
            {
                "nodes": [{"id": "a", "labels": ["A"], "properties": {"a_id": "A1"}}],
                "relationships": [{"type": "LINKS", "source": "a", "target": "ghost"}],
            }
        ),
        encoding="utf-8",
    )

    graph = JsonGraphSource(path).load()

    assert graph.stats.edge_count == 0
    assert graph.stats.dangling_edge_count == 1


def test_neo4j_source_projects_into_the_same_type():
    """The Neo4j path must be indistinguishable downstream from the JSON path."""

    class FakeClient:
        def run_query(self, cypher: str, parameters=None):
            if "labels(n)" in cypher:
                return [
                    {"node_id": "1", "labels": ["A"], "properties": {"a_id": "A1"}},
                    {"node_id": "2", "labels": ["B"], "properties": {"b_id": "B1"}},
                ]
            return [{"rel_type": "LINKS", "source_id": "1", "target_id": "2"}]

    graph = Neo4jGraphSource(FakeClient()).load()

    assert isinstance(graph, KnowledgeGraph)
    assert graph.stats.node_count == 2
    assert graph.stats.edge_count == 1


def test_neo4j_failure_is_wrapped():
    class FailingClient:
        def run_query(self, cypher: str, parameters=None):
            raise RuntimeError("connection refused")

    with pytest.raises(GraphSourceError, match="connection refused"):
        Neo4jGraphSource(FailingClient()).load()


# -- traversal --------------------------------------------------------------


def test_traversal_returns_shortest_hop_distance(tiny_graph):
    """``d`` is reachable in 1 hop via REL_W and 3 via the chain; 1 must win."""
    neighbours = {n.node.node_id: n.hops for n in tiny_graph.traverse("a", max_hops=3, max_neighbors=10)}

    assert neighbours == {"b": 1, "d": 1, "c": 2}


def test_traversal_excludes_the_anchor(tiny_graph):
    assert all(n.node.node_id != "a" for n in tiny_graph.traverse("a", max_hops=3, max_neighbors=10))


def test_traversal_respects_max_hops(tiny_graph):
    one_hop = tiny_graph.traverse("a", max_hops=1, max_neighbors=10)
    assert {n.node.node_id for n in one_hop} == {"b", "d"}


def test_traversal_is_deterministic(graph):
    """Identical input must produce an identical rank order, every run."""
    anchor = next(iter(graph)).node_id
    runs = {
        tuple(n.node.node_id for n in graph.traverse(anchor, max_hops=2, max_neighbors=20))
        for _ in range(5)
    }
    assert len(runs) == 1


def test_traversal_orders_by_hop_then_id(graph):
    """Ordering is (hops, node_id): the contract fusion ranks depend on."""
    anchor = max(graph.nodes, key=graph.degree)
    neighbours = graph.traverse(anchor, max_hops=2, max_neighbors=50)

    keys = [(n.hops, n.node.node_id) for n in neighbours]
    assert keys == sorted(keys)


def test_traversal_records_the_relationship_path(tiny_graph):
    """
    The walked path is the provenance a grounded explanation cites.

    ``c`` sits at the end of two equal-length paths (a-b-c and a-d-c). The
    contract is that the recorded path has exactly ``hops`` relationships and
    is genuinely one of the valid routes -- and that the choice between equal
    routes is stable, which the determinism test covers separately.
    """
    neighbour = next(
        n for n in tiny_graph.traverse("a", max_hops=3, max_neighbors=10) if n.node.node_id == "c"
    )

    assert len(neighbour.path) == neighbour.hops == 2
    assert neighbour.path in {("REL_X", "REL_Y"), ("REL_W", "REL_Z")}


def test_single_route_path_is_exact(tiny_graph):
    """Where only one route exists, the recorded path is fully determined."""
    neighbour = next(
        n
        for n in tiny_graph.traverse(
            "a", max_hops=3, max_neighbors=10, relationship_types=frozenset({"REL_X", "REL_Y"})
        )
        if n.node.node_id == "c"
    )
    assert neighbour.path == ("REL_X", "REL_Y")


def test_relationship_type_filter(tiny_graph):
    neighbours = tiny_graph.traverse(
        "a", max_hops=3, max_neighbors=10, relationship_types=frozenset({"REL_W"})
    )
    assert {n.node.node_id for n in neighbours} == {"d"}


def test_directed_traversal_follows_declared_direction(tiny_graph):
    """Undirected reaches ``a`` from ``b``; directed does not."""
    undirected = tiny_graph.traverse("b", max_hops=1, max_neighbors=10, directed=False)
    directed = tiny_graph.traverse("b", max_hops=1, max_neighbors=10, directed=True)

    assert "a" in {n.node.node_id for n in undirected}
    assert "a" not in {n.node.node_id for n in directed}


def test_unknown_anchor_returns_empty(graph):
    assert graph.traverse("no_such_node", max_hops=2, max_neighbors=10) == []


def test_self_loops_do_not_revisit_the_anchor():
    """The AXIOM graph contains a self-referential REVISES edge."""
    graph = KnowledgeGraph(
        nodes=[GraphNode("a", ("A",), {"a_id": "A1"})],
        edges=[GraphEdge("REVISES", "a", "a")],
    )
    assert graph.traverse("a", max_hops=3, max_neighbors=10) == []


def test_max_neighbors_keeps_the_closest(graph):
    """Truncation must drop the far records, not an arbitrary slice."""
    anchor = max(graph.nodes, key=graph.degree)
    full = graph.traverse(anchor, max_hops=2, max_neighbors=100)
    capped = graph.traverse(anchor, max_hops=2, max_neighbors=2)

    assert len(capped) == 2
    assert max(n.hops for n in capped) <= max(n.hops for n in full)


def test_duplicate_node_ids_are_deduplicated():
    graph = KnowledgeGraph(
        nodes=[GraphNode("a", ("A",), {"v": 1}), GraphNode("a", ("A",), {"v": 2})]
    )
    assert graph.stats.node_count == 1
    assert graph.stats.duplicate_node_count == 1
