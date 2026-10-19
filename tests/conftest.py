"""
Shared pytest fixtures for the retrieval layer.

Two principles drive the design:

**Real data, no network.** Fixtures load the actual knowledge-graph JSON, so
tests exercise the real label set, real identifier conventions, and real
relationship structure. Nothing asserts against an invented corpus that could
drift from what the system actually indexes.

**Deterministic stand-ins for the transformers.** The encoder and cross-encoder
are replaced by small, explainable fakes. This is not to avoid testing them --
it is because a 90 MB download makes the suite non-hermetic, and because a real
model gives no *predictable* score to assert on. The fakes let the ranking,
caching, thresholding, and failure-handling logic be tested exactly.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv

from src.config.env import process_env
from src.config.settings import PROJECT_ROOT, Neo4jConfig, Settings
from src.graph.model import GraphEdge, GraphNode, KnowledgeGraph
from src.graph.sources import JsonGraphSource
from src.retrieval.base import RetrievalResult
from src.retrieval.corpus import Corpus, RecordSerializer, build_corpus

# -- configuration ----------------------------------------------------------


#: The knowledge graph the whole suite runs against: the PoC export, which is
#: the source of truth for this project. Tests assert against the same data
#: the system actually serves, so a schema detail that holds in a test holds
#: in production.
#:
#: The older ``data/knowledge_graph_ai_sample.json`` extract is deliberately
#: NOT used: its schema has drifted from this export (its ``PriceComponent``
#: has no ``applies_to_id``), so testing against it would assert properties
#: the real graph does not have.
GRAPH_FILENAME = "axiom_pricing_module_Knowledge-graph_PoC.json"


@pytest.fixture(scope="session")
def graph_path() -> Path:
    """Path to the knowledge-graph JSON the whole suite runs against."""
    path = PROJECT_ROOT / GRAPH_FILENAME
    if not path.exists():  # pragma: no cover - guards a misconfigured checkout
        pytest.skip(f"Knowledge graph not found: {path}")
    return path


@pytest.fixture(scope="session")
def settings(graph_path: Path) -> Settings:
    """
    Settings built from an explicit dict rather than the process environment.

    A developer's local ``.env`` therefore cannot change test outcomes.
    """
    return Settings.from_env(
        {
            "CORPUS_SOURCE": "json",
            "CORPUS_JSON_PATH": str(graph_path),
            "DEFAULT_TOP_K": "10",
            "RRF_K": "60",
            "GRAPH_MAX_HOPS": "2",
            "RERANK_ENABLED": "true",
            "RERANK_MIN_SCORE": "0.0",
        }
    )


@pytest.fixture(scope="session")
def live_neo4j_config() -> Neo4jConfig:
    """
    Real connection details for a *live* Neo4j test, read from ``.env``.

    Deliberately separate from ``settings``: that fixture is built from an
    explicit dict precisely so a developer's ``.env`` cannot change hermetic
    test outcomes, which means it can never hold real database credentials.
    A live test needs the opposite -- the actual URI and password Neo4j
    Desktop is running with -- so it reads the process environment directly,
    the same way ``get_settings()`` does in production.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Neo4jConfig.from_env(process_env())


# -- graph and corpus -------------------------------------------------------


@pytest.fixture(scope="session")
def graph(settings: Settings) -> KnowledgeGraph:
    """The real AXIOM sample graph, loaded once per session."""
    return JsonGraphSource.from_config(settings.corpus).load()


@pytest.fixture(scope="session")
def serializer(settings: Settings) -> RecordSerializer:
    """Record serializer bound to the test configuration."""
    return RecordSerializer(settings.corpus)


@pytest.fixture(scope="session")
def corpus(graph: KnowledgeGraph, serializer: RecordSerializer) -> Corpus:
    """Retrieval corpus derived from the real graph."""
    return build_corpus(graph, serializer)


@pytest.fixture
def tiny_graph() -> KnowledgeGraph:
    """
    A minimal hand-built graph for traversal edge cases.

    Shape (all undirected for traversal purposes)::

        a --REL_X--> b --REL_Y--> c --REL_Z--> d
        a --REL_W--> d
    """
    nodes = [
        GraphNode(node_id=key, labels=("Thing",), properties={"thing_id": key.upper()})
        for key in ("a", "b", "c", "d")
    ]
    edges = [
        GraphEdge("REL_X", "a", "b"),
        GraphEdge("REL_Y", "b", "c"),
        GraphEdge("REL_Z", "c", "d"),
        GraphEdge("REL_W", "a", "d"),
    ]
    return KnowledgeGraph(nodes=nodes, edges=edges)


# -- deterministic model stand-ins ------------------------------------------


class HashingEncoder:
    """
    Deterministic bag-of-tokens encoder.

    Projects tokens into a fixed-width space by hashing. Two texts sharing
    vocabulary get similar vectors, which is enough to verify that the dense
    retriever ranks, normalises, and caches correctly -- without a download and
    without run-to-run variance.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.call_count = 0
        self.encoded_texts: list[str] = []

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        self.call_count += 1
        self.encoded_texts.extend(texts)
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                # Python's hash() is salted per process; zlib.crc32 is stable,
                # so a failure reproduces on the next run.
                from zlib import crc32

                matrix[row, crc32(token.encode()) % self.dim] += 1.0
        return matrix


class KeywordCrossEncoder:
    """
    Deterministic cross-encoder returning logit-scale scores.

    Scores by token overlap mapped onto roughly ``[-8, +8]``, mimicking the
    unbounded logit a real MiniLM emits -- which is what makes the sigmoid
    calibration and threshold logic meaningfully testable.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def predict_pairs(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        self.call_count += 1
        scores = []
        for query, document in pairs:
            query_tokens = set(query.lower().split())
            document_tokens = set(document.lower().split())
            if not query_tokens:
                scores.append(-8.0)
                continue
            overlap = len(query_tokens & document_tokens) / len(query_tokens)
            scores.append(overlap * 16.0 - 8.0)
        return scores


class ExplodingCrossEncoder:
    """Cross-encoder that always fails, for exercising fail-open behaviour."""

    def predict_pairs(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        raise RuntimeError("model backend unavailable")


@pytest.fixture
def encoder() -> HashingEncoder:
    return HashingEncoder()


@pytest.fixture
def cross_encoder() -> KeywordCrossEncoder:
    return KeywordCrossEncoder()


# -- helpers ----------------------------------------------------------------


@pytest.fixture
def make_result():
    """Factory for synthetic results, for fusion and reranking unit tests."""

    def _make(
        record_id: str,
        source: str = "test",
        score: float = 1.0,
        text: str | None = None,
        record_type: str = "TestRecord",
    ) -> RetrievalResult:
        return RetrievalResult(
            record_id=record_id,
            record_type=record_type,
            text=text if text is not None else f"text for {record_id}",
            score=score,
            source=source,
        )

    return _make


@pytest.fixture
def settings_with(settings: Settings):
    """Derive a settings variant by overriding one section, e.g. ``fusion``."""

    def _with(**sections) -> Settings:
        return replace(settings, **sections)

    return _with
