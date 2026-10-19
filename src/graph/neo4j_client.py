"""
Reusable Neo4j client.

Every module that talks to the graph goes through this client rather than
opening its own driver, so connection handling, retry semantics, and read-only
enforcement stay in one reviewable place.

Sessions are opened per query and closed immediately; the driver underneath
pools connections, so this is both correct and cheap. Queries are executed in
explicit **read** transactions, which means a generated Cypher statement that
attempts a write is rejected by the server rather than relied upon to be
well-behaved -- the guarantee the text-to-Cypher layer depends on.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from neo4j import Driver, GraphDatabase
from neo4j.graph import Node
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from src.config.settings import Neo4jConfig, get_settings

__all__ = ["Neo4jClient", "Neo4jQueryError", "WriteCounters"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WriteCounters:
    """What a write statement actually changed, as reported by the server."""

    nodes_created: int = 0
    relationships_created: int = 0
    properties_set: int = 0
    labels_added: int = 0
    constraints_added: int = 0
    indexes_added: int = 0

    def __add__(self, other: "WriteCounters") -> "WriteCounters":
        return WriteCounters(
            nodes_created=self.nodes_created + other.nodes_created,
            relationships_created=self.relationships_created + other.relationships_created,
            properties_set=self.properties_set + other.properties_set,
            labels_added=self.labels_added + other.labels_added,
            constraints_added=self.constraints_added + other.constraints_added,
            indexes_added=self.indexes_added + other.indexes_added,
        )


class Neo4jQueryError(RuntimeError):
    """
    Raised when a Cypher statement fails.

    Carries the server's own message and status code unmodified. The
    self-healing text-to-Cypher loop feeds that exact text back to the model,
    so paraphrasing it here would discard the signal the repair depends on.
    """

    def __init__(self, message: str, *, code: str | None = None, cypher: str | None = None):
        super().__init__(message)
        self.code = code
        self.cypher = cypher


class Neo4jClient:
    """
    Thin, configuration-driven wrapper around the official Neo4j driver.

    Usage::

        with Neo4jClient() as client:
            rows = client.run_query("MATCH (n) RETURN count(n) AS total")
    """

    __slots__ = ("_config", "_driver")

    def __init__(self, config: Neo4jConfig | None = None) -> None:
        self._config = config or get_settings().neo4j
        self._driver: Driver | None = None

    @property
    def database(self) -> str:
        """Name of the database this client targets."""
        return self._config.database

    def connect(self) -> None:
        """Open the driver. Idempotent."""
        if self._driver is not None:
            return
        self._driver = GraphDatabase.driver(
            self._config.uri,
            auth=(self._config.user, self._config.password),
            connection_timeout=self._config.connection_timeout_s,
        )
        logger.info("Neo4j driver opened for %s", self._config.uri)

    def verify_connectivity(self) -> bool:
        """
        Return whether the database is reachable.

        Returns a bool rather than raising: callers use this to decide between
        the live-graph path and the JSON path, which is a routine branch, not
        an error condition.
        """
        try:
            self.connect()
            self._driver.verify_connectivity()
            return True
        except (ServiceUnavailable, Neo4jError, OSError) as exc:
            logger.warning("Neo4j is not reachable at %s: %s", self._config.uri, exc)
            return False

    def run_query(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute ``cypher`` in a read transaction and return plain dict rows.

        A node or relationship returned by the query is flattened to its
        properties alone (its labels and identity are dropped) -- fine for
        callers who already know what they asked for, such as
        :class:`~src.graph.sources.Neo4jGraphSource`, whose Cypher explicitly
        projects ``labels(n)`` and ``properties(n)`` as separate fields. A
        caller that does not control the query's shape this precisely --
        text-to-Cypher, where the query comes from an LLM -- should use
        :meth:`run_query_with_entities` instead, which preserves labels.

        Args:
            cypher: the statement. Never interpolate values into it -- pass
                them via ``parameters`` so the server parameterises them and
                the query plan stays cacheable.
            parameters: query parameters.

        Returns:
            One dict per record.

        Raises:
            Neo4jQueryError: wrapping the server's message and status code.
        """
        rows = self._execute_read(cypher, parameters, row=lambda record: record.data())
        return rows

    def run_query_with_entities(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute ``cypher`` and return rows that preserve node identity.

        Unlike :meth:`run_query`, a node-valued field is not flattened to its
        bare properties: it becomes ``{"labels": [...], "properties": {...}}``.
        This is what lets a query written by an LLM -- which cannot be relied
        on to always add an explicit ``labels(x)`` projection -- still be
        turned into a citable record afterwards, by feeding that shape into
        the same identity logic retrieval already uses.

        Returns:
            One dict per record; scalar fields are passed through unchanged,
            node-valued fields become the label/property structure above.

        Raises:
            Neo4jQueryError: wrapping the server's message and status code.
        """
        return self._execute_read(cypher, parameters, row=_entity_aware_row)

    def execute_write(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> WriteCounters:
        """
        Execute a write statement and report what it changed.

        **This is for data ingestion only.** It is deliberately a separate,
        explicitly-named method rather than a flag on :meth:`run_query`,
        because the read-only guarantee that method provides is a security
        control the text-to-Cypher layer depends on: a query written by a
        language model reaches the database through :meth:`run_query` or
        :meth:`run_query_with_entities`, both of which run in a read
        transaction that the server itself refuses to let write.

        Nothing in ``src/text_to_query`` calls this method, and nothing
        should. Its only caller is ``scripts/import_synthetic_data.py``,
        which imports curated CSV files the maintainer chose to load.

        Returns:
            :class:`WriteCounters` with the server's own report of nodes and
            relationships created, so an importer can distinguish a row it
            created from one that already existed -- which is what makes a
            MERGE-based import safely rerunnable *and* honestly reportable.

        Raises:
            Neo4jQueryError: wrapping the server's message and status code.
        """
        self.connect()
        try:
            with self._driver.session(database=self._config.database) as session:
                return session.execute_write(_write_and_count, cypher, parameters or {})
        except Neo4jError as exc:
            raise Neo4jQueryError(
                str(exc.message or exc), code=exc.code, cypher=cypher
            ) from exc
        except ServiceUnavailable as exc:
            raise Neo4jQueryError(f"Neo4j unavailable: {exc}", cypher=cypher) from exc

    def _execute_read(
        self,
        cypher: str,
        parameters: dict[str, Any] | None,
        *,
        row: "Callable[[Any], dict[str, Any]]",
    ) -> list[dict[str, Any]]:
        """Shared connect/execute/error-translation path for both read methods."""
        self.connect()
        try:
            with self._driver.session(database=self._config.database) as session:
                # execute_read (not session.run) is what actually enforces read
                # access mode: a statement containing CREATE/MERGE/DELETE is
                # refused by the server. That server-side guarantee is what the
                # text-to-Cypher layer relies on -- prompt instructions alone
                # are not a security control.
                return session.execute_read(
                    lambda tx: [row(record) for record in tx.run(cypher, parameters or {})]
                )
        except Neo4jError as exc:
            # exc.message / exc.code are the server's own words: the repair
            # loop needs them verbatim.
            raise Neo4jQueryError(
                str(exc.message or exc), code=exc.code, cypher=cypher
            ) from exc
        except ServiceUnavailable as exc:
            raise Neo4jQueryError(f"Neo4j unavailable: {exc}", cypher=cypher) from exc

    def close(self) -> None:
        """Close the driver. Idempotent."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    def __enter__(self) -> "Neo4jClient":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _entity_aware_row(record: Any) -> dict[str, Any]:
    """
    Convert one driver record to a dict, preserving node labels.

    ``record.data()`` would flatten a node value to its bare properties; this
    instead keeps ``{"labels": [...], "properties": {...}}`` for anything the
    driver represents as a :class:`neo4j.graph.Node`, and passes every other
    value through as-is (Python's native ``int``/``float``/``str`` for a
    scalar, recursively-converted lists/dicts for a collection).
    """
    return {key: _entity_aware_value(value) for key, value in record.items()}


def _entity_aware_value(value: Any) -> Any:
    """Recursively convert one field, preserving node labels where present."""
    if isinstance(value, Node):
        return {"labels": list(value.labels), "properties": dict(value)}
    if isinstance(value, list):
        return [_entity_aware_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _entity_aware_value(item) for key, item in value.items()}
    return value


def _write_and_count(tx: Any, cypher: str, parameters: dict[str, Any]) -> WriteCounters:
    """Transaction function: run a write and return the server's own counters."""
    # consume() drains the stream and returns the summary in one step; calling
    # it twice would report against an already-exhausted result.
    counters = tx.run(cypher, parameters).consume().counters
    return WriteCounters(
        nodes_created=counters.nodes_created,
        relationships_created=counters.relationships_created,
        properties_set=counters.properties_set,
        labels_added=counters.labels_added,
        constraints_added=counters.constraints_added,
        indexes_added=counters.indexes_added,
    )
