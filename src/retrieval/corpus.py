"""
Derivation of the retrieval corpus from a knowledge graph.

This module replaces the two places where graph schema used to be hard-coded:
the ``coalesce(...)`` identifier ladder in the Cypher, and the per-label
``preferred_keys`` map in the graph retriever. Both encoded knowledge that is
already present in the data, and both silently produced wrong output the moment
a new label appeared.

Identifier derivation
---------------------
A record's citable id is ``<Label>:<value>``. The value is chosen by scoring
each property name against the node's own label, splitting both into tokens:

    OrderLine  -> {order, line}
      order_line_id -> {order, line}   overlap 2  <- selected
      order_id      -> {order}         overlap 1
      product_id    -> {product}       overlap 0

The strategy falls through a documented ladder (identifier keys by label
affinity, then any identifier key, then any key by label affinity, then a
naming key, then the raw node id), so every node in any graph resolves to
*something* stable. Collisions are then de-duplicated deterministically, which
matters because several AXIOM period-style labels legitimately share a business
identifier across validity windows.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any

from src.config.settings import CorpusConfig
from src.graph.model import GraphNode, KnowledgeGraph

__all__ = [
    "Corpus",
    "CorpusRecord",
    "LabelIdentityPolicy",
    "RecordSerializer",
    "build_corpus",
]

logger = logging.getLogger(__name__)

#: Splits CamelCase / snake_case / digit boundaries into comparable tokens.
_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-z]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

#: Separates the label from the identifying value in a citable record id.
ID_SEPARATOR = ":"
#: Appended, with the node id, only when two records would otherwise collide.
DISAMBIGUATOR = "#"


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    """
    One retrievable, citable unit of evidence.

    ``record_id`` is the citation handle that flows all the way through fusion,
    reranking, and generation. ``node_id`` is kept alongside it so any record
    can be traced back to the exact graph node it was derived from.
    """

    record_id: str
    record_type: str
    text: str
    node_id: str
    properties: Mapping[str, Any]


class Corpus(Sequence[CorpusRecord]):
    """
    An indexed, immutable sequence of :class:`CorpusRecord`.

    Behaves as a plain sequence for the retrievers (which need positional
    access to align scores with documents) while also offering id lookups for
    the graph retriever and the evidence formatter.
    """

    __slots__ = ("_records", "_by_record_id", "_by_node_id")

    def __init__(self, records: Sequence[CorpusRecord]) -> None:
        self._records: tuple[CorpusRecord, ...] = tuple(records)
        self._by_record_id: Mapping[str, CorpusRecord] = MappingProxyType(
            {record.record_id: record for record in self._records}
        )
        self._by_node_id: Mapping[str, CorpusRecord] = MappingProxyType(
            {record.node_id: record for record in self._records}
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> CorpusRecord:  # type: ignore[override]
        return self._records[index]

    def __iter__(self) -> Iterator[CorpusRecord]:
        return iter(self._records)

    def by_record_id(self, record_id: str) -> CorpusRecord | None:
        """Look up a record by its citation handle."""
        return self._by_record_id.get(record_id)

    def by_node_id(self, node_id: str) -> CorpusRecord | None:
        """Look up a record by the graph node it was derived from."""
        return self._by_node_id.get(node_id)

    @property
    def record_ids(self) -> tuple[str, ...]:
        """All citation handles, in corpus order."""
        return tuple(record.record_id for record in self._records)


class LabelIdentityPolicy:
    """
    The identity property chosen for each label, decided once per graph.

    Deriving identity per *node* is not sufficient: AXIOM labels have optional
    properties, so two nodes sharing a label can present different property
    sets and end up with differently-shaped handles. On the full PoC export
    that produced ``CompetitorPresence`` records keyed on ``competitor_id``
    (124 of them) alongside others keyed on ``opportunity_id`` (42) -- the same
    label, two incompatible citation schemes.

    Deciding per label fixes that by construction. The winning key is chosen by
    majority vote across every node carrying the label, so a handful of sparse
    or irregular nodes cannot redefine the scheme for the whole type.
    """

    __slots__ = ("_by_label",)

    def __init__(self, by_label: Mapping[str, str]) -> None:
        self._by_label: Mapping[str, str] = MappingProxyType(dict(by_label))

    @classmethod
    def learn(
        cls, graph: KnowledgeGraph, config: CorpusConfig
    ) -> "LabelIdentityPolicy":
        """Derive the per-label identity key by majority vote over the graph."""
        votes: dict[str, Counter[str]] = {}
        probe = RecordSerializer(config)
        for node in graph:
            key = probe.node_identity_key(node)
            if key is not None:
                votes.setdefault(node.primary_label, Counter())[key] += 1

        chosen: dict[str, str] = {}
        for label, counter in votes.items():
            # Ties break alphabetically so the policy is reproducible.
            chosen[label] = min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if len(counter) > 1:
                logger.info(
                    "Label %s: identity key %r chosen over %s",
                    label,
                    chosen[label],
                    ", ".join(
                        f"{k} ({n})" for k, n in counter.most_common() if k != chosen[label]
                    ),
                )
        return cls(chosen)

    def key_for(self, label: str) -> str | None:
        """Return the identity property for ``label``, or ``None`` if unknown."""
        return self._by_label.get(label)

    def __len__(self) -> int:
        return len(self._by_label)

    def as_dict(self) -> Mapping[str, str]:
        """The full label -> property mapping, for inspection and reporting."""
        return self._by_label


class RecordSerializer:
    """
    Turns graph nodes into citable, indexable corpus records.

    Stateless apart from its configuration and (optionally) a
    :class:`LabelIdentityPolicy`, so a single instance is shared by the corpus
    builder, the anchor resolver, and the graph retriever -- guaranteeing that
    a record reached by traversal carries exactly the same ``record_id`` as the
    one reached lexically or semantically. Without that guarantee RRF could not
    recognise cross-modal agreement, which is the entire point of fusing.
    """

    __slots__ = ("_config", "_policy")

    def __init__(
        self, config: CorpusConfig, policy: LabelIdentityPolicy | None = None
    ) -> None:
        self._config = config
        self._policy = policy

    # -- public API ---------------------------------------------------------

    def identity_key(self, node: GraphNode) -> str | None:
        """
        Return the property name identifying ``node``, or ``None``.

        The label-wide policy wins when one is available and the node actually
        carries that property. Otherwise the per-node ladder applies, so a node
        missing its label's usual identifier still resolves to something stable
        rather than falling through to the raw node id.
        """
        if self._policy is not None:
            key = self._policy.key_for(node.primary_label)
            if key is not None and self._visible_properties(node).get(key) is not None:
                return key
        return self.node_identity_key(node)

    def node_identity_key(self, node: GraphNode) -> str | None:
        """
        Derive an identity property from this node alone, ignoring any policy.

        This is the ladder described in the module docstring, and the input
        :meth:`LabelIdentityPolicy.learn` votes over.
        """
        properties = self._visible_properties(node)
        if not properties:
            return None

        label_tokens = _tokenise(node.primary_label)
        suffix = self._config.id_key_suffix
        id_keys = [key for key in properties if key.endswith(suffix)]

        # 1. identifier-shaped keys, ranked by affinity with the node's label.
        best = _best_by_affinity(id_keys, label_tokens, suffix)
        if best is not None:
            return best
        # 2. any identifier-shaped key, first in source order.
        if id_keys:
            return id_keys[0]
        # 3. any key at all, ranked by affinity with the label
        #    (covers ProductLine.product_line, City.city_name, ...).
        best = _best_by_affinity(list(properties), label_tokens, suffix)
        if best is not None:
            return best
        # 4. a naming key.
        for key in properties:
            if any(key.endswith(s) for s in self._config.name_key_suffixes):
                return key
        # 5. no usable property -- caller falls back to the raw node id.
        return None

    def record_id(self, node: GraphNode) -> str:
        """
        Build the citation handle for ``node``: ``<Label>:<identifying value>``.

        Falls back to the raw node id when no property identifies the node, so
        the handle is always present and always resolvable.
        """
        key = self.identity_key(node)
        if key is None:
            return f"{node.primary_label}{ID_SEPARATOR}{node.node_id}"
        value = self._format_value(node.properties.get(key))
        if not value:
            return f"{node.primary_label}{ID_SEPARATOR}{node.node_id}"
        return f"{node.primary_label}{ID_SEPARATOR}{value}"

    def to_text(self, node: GraphNode) -> str:
        """
        Render ``node`` as the flat text that BM25 and the encoder both index.

        The identifying property leads, so an exact-identifier query scores
        against the front of the document, and so a truncated document never
        loses the field that makes it citable.
        """
        properties = self._visible_properties(node)
        identity = self.identity_key(node)
        ordered_keys = (
            [identity, *(k for k in properties if k != identity)]
            if identity in properties
            else list(properties)
        )

        parts: list[str] = [node.primary_label]
        for key in ordered_keys:
            value = self._format_value(properties.get(key))
            if value:
                parts.append(f"{key} {value}")
        return _truncate(" ".join(parts), self._config.max_text_chars)

    def to_record(self, node: GraphNode) -> CorpusRecord:
        """Build a full :class:`CorpusRecord` from ``node``."""
        return CorpusRecord(
            record_id=self.record_id(node),
            record_type=node.primary_label,
            text=self.to_text(node),
            node_id=node.node_id,
            properties=MappingProxyType(dict(node.properties)),
        )

    def accepts(self, node: GraphNode) -> bool:
        """Apply the configured label allow/deny lists to ``node``."""
        label = node.primary_label
        if self._config.include_labels and label not in self._config.include_labels:
            return False
        return label not in self._config.exclude_labels

    # -- internals ----------------------------------------------------------

    def _visible_properties(self, node: GraphNode) -> Mapping[str, Any]:
        """Node properties minus the configured exclusions and null values."""
        excluded = self._config.excluded_property_keys
        return {
            key: value
            for key, value in node.properties.items()
            if key not in excluded and value is not None and value != ""
        }

    def _format_value(self, value: Any) -> str:
        """
        Render a property value as stable, searchable text.

        Floats are rounded to a configured precision and stripped of trailing
        zeros so that ``162.5`` indexes as ``162.5`` rather than
        ``162.50000000000003`` -- which would otherwise make an exact-price
        lexical match impossible.
        """
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            rendered = f"{value:.{self._config.float_precision}f}".rstrip("0").rstrip(".")
            return rendered or "0"
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, (list, tuple, set, frozenset)):
            return " ".join(self._format_value(item) for item in value if item is not None)
        if isinstance(value, Mapping):
            return " ".join(
                f"{key} {self._format_value(item)}" for key, item in value.items()
            )
        return str(value).strip()


def build_corpus(graph: KnowledgeGraph, serializer: RecordSerializer) -> Corpus:
    """
    Project every accepted node in ``graph`` into a :class:`Corpus`.

    Record ids that collide -- common for AXIOM's period-style labels, where
    several validity windows share one business identifier -- are disambiguated
    by appending the originating node id. The suffix is applied only to the
    records that actually collide, so the common case keeps a clean handle, and
    it is derived from the data, so it is stable across runs.
    """
    records: list[CorpusRecord] = []
    seen: dict[str, int] = {}
    collisions: list[str] = []

    for node in graph:
        if not serializer.accepts(node):
            continue
        record = serializer.to_record(node)
        count = seen.get(record.record_id, 0)
        seen[record.record_id] = count + 1
        if count:
            collisions.append(record.record_id)
            record = CorpusRecord(
                record_id=f"{record.record_id}{DISAMBIGUATOR}{record.node_id}",
                record_type=record.record_type,
                text=record.text,
                node_id=record.node_id,
                properties=record.properties,
            )
        records.append(record)

    if collisions:
        logger.info(
            "Disambiguated %d colliding record id(s), e.g. %s",
            len(collisions),
            ", ".join(sorted(set(collisions))[:3]),
        )
    logger.info("Built retrieval corpus: %d records", len(records))
    return Corpus(records)


def _tokenise(text: str) -> frozenset[str]:
    """Split a label or property name into lowercase comparison tokens."""
    return frozenset(token.lower() for token in _TOKEN_SPLIT.split(text) if token)


def _best_by_affinity(
    keys: Sequence[str], label_tokens: frozenset[str], suffix: str
) -> str | None:
    """
    Pick the key whose name best matches the node's own label.

    Ranking, in order:

    1. **Most shared tokens.** ``order_line_id`` (2 shared with ``OrderLine``)
       beats ``order_id`` (1).
    2. **Fewest extraneous tokens.** A key carrying tokens *absent* from the
       label is very likely a foreign key rather than the node's own identity.
       This is what keeps an ``Opportunity`` that carries both
       ``opportunity_id`` and ``revises_opportunity_id`` from being named after
       the opportunity it revises -- a citation pointing at the wrong record.
    3. **Source order**, so the choice is deterministic for any given graph.
    """
    best_key: str | None = None
    best_rank: tuple[int, int] | None = None
    for key in keys:
        stem = key[: -len(suffix)] if suffix and key.endswith(suffix) else key
        tokens = _tokenise(stem)
        overlap = len(tokens & label_tokens)
        if overlap == 0:
            continue
        extraneous = len(tokens - label_tokens)
        # Higher overlap wins; among equals, fewer extraneous tokens wins.
        rank = (overlap, -extraneous)
        if best_rank is None or rank > best_rank:
            best_rank, best_key = rank, key
    return best_key


def _truncate(text: str, max_chars: int) -> str:
    """Trim ``text`` to ``max_chars`` on a word boundary where possible."""
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    boundary = clipped.rfind(" ")
    return clipped[:boundary] if boundary > 0 else clipped
