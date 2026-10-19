"""
Live schema derivation and pruning for text-to-Cypher.

The schema shown to the LLM is derived from the actual :class:`KnowledgeGraph`
at startup -- every label, every relationship, every property that genuinely
exists. Nothing here names an AXIOM label, so this module is correct for
whatever graph is loaded, and stays correct if the graph changes: there is
no separate schema document that can drift out of sync with the data.

Pruning exists because showing an LLM the entire schema on every call wastes
prompt budget and gives it more surface area to reference something
irrelevant. It selects only the labels whose name or properties share
vocabulary with the question -- a cheap, zero-extra-call heuristic with a
clear upgrade path to embedding similarity if the schema ever grows large
enough to need it.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from src.config.settings import TextToCypherConfig
from src.graph.model import KnowledgeGraph

__all__ = ["GraphSchema", "build_schema_from_graph", "prune_schema"]

#: Splits CamelCase / snake_case into comparable lowercase tokens. Shared with
#: src/retrieval/corpus.py's convention so a question's vocabulary matches the
#: same way in both places.
_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-z]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

#: A property with at most this many distinct values is treated as categorical
#: and its values are shown in the prompt. Above it, the property is an
#: identifier or a measure, and listing values would be noise.
MAX_SAMPLE_VALUES = 8


@dataclass(frozen=True, slots=True)
class GraphSchema:
    """
    A description of the graph's structure, derived from real data.

    ``relationship_types`` maps a relationship type to every ``(source_label,
    target_label)`` pair actually observed for it -- a set, not a single pair,
    because nothing guarantees a relationship type connects only one pair of
    labels in a real graph.
    """

    node_labels: Mapping[str, tuple[str, ...]]
    relationship_types: Mapping[str, frozenset[tuple[str, str]]]
    #: label -> property -> the distinct values observed, for properties with
    #: few enough of them to be worth showing. See :func:`build_schema_from_graph`.
    value_samples: Mapping[str, Mapping[str, tuple[str, ...]]] = field(
        default_factory=dict
    )

    @property
    def label_names(self) -> frozenset[str]:
        return frozenset(self.node_labels)

    @property
    def relationship_names(self) -> frozenset[str]:
        return frozenset(self.relationship_types)

    def render(self) -> str:
        """
        Render as compact text for the prompt, with observed values.

        Showing the values a categorical property actually takes is what
        stops a plan filtering ``is_tender = true`` when this export stores
        the string ``'true'``, or guessing ``result = 'lost'`` when the data
        says ``'lost_to_competitor'``. Knowing that a property *exists* is
        not enough to filter on it correctly.
        """
        lines: list[str] = []
        for label, properties in self.node_labels.items():
            samples = self.value_samples.get(label, {})
            rendered = [
                f"{prop}={'|'.join(samples[prop])}" if prop in samples else prop
                for prop in properties
            ]
            lines.append(f"  ({label} {{{', '.join(rendered)}}})")
        for rel_type, pairs in self.relationship_types.items():
            for source, target in sorted(pairs):
                lines.append(f"  ({source})-[:{rel_type}]->({target})")
        return "\n".join(lines)


def build_schema_from_graph(graph: KnowledgeGraph) -> GraphSchema:
    """
    Derive the full schema by inspecting every node and edge once.

    Property lists are the union of keys seen across all nodes of a label, so
    a property that only some records carry still appears -- the LLM should
    know it exists, even if a given row lacks it.
    """
    properties_by_label: dict[str, set[str]] = {}
    # label -> property -> every distinct value seen, capped while collecting
    # so a high-cardinality property (an id, a price) cannot blow up memory
    # on a large graph.
    values_by_label: dict[str, dict[str, set[str]]] = {}

    for node in graph:
        label = node.primary_label
        properties_by_label.setdefault(label, set()).update(node.properties.keys())
        seen = values_by_label.setdefault(label, {})
        for key, value in node.properties.items():
            if value is None:
                continue
            observed = seen.setdefault(key, set())
            if len(observed) <= MAX_SAMPLE_VALUES:
                observed.add(str(value))

    node_labels = {
        label: tuple(sorted(properties))
        for label, properties in properties_by_label.items()
    }
    # Keep only genuinely categorical properties. A property with one value
    # per record is an identifier, and listing those would be noise that
    # crowds out the values a filter can usefully match on.
    value_samples = {
        label: MappingProxyType(
            {
                key: tuple(sorted(values))
                for key, values in sorted(properties.items())
                if len(values) <= MAX_SAMPLE_VALUES
            }
        )
        for label, properties in values_by_label.items()
    }

    pairs_by_type: dict[str, set[tuple[str, str]]] = {}
    for edge in graph.edges:
        source = graph.node(edge.source_id)
        target = graph.node(edge.target_id)
        if source is None or target is None:  # pragma: no cover - graph invariant
            continue
        pairs_by_type.setdefault(edge.rel_type, set()).add(
            (source.primary_label, target.primary_label)
        )
    relationship_types = {
        rel_type: frozenset(pairs) for rel_type, pairs in pairs_by_type.items()
    }

    return GraphSchema(
        node_labels=MappingProxyType(node_labels),
        relationship_types=MappingProxyType(relationship_types),
        value_samples=MappingProxyType(value_samples),
    )


def prune_schema(
    question: str, full_schema: GraphSchema, config: TextToCypherConfig
) -> GraphSchema:
    """
    Reduce ``full_schema`` to the labels and relationships relevant to
    ``question``.

    A label is kept when its name or any of its property names shares at
    least ``config.min_schema_overlap`` token(s) with the question. A
    relationship is kept when its own name matches the question, or when
    either endpoint label was kept -- so a join is not silently dropped just
    because the question named only one side of it. Kept labels are ranked by
    overlap strength and capped at ``config.max_schema_labels``.

    Falls back to the full schema when nothing matches at all: an LLM given an
    empty schema cannot possibly write a correct query, so over-including is
    the safer failure mode than under-including.
    """
    question_tokens = _stem_all(_tokenise(question))

    scored: list[tuple[int, str]] = []
    for label, properties in full_schema.node_labels.items():
        label_tokens = _stem_all(
            _tokenise(label) | {t for prop in properties for t in _tokenise(prop)}
        )
        overlap = len(question_tokens & label_tokens)
        if overlap >= config.min_schema_overlap:
            scored.append((overlap, label))

    if not scored:
        return full_schema

    # Highest overlap first; ties broken alphabetically for a reproducible
    # prompt across identical questions.
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    kept_labels: dict[str, tuple[str, ...]] = {
        label: full_schema.node_labels[label]
        for _, label in scored[: config.max_schema_labels]
    }

    kept_relationships: dict[str, frozenset[tuple[str, str]]] = {}
    for rel_type, pairs in full_schema.relationship_types.items():
        name_matches = bool(question_tokens & _stem_all(_tokenise(rel_type)))
        relevant_pairs = frozenset(
            (source, target)
            for source, target in pairs
            if name_matches or source in kept_labels or target in kept_labels
        )
        if not relevant_pairs:
            continue
        kept_relationships[rel_type] = relevant_pairs
        # Pull in both endpoint labels, even if they didn't match on their
        # own -- a relationship the LLM can see but cannot anchor on either
        # side is useless.
        for source, target in relevant_pairs:
            kept_labels.setdefault(source, full_schema.node_labels.get(source, ()))
            kept_labels.setdefault(target, full_schema.node_labels.get(target, ()))

    return GraphSchema(
        node_labels=MappingProxyType(kept_labels),
        relationship_types=MappingProxyType(kept_relationships),
        # Carry the samples through: a pruned schema without them would lose
        # exactly the detail that makes a filter value correct.
        value_samples=MappingProxyType(
            {
                label: full_schema.value_samples[label]
                for label in kept_labels
                if label in full_schema.value_samples
            }
        ),
    )


def _tokenise(text: str) -> frozenset[str]:
    """Split a label, property, or question into lowercase comparison tokens."""
    return frozenset(token.lower() for token in _TOKEN_SPLIT.split(text) if token)


def _stem_all(tokens: frozenset[str]) -> frozenset[str]:
    """
    Fold each token to a crude stem, so ``opportunities`` matches
    ``opportunity``.

    Without this, ``DealOutcome`` -- the label that actually answers "how many
    opportunities did we lose" -- was excluded from pruning entirely, because
    the question says "opportunities" and the schema says "opportunity" and
    plain string overlap never sees the two as related. This is deliberately
    a crude suffix rule, not a real stemmer: schema pruning is a cheap
    relevance filter that falls back to the full schema when unsure, not a
    precision-critical component, so a rule this simple is proportionate.
    """
    stemmed = set(tokens)
    for token in tokens:
        if len(token) > 4 and token.endswith("ies"):
            stemmed.add(token[:-3] + "y")
        elif len(token) > 3 and token.endswith("es"):
            stemmed.add(token[:-2])
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            stemmed.add(token[:-1])
    return frozenset(stemmed)
