"""
Resolving the records Track A cites, against the graph.

This is the one place Track B touches the graph during narrative generation,
and it is retrieval, not calculation. Section 7 of the onboarding plan
describes the job exactly:

    "You consume that object: you retrieve the referenced records and the
    surrounding evidence from the graph, and generate the cited narrative
    that explains it."

So a payload arrives carrying handles such as ``OrderLine:OL000248``, and
this module answers two questions about them:

**Does the record exist?** A handle that resolves to nothing is a citation
that would send a reader to a record that is not there. That is worse than an
uncited claim, because it looks verifiable. The generator treats unresolvable
evidence as insufficient evidence and abstains.

**What does the record say?** The resolved text is what lets a reviewer check
a narrative's claim against the underlying record without opening the
database. It is carried alongside the narrative, not fed to the model as
extra figures -- every number the model may state comes from the fact pack,
and widening its input would widen what it can quote.

Handles are produced by the same :class:`RecordSerializer` the retrieval and
text-to-query layers use, so a citation in a narrative is the same string a
retrieval hit or a text-to-query answer would carry for that record. One
citation scheme across all three components is what makes a reader able to
follow a sentence back to one exact record.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from src.graph.model import KnowledgeGraph
from src.retrieval.corpus import RecordSerializer

__all__ = ["EvidenceIndex", "ResolvedRecord"]


@dataclass(frozen=True, slots=True)
class ResolvedRecord:
    """One cited record, found in the graph."""

    record_id: str
    record_type: str
    text: str

    def preview(self, width: int = 88) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= width else flat[: width - 3] + "..."


class EvidenceIndex:
    """
    Looks up citation handles in the graph.

    Built once per graph and reused: resolving handles one at a time by
    scanning would make every narrative quadratic in the size of the graph.
    """

    __slots__ = ("_by_handle",)

    def __init__(self, by_handle: Mapping[str, ResolvedRecord]) -> None:
        self._by_handle: Mapping[str, ResolvedRecord] = MappingProxyType(dict(by_handle))

    @classmethod
    def build(cls, graph: KnowledgeGraph, serializer: RecordSerializer) -> "EvidenceIndex":
        """Index every node in the graph by its citation handle."""
        records: dict[str, ResolvedRecord] = {}
        for node in graph:
            if not serializer.accepts(node):
                continue
            handle = serializer.record_id(node)
            # First wins: handles are label-scoped and collisions would mean
            # two records share an identity, which the identity policy
            # already resolves upstream.
            records.setdefault(
                handle,
                ResolvedRecord(
                    record_id=handle,
                    record_type=node.primary_label,
                    text=serializer.to_text(node),
                ),
            )
        return cls(records)

    def __len__(self) -> int:
        return len(self._by_handle)

    def __contains__(self, handle: object) -> bool:
        return handle in self._by_handle

    def resolve(self, handle: str) -> ResolvedRecord | None:
        """The record behind one handle, or ``None`` if it is not in the graph."""
        return self._by_handle.get(handle)

    def resolve_all(self, handles: Iterable[str]) -> tuple[ResolvedRecord, ...]:
        """Every handle that resolves, in the order given."""
        found = (self.resolve(handle) for handle in handles)
        return tuple(record for record in found if record is not None)

    def missing(self, handles: Iterable[str]) -> tuple[str, ...]:
        """
        Handles that resolve to nothing.

        A non-empty result means Track A cited a record this graph does not
        contain -- a contract mismatch worth surfacing loudly, because the
        narrative built on it would carry citations a reader cannot follow.
        """
        return tuple(handle for handle in handles if handle not in self._by_handle)
