"""
Composition root for the reliability layer.

Two wiring decisions live here rather than in the narrator, because both are
easy to get wrong silently:

**The generator samples above zero.** D4's generator defaults to temperature 0
for reproducibility. Handed unchanged to a k-sample loop it returns k
identical texts, and identical samples agree perfectly while proving nothing
-- self-consistency would report 1.00 agreement on a model that had been
asked the same question once. The generator built here therefore carries
``sample_temperature`` as its base.

**The retriever is adapted, not imported.** Corrective retrieval needs only
"give me record handles for this query", so the hybrid stack is narrowed to
that one callable. The reliability layer never sees the pipeline, and a test
can pass a stub instead of building an index.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

from src.config.settings import Settings, get_settings
from src.graph.model import KnowledgeGraph
from src.narrative.evidence import EvidenceIndex
from src.narrative.factory import build_serializer
from src.narrative.generator import NarrativeGenerator
from src.reliability.corrective import SupportingRecordFinder
from src.reliability.narrator import ReliableNarrator
from src.text_to_query.llm_client import LLMClient, OllamaClient

__all__ = ["build_reliable_narrator", "retrieval_finder"]

logger = logging.getLogger(__name__)


def retrieval_finder(pipeline) -> SupportingRecordFinder:
    """
    Adapt a hybrid retrieval pipeline to the narrow finder protocol.

    Returns record handles only. Scores and stream provenance are dropped:
    corrective retrieval decides what to cite, not how confident retrieval
    was, and passing the richer object through would couple the two layers
    for no gain.
    """

    def find(query: str, top_k: int) -> Sequence[str]:
        outcome = pipeline.run(query, top_k=top_k)
        return [result.record_id for result in outcome.final]

    return find


def build_reliable_narrator(
    graph: KnowledgeGraph,
    settings: Settings | None = None,
    *,
    llm_client: LLMClient | None = None,
    finder: SupportingRecordFinder | None = None,
    evidence: EvidenceIndex | None = None,
) -> ReliableNarrator:
    """
    Build a ready-to-use :class:`ReliableNarrator`.

    Args:
        graph: the knowledge graph, for resolving citations.
        settings: application configuration; defaults to :func:`get_settings`.
        llm_client: overrides the production Ollama client.
        finder: supplies corrective retrieval. Without one, a weak-evidence
            pack is reported as weak and generated from anyway -- the
            assessment still runs, only the repair is skipped.
        evidence: overrides the index built from ``graph``.
    """
    settings = settings or get_settings()
    index = evidence or EvidenceIndex.build(graph, build_serializer(graph, settings))

    generator = NarrativeGenerator(
        llm_client or OllamaClient(settings.llm),
        settings.narrative,
        base_temperature=settings.reliability.sample_temperature,
    )
    logger.info(
        "Reliable narrator ready: k=%d, quorum %d, agreement floor %.2f, "
        "temperature %.2f, corrective %s",
        settings.reliability.samples,
        settings.reliability.min_verified,
        settings.reliability.min_agreement,
        settings.reliability.sample_temperature,
        "on" if settings.reliability.corrective_enabled and finder else "off",
    )
    return ReliableNarrator(
        generator,
        settings.reliability,
        settings.narrative,
        evidence=index,
        finder=finder,
    )
