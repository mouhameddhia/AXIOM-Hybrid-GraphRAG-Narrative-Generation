"""
Composition root for the narrative layer.

Mirrors :mod:`src.text_to_query.factory`: collaborators are injected, nothing
is constructed implicitly, and this is the one place that wires the concrete
production pieces together. The serializer is built with a label-identity
policy learned from the graph, exactly as the retrieval and text-to-query
factories do -- that shared construction is what guarantees a citation in a
narrative is the same handle a retrieval hit would carry.
"""
from __future__ import annotations

import logging

from src.config.settings import Settings, get_settings
from src.graph.model import KnowledgeGraph
from src.narrative.generator import NarrativeGenerator
from src.retrieval.corpus import LabelIdentityPolicy, RecordSerializer
from src.text_to_query.llm_client import LLMClient, OllamaClient

__all__ = ["build_narrative_generator", "build_serializer"]

logger = logging.getLogger(__name__)


def build_narrative_generator(
    settings: Settings | None = None,
    *,
    llm_client: LLMClient | None = None,
) -> NarrativeGenerator:
    """
    Build a ready-to-use :class:`NarrativeGenerator`.

    Args:
        settings: application configuration; defaults to :func:`get_settings`.
        llm_client: overrides the production Ollama client. Injecting a fake
            keeps a network call and a running model out of the test path.
    """
    settings = settings or get_settings()
    generator = NarrativeGenerator(
        llm_client or OllamaClient(settings.llm),
        settings.narrative,
        base_temperature=settings.llm.temperature,
    )
    logger.info(
        "Narrative generator ready: up to %d corrective attempt(s), %d-word limit",
        settings.narrative.max_retries,
        settings.narrative.max_words,
    )
    return generator


def build_serializer(graph: KnowledgeGraph, settings: Settings | None = None) -> RecordSerializer:
    """The shared record serializer, so citations match the other components."""
    settings = settings or get_settings()
    return RecordSerializer(
        settings.corpus, LabelIdentityPolicy.learn(graph, settings.corpus)
    )
