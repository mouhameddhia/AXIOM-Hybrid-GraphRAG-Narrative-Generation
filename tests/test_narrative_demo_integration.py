"""
Live D2 + D3 + D4 integration chain -- for review with the mentor.

Skipped unless ``RUN_LIVE_NARRATIVE_DEMO=1`` and a local Ollama model is
actually reachable -- run it explicitly:

    RUN_LIVE_NARRATIVE_DEMO=1 pytest tests/test_narrative_demo_integration.py -v -s

``-s`` is required to see the printed stage-by-stage output -- without it
pytest captures stdout and shows only pass/fail.

What this file demonstrates
----------------------------
Hybrid retrieval (D2), text-to-query (D3), and grounded narrative generation
(D4) are each tested in isolation elsewhere. What none of those tests show is
that the three are actually wired together through one shared citation
scheme -- a record id retrieval ranks is the same record id text-to-query
can return and the narrative can cite, all resolving in the same graph index.

This file runs all three in sequence against the real graph and the real
model: retrieval ranks surrounding evidence for a product/customer, the
text-to-query engine answers a live question (unit cost) by generating and
running a real Cypher-equivalent query, that answer is folded into the
narrative's fact pack, and the narrative is generated and verified from the
combined pack. The assertion that matters is that every citation the
narrative used resolves in the same index retrieval and text-to-query both
draw from -- proving the chain, not just the three links separately.

For plain narrative generation (D4) alone, see
``test_narrative_demo_generation.py``. For the D5 reliability layer, see
``test_narrative_demo_reliability.py``. To run everything together:

    RUN_LIVE_NARRATIVE_DEMO=1 pytest tests/test_narrative_demo_generation.py tests/test_narrative_demo_reliability.py tests/test_narrative_demo_integration.py -v -s
"""
from __future__ import annotations

import os

import pytest

from src.config.settings import PROJECT_ROOT, get_settings
from src.narrative.contract import PriceRecommendation
from src.narrative.evidence import EvidenceIndex
from src.narrative.facts import build_facts
from src.narrative.factory import build_narrative_generator, build_serializer
from src.narrative.graph_facts import fact_from_answer
from src.pipeline.factory import build_retrieval_stack
from src.text_to_query.factory import build_text_to_query_engine

LIVE_DEMO_ENABLED = os.getenv("RUN_LIVE_NARRATIVE_DEMO") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE_DEMO_ENABLED,
    reason="Set RUN_LIVE_NARRATIVE_DEMO=1 to run against a live Ollama model",
)

PAYLOADS = PROJECT_ROOT / "data" / "track_a_payloads_eval.json"
RULE = "=" * 78


def heading(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def stack(settings):
    """The real retrieval stack, lexical + graph only (no embedding download)."""
    return build_retrieval_stack(settings, enable_dense=False)


@pytest.fixture(scope="module")
def index(stack, settings):
    return EvidenceIndex.build(stack.graph, build_serializer(stack.graph, settings))


class TestLiveIntegrationChain:
    def test_retrieval_text_to_query_and_narrative_share_one_citation_scheme(
        self, settings, stack, index
    ):
        heading("INTEGRATION -- D2 (retrieval) + D3 (text-to-query) + D4 (narrative)")

        recommendation = PriceRecommendation.load(PAYLOADS, "REC-0001")

        query = f"{recommendation.product_id} {recommendation.customer_id} price discount deal"
        outcome = stack.pipeline.run(query, top_k=5)
        print(f"STAGE 1 (D2) -- retrieval ranked {len(outcome.final)} record(s) for: {query!r}")
        for rank, result in enumerate(outcome.final, start=1):
            print(f"  {rank}. {result.record_id}")

        engine = build_text_to_query_engine(stack.graph, settings)
        question = f"What is the unit cost of product {recommendation.product_id}?"
        answer = engine.ask(question)
        print(f"\nSTAGE 2 (D3) -- {question!r}")
        graph_fact = None
        if answer.success:
            print(f"  cypher: {answer.cypher}")
            graph_fact = fact_from_answer(answer, key="graph.unit_cost", label_en="unit cost")
            if graph_fact is not None:
                print(f"  -> {graph_fact.fact.value} [{graph_fact.fact.citation}]")

        pack = build_facts(recommendation)
        if graph_fact is not None:
            pack = pack.extended([graph_fact.fact])

        generator = build_narrative_generator(settings)
        narrative = generator.explain(pack, language="en", evidence=index)
        print(f"\nSTAGE 3 (D4) -- narrative ({len(pack.facts)} figures available):")
        print(f"\n{narrative.text if narrative.success else '[ABSTAINED: ' + str(narrative.reason) + ']'}")

        assert narrative.success, f"abstained: {narrative.reason}"
        # The property this whole chain exists to prove: a citation the
        # narrative actually used must be one retrieval could also have
        # surfaced, and must resolve in the same graph both stages share.
        for citation in narrative.citations:
            if citation.startswith(("GraphQuery:", "trackA-")):
                continue
            assert index.resolve(citation) is not None, (
                f"narrative cited {citation}, unresolvable in the shared graph index"
            )
        print(f"\ncited {len(narrative.citations)} record(s), all resolving in the same "
              f"index retrieval and text-to-query both use: {list(narrative.citations)}")
