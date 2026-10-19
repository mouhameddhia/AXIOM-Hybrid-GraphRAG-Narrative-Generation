"""
Tests for the three retrieval modalities.

Each modality is tested for the property that justifies its presence in the
hybrid: BM25 for exact identifier matching, dense for vocabulary-independent
matching, graph for structural reachability with meaningful rank order.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.retrieval.anchors import AnchorResolver
from src.retrieval.base import RetrievalQuery
from src.retrieval.bm25_retriever import BM25Retriever, Tokenizer
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.graph_retriever import GraphRetriever


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def bm25(corpus, settings):
    return BM25Retriever(corpus, settings.lexical)


@pytest.fixture
def dense(corpus, settings, encoder):
    return DenseRetriever(corpus, settings.embedding, encoder)


@pytest.fixture
def resolver(graph, corpus, serializer, settings):
    return AnchorResolver(graph, corpus, serializer, settings.graph, settings.corpus)


@pytest.fixture
def graph_retriever(graph, corpus, settings):
    return GraphRetriever(graph, corpus, settings.graph)


def query(text: str, resolver: AnchorResolver | None = None) -> RetrievalQuery:
    anchors = resolver.resolve(text) if resolver else ()
    return RetrievalQuery(text=text, anchors=anchors)


# -- tokenisation -----------------------------------------------------------


def test_tokenizer_keeps_identifiers_intact(settings):
    """``SKU0001,`` must tokenise to ``sku0001``, not ``sku0001,``."""
    tokens = Tokenizer(settings.lexical)("Price for SKU0001, quoted at 162.5 SAR.")
    assert "sku0001" in tokens
    assert "162" in tokens and "5" in tokens


def test_tokenizer_segments_arabic(settings):
    """The bilingual requirement needs Arabic to segment, not vanish."""
    tokens = Tokenizer(settings.lexical)("ما هو سعر المنتج SKU0001")
    assert "sku0001" in tokens
    assert any(not token.isascii() for token in tokens)


def test_tokenizer_strips_arabic_diacritics(settings):
    """Decorated and undecorated Arabic must produce identical tokens."""
    tokenize = Tokenizer(settings.lexical)
    assert tokenize("سِعْر") == tokenize("سعر")


# -- BM25 -------------------------------------------------------------------

def test_bm25_finds_an_exact_identifier(bm25, corpus):
    """The modality's core job: literal identifiers must hit."""
    results = bm25.search(query("SKU0001"), top_k=10)

    assert results
    assert any("SKU0001" in result.record_id for result in results)


def test_bm25_drops_zero_scoring_documents(bm25):
    """
    An irrelevant document must not occupy a rank slot.

    Passing it on would let noise accrue RRF weight purely by being listed.
    """
    results = bm25.search(query("zzzz_nonexistent_term_qqqq"), top_k=10)
    assert results == []


def test_bm25_respects_top_k(bm25):
    assert len(bm25.search(query("product price"), top_k=3)) <= 3


def test_bm25_is_deterministic(bm25):
    runs = {
        tuple(r.record_id for r in bm25.search(query("price SAR product"), top_k=10))
        for _ in range(3)
    }
    assert len(runs) == 1


def test_bm25_handles_an_empty_query(bm25):
    assert bm25.search(query("   "), top_k=5) == []


# -- dense ------------------------------------------------------------------


def test_dense_returns_ranked_results(dense):
    results = dense.search(query("product price"), top_k=5)
    assert results
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_dense_embeds_the_corpus_once(dense, encoder):
    """The index must be cached; re-embedding per query would be a latency bug."""
    dense.search(query("first"), top_k=3)
    after_first = encoder.call_count
    dense.search(query("second"), top_k=3)

    # One corpus pass plus one query encode each: never a second corpus pass.
    assert after_first == 2
    assert encoder.call_count == 3


def test_dense_warmup_builds_the_index(corpus, settings, encoder):
    retriever = DenseRetriever(corpus, settings.embedding, encoder)
    retriever.warmup()
    assert encoder.call_count == 1
    assert len(encoder.encoded_texts) == len(corpus)


def test_dense_rejects_a_malformed_encoder(corpus, settings):
    """A wrong-shaped encoder must fail loudly, not rank garbage."""

    class BadEncoder:
        def encode_batch(self, texts):
            import numpy as np

            return np.zeros((len(texts) + 1, 8), dtype="float32")

    retriever = DenseRetriever(corpus, settings.embedding, BadEncoder())
    with pytest.raises(ValueError, match="expected"):
        retriever.search(query("anything"), top_k=3)


# -- anchors ----------------------------------------------------------------


def test_anchor_resolution_grounds_a_real_identifier(resolver):
    anchors = resolver.resolve("What is the list price for SKU0001?")

    assert anchors
    assert all(anchor.mention.upper() == "SKU0001" for anchor in anchors)
    assert all(anchor.matched_property.endswith("_id") for anchor in anchors)


def test_unknown_identifier_resolves_to_nothing(resolver):
    """A hallucinated id must ground nothing rather than anchor at random."""
    assert resolver.resolve("Tell me about SKU9999") == ()


def test_anchor_resolution_is_case_insensitive(resolver):
    """
    Casing must not change what a mention grounds to.

    ``mention`` itself is deliberately preserved as the user wrote it, so an
    explanation can quote the question verbatim; the comparison is therefore on
    the resolved nodes.
    """
    lowered = resolver.resolve("sku0001")
    upper = resolver.resolve("SKU0001")

    assert lowered and upper
    assert [a.node_id for a in lowered] == [a.node_id for a in upper]
    assert [a.record_id for a in lowered] == [a.record_id for a in upper]
    assert (lowered[0].mention, upper[0].mention) == ("sku0001", "SKU0001")


def test_anchor_count_is_capped(graph, corpus, serializer, settings):
    capped = replace(settings.graph, max_anchors=1)
    resolver = AnchorResolver(graph, corpus, serializer, capped, settings.corpus)

    assert len(resolver.resolve("SKU0001 and OPP00003 and CUST0015")) == 1


# -- graph ------------------------------------------------------------------


def test_graph_retriever_needs_an_anchor(graph_retriever):
    """Without a grounded entity the modality sits out; it does not guess."""
    assert graph_retriever.search(RetrievalQuery(text="general pricing question"), top_k=5) == []


def test_graph_results_are_ordered_by_proximity(graph_retriever, resolver):
    """
    The defect this replaces: every hit previously scored 1.0, so RRF consumed
    an arbitrary order. Hop distance must now be non-decreasing down the list.
    """
    results = graph_retriever.search(query("evidence for CUST0015", resolver), top_k=10)

    assert results
    hops = [result.metadata["hops"] for result in results]
    assert hops == sorted(hops)
    assert results[0].score >= results[-1].score


def test_graph_scores_decay_with_distance(graph_retriever, resolver):
    results = graph_retriever.search(query("evidence for CUST0015", resolver), top_k=20)

    by_hop = {}
    for result in results:
        by_hop.setdefault(result.metadata["hops"], []).append(result.score)
    if len(by_hop) > 1:
        near, far = sorted(by_hop)[:2]
        assert max(by_hop[far]) <= max(by_hop[near])


def test_graph_carries_traversal_provenance(graph_retriever, resolver):
    """Each hit must say how it was reached, for citation."""
    results = graph_retriever.search(query("evidence for CUST0015", resolver), top_k=5)

    for result in results:
        assert len(result.metadata["traversal_path"]) == result.metadata["hops"]
        assert result.metadata["anchor_record_ids"]


def test_graph_record_ids_match_the_corpus(graph_retriever, resolver, corpus):
    """
    Traversal hits must carry the same handle the lexical index uses.

    Without this, RRF cannot detect cross-modal agreement at all.
    """
    results = graph_retriever.search(query("evidence for OPP00003", resolver), top_k=10)

    assert results
    for result in results:
        assert corpus.by_record_id(result.record_id) is not None


def test_graph_is_deterministic(graph_retriever, resolver):
    runs = {
        tuple(
            r.record_id
            for r in graph_retriever.search(query("evidence for OPP00003", resolver), top_k=10)
        )
        for _ in range(3)
    }
    assert len(runs) == 1


# -- stopword filtering (regression found by the question bank) --------------


def test_stopwords_are_removed(settings):
    """
    Function words must not reach the index.

    This corpus is key-value records, so English stopwords are *rare* in it
    ("the" appears in 80 of 5,794 records, only in prose descriptions). BM25
    therefore gives them a high IDF and scores them as discriminative, which
    let a long natural-language question rank 40 SellableProduct records on
    the strength of "the" x4 while missing the record it named.
    """
    tokens = Tokenizer(settings.lexical)(
        "Give me the full picture of opportunity OPP00252 including the outcome."
    )

    assert "the" not in tokens
    assert "of" not in tokens
    assert "give" not in tokens
    # The discriminative terms must survive.
    assert "opp00252" in tokens
    assert "opportunity" in tokens
    assert "outcome" in tokens


def test_pricing_vocabulary_is_never_treated_as_a_stopword(settings):
    """Domain terms must survive: removing them would destroy recall."""
    tokens = Tokenizer(settings.lexical)(
        "price cost discount margin rebate freight quote order product customer"
    )

    for term in (
        "price", "cost", "discount", "margin", "rebate",
        "freight", "quote", "order", "product", "customer",
    ):
        assert term in tokens, f"{term} must not be filtered"


def test_arabic_stopwords_are_removed(settings):
    """The bilingual requirement means Arabic function words too."""
    tokens = Tokenizer(settings.lexical)("ما هو سعر القائمة للمنتج SKU0001؟")

    assert "ما" not in tokens
    assert "هو" not in tokens
    assert "sku0001" in tokens
    assert "سعر" in tokens


def test_stopword_filtering_can_be_disabled(settings):
    """Behaviour stays configuration-driven, including switching it off."""
    from dataclasses import replace

    disabled = replace(settings.lexical, stopwords=frozenset())

    assert "the" in Tokenizer(disabled)("the price")


def test_long_natural_language_query_finds_the_named_entity(corpus, settings):
    """
    End-to-end guard for the defect above.

    A verbose question must still rank the record it explicitly names, rather
    than records that merely share function words.
    """
    retriever = BM25Retriever(corpus, settings.lexical)

    results = retriever.search(
        query(
            "Give me the full picture of the product SKU0001 including "
            "all of the details that we have on it."
        ),
        top_k=10,
    )

    assert results
    # "About SKU0001" means the identifier appears in the handle or the text:
    # ProductSubstitute:SKU0002 records `product_id SKU0001` and is a correct
    # hit even though its own handle names the substitute.
    def mentions_target(result) -> bool:
        return "SKU0001" in result.record_id or "SKU0001" in result.text

    assert mentions_target(results[0]), (
        f"verbose query ranked {results[0].record_id} first, which does not "
        f"mention the entity the question named"
    )
    assert sum(mentions_target(r) for r in results[:3]) >= 2
