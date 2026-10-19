"""
Bridging text-to-query answers into narrative facts.

A narrative explains Track A's pricing decision, but a pricing decision is
easier to read against its context: what the list price actually is, how many
times this deal was quoted, what the product costs. Those are not Track A's
numbers -- they are facts about the graph, and D3's text-to-query layer
already computes them correctly, by generating a query and letting the graph
do the arithmetic.

This module lets such an answer become a fact the narrative may state, under
the same closure rule as everything else: it goes into the fact pack, or it
cannot be used.

**Two attested sources, one rule.** After this bridge a fact pack can hold
figures from two places, and neither is the language model:

===================  ==========================================  ==============
Source               Attribution                                 Who computed it
===================  ==========================================  ==============
Track A's payload    ``trackA-<version>``                        the pricing engine
Text-to-query        a record handle, or ``GraphQuery:<key>``    the graph itself
===================  ==========================================  ==============

The distinction is kept visible rather than blurred. Pricing decisions remain
Track A's alone; graph facts are context, they carry their own provenance, and
the Cypher that produced each one is retained so a reviewer can re-run it.

A pure aggregate has no single record behind it -- ``count(n)`` over eighty
products is not "about" any one of them -- so those are attributed to
``GraphQuery:<key>`` and the query text is reported alongside the narrative.
That is a weaker citation than a record id, and it is labelled as such rather
than dressed up as one.

An answer that failed, or returned nothing, produces no fact at all. A
narrative is never given a placeholder to explain.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.narrative.facts import Fact
from src.text_to_query.engine import QueryResult

__all__ = ["GraphFact", "fact_from_answer"]


@dataclass(frozen=True, slots=True)
class GraphFact:
    """One fact computed by the graph, with the query that produced it."""

    fact: Fact
    question: str
    cypher: str

    @property
    def is_record_backed(self) -> bool:
        """Whether this cites a specific record rather than a query."""
        return not self.fact.citation.startswith("GraphQuery:")


def fact_from_answer(
    result: QueryResult,
    *,
    key: str,
    label_en: str,
    kind: str = "money",
    label_ar: str = "",
) -> GraphFact | None:
    """
    Turn one text-to-query answer into a citable fact.

    Args:
        result: the answer from :meth:`TextToQueryEngine.ask`.
        key: stable identifier, also used as the fallback citation.
        label_en: how the quantity is named in the fact table.
        kind: ``money``, ``percent`` or ``count`` -- display formatting only.
        label_ar: optional Arabic label, for the deferred bilingual path.

    Returns:
        A :class:`GraphFact`, or ``None`` when the query failed, returned no
        value, or returned something that is not a number. A narrative is
        given nothing rather than a placeholder.
    """
    if not result.success:
        return None

    value = result.value
    # bool is an int subclass; a True must never become the quantity 1.
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    if not _plan_shape_matches(result, kind):
        return None

    citation = result.citations[0] if result.citations else f"GraphQuery:{key}"
    return GraphFact(
        fact=Fact(
            key=key,
            value=round(float(value), 2),
            kind=kind,
            citation=citation,
            label_en=label_en,
            label_ar=label_ar,
        ),
        question=result.question,
        cypher=result.cypher or "",
    )


def _plan_shape_matches(result: QueryResult, kind: str) -> bool:
    """
    Whether the query that ran could answer the question that was asked.

    Closure guarantees a narrative's numbers come from an attested source. It
    does **not** guarantee that source answered the right question, and this
    is where the two come apart: asked for the list price of a product, the
    text-to-query engine once returned ``count(n)`` over ``SellableProduct``
    -- the number 1, correctly computed by the graph, attributed to a real
    record, and completely wrong as a price. Admitting it would have put a
    plausible wrong figure into the set the narrative may quote.

    The check is structural rather than semantic: a monetary or percentage
    quantity must come from aggregating an actual property (``min``, ``max``,
    ``avg``, ``sum`` over a named field), never from counting rows. Counting
    rows answers "how many", and "how many" is not a price.

    This detects a mismatch of *shape*. It cannot tell that the engine
    aggregated the wrong property, so it narrows the hole rather than closing
    it -- see the limitations section of the D4 document.
    """
    plan = result.plan
    if plan is None:
        return False
    if kind == "count":
        return True
    # A money or percent figure must aggregate a named property.
    return bool(plan.aggregate_property)
