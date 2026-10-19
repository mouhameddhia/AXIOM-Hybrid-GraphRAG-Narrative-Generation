"""
Standard ranking metrics for retrieval evaluation.

Deliberately dependency-free and written from the definitions, so the numbers
in a report can be checked by hand against a printed ranking. Each function
takes the ranked citation handles and the labelled relevant set, and nothing
else -- no pipeline objects, no configuration, no hidden state.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

__all__ = [
    "average_precision",
    "hit_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]


def reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    """
    ``1 / rank`` of the first relevant result, or ``0.0`` if none appears.

    Averaged over a query set this is MRR. It answers "how far down must a
    reviewer read before finding something that answers the question" -- the
    metric that matters most when evidence feeds a generated explanation.
    """
    for position, record_id in enumerate(ranked, start=1):
        if record_id in relevant:
            return 1.0 / position
    return 0.0


def hit_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """1.0 if any relevant record appears in the top ``k``, else 0.0."""
    return float(any(record_id in relevant for record_id in ranked[:k]))


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """
    Fraction of the relevant set retrieved within the top ``k``.

    Returns 0.0 for an empty relevant set rather than dividing by zero; the
    harness rejects unlabelled cases before they reach here.
    """
    if not relevant:
        return 0.0
    found = sum(1 for record_id in ranked[:k] if record_id in relevant)
    return found / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of the top ``k`` that is relevant."""
    if k < 1 or not ranked:
        return 0.0
    window = ranked[:k]
    return sum(1 for record_id in window if record_id in relevant) / len(window)


def average_precision(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    """
    Mean of the precision values observed at each relevant hit.

    Rewards placing *all* relevant records high, not merely the first one, so
    it separates a ranking that front-loads the evidence from one that scatters
    it through the list.
    """
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for position, record_id in enumerate(ranked, start=1):
        if record_id in relevant:
            hits += 1
            total += hits / position
    return total / len(relevant)


def ndcg_at_k(
    ranked: Sequence[str],
    gains: Mapping[str, float],
    k: int,
) -> float:
    """
    Normalised discounted cumulative gain over graded relevance.

    Graded, not binary: the record that actually answers the question carries a
    higher gain than a merely supporting one, so a ranking that puts context
    above the answer is scored lower even though both are "relevant".

    Uses the standard ``log2(rank + 1)`` discount and normalises against the
    ideal ordering of the same gains.
    """
    if not gains or k < 1:
        return 0.0

    dcg = sum(
        gains.get(record_id, 0.0) / math.log2(position + 1)
        for position, record_id in enumerate(ranked[:k], start=1)
    )
    ideal = sum(
        gain / math.log2(position + 1)
        for position, gain in enumerate(
            sorted(gains.values(), reverse=True)[:k], start=1
        )
    )
    return dcg / ideal if ideal > 0 else 0.0
