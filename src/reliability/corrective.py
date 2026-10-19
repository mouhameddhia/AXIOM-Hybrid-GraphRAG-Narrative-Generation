"""
Corrective retrieval: strengthen weak evidence before generating (D5).

The onboarding plan asks for a fallback that fires "when retrieved evidence is
judged weak: re-retrieve or rewrite the query before generating". This module
is the judging and the re-retrieving; :mod:`src.reliability.narrator` decides
when to call it.

Two measures, only one of which is actionable
--------------------------------------------
Measuring real fact packs separated these, and conflating them produced a
correction that fired on every explanation and improved nothing:

**Citation density** -- the share of figures carrying a record of their own.
A typical recommendation scores 0.67, because ``invoice_price``,
``pocket_price``, ``total_deductions``, ``erosion_pct`` and
``deduction_count`` arrive from Track A as subtotals of the walk with no row
behind them. Those are exactly the headline numbers a reader most wants to
check. **Retrieval cannot fix this** -- the missing link is per-figure
provenance in the upstream contract, not a record that exists and was not
found. It is reported as a diagnostic and never used as a trigger, because a
trigger its own remedy cannot move is a loop that fires forever.

**Support ratio** -- citable records per figure. This one retrieval *can*
move, and it is what ``weak`` is judged on.

Subject relevance is the whole game
-----------------------------------
The first working version admitted whatever retrieval ranked highest, which
for an explanation of order line ``OL000001`` meant adding ``OL000085``,
``OL000183`` and ``OL000011`` -- three *other* order lines. Handing those to a
narrative under a rule that every claim must be cited does not improve
grounding; it invites a citation that looks valid, resolves cleanly, and
points at the wrong deal. Worse than no citation, for the same reason an
unresolvable handle is worse than none: it survives inspection.

A retrieved record is therefore admitted only when its handle names this
recommendation's product, customer or order line. On a complete Track A
payload that filter usually admits nothing, because the payload already cites
those records -- which is the correct outcome and worth stating plainly: this
fallback earns its keep on *degraded* evidence, not on healthy evidence.

What the correction never does
------------------------------
It adds **no figures**. The closed fact set is what keeps the model from
inventing numbers, and widening it here would trade the track's central
guarantee for prose quality. More records to cite, never more numbers to
state.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from src.narrative.contract import PriceRecommendation
from src.narrative.evidence import EvidenceIndex
from src.narrative.facts import FactPack

__all__ = [
    "CorrectionOutcome",
    "EvidenceAssessment",
    "SupportingRecordFinder",
    "apply_corrective_retrieval",
    "assess_evidence",
    "subject_terms",
]

logger = logging.getLogger(__name__)


class SupportingRecordFinder(Protocol):
    """
    Anything that can return record handles for a query.

    Narrowed to this one method so the reliability layer does not depend on
    the whole retrieval stack, and so a test can drive it with a stub instead
    of building an index.
    """

    def __call__(self, query: str, top_k: int) -> Sequence[str]:
        ...


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """How well supported a fact pack is, before anything is generated."""

    facts_total: int
    facts_with_citation: int
    citations_total: int
    unresolvable: tuple[str, ...]
    #: The support-ratio floor this assessment was judged against.
    minimum_support: float

    @property
    def citation_density(self) -> float:
        """
        Share of figures carrying a record of their own.

        A diagnostic, never a trigger: the gap is upstream provenance, which
        retrieval cannot supply. See the module docstring.
        """
        if self.facts_total == 0:
            return 0.0
        return self.facts_with_citation / self.facts_total

    @property
    def support_ratio(self) -> float:
        """Citable records per figure -- the measure retrieval can move."""
        if self.facts_total == 0:
            return 0.0
        return self.citations_total / self.facts_total

    @property
    def weak(self) -> bool:
        """Too few records to attribute this many figures to."""
        return self.support_ratio < self.minimum_support

    def describe(self) -> str:
        return (
            f"support {self.citations_total}/{self.facts_total} records per figure "
            f"({self.support_ratio:.2f}, floor {self.minimum_support:.2f}); "
            f"density {self.citation_density:.2f} "
            f"({self.facts_with_citation} figure(s) carry their own record); "
            f"{len(self.unresolvable)} unresolvable"
        )


@dataclass(frozen=True, slots=True)
class CorrectionOutcome:
    """The result of trying to strengthen a pack's evidence."""

    before: EvidenceAssessment
    after: EvidenceAssessment
    pack: FactPack
    added: tuple[str, ...]
    #: Retrieved but refused for naming a different subject.
    rejected: tuple[str, ...]
    queries_tried: tuple[str, ...]

    @property
    def improved(self) -> bool:
        return bool(self.added)

    def describe(self) -> str:
        if not self.added:
            return (
                f"corrective retrieval added nothing "
                f"({len(self.queries_tried)} quer(y|ies), "
                f"{len(self.rejected)} off-subject candidate(s) refused)"
            )
        return (
            f"corrective retrieval added {len(self.added)} record(s) "
            f"[{', '.join(self.added)}]; support "
            f"{self.before.support_ratio:.2f} -> {self.after.support_ratio:.2f}"
            + (f", {len(self.rejected)} off-subject refused" if self.rejected else "")
        )


def subject_terms(recommendation: PriceRecommendation) -> tuple[str, ...]:
    """The identifiers a record must name to count as on-subject."""
    return tuple(
        term
        for term in (
            recommendation.product_id,
            recommendation.customer_id,
            recommendation.order_line_id,
        )
        if term
    )


def assess_evidence(
    pack: FactPack,
    index: EvidenceIndex | None = None,
    *,
    minimum_support: float = 0.75,
) -> EvidenceAssessment:
    """
    Judge how well the pack's figures are backed by records.

    Args:
        pack: the fact set about to be explained.
        index: when supplied, citations are resolved against the graph so
            unresolvable handles are reported.
        minimum_support: citable records per figure below which the evidence
            counts as thin.

    Returns:
        An :class:`EvidenceAssessment`; check ``weak``.
    """
    unresolvable: tuple[str, ...] = ()
    if index is not None:
        unresolvable = tuple(
            handle
            for handle in index.missing(pack.citations)
            # The engine version is an attribution, not a graph record, so it
            # is not expected to resolve and must not be reported as missing.
            if handle != pack.engine_version
        )
    return EvidenceAssessment(
        facts_total=len(pack.facts),
        facts_with_citation=sum(1 for fact in pack.facts if fact.citation),
        citations_total=len(pack.citations),
        unresolvable=unresolvable,
        minimum_support=minimum_support,
    )


def apply_corrective_retrieval(
    pack: FactPack,
    recommendation: PriceRecommendation,
    finder: SupportingRecordFinder,
    index: EvidenceIndex,
    *,
    top_k: int = 6,
    max_added: int = 4,
    minimum_support: float = 0.75,
) -> CorrectionOutcome:
    """
    Re-retrieve supporting records and admit the on-subject ones.

    Adds records only -- never figures.

    Args:
        pack: the pack to strengthen.
        recommendation: the Track A payload, for the query and the subject.
        finder: the retrieval callable.
        index: used to discard any handle that does not resolve.
        top_k: how many candidates to request per query.
        max_added: cap on new citable records, so a narrative is not handed a
            long list of loosely-related records to choose from.
        minimum_support: the floor used for the before/after assessment.

    Returns:
        A :class:`CorrectionOutcome` carrying the strengthened pack.
    """
    before = assess_evidence(pack, index, minimum_support=minimum_support)
    terms = subject_terms(recommendation)
    known = set(pack.citations)
    added: list[str] = []
    rejected: list[str] = []
    tried: list[str] = []

    for query in _query_ladder(recommendation):
        if len(added) >= max_added:
            break
        tried.append(query)
        try:
            candidates = finder(query, top_k)
        except Exception as exc:  # retrieval is best-effort here
            logger.warning("Corrective retrieval failed for %r: %s", query, exc)
            continue
        for handle in candidates:
            if len(added) >= max_added:
                break
            if handle in known or handle not in index:
                continue
            if not _on_subject(handle, terms):
                if handle not in rejected:
                    rejected.append(handle)
                continue
            known.add(handle)
            added.append(handle)

    strengthened = (
        FactPack(
            subject=pack.subject,
            kind=pack.kind,
            currency_code=pack.currency_code,
            facts=pack.facts,
            citations=pack.citations + tuple(added),
            engine_version=pack.engine_version,
        )
        if added
        else pack
    )
    after = assess_evidence(strengthened, index, minimum_support=minimum_support)
    if added:
        logger.info("Corrective retrieval admitted %d record(s): %s", len(added), added)
    if rejected:
        logger.info(
            "Corrective retrieval refused %d off-subject record(s): %s",
            len(rejected), rejected,
        )
    return CorrectionOutcome(
        before=before,
        after=after,
        pack=strengthened,
        added=tuple(added),
        rejected=tuple(rejected),
        queries_tried=tuple(tried),
    )


def _on_subject(handle: str, terms: Sequence[str]) -> bool:
    """
    Whether a record handle names this recommendation's subject.

    Without this, retrieval for order line OL000001 admitted OL000085 and
    OL000183 -- valid, resolvable citations pointing at other deals.
    """
    return any(term in handle for term in terms)


def _query_ladder(recommendation: PriceRecommendation) -> list[str]:
    """
    Progressively broader queries: precise first, then whatever returns.

    Rewriting only matters when the precise query comes back thin, so the
    narrow form leads and the broad form is the fallback -- the reverse order
    would dilute good evidence with loosely-related records.
    """
    product = recommendation.product_id
    customer = recommendation.customer_id
    line = recommendation.order_line_id

    queries = []
    if line:
        queries.append(f"{product} {customer} {line} price discount")
        queries.append(f"{product} {line} price component discount")
    queries.append(f"{product} {customer} price")
    queries.append(f"{product} list price cost")
    return list(dict.fromkeys(queries))
