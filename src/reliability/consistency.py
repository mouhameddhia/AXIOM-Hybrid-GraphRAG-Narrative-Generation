"""
Self-consistency with abstain-on-disagreement (D5).

The onboarding plan names this as the mechanism that makes generation
trustworthy: "sample several answers (k=3) and, if they disagree beyond a
threshold, abstain instead of guessing."

What disagreement means here
----------------------------
Borrowing the textbook formulation unchanged would not work in this pipeline,
and measuring first is what showed why. Sampling one fact pack three times at
temperature 0.7 produced claim sets of 7, 11 and 9 numbers with *identical*
citations and no contradiction anywhere: the samples differed in how much
they said, not in what they said. A naive set-equality or Jaccard test would
have called that a disagreement and abstained on healthy output every time.

So agreement is measured on the **consensus core** -- the figures a majority
of samples state -- as a fraction of everything any sample states:

    agreement = |stated by a majority| / |stated by anyone|

A sample that adds the price range to what the others said lowers this
slightly and correctly: it is extra breadth, not conflict. Samples that talk
about genuinely different figures drive the core towards empty and the score
towards zero, which is the case worth abstaining on.

Three filters, then the vote
----------------------------
A sample must clear all three checks to join the vote at all:

1. **Verification** -- every number traces to the pack, every citation
   resolves, every claim is cited (D4).
2. **Attribution** -- the sentence asserting a role states that role's value
   (:mod:`src.reliability.attribution`).
3. **Quorum** -- if fewer than ``min_verified`` samples survive, the model is
   unstable on this input and the answer is withheld regardless of what the
   survivors agree on.

The served narrative is then the surviving sample that covers the most of the
consensus core, with the shortest text breaking ties -- brevity being the
tiebreak that avoids rewarding a sample for padding.

Cost is real: k samples multiply inference cost k-fold, which the plan flags
explicitly. ``k`` is configuration, and setting it to 1 degrades this to the
D4 behaviour with the attribution check still applied.
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from src.narrative.facts import FactPack
from src.narrative.generator import Narrative
from src.reliability.attribution import AttributionReport, check_attribution

__all__ = [
    "ConsistencyReport",
    "Sample",
    "assess_consistency",
]

logger = logging.getLogger(__name__)

#: Numbers are compared at this precision when forming claim sets. Tolerance
#: is already applied upstream by the verifier; this only prevents float
#: representation noise from splitting one figure into two.
_CLAIM_DP = 2


@dataclass(frozen=True, slots=True)
class Sample:
    """One generated candidate and everything known about it."""

    index: int
    narrative: Narrative
    attribution: AttributionReport | None
    #: Whether a misattribution disqualifies this sample from voting.
    #: Always measured; enforcement is what the D4 baseline turns off, so
    #: that the two configurations can be compared on the same instrument.
    require_attribution: bool = True

    @property
    def attribution_ok(self) -> bool:
        return self.attribution is None or self.attribution.ok

    @property
    def usable(self) -> bool:
        """Verified, attributed correctly, and therefore eligible to vote."""
        if not self.narrative.success:
            return False
        return self.attribution_ok or not self.require_attribution

    @property
    def claims(self) -> frozenset[float]:
        """The distinct figures this sample states."""
        report = self.narrative.verification
        if report is None:
            return frozenset()
        return frozenset(round(value, _CLAIM_DP) for value in report.numbers_found)

    @property
    def rejection(self) -> str:
        """Why this sample cannot vote, for the abstention reason."""
        if not self.narrative.success:
            return self.narrative.reason or "failed verification"
        if not self.attribution_ok:
            first = self.attribution.misattributions[0]
            verb = "misattributed" if self.require_attribution else "misattributed (not enforced)"
            return f"{verb} {first.role_phrase!r}"
        return "usable"


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    """The outcome of sampling one fact pack k times."""

    samples: tuple[Sample, ...]
    #: Samples that passed verification and attribution.
    usable: tuple[Sample, ...]
    #: Figures a majority of usable samples state.
    core: frozenset[float]
    #: Every figure any usable sample states.
    union: frozenset[float]
    agreement: float
    consensus: Sample | None
    abstained: bool
    reason: str | None = None
    #: ``core`` (figures a majority state / figures any sample states; for a general
    #: explanation) or ``overlap`` (for an answer to a question; see :func:`_overlap_agreement`).
    agreement_kind: str = "core"

    @property
    def k(self) -> int:
        return len(self.samples)

    def describe(self) -> str:
        lines = [
            f"k={self.k}, usable={len(self.usable)}, "
            f"agreement={self.agreement:.2f} "
            f"(core {len(self.core)} of union {len(self.union)})"
        ]
        for sample in self.samples:
            mark = "ok " if sample.usable else "-- "
            lines.append(
                f"  {mark}sample {sample.index}: "
                f"{len(sample.claims)} figure(s), {sample.rejection}"
            )
        return "\n".join(lines)


def assess_consistency(
    samples: Sequence[Narrative],
    pack: FactPack,
    *,
    min_verified: int = 2,
    min_agreement: float = 0.6,
    absolute_tolerance: float = 0.01,
    relative_tolerance: float = 0.005,
    require_attribution: bool = True,
    agreement_kind: str = "core",
) -> ConsistencyReport:
    """
    Filter, score and vote on k sampled narratives.

    Args:
        samples: the candidates, in generation order.
        pack: the fact pack they were all generated from.
        min_verified: how many must survive verification and attribution for
            any answer to be served.
        min_agreement: the consensus-core fraction below which the samples
            are judged to disagree.
        absolute_tolerance: slack in currency minor units, for attribution.
        relative_tolerance: slack for a rounded figure, for attribution.
        require_attribution: whether a misattributed sample is disqualified.
            Attribution is always *measured*; turning enforcement off is what
            produces a genuine D4 baseline, where the gate exists only as an
            observation. Comparing D4 against D5 with the gate silently
            active on both sides measures nothing.

    Returns:
        A :class:`ConsistencyReport`. When ``abstained`` is True, ``consensus``
        is ``None`` and ``reason`` says which filter rejected the batch.
    """
    if not samples:
        raise ValueError("assess_consistency() needs at least one sample")

    scored = tuple(
        Sample(
            index=position,
            narrative=narrative,
            attribution=(
                check_attribution(
                    narrative.text,
                    pack,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                if narrative.success
                else None
            ),
            require_attribution=require_attribution,
        )
        for position, narrative in enumerate(samples, start=1)
    )
    usable = tuple(sample for sample in scored if sample.usable)

    if len(usable) < min_verified:
        # When every sample declined for the same stated reason (the question cannot
        # be answered from the facts, the model was unreachable, ...), that reason is
        # more useful than a count of failed checks.
        declined = {s.narrative.reason for s in scored if s.narrative.abstained}
        if all(s.narrative.abstained for s in scored) and len(declined) == 1 and None not in declined:
            reason = f"every sample declined: {declined.pop()}"
        else:
            reason = (
                f"only {len(usable)} of {len(scored)} samples passed verification "
                f"and attribution; {min_verified} are required"
            )
        return ConsistencyReport(
            samples=scored,
            usable=usable,
            core=frozenset(),
            union=frozenset(),
            agreement=0.0,
            consensus=None,
            abstained=True,
            reason=reason,
        )

    core, union, agreement = _agreement(usable)
    if agreement_kind == "overlap":
        agreement = _overlap_agreement(usable)
    if agreement < min_agreement:
        detail = (
            f"they state different figures (overlap {agreement:.2f}, minimum {min_agreement:.2f})"
            if agreement_kind == "overlap"
            else f"only {len(core)} of {len(union)} figures are stated by a majority "
            f"(agreement {agreement:.2f}, minimum {min_agreement:.2f})"
        )
        return ConsistencyReport(
            samples=scored,
            usable=usable,
            core=core,
            union=union,
            agreement=agreement,
            consensus=None,
            abstained=True,
            reason=f"the samples disagree: {detail}",
            agreement_kind=agreement_kind,
        )

    consensus = _elect(usable, core)
    logger.info(
        "Self-consistency: %d/%d usable, agreement %.2f, serving sample %d",
        len(usable), len(scored), agreement, consensus.index,
    )
    return ConsistencyReport(
        samples=scored,
        usable=usable,
        core=core,
        union=union,
        agreement=agreement,
        consensus=consensus,
        abstained=False,
        agreement_kind=agreement_kind,
    )


def _overlap_agreement(usable: Sequence[Sample]) -> float:
    """
    Agreement between short answers: how much of the shorter answer the longer one contains.

    The core-over-union measure suits a general explanation, where samples state most of the
    same figures. An answer to a question states only the figures that answer it, and two
    correct answers legitimately differ in how many they mention (one gives the pocket price,
    the invoice price and the gap; another only the gap). Over the union that reads as
    disagreement (measured: 2 of 3 runs of an ordinary question withheld at 0.33), although
    one answer is simply contained in the other. For every pair of samples that state
    figures, take the size of the intersection over the size of the smaller set, and average:
    a subset agrees (1.0), disjoint figures disagree (0.0). A sample that states no figure
    neither confirms nor contradicts and is left out; if none states any, they agree.
    """
    stated = [sample.claims for sample in usable if sample.claims]
    if len(stated) < 2:
        return 1.0
    scores = []
    for i in range(len(stated)):
        for j in range(i + 1, len(stated)):
            scores.append(len(stated[i] & stated[j]) / min(len(stated[i]), len(stated[j])))
    return sum(scores) / len(scores)


def _agreement(usable: Sequence[Sample]) -> tuple[frozenset[float], frozenset[float], float]:
    """
    The consensus core, the union, and the ratio between them.

    A single usable sample agrees with itself by definition: with nothing to
    compare against, self-consistency has no opinion, and the verification and
    attribution checks are what stand behind the answer.
    """
    counts: Counter[float] = Counter()
    union: set[float] = set()
    for sample in usable:
        claims = sample.claims
        union |= claims
        counts.update(claims)

    if not union:
        return frozenset(), frozenset(), 1.0

    majority = len(usable) // 2 + 1
    core = frozenset(value for value, seen in counts.items() if seen >= majority)
    return core, frozenset(union), len(core) / len(union)


def _elect(usable: Sequence[Sample], core: frozenset[float]) -> Sample:
    """
    The sample that best represents the consensus.

    Most of the core first, then the shortest text: a sample should not win
    by saying more than the others, only by covering what they agreed on.
    """
    return max(
        usable,
        key=lambda sample: (
            len(sample.claims & core),
            -len(sample.narrative.text),
            -sample.index,
        ),
    )
