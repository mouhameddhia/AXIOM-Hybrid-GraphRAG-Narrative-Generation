r"""
The reliability layer, assembled (D5).

    Track A payload
          |
    build fact pack  (D4)
          |
    assess evidence ------ weak? ------> corrective re-retrieval
          |                                      |
          +<-------- strengthened pack ----------+
          |
    generate k samples at temperature
          |
    per sample: verify (D4) + check attribution
          |
    quorum? ---- no ----> abstain
          |
    agreement >= threshold? ---- no ----> abstain
          |
    elect the consensus sample
          |
    served narrative

Every gate here answers the plan's D5 definition of done -- "self-consistency
(k=3) with abstain-on-disagreement; corrective-retrieval fallback;
hallucination checks; measurable move toward the <5% served-wrong target" --
and every one of them can only ever *withhold* an answer D4 would have
served. That direction is the point: this layer trades availability for
correctness, and the trade is measured rather than assumed
(``evaluate_narratives.py`` reports both sides).

Sampling temperature deserves a note. D4 generates at temperature 0 for
determinism. Sampling k times at 0 returns k identical texts, which measures
nothing and costs k times as much, so this layer samples above zero on
purpose. The consequence is that a served narrative is no longer bit-for-bit
reproducible from the same inputs; what *is* reproducible is the decision to
serve or abstain, which is what the governance claim rests on.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.config.settings import NarrativeConfig, ReliabilityConfig
from src.narrative.contract import PriceRecommendation
from src.narrative.evidence import EvidenceIndex
from src.narrative.facts import FactPack, build_facts
from src.narrative.generator import Narrative, NarrativeGenerator
from src.reliability.consistency import ConsistencyReport, assess_consistency
from src.reliability.corrective import (
    CorrectionOutcome,
    EvidenceAssessment,
    SupportingRecordFinder,
    apply_corrective_retrieval,
    assess_evidence,
)

__all__ = ["Observer", "ReliableNarrative", "ReliableNarrator"]

#: Receives ``(stage, payload)`` as each stage of ``explain`` completes. It is
#: purely observational (a front end can show progress); an observer that
#: raises is logged and ignored, so it can never change a decision.
Observer = Callable[[str, Any], None]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReliableNarrative:
    """One explanation that survived the reliability layer, or a refusal."""

    subject: str
    served: bool
    narrative: Narrative | None
    consistency: ConsistencyReport | None
    assessment: EvidenceAssessment
    correction: CorrectionOutcome | None
    pack: FactPack
    reason: str | None = None

    @property
    def text(self) -> str:
        """The served explanation, or the empty string on abstention."""
        return self.narrative.text if self.narrative is not None else ""

    @property
    def citations(self) -> tuple[str, ...]:
        return self.narrative.citations if self.narrative is not None else ()

    @property
    def samples_used(self) -> int:
        return self.consistency.k if self.consistency is not None else 0

    @property
    def agreement(self) -> float:
        return self.consistency.agreement if self.consistency is not None else 0.0

    def describe(self) -> str:
        lines = [f"evidence: {self.assessment.describe()}"]
        if self.correction is not None:
            lines.append(self.correction.describe())
        if self.consistency is not None:
            lines.append(self.consistency.describe())
        lines.append("served" if self.served else f"ABSTAINED: {self.reason}")
        return "\n".join(lines)


class ReliableNarrator:
    """
    Wraps a :class:`NarrativeGenerator` with the D5 safeguards.

    Usage::

        narrator = ReliableNarrator(generator, settings.reliability,
                                    settings.narrative, evidence=index)
        result = narrator.explain(recommendation)
        if result.served:
            print(result.text)
    """

    __slots__ = ("_generator", "_config", "_narrative_config", "_evidence", "_finder")

    def __init__(
        self,
        generator: NarrativeGenerator,
        config: ReliabilityConfig,
        narrative_config: NarrativeConfig,
        *,
        evidence: EvidenceIndex | None = None,
        finder: SupportingRecordFinder | None = None,
    ) -> None:
        self._generator = generator
        self._config = config
        self._narrative_config = narrative_config
        self._evidence = evidence
        self._finder = finder

    def explain(
        self,
        recommendation: PriceRecommendation,
        *,
        pack: FactPack | None = None,
        observer: Observer | None = None,
        question: str | None = None,
    ) -> ReliableNarrative:
        """
        Explain one Track A result under the reliability safeguards.

        Args:
            recommendation: the payload to explain.
            pack: overrides the fact pack built from it, for a caller that
                has already folded in graph-computed context.
            observer: optional progress callback, called with ``(stage, data)``
                for the stages ``pack``, ``assessment``, ``correction``,
                ``sample`` (``{"index", "narrative"}``), ``consistency`` and
                ``decision``. It cannot influence the outcome.
            question: when given, each sample answers this question from the
                fact pack instead of writing the general explanation; the
                evidence check, quorum, agreement and wrong-label check apply
                unchanged.

        Returns:
            A :class:`ReliableNarrative`; check ``served`` before ``text``.
        """
        def notify(stage: str, data: Any) -> None:
            if observer is None:
                return
            try:
                observer(stage, data)
            except Exception:  # an observer must never alter a decision
                logger.exception("Reliability observer failed at stage %r", stage)

        working = pack if pack is not None else build_facts(recommendation)
        notify("pack", working)
        assessment = assess_evidence(
            working, self._evidence, minimum_support=self._config.min_support_ratio
        )
        notify("assessment", assessment)

        correction: CorrectionOutcome | None = None
        if assessment.weak and self._should_correct():
            logger.info("Evidence judged weak: %s", assessment.describe())
            correction = apply_corrective_retrieval(
                working,
                recommendation,
                self._finder,
                self._evidence,
                top_k=self._config.corrective_top_k,
                max_added=self._config.corrective_max_added,
                minimum_support=self._config.min_support_ratio,
            )
            working = correction.pack
            notify("correction", correction)

        samples = []
        for index in range(1, self._config.samples + 1):
            sample = self._generator.explain(
                working, language="en", evidence=self._evidence, question=question
            )
            samples.append(sample)
            notify("sample", {"index": index, "narrative": sample, "pack": working})

        consistency = assess_consistency(
            samples,
            working,
            min_verified=self._config.min_verified,
            min_agreement=self._config.min_agreement,
            absolute_tolerance=self._narrative_config.absolute_tolerance,
            relative_tolerance=self._narrative_config.relative_tolerance,
            require_attribution=self._config.enforce_attribution,
            agreement_kind="overlap" if question else "core",
        )

        notify("consistency", consistency)

        if consistency.abstained or consistency.consensus is None:
            result = ReliableNarrative(
                subject=working.subject,
                served=False,
                narrative=None,
                consistency=consistency,
                assessment=assessment,
                correction=correction,
                pack=working,
                reason=consistency.reason,
            )
            notify("decision", result)
            return result

        result = ReliableNarrative(
            subject=working.subject,
            served=True,
            narrative=consistency.consensus.narrative,
            consistency=consistency,
            assessment=assessment,
            correction=correction,
            pack=working,
        )
        notify("decision", result)
        return result

    def _should_correct(self) -> bool:
        return (
            self._config.corrective_enabled
            and self._finder is not None
            and self._evidence is not None
        )
