"""
Narrative evaluation harness (D6).

The plan asks for generation-side metrics -- "faithfulness/groundedness (is
each claim supported by the cited evidence?), answer relevance, and citation
coverage" -- plus a measurable move toward the under-5% served-wrong target,
wired into regression checks. This module is that measurement. Retrieval
metrics stay where they already live, in :mod:`src.evaluation.harness`.

Independence is the design constraint
-------------------------------------
A faithfulness score computed against the same fact pack the generator was
handed is close to tautological: the pipeline verifies numbers against that
pack before serving anything, so the score can only ever catch a broken
verifier. It would read 1.00 on a system that was confidently wrong about
what the pack should have contained.

So every expected figure reaches this harness from the frozen gold set, read
straight off disk, never through :func:`build_facts`. The two paths are then
compared rather than assumed equal, and
:attr:`NarrativeCaseResult.pack_matches_gold` reports any divergence between
what the gold set says a recommendation contains and what the fact pack built
from it actually offered. A drift there invalidates the run, and saying so is
more useful than a clean-looking number.

What is measured
----------------
``faithfulness``
    Share of stated figures that appear in the gold set's expected values.
    This is the groundedness question asked against independent truth.

``answer_relevance``
    Share of the case's ``required_facts`` the narrative actually states. A
    waterfall explanation quoting only the list price is perfectly faithful
    and useless, and nothing else here would notice.

``citation_coverage``
    Share of numeric claims carrying a citation, from the D4 verifier.

``attribution_ok``
    Whether every asserted role carried its own value (D5).

``served_wrong``
    The headline, and deliberately narrow: a *served* narrative counts as
    wrong when it states a figure absent from the gold truth, or puts a
    figure on the wrong noun. Both are **false content**. Abstentions are
    excluded by construction -- withholding an answer is not serving a wrong
    one, which is exactly the trade the reliability layer exists to make.

``weak_citation``
    Kept separate from served-wrong, and the separation was forced by
    auditing real output. Two baseline narratives were being counted as
    wrong while stating every figure correctly; what they had done was cite
    one record -- the order line -- after all four claims. That is a genuine
    defect, because the order line does not contain the freight figure and a
    reader cannot check it there. It is not the same event as asserting a
    false number, and folding the two together made the headline metric mean
    two things at once, inflating the baseline in a way that happened to
    flatter the reliability layer. Reported alongside, never inside.

Reproducibility, stated precisely
---------------------------------
The reliability layer samples above temperature zero, so narrative *text* is
not reproducible run to run. What the harness reports as stable is the
decision profile: service rate, served-wrong rate, and the metric means over
a frozen gold set. Anyone expecting byte-identical prose from a rerun is
measuring the wrong thing, and the report prints the sampling settings
alongside the numbers so the distinction is visible.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from src.config.settings import PROJECT_ROOT
from src.narrative.contract import ContractError, PriceRecommendation
from src.narrative.facts import FactPack, build_facts
from src.reliability.attribution import check_attribution
from src.reliability.narrator import ReliableNarrator

__all__ = [
    "NarrativeCase",
    "NarrativeCaseResult",
    "NarrativeEvaluationReport",
    "NarrativeGoldSet",
    "NarrativeGoldSetError",
    "evaluate_narratives",
    "score_case",
]

logger = logging.getLogger(__name__)

#: Slack when matching a stated figure against gold truth. Mirrors the
#: verifier's tolerances so the two layers cannot disagree over rounding.
ABSOLUTE_TOLERANCE = 0.01
RELATIVE_TOLERANCE = 0.005

#: The plan's attestation-grade target for served-wrong answers.
SERVED_WRONG_TARGET = 0.05


class NarrativeGoldSetError(ValueError):
    """Raised when the gold set is malformed or its payloads are unreadable."""


@dataclass(frozen=True, slots=True)
class NarrativeCase:
    """One labelled recommendation with known-correct figures."""

    case_id: str
    recommendation_id: str
    payloads: Path
    subject: str
    required_facts: tuple[str, ...]
    min_citations: int
    expected_values: Mapping[str, float]

    @property
    def gold_figures(self) -> tuple[float, ...]:
        """Every figure a faithful narrative of this case may state."""
        return tuple(self.expected_values.values())

    def load_recommendation(self) -> PriceRecommendation:
        try:
            return PriceRecommendation.load(self.payloads, self.recommendation_id)
        except ContractError as exc:
            raise NarrativeGoldSetError(
                f"{self.case_id}: cannot load {self.recommendation_id} "
                f"from {self.payloads}: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class NarrativeGoldSet:
    """A validated collection of narrative cases."""

    cases: tuple[NarrativeCase, ...]
    source: Path
    description: str = ""
    language: str = "en"
    version: str = ""

    @classmethod
    def load(cls, path: Path) -> "NarrativeGoldSet":
        """
        Read and validate the frozen gold set.

        Raises:
            NarrativeGoldSetError: unreadable, malformed, or carrying a case
                with no expected values -- which would score a perfect
                faithfulness of 1.00 against an empty truth set and quietly
                flatter the system.
        """
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise NarrativeGoldSetError(f"Gold set not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise NarrativeGoldSetError(
                f"{path} is not valid JSON (line {exc.lineno}): {exc.msg}"
            ) from exc

        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise NarrativeGoldSetError(f"{path} contains no 'cases' array")

        cases = tuple(
            cls._parse_case(entry, index, path)
            for index, entry in enumerate(raw_cases)
        )
        return cls(
            cases=cases,
            source=path,
            description=str(payload.get("description", "")),
            language=str(payload.get("language", "en")),
            version=str(payload.get("version", "")),
        )

    @staticmethod
    def _parse_case(entry: object, index: int, source: Path) -> NarrativeCase:
        if not isinstance(entry, Mapping):
            raise NarrativeGoldSetError(f"{source}: case {index} is not an object")

        case_id = str(entry.get("case_id") or f"case-{index}")
        expected = entry.get("expected_values")
        if not isinstance(expected, Mapping) or not expected:
            raise NarrativeGoldSetError(
                f"{source}: {case_id} has no expected_values. A case with no "
                f"truth scores a perfect faithfulness against nothing."
            )
        required = entry.get("required_facts")
        if not isinstance(required, Sequence) or isinstance(required, str):
            raise NarrativeGoldSetError(
                f"{source}: {case_id} needs 'required_facts' as a list"
            )
        unknown = [key for key in required if key not in expected]
        if unknown:
            raise NarrativeGoldSetError(
                f"{source}: {case_id} requires fact(s) {unknown} that carry no "
                f"expected value, so relevance could never be scored for them"
            )

        raw_payloads = str(entry.get("payloads", ""))
        if not raw_payloads:
            raise NarrativeGoldSetError(f"{source}: {case_id} names no payload file")
        payload_path = Path(raw_payloads)
        if not payload_path.is_absolute():
            payload_path = PROJECT_ROOT / payload_path

        return NarrativeCase(
            case_id=case_id,
            recommendation_id=str(entry.get("recommendation_id", "")),
            payloads=payload_path,
            subject=str(entry.get("subject", "")),
            required_facts=tuple(str(key) for key in required),
            min_citations=int(entry.get("min_citations", 1)),
            expected_values={str(k): float(v) for k, v in expected.items()},
        )


@dataclass(frozen=True, slots=True)
class NarrativeCaseResult:
    """Metrics for one evaluated case."""

    case_id: str
    recommendation_id: str
    served: bool
    text: str
    #: Figures the narrative stated that are absent from the gold truth.
    unfaithful: tuple[float, ...]
    faithfulness: float
    answer_relevance: float
    missing_required: tuple[str, ...]
    citation_coverage: float
    citations: int
    #: The case's floor on distinct cited records.
    min_citations: int
    attribution_ok: bool
    misattributed: tuple[str, ...]
    agreement: float
    samples_usable: int
    samples_total: int
    corrective_added: int
    #: False when the fact pack disagrees with the gold set about this
    #: recommendation -- the run is not trustworthy until that is resolved.
    pack_matches_gold: bool
    pack_gold_diff: tuple[str, ...]
    abstain_reason: str | None = None

    @property
    def meets_citation_floor(self) -> bool:
        """
        Whether enough distinct records were cited to be informative.

        One record repeated across every claim satisfies coverage while
        telling a reader nothing about which figure rests on what, so the
        gold set sets a per-case floor.
        """
        return self.citations >= self.min_citations

    @property
    def served_wrong(self) -> bool:
        """
        Whether a *served* narrative carries a detectable error.

        Abstentions are not wrong answers: withholding is the designed
        outcome when the checks fail, and counting it here would make the
        headline metric punish exactly the behaviour the layer exists for.
        """
        if not self.served:
            return False
        return bool(self.unfaithful) or not self.attribution_ok

    @property
    def weak_citation(self) -> bool:
        """
        Served, correct, but citing too few distinct records to be checkable.

        A separate axis from :attr:`served_wrong`: nothing stated is false.
        """
        return self.served and not self.meets_citation_floor

    def describe(self) -> str:
        if not self.served:
            return (
                f"{self.case_id}  ABSTAINED  "
                f"({self.samples_usable}/{self.samples_total} usable) "
                f"{self.abstain_reason or ''}"
            )
        flags = []
        if self.unfaithful:
            flags.append(f"unfaithful={[f'{v:,.2f}' for v in self.unfaithful]}")
        if not self.attribution_ok:
            flags.append(f"misattributed={list(self.misattributed)}")
        if self.missing_required:
            flags.append(f"missing={list(self.missing_required)}")
        if not self.meets_citation_floor:
            flags.append(f"cites {self.citations} < floor {self.min_citations}")
        return (
            f"{self.case_id}  SERVED  faith={self.faithfulness:.2f} "
            f"relev={self.answer_relevance:.2f} cover={self.citation_coverage:.2f} "
            f"cites={self.citations} agree={self.agreement:.2f}"
            + ("  " + "; ".join(flags) if flags else "")
        )


@dataclass(frozen=True, slots=True)
class NarrativeEvaluationReport:
    """Aggregate results across the whole gold set."""

    results: tuple[NarrativeCaseResult, ...]
    gold_set: Path
    label: str = ""
    config_summary: Mapping[str, object] = field(default_factory=dict)

    @property
    def cases(self) -> int:
        return len(self.results)

    @property
    def served(self) -> tuple[NarrativeCaseResult, ...]:
        return tuple(row for row in self.results if row.served)

    @property
    def service_rate(self) -> float:
        """Share of cases that produced an explanation."""
        return len(self.served) / self.cases if self.cases else 0.0

    @property
    def abstention_rate(self) -> float:
        return 1.0 - self.service_rate

    @property
    def served_wrong_rate(self) -> float:
        """
        Share of *served* narratives carrying a detectable error.

        The denominator is deliberately the served set, not every case: the
        question this answers is "if the system gives me an answer, how often
        is it wrong", which is what an attestation claim rests on.
        """
        served = self.served
        if not served:
            return 0.0
        return sum(1 for row in served if row.served_wrong) / len(served)

    @property
    def weak_citation_rate(self) -> float:
        """
        Share of *served* narratives citing too few distinct records.

        Tracked beside the served-wrong rate rather than inside it: these
        narratives state nothing false, and merging the two would make a
        single number stand for two different failures.
        """
        served = self.served
        if not served:
            return 0.0
        return sum(1 for row in served if row.weak_citation) / len(served)

    @property
    def meets_served_wrong_target(self) -> bool:
        return self.served_wrong_rate <= SERVED_WRONG_TARGET

    @property
    def faithfulness(self) -> float:
        return self._mean("faithfulness")

    @property
    def answer_relevance(self) -> float:
        return self._mean("answer_relevance")

    @property
    def citation_coverage(self) -> float:
        return self._mean("citation_coverage")

    @property
    def attribution_rate(self) -> float:
        """Share of served narratives with every role correctly attributed."""
        served = self.served
        if not served:
            return 0.0
        return sum(1 for row in served if row.attribution_ok) / len(served)

    @property
    def mean_agreement(self) -> float:
        return self._mean("agreement")

    @property
    def gold_drift(self) -> tuple[NarrativeCaseResult, ...]:
        """Cases where the fact pack and the gold set disagree."""
        return tuple(row for row in self.results if not row.pack_matches_gold)

    @property
    def trustworthy(self) -> bool:
        """
        Whether the run's numbers can be believed at all.

        A gold set that has drifted from the payloads measures the wrong
        thing; every other metric in the report is meaningless until that is
        fixed, so it is surfaced as a single flag rather than buried.
        """
        return not self.gold_drift

    def _mean(self, attribute: str) -> float:
        served = self.served
        return mean(getattr(row, attribute) for row in served) if served else 0.0

    def summary(self) -> str:
        lines = [
            f"Narrative evaluation{f' [{self.label}]' if self.label else ''}",
            f"  gold set          : {self.gold_set.name} ({self.cases} cases)",
            f"  service rate      : {self.service_rate:.0%} "
            f"({len(self.served)}/{self.cases} served, "
            f"{self.abstention_rate:.0%} withheld)",
            f"  served-wrong rate : {self.served_wrong_rate:.1%} "
            f"(target <= {SERVED_WRONG_TARGET:.0%}) "
            f"{'PASS' if self.meets_served_wrong_target else 'FAIL'}",
            f"  faithfulness      : {self.faithfulness:.3f}",
            f"  answer relevance  : {self.answer_relevance:.3f}",
            f"  citation coverage : {self.citation_coverage:.3f}",
            f"  attribution ok    : {self.attribution_rate:.0%} of served",
            f"  weak citation     : {self.weak_citation_rate:.1%} of served "
            f"(correct, but too few distinct records to check)",
            f"  mean agreement    : {self.mean_agreement:.3f}",
        ]
        if not self.trustworthy:
            lines.append(
                f"  !! GOLD DRIFT     : {len(self.gold_drift)} case(s) disagree with "
                f"the fact pack; metrics above are not trustworthy"
            )
        if self.config_summary:
            rendered = ", ".join(f"{k}={v}" for k, v in self.config_summary.items())
            lines.append(f"  configuration     : {rendered}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable form, for a frozen results report."""
        return {
            "label": self.label,
            "gold_set": self.gold_set.name,
            "cases": self.cases,
            "service_rate": round(self.service_rate, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "served_wrong_rate": round(self.served_wrong_rate, 4),
            "served_wrong_target": SERVED_WRONG_TARGET,
            "meets_served_wrong_target": self.meets_served_wrong_target,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "citation_coverage": round(self.citation_coverage, 4),
            "attribution_rate": round(self.attribution_rate, 4),
            "weak_citation_rate": round(self.weak_citation_rate, 4),
            "mean_agreement": round(self.mean_agreement, 4),
            "trustworthy": self.trustworthy,
            "configuration": dict(self.config_summary),
            "results": [
                {
                    "case_id": row.case_id,
                    "served": row.served,
                    "faithfulness": round(row.faithfulness, 4),
                    "answer_relevance": round(row.answer_relevance, 4),
                    "citation_coverage": round(row.citation_coverage, 4),
                    "citations": row.citations,
                    "meets_citation_floor": row.meets_citation_floor,
                    "attribution_ok": row.attribution_ok,
                    "served_wrong": row.served_wrong,
                    "weak_citation": row.weak_citation,
                    "agreement": round(row.agreement, 4),
                    "missing_required": list(row.missing_required),
                    "unfaithful": [round(v, 2) for v in row.unfaithful],
                    "misattributed": list(row.misattributed),
                    "abstain_reason": row.abstain_reason,
                    # A served-wrong answer is the one thing a reviewer must
                    # be able to read for themselves; a rate with no example
                    # behind it cannot be audited or disputed.
                    "text": (
                        row.text if row.served_wrong or row.weak_citation else None
                    ),
                }
                for row in self.results
            ],
        }


def evaluate_narratives(
    gold_set: NarrativeGoldSet,
    narrator: ReliableNarrator,
    *,
    label: str = "",
    config_summary: Mapping[str, object] | None = None,
) -> NarrativeEvaluationReport:
    """
    Run every gold case through the reliability layer and score the results.

    Args:
        gold_set: the frozen cases, carrying independent ground truth.
        narrator: the system under test.
        label: a tag for the report header.
        config_summary: settings worth recording beside the numbers, since a
            served-wrong rate means nothing without the sampling parameters
            that produced it.

    Returns:
        A :class:`NarrativeEvaluationReport`. Check ``trustworthy`` before
        reading any metric.
    """
    results = []
    for case in gold_set.cases:
        recommendation = case.load_recommendation()
        pack = build_facts(recommendation)
        outcome = narrator.explain(recommendation)
        results.append(_score(case, pack, outcome))
        logger.info("%s", results[-1].describe())

    return NarrativeEvaluationReport(
        results=tuple(results),
        gold_set=gold_set.source,
        label=label,
        config_summary=dict(config_summary or {}),
    )


def _score(case: NarrativeCase, pack: FactPack, outcome) -> NarrativeCaseResult:
    """Score one case against gold truth that never passed through the pack."""
    matches, diff = _compare_pack_to_gold(case, pack)
    consistency = outcome.consistency
    samples_total = consistency.k if consistency else 0
    samples_usable = len(consistency.usable) if consistency else 0
    corrective_added = len(outcome.correction.added) if outcome.correction else 0

    if not outcome.served or outcome.narrative is None:
        return NarrativeCaseResult(
            case_id=case.case_id,
            recommendation_id=case.recommendation_id,
            served=False,
            text="",
            unfaithful=(),
            faithfulness=0.0,
            answer_relevance=0.0,
            missing_required=case.required_facts,
            citation_coverage=0.0,
            citations=0,
            min_citations=case.min_citations,
            attribution_ok=True,
            misattributed=(),
            agreement=outcome.agreement,
            samples_usable=samples_usable,
            samples_total=samples_total,
            corrective_added=corrective_added,
            pack_matches_gold=matches,
            pack_gold_diff=diff,
            abstain_reason=outcome.reason,
        )

    narrative = outcome.narrative
    report = narrative.verification
    stated = tuple(report.numbers_found) if report else ()
    gold = case.gold_figures

    unfaithful = tuple(
        value for value in stated if not _matches_any(value, gold)
    )
    faithfulness = (
        (len(stated) - len(unfaithful)) / len(stated) if stated else 1.0
    )

    missing = tuple(
        key
        for key in case.required_facts
        if not _matches_any(case.expected_values[key], stated)
    )
    relevance = (
        (len(case.required_facts) - len(missing)) / len(case.required_facts)
        if case.required_facts
        else 1.0
    )

    attribution = check_attribution(
        narrative.text,
        pack,
        absolute_tolerance=ABSOLUTE_TOLERANCE,
        relative_tolerance=RELATIVE_TOLERANCE,
    )

    return NarrativeCaseResult(
        case_id=case.case_id,
        recommendation_id=case.recommendation_id,
        served=True,
        text=narrative.text,
        unfaithful=unfaithful,
        faithfulness=faithfulness,
        answer_relevance=relevance,
        missing_required=missing,
        citation_coverage=report.citation_coverage if report else 0.0,
        citations=len(narrative.citations),
        min_citations=case.min_citations,
        attribution_ok=attribution.ok,
        misattributed=tuple(m.role_key for m in attribution.misattributions),
        agreement=outcome.agreement,
        samples_usable=samples_usable,
        samples_total=samples_total,
        corrective_added=corrective_added,
        pack_matches_gold=matches,
        pack_gold_diff=diff,
    )


#: Public name for scoring one outcome against its gold case (used by the live demo,
#: which scores a single answer the same way the harness scores all 24).
score_case = _score


def _compare_pack_to_gold(
    case: NarrativeCase, pack: FactPack
) -> tuple[bool, tuple[str, ...]]:
    """
    Check the fact pack against the gold set's independent record.

    The two are built from the same payload by different code, so they should
    agree on every shared key. When they do not, the gold set has drifted from
    the payloads and every metric in the run is suspect.
    """
    by_key = {fact.key: fact.value for fact in pack.facts}
    differences = []
    for key, expected in case.expected_values.items():
        actual = by_key.get(key)
        if actual is None:
            differences.append(f"{key}: gold {expected:,.2f}, absent from pack")
        elif not _close(actual, expected):
            differences.append(f"{key}: gold {expected:,.2f} vs pack {actual:,.2f}")
    return not differences, tuple(differences)


def _matches_any(value: float, candidates: Sequence[float]) -> bool:
    return any(_close(value, candidate) for candidate in candidates)


def _close(left: float, right: float) -> bool:
    tolerance = max(ABSOLUTE_TOLERANCE, abs(right) * RELATIVE_TOLERANCE)
    return abs(abs(left) - abs(right)) <= tolerance
