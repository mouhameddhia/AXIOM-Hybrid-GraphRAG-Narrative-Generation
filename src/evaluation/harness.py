"""
Gold-set evaluation harness for the retrieval pipeline.

Loads a labelled question set, runs it through a configured pipeline, and
reports recall@k, MRR, nDCG@k and related metrics -- per case and in aggregate,
per language.

Two design decisions keep the numbers honest:

**Gold ids are validated against the live corpus.** A stale or mistyped handle
fails the run immediately. Silently scoring zero for a record that no longer
exists would look like a retrieval regression and send the next investigation
in entirely the wrong direction.

**Graded relevance.** The record that actually answers the question carries a
higher gain than a supporting one, so a ranking that surfaces context above the
answer scores lower even though both are "relevant".
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from src.evaluation.metrics import (
    average_precision,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.pipeline.retriever_pipeline import HybridRetrievalPipeline
from src.retrieval.corpus import Corpus

__all__ = [
    "CaseResult",
    "EvaluationReport",
    "GoldCase",
    "GoldSet",
    "GoldSetError",
    "evaluate",
]

logger = logging.getLogger(__name__)

#: Graded gains. The answering record is worth substantially more than context,
#: so nDCG separates "found the answer" from "found the neighbourhood".
PRIMARY_GAIN = 3.0
SUPPORTING_GAIN = 1.0

#: Language codes the gold set may carry, mapped to their question field.
QUESTION_FIELDS: Mapping[str, str] = {"en": "question_en", "ar": "question_ar"}


class GoldSetError(ValueError):
    """Raised when the gold set is malformed or references unknown records."""


@dataclass(frozen=True, slots=True)
class GoldCase:
    """One labelled question with its expected evidence."""

    case_id: str
    questions: Mapping[str, str]
    primary: str
    supporting: tuple[str, ...]
    probes: str = ""

    @property
    def relevant(self) -> frozenset[str]:
        """Every record counted as relevant for recall and MRR."""
        return frozenset({self.primary, *self.supporting})

    @property
    def gains(self) -> Mapping[str, float]:
        """Graded relevance for nDCG."""
        return {
            self.primary: PRIMARY_GAIN,
            **{record_id: SUPPORTING_GAIN for record_id in self.supporting},
        }


@dataclass(frozen=True, slots=True)
class GoldSet:
    """A validated collection of labelled cases."""

    cases: tuple[GoldCase, ...]
    source: Path
    description: str = ""

    @property
    def languages(self) -> tuple[str, ...]:
        """Language codes present across the whole set, in a stable order."""
        seen = {code for case in self.cases for code in case.questions}
        return tuple(code for code in QUESTION_FIELDS if code in seen)

    @classmethod
    def load(cls, path: Path, corpus: Corpus | None = None) -> "GoldSet":
        """
        Read and validate a gold set.

        Args:
            path: the JSON gold-set file.
            corpus: when supplied, every referenced record id is checked
                against it and an unknown id raises.

        Raises:
            GoldSetError: the file is unreadable, malformed, or references a
                record that does not exist in ``corpus``.
        """
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise GoldSetError(f"Gold set not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise GoldSetError(f"{path} is not valid JSON (line {exc.lineno})") from exc

        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise GoldSetError(f"{path} contains no 'cases' array")

        cases = tuple(cls._parse_case(entry, index) for index, entry in enumerate(raw_cases))
        gold_set = cls(
            cases=cases, source=Path(path), description=str(payload.get("description", ""))
        )
        if corpus is not None:
            gold_set.validate_against(corpus)
        return gold_set

    @staticmethod
    def _parse_case(entry: object, index: int) -> GoldCase:
        if not isinstance(entry, Mapping):
            raise GoldSetError(f"Case at index {index} is not an object")
        case_id = str(entry.get("id") or f"case_{index}")
        primary = entry.get("primary")
        if not primary:
            raise GoldSetError(f"Case {case_id!r} has no 'primary' record id")

        questions = {
            code: str(entry[field]).strip()
            for code, field in QUESTION_FIELDS.items()
            if entry.get(field)
        }
        if not questions:
            raise GoldSetError(
                f"Case {case_id!r} has no question in any known language "
                f"({', '.join(QUESTION_FIELDS.values())})"
            )

        supporting = entry.get("supporting") or []
        if not isinstance(supporting, list):
            raise GoldSetError(f"Case {case_id!r}: 'supporting' must be an array")

        return GoldCase(
            case_id=case_id,
            questions=questions,
            primary=str(primary),
            supporting=tuple(str(item) for item in supporting),
            probes=str(entry.get("probes", "")),
        )

    def validate_against(self, corpus: Corpus) -> None:
        """Fail loudly if any labelled record id is absent from the corpus."""
        unknown: list[str] = []
        for case in self.cases:
            for record_id in case.relevant:
                if corpus.by_record_id(record_id) is None:
                    unknown.append(f"{case.case_id} -> {record_id}")
        if unknown:
            raise GoldSetError(
                "Gold set references records absent from the corpus "
                "(the graph or the identifier scheme may have changed):\n  "
                + "\n  ".join(unknown)
            )
        logger.info(
            "Gold set validated: %d cases, %d record references",
            len(self.cases),
            sum(len(case.relevant) for case in self.cases),
        )


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Metrics for one case in one language."""

    case_id: str
    language: str
    question: str
    ranked: tuple[str, ...]
    primary: str
    primary_rank: int | None
    reciprocal_rank: float
    recall: float
    precision: float
    average_precision: float
    ndcg: float
    hit: float
    latency_ms: float

    @property
    def found_primary(self) -> bool:
        """Whether the answering record appeared at all within the top-k."""
        return self.primary_rank is not None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregate results across the whole gold set."""

    results: tuple[CaseResult, ...]
    k: int
    label: str = ""
    config_summary: Mapping[str, object] = field(default_factory=dict)

    def _mean(self, attribute: str, results: Sequence[CaseResult] | None = None) -> float:
        rows = self.results if results is None else results
        return mean(getattr(row, attribute) for row in rows) if rows else 0.0

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank over every case and language."""
        return self._mean("reciprocal_rank")

    @property
    def recall_at_k(self) -> float:
        return self._mean("recall")

    @property
    def precision_at_k(self) -> float:
        return self._mean("precision")

    @property
    def map_score(self) -> float:
        return self._mean("average_precision")

    @property
    def ndcg_at_k(self) -> float:
        return self._mean("ndcg")

    @property
    def hit_rate(self) -> float:
        return self._mean("hit")

    @property
    def primary_found_rate(self) -> float:
        """Share of cases where the answering record was retrieved at all."""
        if not self.results:
            return 0.0
        return sum(1 for row in self.results if row.found_primary) / len(self.results)

    @property
    def mean_latency_ms(self) -> float:
        return self._mean("latency_ms")

    def by_language(self) -> Mapping[str, "EvaluationReport"]:
        """Split the report per language, for the bilingual comparison."""
        languages = {row.language for row in self.results}
        return {
            language: EvaluationReport(
                results=tuple(row for row in self.results if row.language == language),
                k=self.k,
                label=f"{self.label} [{language}]".strip(),
                config_summary=self.config_summary,
            )
            for language in sorted(languages)
        }

    def failures(self) -> tuple[CaseResult, ...]:
        """Cases whose answering record never surfaced -- the ones to work on."""
        return tuple(row for row in self.results if not row.found_primary)

    def as_dict(self) -> dict[str, object]:
        """Machine-readable summary, for committing results alongside a report."""
        return {
            "label": self.label,
            "k": self.k,
            "config": dict(self.config_summary),
            "aggregate": {
                "mrr": round(self.mrr, 4),
                f"recall@{self.k}": round(self.recall_at_k, 4),
                f"precision@{self.k}": round(self.precision_at_k, 4),
                f"ndcg@{self.k}": round(self.ndcg_at_k, 4),
                "map": round(self.map_score, 4),
                f"hit@{self.k}": round(self.hit_rate, 4),
                "primary_found_rate": round(self.primary_found_rate, 4),
                "mean_latency_ms": round(self.mean_latency_ms, 2),
            },
            "per_language": {
                language: {
                    "mrr": round(report.mrr, 4),
                    f"ndcg@{self.k}": round(report.ndcg_at_k, 4),
                    "primary_found_rate": round(report.primary_found_rate, 4),
                }
                for language, report in self.by_language().items()
            },
            "cases": [
                {
                    "id": row.case_id,
                    "language": row.language,
                    "primary": row.primary,
                    "primary_rank": row.primary_rank,
                    "rr": round(row.reciprocal_rank, 4),
                    f"ndcg@{self.k}": round(row.ndcg, 4),
                }
                for row in self.results
            ],
        }


def evaluate(
    pipeline: HybridRetrievalPipeline,
    gold_set: GoldSet,
    *,
    k: int = 10,
    languages: Iterable[str] | None = None,
    label: str = "",
    config_summary: Mapping[str, object] | None = None,
) -> EvaluationReport:
    """
    Run every gold case through ``pipeline`` and score the rankings.

    Args:
        pipeline: the assembled retrieval pipeline.
        gold_set: validated labelled cases.
        k: cut-off for the @k metrics and the pipeline's own ``top_k``.
        languages: restrict to these language codes; defaults to all present.
        label: name for this run, shown in comparison tables.
        config_summary: configuration values worth recording beside the scores.

    Returns:
        An :class:`EvaluationReport` holding per-case and aggregate metrics.
    """
    selected = tuple(languages) if languages is not None else gold_set.languages
    results: list[CaseResult] = []

    for case in gold_set.cases:
        for language in selected:
            question = case.questions.get(language)
            if not question:
                continue

            outcome = pipeline.run(question, top_k=k)
            ranked = outcome.record_ids
            relevant = case.relevant

            primary_rank = next(
                (
                    position
                    for position, record_id in enumerate(ranked, start=1)
                    if record_id == case.primary
                ),
                None,
            )
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    language=language,
                    question=question,
                    ranked=ranked,
                    primary=case.primary,
                    primary_rank=primary_rank,
                    # Reciprocal rank of the *answering* record specifically,
                    # not of any relevant record: finding context first is not
                    # the same as answering the question.
                    reciprocal_rank=1.0 / primary_rank if primary_rank else 0.0,
                    recall=recall_at_k(ranked, relevant, k),
                    precision=precision_at_k(ranked, relevant, k),
                    average_precision=average_precision(ranked, relevant),
                    ndcg=ndcg_at_k(ranked, case.gains, k),
                    hit=hit_at_k(ranked, relevant, k),
                    latency_ms=float(outcome.timings_ms.get("total", 0.0)),
                )
            )

    return EvaluationReport(
        results=tuple(results),
        k=k,
        label=label,
        config_summary=dict(config_summary or {}),
    )


def _reciprocal_rank_of_any(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    """Reciprocal rank of the first *any*-relevant hit (kept for comparison)."""
    return reciprocal_rank(ranked, relevant)
