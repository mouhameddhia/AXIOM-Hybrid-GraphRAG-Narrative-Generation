"""
Question-bank loading and execution.

The question bank is broader and softer than the scored gold set. The gold set
measures ranking quality against one answering record per question; this bank
explores behaviour across the whole surface the onboarding plan describes --
reference prices, applicable policies, comparable deals, waterfall components,
win/loss reasoning, multi-hop traversal, and honest failure on out-of-domain
input.

Expectations are deliberately graded:

``expect_records``
    Specific ids that must appear. Used where exactly one record answers.
``expect_types``
    A record type that must appear. Used where several records are equally
    valid answers and naming one would be arbitrary.
``expect_no_anchor``
    The question must ground no entity. This is the honest-failure check: an
    unknown identifier must resolve to nothing rather than drift to a
    superficially similar record.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.pipeline.retriever_pipeline import HybridRetrievalPipeline, RetrievalOutcome
from src.retrieval.corpus import Corpus

__all__ = [
    "Category",
    "Probe",
    "ProbeOutcome",
    "QuestionBank",
    "QuestionBankError",
    "run_bank",
]

logger = logging.getLogger(__name__)

#: Language codes, mapped to the field holding that language's question.
LANGUAGES: Mapping[str, str] = {"en": "en", "ar": "ar"}


class QuestionBankError(ValueError):
    """Raised when the question bank is malformed or references unknown records."""


@dataclass(frozen=True, slots=True)
class Probe:
    """One question with its graded expectations."""

    probe_id: str
    category_id: str
    questions: Mapping[str, str]
    expect_records: tuple[str, ...] = ()
    expect_types: tuple[str, ...] = ()
    expect_no_anchor: bool = False
    probes: str = ""

    @property
    def has_expectations(self) -> bool:
        """Whether anything about this probe can be checked automatically."""
        return bool(self.expect_records or self.expect_types or self.expect_no_anchor)


@dataclass(frozen=True, slots=True)
class Category:
    """A themed group of probes."""

    category_id: str
    title: str
    onboarding_ref: str
    why: str
    probes: tuple[Probe, ...]


@dataclass(frozen=True, slots=True)
class QuestionBank:
    """A validated collection of categories."""

    categories: tuple[Category, ...]
    source: Path
    description: str = ""

    @property
    def all_probes(self) -> tuple[Probe, ...]:
        return tuple(probe for category in self.categories for probe in category.probes)

    @property
    def languages(self) -> tuple[str, ...]:
        seen = {code for probe in self.all_probes for code in probe.questions}
        return tuple(code for code in LANGUAGES if code in seen)

    @classmethod
    def load(cls, path: Path, corpus: Corpus | None = None) -> "QuestionBank":
        """
        Read and validate the question bank.

        When ``corpus`` is supplied, every ``expect_records`` id is checked
        against it. A stale expectation therefore fails loudly instead of
        quietly reporting a retrieval miss that never happened.
        """
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise QuestionBankError(f"Question bank not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise QuestionBankError(
                f"{path} is not valid JSON (line {exc.lineno}): {exc.msg}"
            ) from exc

        raw_categories = payload.get("categories")
        if not isinstance(raw_categories, list) or not raw_categories:
            raise QuestionBankError(f"{path} contains no 'categories' array")

        categories = tuple(cls._parse_category(entry, index) for index, entry in enumerate(raw_categories))
        bank = cls(
            categories=categories,
            source=Path(path),
            description=str(payload.get("description", "")),
        )
        if corpus is not None:
            bank.validate_against(corpus)
        return bank

    @staticmethod
    def _parse_category(entry: object, index: int) -> Category:
        if not isinstance(entry, Mapping):
            raise QuestionBankError(f"Category at index {index} is not an object")
        category_id = str(entry.get("id") or f"category_{index}")

        raw_questions = entry.get("questions")
        if not isinstance(raw_questions, list) or not raw_questions:
            raise QuestionBankError(f"Category {category_id!r} has no questions")

        probes: list[Probe] = []
        for position, raw in enumerate(raw_questions):
            if not isinstance(raw, Mapping):
                raise QuestionBankError(
                    f"Question {position} in category {category_id!r} is not an object"
                )
            questions = {
                code: str(raw[field_name]).strip()
                for code, field_name in LANGUAGES.items()
                if raw.get(field_name)
            }
            if not questions:
                raise QuestionBankError(
                    f"Question {position} in category {category_id!r} has no text "
                    f"in any known language"
                )
            probes.append(
                Probe(
                    probe_id=str(raw.get("id") or f"{category_id}_{position}"),
                    category_id=category_id,
                    questions=questions,
                    expect_records=tuple(raw.get("expect_records") or ()),
                    expect_types=tuple(raw.get("expect_types") or ()),
                    expect_no_anchor=bool(raw.get("expect_no_anchor", False)),
                    probes=str(raw.get("probes", "")),
                )
            )

        return Category(
            category_id=category_id,
            title=str(entry.get("title", category_id)),
            onboarding_ref=str(entry.get("onboarding_ref", "")),
            why=str(entry.get("why", "")),
            probes=tuple(probes),
        )

    def validate_against(self, corpus: Corpus) -> None:
        """Fail loudly if an expected record id is absent from the corpus."""
        unknown = [
            f"{probe.probe_id} -> {record_id}"
            for probe in self.all_probes
            for record_id in probe.expect_records
            if corpus.by_record_id(record_id) is None
        ]
        if unknown:
            raise QuestionBankError(
                "Question bank expects records absent from the corpus "
                "(the graph or identifier scheme may have changed):\n  "
                + "\n  ".join(unknown)
            )
        logger.info(
            "Question bank validated: %d categories, %d questions",
            len(self.categories),
            len(self.all_probes),
        )


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    """The result of running one probe in one language."""

    probe: Probe
    language: str
    question: str
    outcome: RetrievalOutcome
    missing_records: tuple[str, ...]
    missing_types: tuple[str, ...]
    anchor_violation: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ranked(self) -> tuple[str, ...]:
        return self.outcome.record_ids

    @property
    def checked(self) -> bool:
        """Whether this probe asserted anything automatically checkable."""
        return self.probe.has_expectations

    @property
    def passed(self) -> bool:
        """
        Whether every stated expectation held.

        A probe with no expectations is reported as passing: it is an
        exploratory question included to be read, not scored.
        """
        return not (self.missing_records or self.missing_types or self.anchor_violation)

    def rank_of(self, record_id: str) -> int | None:
        """1-based position of ``record_id`` in the final ranking."""
        for position, candidate in enumerate(self.ranked, start=1):
            if candidate == record_id:
                return position
        return None


def run_probe(
    pipeline: HybridRetrievalPipeline, probe: Probe, language: str, top_k: int
) -> ProbeOutcome:
    """Run one probe in one language and evaluate its expectations."""
    question = probe.questions[language]
    outcome = pipeline.run(question, top_k=top_k)
    ranked = set(outcome.record_ids)
    returned_types = {result.record_type for result in outcome.final}

    missing_records = tuple(r for r in probe.expect_records if r not in ranked)
    missing_types = tuple(t for t in probe.expect_types if t not in returned_types)
    anchor_violation = probe.expect_no_anchor and bool(outcome.anchors)

    notes: list[str] = []
    if anchor_violation:
        notes.append(
            "expected no anchor, but grounded: "
            + ", ".join(f"{a.mention}->{a.record_id}" for a in outcome.anchors[:3])
        )
    if not outcome.final:
        notes.append("no evidence returned")

    return ProbeOutcome(
        probe=probe,
        language=language,
        question=question,
        outcome=outcome,
        missing_records=missing_records,
        missing_types=missing_types,
        anchor_violation=anchor_violation,
        notes=tuple(notes),
    )


def run_bank(
    pipeline: HybridRetrievalPipeline,
    bank: QuestionBank,
    *,
    top_k: int = 5,
    languages: Sequence[str] | None = None,
    category_ids: Sequence[str] | None = None,
) -> list[ProbeOutcome]:
    """
    Run every probe in the bank.

    Args:
        pipeline: the assembled retrieval pipeline.
        bank: validated question bank.
        top_k: results to request per question.
        languages: restrict to these codes; defaults to every language present.
        category_ids: restrict to these categories; defaults to all.

    Returns:
        One :class:`ProbeOutcome` per (probe, language) pair, in bank order.
    """
    selected_languages = tuple(languages) if languages else bank.languages
    wanted = set(category_ids) if category_ids else None

    results: list[ProbeOutcome] = []
    for category in bank.categories:
        if wanted is not None and category.category_id not in wanted:
            continue
        for probe in category.probes:
            for language in selected_languages:
                if language in probe.questions:
                    results.append(run_probe(pipeline, probe, language, top_k))
    return results
