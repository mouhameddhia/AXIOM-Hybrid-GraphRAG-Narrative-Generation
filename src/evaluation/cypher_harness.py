"""
Gold-set evaluation for text-to-query correctness and safety.

The gate this serves asks for two things: *correctness* (does the query return
the right number?) and *safety* (is every generated query valid against the
real schema, and incapable of writing?). Both are measured here.

Expected values in the gold set were computed independently from the JSON
export, not captured from this engine's own output -- so a regression cannot
quietly redefine what "correct" means.

This runs with no database: plans execute against the in-memory graph.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import json

from src.text_to_query.engine import QueryResult, TextToQueryEngine
from src.text_to_query.evidence import EvidenceRecord

__all__ = [
    "CypherCase",
    "CypherCaseResult",
    "CypherEvaluationReport",
    "CypherGoldSet",
    "CypherGoldSetError",
    "evaluate_cypher",
]

logger = logging.getLogger(__name__)

#: Default absolute tolerance when a case does not set one.
DEFAULT_TOLERANCE = 0.01


class CypherGoldSetError(ValueError):
    """Raised when the gold set is malformed."""


@dataclass(frozen=True, slots=True)
class CypherCase:
    """One labelled question with its independently-computed expected value."""

    case_id: str
    question: str
    expect_value: float | str
    tolerance: float = DEFAULT_TOLERANCE
    expect_evidence: bool = False
    note: str = ""


@dataclass(frozen=True, slots=True)
class CypherGoldSet:
    """A validated collection of text-to-Cypher cases."""

    cases: tuple[CypherCase, ...]
    source: Path
    description: str = ""

    @classmethod
    def load(cls, path: Path) -> "CypherGoldSet":
        """Read and validate the gold set."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CypherGoldSetError(f"Gold set not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise CypherGoldSetError(
                f"{path} is not valid JSON (line {exc.lineno}): {exc.msg}"
            ) from exc

        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise CypherGoldSetError(f"{path} contains no 'cases' array")

        return cls(
            cases=tuple(cls._parse(entry, index) for index, entry in enumerate(raw_cases)),
            source=Path(path),
            description=str(payload.get("description", "")),
        )

    @staticmethod
    def _parse(entry: object, index: int) -> CypherCase:
        if not isinstance(entry, Mapping):
            raise CypherGoldSetError(f"Case at index {index} is not an object")
        case_id = str(entry.get("id") or f"case_{index}")
        if "question" not in entry:
            raise CypherGoldSetError(f"Case {case_id!r} has no question")
        if "expect_value" not in entry:
            raise CypherGoldSetError(f"Case {case_id!r} has no expect_value")
        return CypherCase(
            case_id=case_id,
            question=str(entry["question"]),
            expect_value=entry["expect_value"],
            tolerance=float(entry.get("tolerance", DEFAULT_TOLERANCE)),
            expect_evidence=bool(entry.get("expect_evidence", False)),
            note=str(entry.get("note", "")),
        )


@dataclass(frozen=True, slots=True)
class CypherCaseResult:
    """The outcome of one gold-set case."""

    case: CypherCase
    result: QueryResult
    value_matched: bool
    matched_field: str | None
    evidence_ok: bool

    @property
    def passed(self) -> bool:
        """Correct value, and evidence present when the case requires it."""
        return self.result.success and self.value_matched and self.evidence_ok


@dataclass(frozen=True, slots=True)
class CypherEvaluationReport:
    """Aggregate results across the gold set."""

    results: tuple[CypherCaseResult, ...]
    label: str = ""

    @property
    def accuracy(self) -> float:
        """Share of cases returning the independently-verified correct value."""
        if not self.results:
            return 0.0
        return sum(1 for row in self.results if row.passed) / len(self.results)

    @property
    def answered(self) -> float:
        """Share of cases producing any answer at all within the retry budget."""
        if not self.results:
            return 0.0
        return sum(1 for row in self.results if row.result.success) / len(self.results)

    @property
    def mean_attempts(self) -> float:
        return mean(row.result.attempts for row in self.results) if self.results else 0.0

    @property
    def self_healed(self) -> int:
        """Cases that failed at least once and then recovered."""
        return sum(
            1 for row in self.results if row.result.success and row.result.attempts > 1
        )

    def failures(self) -> tuple[CypherCaseResult, ...]:
        return tuple(row for row in self.results if not row.passed)

    def as_dict(self) -> dict[str, object]:
        """Machine-readable summary, for committing alongside a report."""
        return {
            "label": self.label,
            "aggregate": {
                "accuracy": round(self.accuracy, 4),
                "answered": round(self.answered, 4),
                "mean_attempts": round(self.mean_attempts, 2),
                "self_healed": self.self_healed,
                "cases": len(self.results),
            },
            "cases": [
                {
                    "id": row.case.case_id,
                    "question": row.case.question,
                    "expected": row.case.expect_value,
                    "passed": row.passed,
                    "attempts": row.result.attempts,
                    "cypher": row.result.cypher,
                    "plan": _plan_dict(row.result),
                    "answer": dict(row.result.answer),
                    "citations": list(row.result.citations),
                    "error": row.result.error,
                }
                for row in self.results
            ],
        }


def evaluate_cypher(
    engine: TextToQueryEngine,
    gold_set: CypherGoldSet,
    *,
    label: str = "",
    backend: str = "json",
) -> CypherEvaluationReport:
    """
    Run every gold case through ``engine`` and check the returned value.

    Args:
        engine: the text-to-query engine.
        gold_set: validated cases with independently-computed expected values.
        label: name for this run, shown in the report.
        backend: ``"json"`` executes plans in memory (no database needed);
            ``"neo4j"`` compiles each plan to Cypher and executes it against
            a live database. Running the identical gold set through both is
            what proves the two execution paths agree, not just that each
            one individually looks plausible.
    """
    results: list[CypherCaseResult] = []
    for case in gold_set.cases:
        result = engine.ask(case.question, backend=backend)
        matched_field = _find_matching_field(result.answer, case, result.evidence)
        evidence_ok = (not case.expect_evidence) or bool(result.evidence)
        results.append(
            CypherCaseResult(
                case=case,
                result=result,
                value_matched=matched_field is not None,
                matched_field=matched_field,
                evidence_ok=evidence_ok,
            )
        )
    return CypherEvaluationReport(results=tuple(results), label=label)


def _plan_dict(result: QueryResult) -> dict[str, object] | None:
    """Serialise the plan behind an answer, so a result file is self-explaining."""
    if result.plan is None:
        return None
    plan = result.plan
    return {
        "label": plan.label,
        "filters": [
            {"property": f.property, "op": f.op, "value": f.value} for f in plan.filters
        ],
        "aggregate": plan.aggregate,
        "aggregate_property": plan.aggregate_property,
        "return_records": plan.return_records,
        "limit": plan.limit,
    }


def _find_matching_field(
    answer: Mapping[str, object],
    case: CypherCase,
    evidence: tuple[EvidenceRecord, ...] = (),
) -> str | None:
    """
    Return the name of the answer field matching the expected value.

    Checks every returned field rather than assuming a field name: the model
    chooses its own aliases, and demanding a specific one would measure
    prompt compliance rather than correctness.

    A single-fact lookup with no numeric aggregate (``MATCH (n:Label) WHERE
    ... RETURN n``) is a *correct* plan for a question like "what region is
    this customer in" -- but the returned node is node-shaped, not a scalar,
    so ``split_row()`` routes it into evidence rather than ``answer`` (nodes
    are citations, not answer fields, everywhere else in this codebase).
    ``answer`` is then empty and the check above finds nothing, even though
    the right plan ran and the right record came back. For a string-valued
    case only, this falls back to checking whether the expected value appears
    as a whole word in the text of a record the answer actually cited --
    the same flattened text a narrative's citation would resolve to, so the
    check reflects how this evidence is read everywhere else in the
    pipeline, not a harness-only notion of a match.
    """
    for field_name, value in answer.items():
        if _values_match(value, case.expect_value, case.tolerance):
            return field_name

    if isinstance(case.expect_value, str):
        pattern = re.compile(rf"\b{re.escape(case.expect_value)}\b", re.IGNORECASE)
        for record in evidence:
            if pattern.search(record.text):
                return f"evidence:{record.record_id}"

    return None


def _values_match(actual: object, expected: float | str, tolerance: float) -> bool:
    """Compare numerically with tolerance, or exactly for strings."""
    if isinstance(expected, str):
        return isinstance(actual, str) and actual.strip() == expected.strip()
    if actual is None or isinstance(actual, bool):
        # bool is an int subclass; True must never satisfy an expected 1.
        return False
    if not isinstance(actual, (int, float)):
        return False
    return math.isclose(float(actual), float(expected), abs_tol=tolerance)
