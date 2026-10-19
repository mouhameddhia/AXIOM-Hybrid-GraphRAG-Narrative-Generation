"""
Few-shot examples for the query-plan prompt.

Examples live in a JSON file (``data/cypher_examples.json``), not a Python
literal, for the same reason ``data/gold_set.json`` does: they are content,
editable without touching code. Every example is parsed into a real
:class:`~src.text_to_query.plan.QueryPlan` and validated against the *live*
schema when loaded -- an example referencing a label or property that no
longer exists fails the load loudly, rather than teaching the model a pattern
the current graph would reject.

Selection ranks examples by keyword overlap with the question, capped at a
configured count. Cheap and adequate for a handful of examples, with a clear
upgrade path to embedding similarity if the bank grows.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import TextToCypherConfig
from src.text_to_query.plan import PlanError, QueryPlan
from src.text_to_query.schema import GraphSchema

__all__ = ["CypherExampleBank", "CypherExampleError", "QueryPlanExample"]

logger = logging.getLogger(__name__)


class CypherExampleError(ValueError):
    """Raised when the example file is malformed or references a stale schema."""


@dataclass(frozen=True, slots=True)
class QueryPlanExample:
    """One (question, plan) pair used to few-shot the prompt."""

    example_id: str
    question: str
    plan: Mapping[str, Any]
    note: str = ""

    def render(self) -> str:
        """Render as the exact text the model should learn to produce."""
        return f"Question: {self.question}\nJSON plan: {json.dumps(self.plan)}"


class CypherExampleBank:
    """A validated, rankable collection of few-shot examples."""

    __slots__ = ("_examples",)

    def __init__(self, examples: Sequence[QueryPlanExample]) -> None:
        self._examples: tuple[QueryPlanExample, ...] = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    @classmethod
    def load(cls, path: Path, schema: GraphSchema | None = None) -> "CypherExampleBank":
        """
        Read and validate the example bank.

        Args:
            path: the JSON file.
            schema: when supplied, every example's plan is validated against
                this schema's labels and properties.

        Raises:
            CypherExampleError: the file is malformed, or an example is not a
                valid plan for the current schema.
        """
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CypherExampleError(f"Example bank not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise CypherExampleError(
                f"{path} is not valid JSON (line {exc.lineno}): {exc.msg}"
            ) from exc

        raw_examples = payload.get("examples")
        if not isinstance(raw_examples, list) or not raw_examples:
            raise CypherExampleError(f"{path} contains no 'examples' array")

        bank = cls(
            [cls._parse(entry, index) for index, entry in enumerate(raw_examples)]
        )
        if schema is not None:
            bank.validate_against(schema)
        logger.info("Loaded %d query-plan example(s) from %s", len(bank), path.name)
        return bank

    @staticmethod
    def _parse(entry: object, index: int) -> QueryPlanExample:
        if not isinstance(entry, Mapping):
            raise CypherExampleError(f"Example at index {index} is not an object")
        question = entry.get("question")
        plan = entry.get("plan")
        if not question or not isinstance(plan, Mapping):
            raise CypherExampleError(
                f"Example at index {index} needs a 'question' and a 'plan' object"
            )
        return QueryPlanExample(
            example_id=str(entry.get("id") or f"example_{index}"),
            question=str(question),
            plan=dict(plan),
            note=str(entry.get("note", "")),
        )

    def validate_against(self, schema: GraphSchema) -> None:
        """
        Fail loudly if any example plan is invalid for the current schema.

        An example teaching a label that no longer exists is worse than no
        example: it actively demonstrates something the graph will reject.
        """
        problems: list[str] = []
        for example in self._examples:
            try:
                QueryPlan.from_dict(example.plan).validate_against(schema)
            except PlanError as exc:
                problems.append(f"{example.example_id}: {exc}")

        if problems:
            raise CypherExampleError(
                "One or more query-plan examples are invalid for the current "
                "schema:\n  " + "\n  ".join(problems)
            )

    def select(
        self, question: str, config: TextToCypherConfig
    ) -> tuple[QueryPlanExample, ...]:
        """
        Rank examples by keyword overlap with ``question``, best first.

        Falls back to the first ``max_few_shot_examples`` when nothing
        overlaps, so the prompt always demonstrates the output format.
        """
        if config.max_few_shot_examples <= 0 or not self._examples:
            return ()

        question_tokens = set(question.lower().split())
        scored = [
            (len(question_tokens & set(example.question.lower().split())), example)
            for example in self._examples
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        if scored[0][0] == 0:
            return self._examples[: config.max_few_shot_examples]
        return tuple(example for _, example in scored[: config.max_few_shot_examples])

    @staticmethod
    def render(examples: Sequence[QueryPlanExample]) -> str:
        """Render examples as text ready to drop into the prompt."""
        return "\n\n".join(example.render() for example in examples)
