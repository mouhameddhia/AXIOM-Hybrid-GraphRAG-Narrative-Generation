r"""
The self-healing text-to-query engine.

    question
       |
    prune schema (live, from the graph, with observed values)
       |
    select few-shot examples
       |
    build prompt --> LLM --> parse JSON query plan
       |
    validate plan against the live schema
       |                                    \
       | pass                                \ fail: exact error fed back, retry
       v
    execute the plan            -- against the in-memory graph (json), or
       |                           against the real database (neo4j)
    assemble evidence (retrieval's citation scheme)
       |
    QueryResult  (+ the equivalent Cypher, for review and for Neo4j)

Why a plan and not Cypher text
------------------------------
Cypher executes only against Neo4j, and this project started with only a JSON
export -- emitting Cypher text meant emitting something that could not be run.
The model instead emits a small structured plan, which
:mod:`src.text_to_query.plan` can evaluate two ways: directly against the
in-memory graph (``backend="json"``, no database needed), or compiled to
Cypher and executed against a real Neo4j (``backend="neo4j"``). Both paths
produce rows in the same shape, so evidence assembly and the self-heal loop
are identical either way -- only the last step differs.

The rule this module enforces: **the graph computes the numbers, the model
never does.** The model's only output is a description of *what to compute*.
Every value in ``QueryResult.answer`` was computed over real graph data,
never by the model reading a result and stating a number.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from src.config.settings import TextToCypherConfig
from src.graph.model import KnowledgeGraph
from src.graph.neo4j_client import Neo4jClient, Neo4jQueryError
from src.retrieval.corpus import RecordSerializer
from src.text_to_query.evidence import EvidenceRecord, assemble_evidence, split_row
from src.text_to_query.examples import CypherExampleBank
from src.text_to_query.llm_client import LLMClient, LLMRequestError
from src.text_to_query.plan import PlanError, QueryPlan, execute_plan
from src.text_to_query.schema import GraphSchema, build_schema_from_graph, prune_schema

__all__ = ["QueryResult", "TextToQueryEngine"]

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "You translate a question about a graph database into a JSON query plan.\n"
    "Answer with ONE JSON object and nothing else. Never write prose, never "
    "write Cypher.\n\n"
    "The plan format:\n"
    '{{"label": "<record type to match>",\n'
    ' "filters": [{{"property": "<name>", "op": "<=|!=|>|>=|<|<=|contains|'
    'starts_with|exists>", "value": <value>}}],\n'
    ' "aggregate": "<count|sum|avg|min|max>",   // omit to return records\n'
    ' "aggregate_property": "<name>",           // required unless count\n'
    ' "return_records": <true|false>,           // true to also cite a record\n'
    ' "limit": <number>}}                        // optional\n\n'
    "Rules:\n"
    "- Use only labels and properties from the schema below.\n"
    "- The schema shows observed values as property=a|b|c. Filter using "
    "exactly those values.\n"
    "- For a question asking how many, use aggregate count.\n"
    "- For a question asking for one specific fact, set return_records true "
    "so the answer can be cited.\n\n"
    "Schema:\n{schema}\n\n"
    "Examples:\n{examples}\n\n"
    "Question: {question}\n"
    "JSON plan:"
)

_RETRY_TEMPLATE = (
    "\n\nYour previous plan was:\n{previous}\n\n"
    "It failed with this exact error:\n{error}\n\n"
    "Fix it and answer again with only the corrected JSON plan."
)


@dataclass(frozen=True, slots=True)
class QueryResult:
    """The outcome of one natural-language question."""

    question: str
    success: bool
    plan: QueryPlan | None = None
    cypher: str | None = None
    answer: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[EvidenceRecord, ...] = ()
    attempts: int = 0
    error: str | None = None
    #: Which executor actually ran the plan -- "json" (in-memory graph) or
    #: "neo4j" (a live database) -- so a caller comparing both can label
    #: results without having tracked the call site itself.
    backend: str = "json"

    @property
    def citations(self) -> tuple[str, ...]:
        """The citation handles backing this answer, in first-seen order."""
        return tuple(record.record_id for record in self.evidence)

    @property
    def value(self) -> Any:
        """
        The single answer value, when there is exactly one.

        Convenience for the common case of an aggregate question; returns
        ``None`` when the query returned records rather than a scalar.
        """
        values = list(self.answer.values())
        return values[0] if len(values) == 1 else None


class TextToQueryEngine:
    """
    Turns a question into a validated query plan, runs it, and self-heals.

    Usage::

        engine = TextToQueryEngine(
            graph, serializer=serializer, llm_client=llm,
            examples=examples, config=settings.text_to_cypher,
        )
        result = engine.ask("How many opportunities were lost to a competitor?")
        print(result.value, result.citations, result.cypher)
    """

    __slots__ = (
        "_graph",
        "_schema",
        "_serializer",
        "_llm",
        "_examples",
        "_config",
        "_base_temperature",
        "_neo4j_client",
    )

    def __init__(
        self,
        graph: KnowledgeGraph,
        *,
        serializer: RecordSerializer,
        llm_client: LLMClient,
        examples: CypherExampleBank,
        config: TextToCypherConfig,
        base_temperature: float = 0.0,
        neo4j_client: Neo4jClient | None = None,
    ) -> None:
        """
        Args:
            graph: the knowledge graph -- the live schema is derived from it,
                and plans execute against it when ``backend="json"``.
            serializer: record serializer shared with retrieval, so citations
                are identical across both components.
            llm_client: anything satisfying :class:`LLMClient`.
            examples: the validated few-shot bank.
            config: retry limits and pruning thresholds.
            neo4j_client: optional. Required only to answer with
                ``backend="neo4j"`` -- without one, the engine still answers
                every question by executing plans against ``graph`` in
                memory, which needs no database at all.
        """
        self._graph = graph
        self._schema = build_schema_from_graph(graph)
        self._serializer = serializer
        self._llm = llm_client
        self._examples = examples
        self._config = config
        self._base_temperature = base_temperature
        self._neo4j_client = neo4j_client

    @property
    def schema(self) -> GraphSchema:
        """The full, live schema this engine prunes from."""
        return self._schema

    def ask(
        self, question: str, *, execute: bool = True, backend: str = "json"
    ) -> QueryResult:
        """
        Answer one question with a value computed from the graph.

        Args:
            question: the natural-language question.
            execute: when False, stop after producing a validated plan. The
                plan and its compiled Cypher are returned without being run.
            backend: ``"json"`` executes the plan in memory against the graph
                this engine was built with -- no database needed. ``"neo4j"``
                compiles the plan to Cypher and executes it against a live
                database. Both return rows in the same shape, so evidence
                assembly and the answer format are identical either way;
                only the execution step differs, which is what makes the two
                directly comparable for the same question.

        Returns:
            A :class:`QueryResult`. On success, ``answer`` holds the computed
            value(s), ``evidence`` holds any citable records, and ``cypher``
            holds the equivalent query for review or for Neo4j.

        Raises:
            ValueError: ``backend`` is not ``"json"`` or ``"neo4j"``, or is
                ``"neo4j"`` but no Neo4j client was configured.
        """
        if backend not in ("json", "neo4j"):
            raise ValueError(f"backend must be 'json' or 'neo4j', got {backend!r}")
        if backend == "neo4j" and execute and self._neo4j_client is None:
            raise ValueError(
                "backend='neo4j' needs a Neo4j client; pass one to the engine "
                "(or the factory) or use backend='json'."
            )

        pruned = prune_schema(question, self._schema, self._config)
        examples = self._examples.select(question, self._config)
        prompt_base = _PROMPT_TEMPLATE.format(
            schema=pruned.render(),
            examples=CypherExampleBank.render(examples),
            question=question,
        )

        previous: str | None = None
        last_error: str | None = None
        total_attempts = self._config.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            prompt = prompt_base
            if previous is not None:
                prompt += _RETRY_TEMPLATE.format(previous=previous, error=last_error)

            # Escalate sampling on each retry: re-reading a corrected
            # prompt at temperature 0 reproduces the rejected plan
            # verbatim, so the loop could never escape its first answer.
            temperature = self._config.temperature_for(attempt, self._base_temperature)
            try:
                raw = self._llm.complete(prompt, temperature=temperature)
            except LLMRequestError as exc:
                # A dead model backend produced no plan, so there is nothing
                # to repair; retrying would only repeat the same failure.
                logger.error("LLM request failed: %s", exc)
                return QueryResult(
                    question=question, success=False, attempts=attempt, error=str(exc)
                )

            plan_text = _extract_json(raw)
            try:
                plan = self._build_plan(plan_text)
            except PlanError as exc:
                logger.warning("Attempt %d/%d rejected: %s", attempt, total_attempts, exc)
                previous, last_error = plan_text, str(exc)
                continue

            logger.info(
                "Plan %s on attempt %d/%d (backend=%s)",
                "built" if not execute else "executed",
                attempt,
                total_attempts,
                backend,
            )
            # A plan that passed validate_against() always executes
            # successfully against the in-memory graph -- execute_plan
            # cannot fail on validated input. Against a live database it can
            # still fail (a genuine outage, a compilation edge case), but
            # retrying would hit the same database with the same plan and
            # get the same answer, so that failure is reported, not
            # self-healed. Only plan validation, above, is worth retrying.
            return self._result(question, plan, attempt, execute=execute, backend=backend)

        return QueryResult(
            question=question, success=False, attempts=total_attempts,
            error=last_error, backend=backend,
        )

    def _build_plan(self, plan_text: str) -> QueryPlan:
        """Parse and validate one candidate plan, or raise :class:`PlanError`."""
        try:
            payload = json.loads(plan_text)
        except json.JSONDecodeError as exc:
            raise PlanError(
                f"Not valid JSON ({exc.msg} at position {exc.pos}). "
                f"Answer with one JSON object and nothing else."
            ) from exc

        plan = QueryPlan.from_dict(payload)
        plan.validate_against(self._schema)
        return plan

    def _result(
        self,
        question: str,
        plan: QueryPlan,
        attempt: int,
        *,
        execute: bool,
        backend: str,
    ) -> QueryResult:
        """Execute the plan against the chosen backend (or not) and package the outcome."""
        cypher = plan.to_cypher()
        if not execute:
            return QueryResult(
                question=question,
                success=True,
                plan=plan,
                cypher=cypher,
                attempts=attempt,
                backend=backend,
            )

        if backend == "neo4j":
            try:
                rows = self._neo4j_client.run_query_with_entities(cypher)
            except Neo4jQueryError as exc:
                return QueryResult(
                    question=question,
                    success=False,
                    plan=plan,
                    cypher=cypher,
                    attempts=attempt,
                    error=f"Neo4j execution error: {exc}",
                    backend=backend,
                )
        else:
            rows = execute_plan(plan, self._graph)

        rows = rows[: self._config.max_result_rows]
        scalars: dict[str, Any] = {}
        for row in rows:
            values, _ = split_row(row)
            scalars.update(values)

        return QueryResult(
            question=question,
            success=True,
            plan=plan,
            cypher=cypher,
            answer=MappingProxyType(scalars),
            evidence=assemble_evidence(rows, self._serializer),
            attempts=attempt,
            backend=backend,
        )


def _extract_json(raw_response: str) -> str:
    """
    Recover the JSON object from whatever wrapping the model added.

    Models routinely wrap an answer in ```json fences, echo the prompt's
    trailing label, or add a sentence of explanation. Each of those would
    otherwise cost a self-heal retry to recover from, so they are stripped
    here instead.
    """
    text = raw_response.strip()

    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            fenced = parts[1]
            first_line, _, rest = fenced.partition("\n")
            text = rest if first_line.strip().isalpha() else fenced
        text = text.strip()

    lowered = text.lower()
    for label in ("json plan:", "json:", "plan:"):
        if lowered.startswith(label):
            text = text[len(label) :].strip()
            break

    # Fall back to the outermost braces, which handles a stray sentence
    # before or after the object.
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    return text.strip()
