"""
Tests for the self-healing text-to-query engine, its example bank, and
evidence assembly.

Everything here runs with a scripted LLM and the real JSON graph: no network,
no model, no database. That is not a compromise -- a real model gives no
*predictable* output to assert on, so the self-heal loop's branching could not
be tested exactly against one. The live path is exercised by
``evaluate_text_to_cypher.py``.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.retrieval.corpus import LabelIdentityPolicy, RecordSerializer
from src.text_to_query.engine import QueryResult, TextToQueryEngine, _extract_json
from src.text_to_query.evidence import assemble_evidence, split_row
from src.text_to_query.examples import CypherExampleBank, CypherExampleError
from src.text_to_query.llm_client import LLMRequestError
from src.text_to_query.schema import build_schema_from_graph


# -- doubles ----------------------------------------------------------------


class ScriptedLLM:
    """Returns a scripted sequence of responses, recording prompts and settings."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.temperatures: list[float | None] = []

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        self.prompts.append(prompt)
        self.temperatures.append(temperature)
        return self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]


class UnreachableLLM:
    """An LLM backend that cannot be reached."""

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        raise LLMRequestError("Could not reach Ollama at http://localhost:11434")


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def ttq_serializer(graph, settings):
    return RecordSerializer(
        settings.corpus, LabelIdentityPolicy.learn(graph, settings.corpus)
    )


@pytest.fixture
def schema(graph):
    return build_schema_from_graph(graph)


@pytest.fixture
def a_label(schema):
    """Any label that exists in the fixture graph."""
    return next(iter(schema.label_names))


@pytest.fixture
def example_bank(graph, a_label, tmp_path):
    """A minimal bank built against whatever labels the fixture graph has."""
    path = tmp_path / "examples.json"
    path.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "id": "count",
                        "question": "How many things are there?",
                        "plan": {"label": a_label, "aggregate": "count"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return CypherExampleBank.load(path)


@pytest.fixture
def make_engine(graph, ttq_serializer, example_bank, settings):
    def _make(llm, config=None):
        return TextToQueryEngine(
            graph,
            serializer=ttq_serializer,
            llm_client=llm,
            examples=example_bank,
            config=config or settings.text_to_cypher,
        )

    return _make


@pytest.fixture
def valid_plan(a_label):
    return json.dumps({"label": a_label, "aggregate": "count"})


# -- the happy path ---------------------------------------------------------


def test_counts_are_computed_from_the_graph(make_engine, graph, a_label, valid_plan):
    """
    The number must come from the data, not from the model.

    The scripted model returns only a plan; the count below is whatever the
    graph actually holds for that label.
    """
    engine = make_engine(ScriptedLLM(valid_plan))

    result = engine.ask("How many are there?")

    assert result.success
    assert result.value == graph.stats.label_counts[a_label]


def test_no_database_is_involved(make_engine, valid_plan):
    """The whole point: this answers with the JSON graph alone."""
    engine = make_engine(ScriptedLLM(valid_plan))

    assert engine.ask("How many are there?").success


def test_result_carries_the_equivalent_cypher(make_engine, a_label, valid_plan):
    """The plan is still reviewable as a query, and runs on Neo4j unchanged."""
    engine = make_engine(ScriptedLLM(valid_plan))

    result = engine.ask("How many are there?")

    assert result.cypher == f"MATCH (n:{a_label}) RETURN count(n) AS count"


def test_filters_narrow_the_result(make_engine, graph, corpus):
    """A filtered count must match an independently computed one."""
    record = corpus[0]
    label = record.record_type
    key = next(k for k, v in record.properties.items() if isinstance(v, str))
    value = record.properties[key]

    expected = sum(
        1
        for node in graph
        if node.primary_label == label and str(node.properties.get(key)) == str(value)
    )
    plan = json.dumps(
        {
            "label": label,
            "filters": [{"property": key, "op": "=", "value": value}],
            "aggregate": "count",
        }
    )

    result = make_engine(ScriptedLLM(plan)).ask("How many?")

    assert result.value == expected


# -- the self-heal loop -----------------------------------------------------


def test_unknown_label_is_rejected_and_healed(make_engine, valid_plan):
    bad = json.dumps({"label": "NoSuchLabelHere", "aggregate": "count"})
    llm = ScriptedLLM(bad, valid_plan)

    result = make_engine(llm).ask("How many are there?")

    assert result.success
    assert result.attempts == 2
    assert "NoSuchLabelHere" in llm.prompts[1]


def test_unknown_property_is_rejected_and_healed(make_engine, a_label, valid_plan):
    bad = json.dumps(
        {
            "label": a_label,
            "filters": [{"property": "no_such_property", "op": "=", "value": "x"}],
            "aggregate": "count",
        }
    )
    llm = ScriptedLLM(bad, valid_plan)

    result = make_engine(llm).ask("How many are there?")

    assert result.success
    assert "no_such_property" in llm.prompts[1]


def test_malformed_json_is_rejected_and_healed(make_engine, valid_plan):
    llm = ScriptedLLM("this is not json at all", valid_plan)

    result = make_engine(llm).ask("How many are there?")

    assert result.success
    assert result.attempts == 2
    assert "JSON" in llm.prompts[1]


def test_the_exact_error_is_fed_back(make_engine, valid_plan):
    """The error text is the repair signal, so it must reach the model intact."""
    llm = ScriptedLLM(json.dumps({"label": "Ghost", "aggregate": "count"}), valid_plan)

    make_engine(llm).ask("How many are there?")

    retry = llm.prompts[1]
    assert "Your previous plan was" in retry
    assert "Ghost" in retry
    assert "Unknown label" in retry


def test_retries_are_bounded_by_configuration(make_engine, settings):
    config = replace(settings.text_to_cypher, max_retries=2)
    llm = ScriptedLLM(json.dumps({"label": "StillWrong", "aggregate": "count"}))

    result = make_engine(llm, config).ask("How many?")

    assert not result.success
    assert result.attempts == 3  # one attempt plus two retries
    assert len(llm.prompts) == 3


def test_zero_retries_means_one_attempt(make_engine, settings):
    config = replace(settings.text_to_cypher, max_retries=0)
    llm = ScriptedLLM(json.dumps({"label": "Wrong", "aggregate": "count"}))

    assert make_engine(llm, config).ask("How many?").attempts == 1


def test_retry_temperature_escalates(make_engine, settings, valid_plan):
    """
    Without this, a self-heal loop cannot escape its first answer.

    At temperature 0 the model re-reads the corrected prompt and greedily
    regenerates the same rejected plan -- observed with qwen2.5, which
    returned byte-identical output on all four attempts.
    """
    llm = ScriptedLLM(json.dumps({"label": "Wrong", "aggregate": "count"}))

    make_engine(llm, replace(settings.text_to_cypher, max_retries=2)).ask("How many?")

    assert llm.temperatures == sorted(llm.temperatures)
    assert llm.temperatures[0] < llm.temperatures[-1]


def test_temperature_escalation_can_be_disabled(make_engine, settings):
    config = replace(settings.text_to_cypher, retry_temperature_step=0.0, max_retries=2)
    llm = ScriptedLLM(json.dumps({"label": "Wrong", "aggregate": "count"}))

    make_engine(llm, config).ask("How many?")

    assert set(llm.temperatures) == {0.0}


def test_unreachable_llm_fails_immediately(make_engine):
    """A dead backend produced no plan, so there is nothing to repair."""
    result = make_engine(UnreachableLLM()).ask("How many are there?")

    assert not result.success
    assert result.attempts == 1
    assert "Could not reach Ollama" in (result.error or "")


# -- the value guardrail ----------------------------------------------------


def test_impossible_filter_value_is_rejected(make_engine, graph, schema, valid_plan):
    """
    A filter on a value the property never takes matches nothing.

    Left to run it returns 0 -- a wrong answer indistinguishable from a real
    finding. Rejecting it turns a silent error into a retryable one.
    """
    label, prop, values = next(
        (label, prop, values)
        for label, properties in schema.value_samples.items()
        for prop, values in properties.items()
        if values
    )
    bad = json.dumps(
        {
            "label": label,
            "filters": [
                {"property": prop, "op": "=", "value": "definitely_not_a_real_value_xyz"}
            ],
            "aggregate": "count",
        }
    )
    llm = ScriptedLLM(bad, valid_plan)

    result = make_engine(llm).ask("How many?")

    assert result.attempts == 2, "the impossible filter should have been rejected"
    assert "never has the value" in llm.prompts[1]
    # The real values must be listed, so the model can pick a valid one.
    assert str(values[0]) in llm.prompts[1]


def test_a_real_value_is_not_rejected(make_engine, schema):
    """The guardrail must not reject a legitimate filter."""
    label, prop, values = next(
        (label, prop, values)
        for label, properties in schema.value_samples.items()
        for prop, values in properties.items()
        if values
    )
    plan = json.dumps(
        {
            "label": label,
            "filters": [{"property": prop, "op": "=", "value": values[0]}],
            "aggregate": "count",
        }
    )

    result = make_engine(ScriptedLLM(plan)).ask("How many?")

    assert result.success
    assert result.attempts == 1


# -- generation without execution -------------------------------------------


def test_dry_run_returns_a_plan_without_executing(make_engine, valid_plan):
    result = make_engine(ScriptedLLM(valid_plan)).ask("How many?", execute=False)

    assert result.success
    assert result.plan is not None
    assert result.cypher
    assert result.answer == {}


# -- response extraction ----------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"label": "X"}',
        'JSON plan: {"label": "X"}',
        '```json\n{"label": "X"}\n```',
        '```\n{"label": "X"}\n```',
        'Here is the plan:\n```json\n{"label": "X"}\n```\nThat should do it.',
        'Sure! {"label": "X"}',
    ],
)
def test_extraction_recovers_the_json_object(raw):
    """Each wrapping form costs a retry if not stripped here."""
    assert json.loads(_extract_json(raw)) == {"label": "X"}


# -- evidence ---------------------------------------------------------------


def test_evidence_uses_retrieval_citation_scheme(corpus, ttq_serializer):
    """One citation scheme across both components, not two."""
    record = corpus[0]
    node_field = {"labels": [record.record_type], "properties": dict(record.properties)}

    evidence = assemble_evidence([{"evidence": node_field}], ttq_serializer)

    assert len(evidence) == 1
    assert evidence[0].record_id == record.record_id


def test_return_records_produces_a_citation(make_engine, corpus):
    """A fact answer must be traceable to the record it came from."""
    record = corpus[0]
    key = next(k for k, v in record.properties.items() if isinstance(v, str))
    plan = json.dumps(
        {
            "label": record.record_type,
            "filters": [{"property": key, "op": "=", "value": record.properties[key]}],
            "aggregate": "count",
            "return_records": True,
        }
    )

    result = make_engine(ScriptedLLM(plan)).ask("How many?")

    assert result.citations


def test_a_pure_aggregate_cites_nothing(make_engine, valid_plan):
    """A count has no individual record behind it, and says so."""
    result = make_engine(ScriptedLLM(valid_plan)).ask("How many?")

    assert result.citations == ()


def test_scalars_and_nodes_are_separated():
    node_field = {"labels": ["Thing"], "properties": {"thing_id": "T1"}}
    scalars, nodes = split_row({"count": 5, "evidence": node_field})

    assert scalars == {"count": 5}
    assert nodes == [node_field]


# -- the example bank -------------------------------------------------------


def test_project_examples_are_valid_against_the_real_schema(schema):
    """
    The committed bank must be valid for the graph the engine actually uses.

    The whole suite runs on the PoC export -- the project's source of truth --
    so this validates the examples against exactly the schema production sees.
    """
    from src.config.settings import PROJECT_ROOT

    path = PROJECT_ROOT / "data" / "cypher_examples.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("example bank not present")

    assert len(CypherExampleBank.load(path, schema)) > 0


def test_stale_example_is_rejected(schema, tmp_path):
    """An example teaching a label that no longer exists is worse than none."""
    path = tmp_path / "examples.json"
    path.write_text(
        json.dumps(
            {"examples": [{"id": "x", "question": "q", "plan": {"label": "Gone"}}]}
        ),
        encoding="utf-8",
    )

    with pytest.raises(CypherExampleError, match="Unknown label"):
        CypherExampleBank.load(path, schema)


def test_example_without_a_plan_is_rejected(tmp_path):
    path = tmp_path / "examples.json"
    path.write_text(
        json.dumps({"examples": [{"id": "x", "question": "q"}]}), encoding="utf-8"
    )

    with pytest.raises(CypherExampleError, match="needs a 'question' and a 'plan'"):
        CypherExampleBank.load(path)


def test_missing_example_file_is_reported(tmp_path):
    with pytest.raises(CypherExampleError, match="not found"):
        CypherExampleBank.load(tmp_path / "absent.json")


def test_example_selection_is_bounded(settings):
    from src.config.settings import PROJECT_ROOT

    path = PROJECT_ROOT / "data" / "cypher_examples.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("example bank not present")

    bank = CypherExampleBank.load(path)
    config = replace(settings.text_to_cypher, max_few_shot_examples=2)

    assert len(bank.select("anything at all", config)) <= 2


def test_zero_examples_configured_returns_none(settings):
    from src.config.settings import PROJECT_ROOT

    path = PROJECT_ROOT / "data" / "cypher_examples.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("example bank not present")

    bank = CypherExampleBank.load(path)
    config = replace(settings.text_to_cypher, max_few_shot_examples=0)

    assert bank.select("anything", config) == ()
