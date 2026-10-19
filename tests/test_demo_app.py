"""
Tests for the demo workbench (``demo_app``), run offline with a scripted model.

The service adds no logic of its own, so these tests check what the question
stream exposes: that a question is routed to the right path (a named Track A
result, an exact value from the graph, or a request to choose a result), that each
stage of the architecture emits its real output in order, that safeguards on and
off behave as described, that evaluation is computed with the project's own code
against independent references when they exist, and that refusals are reported
rather than raised.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from demo_app.orchestrator import extract_identifiers, is_overview  # noqa: E402
from demo_app.server import create_app  # noqa: E402
from src.config.settings import Settings  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"

PLAN = {
    "label": "DealOutcome",
    "filters": [{"property": "result", "op": "=", "value": "lost_to_competitor"}],
    "aggregate": "count",
}

#: A narrative that passes verification and the wrong-label check for simulated REC-0001.
NARRATIVE_REC_0001 = (
    "The recommended price is 389.38 SAR [trackA-provisional-0.1]. "
    "The list price is 74,700.00 SAR [PriceComponent:PC0000010]. "
    "An invoice discount of 2,332.47 SAR [PriceComponent:PC0000011] leaves an invoice price "
    "of 72,367.53 SAR [PriceComponent:PC0000011]. "
    "A volume rebate of 1,074.69 SAR [PriceComponent:PC0000012], which does not appear on the "
    "invoice, brings the pocket price to 71,292.84 SAR [OrderLine:OL000003]."
)

#: The same text with the invoice and pocket values exchanged: every number is real and
#: every citation resolves, but two figures sit on the wrong name.
NARRATIVE_SWAPPED = (
    "The recommended price is 389.38 SAR [trackA-provisional-0.1]. "
    "The list price is 74,700.00 SAR [PriceComponent:PC0000010]. "
    "An invoice discount of 2,332.47 SAR [PriceComponent:PC0000011] and a volume rebate of "
    "1,074.69 SAR [PriceComponent:PC0000012] apply. "
    "The invoice price is 71,292.84 SAR [PriceComponent:PC0000011]. "
    "The pocket price is 72,367.53 SAR [OrderLine:OL000003]."
)

ANSWER_POCKET = (
    "The pocket price is 89,489.98 SAR [OrderLine:OL000248], lower than the invoice price of "
    "95,350.07 SAR [OrderLine:OL000248] because deductions of 5,860.09 SAR [OrderLine:OL000248] "
    "are not shown on the invoice."
)
OVERVIEW_ABSTAINED = (
    "The number of comparable past deals is 0 [engine-0.4.0]. "
    "The confidence in this recommendation is 0.0% [engine-0.4.0]."
)
ANSWER_REC1 = (
    "The pocket price is 71,292.84 SAR [OrderLine:OL000003], lower than the invoice price of "
    "72,367.53 SAR [OrderLine:OL000003] because deductions of 1,074.69 SAR [OrderLine:OL000003] "
    "are not shown on the invoice."
)
EXACT_SENTENCE = "There are 101 opportunities lost to a competitor [graph-query]."


class ScriptedModel:
    """Chooses a reply from what the prompt contains."""

    def __init__(self, narrative: str = NARRATIVE_REC_0001) -> None:
        self.narrative, self.prompts = narrative, []

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        self.prompts.append(prompt)
        if "JSON plan:" in prompt:
            return '{"label": "Nothing"}' if "France" in prompt else json.dumps(PLAN)
        if "last year" in prompt:
            return "INSUFFICIENT"
        if "graph-query" in prompt:
            return EXACT_SENTENCE
        if "The reader asked:" in prompt:
            return ANSWER_REC1 if "OL000003" in prompt else ANSWER_POCKET
        if "did not recommend a price" in prompt:
            return OVERVIEW_ABSTAINED
        return self.narrative


@pytest.fixture(scope="module")
def demo_settings(graph_path: Path) -> Settings:
    return Settings.from_env(
        {"CORPUS_SOURCE": "json", "CORPUS_JSON_PATH": str(graph_path), "RERANK_ENABLED": "false"}
    )


@pytest.fixture(scope="module")
def model() -> ScriptedModel:
    return ScriptedModel()


@pytest.fixture(scope="module")
def client(demo_settings: Settings, model: ScriptedModel):
    with TestClient(create_app(settings=demo_settings, llm_client=model, dense=False)) as test_client:
        yield test_client


def query(client, q: str, **params) -> list[dict]:
    response = client.get("/api/query", params={"q": q, **params})
    assert response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def of(events: list[dict], stage: str) -> dict:
    return next(e for e in events if e["stage"] == stage)


def stages(events: list[dict]) -> list[str]:
    return [e["stage"] for e in events]


EXPLAIN_Q = "Why is the pocket price lower than the invoice price for line OL000248?"


# --------------------------------------------------------------------- basics
def test_status_and_the_page(client):
    body = client.get("/api/status").json()
    assert body["graph"]["nodes"] == 5794 and body["retrieval"]["streams"] == ["bm25", "graph"]
    assert body["reliability"]["samples"] == 3
    assert "AXIOM Pricing Module" in client.get("/").text
    for asset in ("app.js", "style.css"):
        assert client.get(f"/static/{asset}").status_code == 200


def test_examples_are_generated_from_the_loaded_results(client):
    examples = client.get("/api/examples").json()
    kinds = {e["kind"] for e in examples}
    assert kinds == {"result", "exact", "refuse"}
    assert any("OL000248" in e["text"] for e in examples), "built from the engine sample, not hard-coded text"


def test_identifier_extraction_and_overview_detection():
    assert extract_identifiers("Explain SKU0026 for cust0009 on line OL000248 (REC-0005)") == [
        "SKU0026", "CUST0009", "OL000248", "REC-0005"]
    assert is_overview("Explain the price recommendation for SKU0037")
    assert not is_overview("Explain why the pocket price is lower than the invoice price")
    assert not is_overview("What is the pocket price?")


# ------------------------------------------------------------------------ runs
def test_the_built_in_runs_and_a_case(client):
    runs = {r["id"]: r for r in client.get("/api/runs").json()}
    assert runs["engine-sample"]["format"] == "engine" and runs["engine-sample"]["cases"] == 2
    assert runs["simulated"]["format"] == "flat" and runs["simulated"]["cases"] == 24
    first = runs["simulated"]["case_list"][0]
    assert {"product", "customer", "line", "price"} <= set(first)
    body = client.get("/api/runs/engine-sample/cases/rec-1").json()
    assert set(body["view"]) >= {"recommendation", "waterfall", "segment", "elasticity", "flags"}
    assert body["statements"] and body["questions"] and body["unresolvable"] == []
    assert client.get("/api/runs/nope/cases/x").status_code == 404
    assert client.get("/api/runs/engine-sample/raw").json()["provenance"]["engine_version"] == "0.4.0"


def test_loading_engine_output_and_refusing_what_does_not_add_up(client):
    text = (DATA / "track_a_real_schema_sample_output.json").read_text(encoding="utf-8")
    ok = client.post("/api/runs/load", json={"name": "mine.json", "text": text}).json()
    assert ok["format"] == "engine" and ok["cases"] == 2 and ok["title"] == "mine.json"
    data = json.loads(text)
    data["waterfalls"][0]["pocket_price"]["amount"] += 50
    bad = client.post("/api/runs/load", json={"text": json.dumps(data)})
    assert bad.status_code == 422 and "does not add up" in bad.json()["detail"]
    assert client.post("/api/runs/load", json={"text": "{ not json"}).status_code == 422


def test_one_bad_flat_result_is_refused_while_its_neighbours_stay_usable(client):
    entries = json.loads((DATA / "track_a_payloads_eval.json").read_text(encoding="utf-8"))["recommendations"][:2]
    entries[1]["waterfall"]["pocket_price"] += 50
    body = client.post("/api/runs/load", json={"text": json.dumps({"recommendations": entries})}).json()
    assert body["cases"] == 2 and body["refused"] == 1
    bad = next(c for c in body["case_list"] if c["error"])
    assert client.get(f"/api/runs/{body['id']}/cases/{bad['id']}").status_code == 422


# ----------------------------------------------------------- a Track A question
def test_a_question_naming_a_line_explains_that_result_through_every_stage(client, model):
    events = query(client, EXPLAIN_Q)
    order = stages(events)
    assert order[:4] == ["route", "retrieval", "trackA", "exact"]
    assert order.index("pack") < order.index("sample") < order.index("consistency") < order.index("decision")
    assert order[-2:] == ["evaluation", "done"]

    route = of(events, "route")
    assert route["path"] == "explain" and route["identifiers"] == ["OL000248"]
    assert route["matched"]["case"] == "rec-1" and route["matched"]["matched_on"] == ["OL000248"]

    assert of(events, "retrieval")["results"][0]["streams"], "each evidence record carries its stream trace"
    track_a = of(events, "trackA")
    assert track_a["view"]["waterfall"]["levels"][0]["amount"] == 103125.0 and track_a["citations"]
    assert of(events, "exact")["skipped"] is True
    assert of(events, "pack")["figures"] == 30

    decision = of(events, "decision")
    assert decision["served"] and decision["text"] == ANSWER_POCKET
    assert any(f'The reader asked: "{EXPLAIN_Q}"' in p for p in model.prompts)


def test_evaluation_scores_retrieval_and_generation_without_needing_a_reference(client):
    evaluation = of(query(client, EXPLAIN_Q), "evaluation")
    track_a = evaluation["retrieval"]["trackA"]
    assert track_a["wanted"] == 9 and 0 <= track_a["found"] <= 9
    free = evaluation["generation"]["reference_free"]
    assert free["figures_stated"] == free["figures_traced"] == 3 and free["labels_correct"] is True
    assert "reference" not in evaluation["generation"], "the hand-built sample has no frozen reference"


def test_a_simulated_result_is_scored_against_its_frozen_reference(client):
    events = query(client, "Explain the price recommendation for SKU0029 and CUST0034")
    assert of(events, "route")["matched"]["case"] == "REC-0001"
    assert of(events, "decision")["text"] == NARRATIVE_REC_0001
    reference = of(events, "evaluation")["generation"]["reference"]
    assert "NAR-0001" in reference["source"]
    assert reference["faithfulness"] == 1.0 and reference["answer_relevance"] == 1.0
    assert reference["served_wrong"] is False


def test_a_specific_question_about_a_frozen_result_skips_the_relevance_score(client):
    events = query(client, "Why is the pocket price lower than the invoice price for line OL000003?")
    reference = of(events, "evaluation")["generation"].get("reference")
    assert reference is not None and reference["answer_relevance"] is None


def test_a_result_selected_beside_the_question_is_used_when_the_question_names_none(client):
    events = query(client, "Explain this result", result="simulated:REC-0003")
    route = of(events, "route")
    assert route["path"] == "explain" and route["matched"]["case"] == "REC-0003"
    assert "selected beside" in route["reason"]


def test_a_price_question_that_names_nothing_asks_for_a_result(client):
    events = query(client, "explain the recommendation")
    assert of(events, "route")["path"] == "need_result"
    assert "decision" not in stages(events) and of(events, "trackA")["skipped"]
    assert of(events, "retrieval")["results"] is not None


def test_a_question_unrelated_to_the_selected_result_is_declined_without_a_model_call(client, model):
    calls = len(model.prompts)
    events = query(client, "What is the weather in Riyadh tomorrow?", result="engine-sample:rec-1")
    decision = of(events, "decision")
    assert decision["served"] is False and "does not relate to this result" in decision["reason"]
    assert len(model.prompts) == calls
    assert "generation" in of(events, "evaluation") and of(events, "evaluation")["generation"]["served"] is False


def test_a_question_the_data_cannot_answer_is_declined_with_the_models_reason(client):
    events = query(client, "What was the discount for line OL000248 last year?")
    decision = of(events, "decision")
    assert decision["served"] is False and decision["reason"].startswith("every sample declined")


# ------------------------------------------------------------- an exact question
def test_a_question_asking_for_a_stored_value_goes_to_the_graph_even_if_it_names_a_product(client):
    events = query(client, "What is the unit cost of product SKU0046?")
    route = of(events, "route")
    assert "SKU0046" in route["identifiers"] and route["path"] == "exact" and route["matched"] is None
    assert "stored in the graph" in route["reason"]
    assert of(events, "trackA")["skipped"] is True


def test_an_exact_question_runs_the_plan_the_cypher_the_cited_sentence_and_the_reference_check(client):
    events = query(client, "How many opportunities were lost to a competitor?")
    exact = of(events, "exact")
    assert exact["success"] and exact["answer"]["count"] == 101 and "MATCH (n:DealOutcome)" in exact["cypher"]
    assert of(events, "pack")["facts"][0]["citation"] == "graph-query"
    assert "correction" not in stages(events), "a computed aggregate has no record to retrieve around"
    decision = of(events, "decision")
    assert decision["served"] and decision["text"] == EXACT_SENTENCE
    reference = of(events, "evaluation")["exact"]["reference"]
    assert reference["correct"] is True and reference["expected"] == 101


def test_an_exact_question_that_cannot_be_planned_is_refused_not_guessed(client):
    events = query(client, "What is the capital of France?")
    exact = of(events, "exact")
    assert exact["success"] is False and exact["attempts"] >= 1 and exact["error"]
    decision = of(events, "decision")
    assert decision["served"] is False and "no value could be computed" in decision["reason"]


def test_retrieval_is_scored_against_a_frozen_reference_when_the_question_is_in_it(client):
    events = query(client, "What is the unit cost of product SKU0001?")
    reference = of(events, "evaluation")["retrieval"]["reference"]
    assert reference["primary"] == "ProductCostPeriod:SKU0001" and reference["mrr"] > 0
    assert "gold set" in reference["source"]


# ----------------------------------------------------------------- safeguards
def test_without_safeguards_one_sample_is_drawn_and_no_correction_is_made(client):
    events = query(client, EXPLAIN_Q, safeguards="false")
    assert stages(events).count("sample") == 1 and "correction" not in stages(events)
    assert of(events, "pack")["k"] == 1


def test_with_safeguards_three_samples_and_corrective_retrieval(client):
    events = query(client, EXPLAIN_Q, safeguards="true")
    assert stages(events).count("sample") == 3 and "correction" in stages(events)
    consistency = of(events, "consistency")
    assert consistency["usable"] == 3 and consistency["agreement"] == pytest.approx(1.0)


def test_a_misattributed_text_is_withheld_with_safeguards_and_served_without(demo_settings):
    app = create_app(settings=demo_settings, llm_client=ScriptedModel(NARRATIVE_SWAPPED), dense=False)
    q = "Explain the price recommendation for SKU0029 and CUST0034"
    with TestClient(app) as local:
        on = query(local, q, safeguards="true")
        off = query(local, q, safeguards="false")
    assert of(off, "sample")["attribution"]["ok"] is False
    assert of(off, "decision")["served"] is True, "the baseline serves a verified but misattributed text"
    assert of(on, "decision")["served"] is False, "the reliability layer withholds it"
    assert of(off, "evaluation")["generation"]["reference_free"]["labels_correct"] is False


def test_a_result_file_can_be_added_at_start_up(demo_settings, tmp_path, monkeypatch):
    entries = json.loads((DATA / "track_a_payloads_eval.json").read_text(encoding="utf-8"))
    path = tmp_path / "mine.json"
    path.write_text(json.dumps({"recommendations": entries["recommendations"][:3]}), encoding="utf-8")
    monkeypatch.setenv("DEMO_PAYLOADS", str(path))
    with TestClient(create_app(settings=demo_settings, llm_client=ScriptedModel(), dense=False)) as local:
        runs = {r["id"]: r for r in local.get("/api/runs").json()}
    assert runs["custom"]["cases"] == 3 and runs["custom"]["title"] == "Your file: mine.json"
