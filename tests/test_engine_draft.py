"""
Tests for the reader of the pricing engine's own output format (draft 0.2.0),
and for the question mode of the generator that uses its cases.

The reader follows the same rules as the flat-result parser: it reads, never
computes; it refuses a result that does not reconcile instead of correcting it;
an unavailable amount contributes a statement, never a zero; an abstained
recommendation is a case in its own right.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.narrative.engine_draft import (
    EngineFormatError,
    is_engine_result,
    parse_engine_result,
    suggested_questions,
)
from src.narrative.facts import Fact, FactPack
from src.narrative.generator import NarrativeGenerator
from src.narrative.verifier import extract_figures, verify
from src.reliability.attribution import check_attribution

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "track_a_real_schema_sample_output.json"


@pytest.fixture()
def sample() -> dict:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


@pytest.fixture()
def run(sample):
    return parse_engine_result(sample)


def keys(case) -> set[str]:
    return {fact.key for fact in case.pack.facts}


# ------------------------------------------------------------------- detection
def test_the_engine_run_is_recognised_and_a_flat_result_is_not(sample):
    assert is_engine_result(sample)
    flat = json.loads((SAMPLE.parent / "track_a_sample_payloads.json").read_text(encoding="utf-8"))
    assert not is_engine_result(flat)
    assert not is_engine_result([1, 2, 3])


# ---------------------------------------------------------------------- cases
def test_the_sample_gives_one_case_per_recommendation(run):
    assert run.contract_version == "0.2.0-draft" and run.engine_version == "0.4.0"
    assert [c.case_id for c in run.cases] == ["rec-1", "rec-2"]


def test_a_case_gathers_its_waterfall_segment_elasticity_and_flag(run):
    case = run.cases[0]
    assert set(case.view) >= {"recommendation", "waterfall", "segment", "elasticity", "flags"}
    assert case.subject.order_line_id == "OL000248"
    assert case.subject.product_id == "SKU0026" and case.subject.customer_id == "CUST0009"
    assert {"recommended_price", "price_range_low", "price_range_high", "acceptance_probability",
            "list_price", "invoice_price", "pocket_price", "pocket_margin", "elasticity_beta",
            "n_observations", "segment_score"} <= keys(case)


def test_figures_are_the_engines_own_with_ratios_published_as_percentages(run):
    facts = {f.key: f for f in run.cases[0].pack.facts}
    assert facts["pocket_price"].value == pytest.approx(89489.98)
    assert facts["recommended_price"].value == pytest.approx(378.5)
    assert facts["acceptance_probability"].value == pytest.approx(64.0)
    assert facts["recommendation_confidence"].value == pytest.approx(78.0)
    assert facts["elasticity_beta"].formatted("SAR") == "-0.62"


def test_waterfall_levels_carry_their_stage_in_the_label(run):
    labels = {f.key: f.label_en for f in run.cases[0].pack.facts}
    assert "NOT shown on the invoice" in labels["off_invoice_leakage"]
    assert "shown on the invoice" in labels["on_invoice_discount"]
    assert labels["step.volume_rebate"].endswith("(NOT shown on the invoice)")
    assert labels["step.freight"].endswith("(shown on the invoice)")


def test_engine_identifiers_become_graph_handles_and_steps_cite_their_component(run):
    case = run.cases[0]
    assert {"OrderLine:OL000248", "TradeCustomer:CUST0009", "SellableProduct:SKU0026",
            "PriceComponent:PC0001167"} <= set(case.pack.citations)
    steps = {f.key: f for f in case.pack.facts if f.key.startswith("step.")}
    assert steps["step.invoice_discount"].citation == "PriceComponent:PC0001167"
    assert case.pack.engine_version == "engine-0.4.0"


def test_every_handle_of_the_sample_resolves_in_the_graph(run, graph, serializer):
    from src.narrative.evidence import EvidenceIndex

    index = EvidenceIndex.build(graph, serializer)
    assert index.missing(run.cases[0].pack.citations) == ()


def test_statements_carry_the_engines_qualifications_and_their_numbers_are_admitted(run):
    case = run.cases[0]
    text = "\n".join(case.pack.context)
    assert "NOT fit for price optimisation" in text and "confidence band low" in text
    assert 90.0 in case.pack.context_values, "a figure quoted inside an engine note is the engine's own"
    assert 90.0 in case.pack.allowed_values


def test_an_abstained_recommendation_is_a_case_that_explains_why(run):
    case = run.cases[1]
    assert case.view["recommendation"]["status"] == "abstained"
    assert not (keys(case) & {"recommended_price", "price_range_low"})
    assert "comparable_deals" in keys(case)
    assert any("did not recommend a price" in line for line in case.pack.context)
    assert any("unavailable" in line for line in case.pack.context), "unavailable is stated, never shown as zero"
    assert "Why did the engine not recommend a price?" in case.questions


def test_suggested_questions_follow_the_sections_present(run):
    q = run.cases[0].questions
    assert "Why is the pocket price lower than the invoice price?" in q
    assert "Is the price elasticity estimate reliable enough to use?" in q
    assert "Did the engine raise any data-quality warning?" in q
    assert suggested_questions({}) == ("Explain this result in plain words.",)


# --------------------------------------------------------------------- refusals
def test_a_waterfall_that_does_not_add_up_is_refused_with_the_mismatch_named(sample):
    sample["waterfalls"][0]["pocket_price"]["amount"] += 50.0
    with pytest.raises(EngineFormatError, match="does not add up.*pocket price"):
        parse_engine_result(sample)


def test_itemised_steps_that_do_not_sum_to_the_stated_total_are_refused(sample):
    sample["waterfalls"][0]["steps"][0]["amount"]["amount"] += 10.0
    with pytest.raises(EngineFormatError, match="steps do not add up"):
        parse_engine_result(sample)


def test_a_complete_amount_without_a_value_is_refused(sample):
    sample["waterfalls"][0]["list_value"]["amount"] = None
    with pytest.raises(EngineFormatError, match="list_value.*no amount"):
        parse_engine_result(sample)


def test_a_probability_outside_zero_to_one_is_refused(sample):
    sample["recommendations"][0]["acceptance_probability"] = 64
    with pytest.raises(EngineFormatError, match="probability in \\[0, 1\\]"):
        parse_engine_result(sample)


def test_an_inverted_price_range_is_refused(sample):
    sample["recommendations"][0]["price_range"].update(low=500.0, high=400.0)
    with pytest.raises(EngineFormatError, match="above high"):
        parse_engine_result(sample)


def test_a_boolean_where_a_number_belongs_is_refused(sample):
    sample["recommendations"][0]["price_range"]["low"] = True
    with pytest.raises(EngineFormatError, match="needs a number"):
        parse_engine_result(sample)


def test_a_run_with_nothing_to_explain_is_refused():
    with pytest.raises(EngineFormatError):
        parse_engine_result({"provenance": {}, "recommendations": [], "waterfalls": []})


def test_an_unavailable_amount_adds_a_statement_not_a_figure(sample):
    sample = copy.deepcopy(sample)
    sample["waterfalls"][0]["cogs"] = {"amount": None, "currency": "SAR", "status": "unavailable", "note": "no cost record"}
    sample["waterfalls"][0]["pocket_margin"] = {"amount": None, "currency": "SAR", "status": "unavailable", "note": None}
    case = parse_engine_result(sample).cases[0]
    assert "cogs" not in keys(case) and "pocket_margin" not in keys(case)
    assert any("cost of goods sold is unavailable" in line for line in case.pack.context)


# ------------------------------------------------------------- generator: questions
class Recorder:
    def __init__(self, reply: str) -> None:
        self.reply, self.prompts = reply, []

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        self.prompts.append(prompt)
        return self.reply


@pytest.fixture()
def generator_for(settings):
    def make(reply: str):
        llm = Recorder(reply)
        return NarrativeGenerator(llm, settings.narrative), llm
    return make


def test_the_question_and_the_engines_statements_reach_the_prompt(run, generator_for):
    generator, llm = generator_for("The pocket price is 89,489.98 SAR [OrderLine:OL000248].")
    generator.explain(run.cases[0].pack, question="Why is the pocket price lower?")
    prompt = llm.prompts[0]
    assert 'The reader asked: "Why is the pocket price lower?"' in prompt
    assert "INSUFFICIENT" in prompt
    assert "Statements from the engine you may restate" in prompt and "NOT fit for price optimisation" in prompt


def test_an_engine_overview_prompt_lists_only_topics_the_result_holds(run, generator_for):
    generator, llm = generator_for("x")
    generator.explain(run.cases[1].pack)          # the abstained recommendation
    prompt = llm.prompts[0]
    assert "that the engine did not recommend a price" in prompt
    assert "walk from list price" not in prompt, "no waterfall, so no instruction to describe one"


def test_the_example_for_an_engine_result_uses_its_own_figures_not_invented_ones(run, generator_for):
    generator, llm = generator_for("x")
    generator.explain(run.cases[1].pack)
    prompt = llm.prompts[0]
    assert "48,123.45" not in prompt, "invented figures in the example were copied into answers"
    assert "using the first figures of your own table" in prompt


def test_an_insufficient_reply_is_an_abstention_with_a_clear_reason(run, generator_for):
    generator, llm = generator_for("INSUFFICIENT")
    result = generator.explain(run.cases[0].pack, question="What was the discount last year?")
    assert result.abstained and not result.success
    assert "cannot answer this question" in result.reason
    assert len(llm.prompts) == 1, "no retry for a refusal"


def test_a_misspelt_refusal_is_still_a_refusal(run, generator_for):
    generator, _ = generator_for("Insufficent.")
    assert generator.explain(run.cases[0].pack, question="Anything off-topic?").abstained


def test_a_figure_quoted_from_an_engine_note_passes_verification(run):
    text = "The segment rests on 90 credit days [engine-0.4.0]."
    assert verify(text, run.cases[0].pack).ok
    assert not verify("The segment rests on 77 credit days [engine-0.4.0].", run.cases[0].pack).ok


def test_a_pack_with_only_the_engine_version_can_be_explained(run, generator_for):
    generator, _ = generator_for("The number of comparable past deals is 0 [engine-0.4.0].")
    assert run.cases[1].pack.citations == ()
    assert generator.explain(run.cases[1].pack).success


def test_extract_figures_masks_identifiers_dates_and_years():
    assert extract_figures("line OL000248 on 2022-09-17 shows 90 days and 12,506.49 SAR in 2026") == (90.0, 12506.49)


# --------------------------------------------------- attribution on engine packs
def test_other_confidences_are_not_mistaken_for_the_recommendation_confidence(run):
    pack = run.cases[0].pack
    text = ("The confidence in this recommendation is 78.0% [engine-0.4.0]. "
            "The confidence in the elasticity estimate is 18.0% [engine-0.4.0].")
    assert check_attribution(text, pack).ok


def test_a_wrong_recommendation_confidence_is_still_caught(run):
    pack = run.cases[0].pack
    report = check_attribution("The confidence in this recommendation is 18.0% [engine-0.4.0].", pack)
    assert not report.ok and report.misattributions[0].role_key == "recommendation_confidence"


def test_a_recommended_price_range_is_not_a_claim_about_the_recommended_price(run):
    pack = run.cases[0].pack
    text = "The recommended price range is 362.06 SAR to 394.94 SAR [engine-0.4.0]."
    assert check_attribution(text, pack).ok


def test_swapping_the_invoice_and_pocket_prices_of_an_engine_result_is_caught(run):
    pack = run.cases[0].pack
    text = "The invoice price is 89,489.98 SAR [OrderLine:OL000248]. The pocket price is 95,350.07 SAR [OrderLine:OL000248]."
    assert not check_attribution(text, pack).ok


def test_the_fact_kind_number_formats_with_two_decimals():
    assert Fact("beta", -0.62, "number", "x", "beta").formatted() == "-0.62"
    pack = FactPack("s", "k", "SAR", (), (), "e", context=("a [e]",), context_values=(3.0,))
    assert pack.allowed_values == (3.0,) and pack.render_context() == "- a [e]"


def test_a_margin_stated_after_the_cost_is_not_a_claim_about_the_cost(run):
    # A correct answer that the checker once rejected: the figure belongs to the margin.
    pack = run.cases[0].pack
    text = "The margin left after the cost of goods sold is 25,552.48 SAR [OrderLine:OL000248]."
    assert check_attribution(text, pack).ok
    wrong = "The cost of goods sold is 25,552.48 SAR [OrderLine:OL000248]."
    assert not check_attribution(wrong, pack).ok, "stating the margin as the cost is still caught"


def test_a_question_unrelated_to_the_result_is_declined_before_the_model_is_called(run, generator_for):
    generator, llm = generator_for("The recommended price is 378.50 SAR [engine-0.4.0].")
    for question in ("What is the capital of France?", "Write me a poem about winter.", "hello"):
        result = generator.explain(run.cases[0].pack, question=question)
        assert result.abstained and "does not relate to this result" in result.reason
    assert llm.prompts == [], "an off-topic question must cost no model call"


def test_on_topic_phrasings_without_the_exact_words_are_still_accepted(run):
    from src.narrative.relevance import is_on_topic

    pack = run.cases[0].pack
    for question in ("How much money do we keep?", "Is this a good deal?", "What is the profit?",
                     "Tell me about the elasticity", "Is the engine sure about this?"):
        assert is_on_topic(pack, question), question
    assert not is_on_topic(pack, "What is the weather in Riyadh tomorrow?")


def test_a_citation_between_the_role_and_a_comparative_does_not_make_it_a_value_claim(run):
    # Found on a correct answer to "why is the pocket price lower than the invoice price?":
    # the citation after "pocket price" hid the comparative from the checker.
    pack = run.cases[0].pack
    text = ("The pocket price [OrderLine:OL000248] is lower than the invoice price [OrderLine:OL000248] "
            "because of additional deductions of 5,860.09 SAR [OrderLine:OL000248].")
    assert check_attribution(text, pack).ok
    wrong = "The pocket price [OrderLine:OL000248] is 5,860.09 SAR."
    assert not check_attribution(wrong, pack).ok, "a real misattribution is still caught"


# ------------------------------------------------- agreement for answers to questions
class _S:
    def __init__(self, *claims):
        self.claims = frozenset(claims)


def test_an_answer_contained_in_a_longer_one_agrees_for_a_question():
    from src.reliability.consistency import _agreement, _overlap_agreement

    # pairs: (1,2,3)&(1,2) -> 2/2 ; (1,2,3)&(3) -> 1/1 ; (1,2)&(3) -> 0/1  => mean 2/3
    assert _overlap_agreement([_S(1, 2, 3), _S(1, 2), _S(3)]) == pytest.approx(2 / 3)
    assert _overlap_agreement([_S(1, 2, 3), _S(1, 2), _S(2, 3)]) == pytest.approx((1 + 1 + 0.5) / 3)
    # a short answer inside a long one reads as breadth, not disagreement
    samples = [_S(1, 2, 3, 4), _S(1), _S(2)]
    assert _agreement(samples)[2] == pytest.approx(0.5)
    assert _overlap_agreement(samples) == pytest.approx(2 / 3)


def test_answers_that_state_different_figures_disagree_for_a_question():
    from src.reliability.consistency import _overlap_agreement

    assert _overlap_agreement([_S(1), _S(2), _S(3)]) == 0.0
    assert _overlap_agreement([_S(1, 2), _S(1, 2), _S(1, 2)]) == 1.0


def test_a_sample_without_figures_neither_confirms_nor_contradicts():
    from src.reliability.consistency import _overlap_agreement

    assert _overlap_agreement([_S(1, 2), _S(), _S(1)]) == 1.0
    assert _overlap_agreement([_S(), _S(), _S()]) == 1.0
    assert _overlap_agreement([_S(1, 2)]) == 1.0


def test_the_reason_for_a_question_names_the_overlap_and_a_general_explanation_keeps_its_wording(run, generator_for, settings):
    from src.reliability.consistency import assess_consistency

    pack = run.cases[0].pack
    texts = ["The pocket price is 89,489.98 SAR [OrderLine:OL000248].",
             "The list price is 103,125.00 SAR [OrderLine:OL000248].",
             "The invoice price is 95,350.07 SAR [OrderLine:OL000248]."]
    narratives = []
    for text in texts:
        generator, _ = generator_for(text)
        narratives.append(generator.explain(pack))
    asked = assess_consistency(narratives, pack, agreement_kind="overlap")
    assert asked.abstained and "state different figures" in asked.reason and asked.agreement_kind == "overlap"
    general = assess_consistency(narratives, pack)
    assert general.abstained and "stated by a majority" in general.reason and general.agreement_kind == "core"


def test_requests_for_advice_are_declined_before_any_model_call(run, generator_for):
    from src.narrative.relevance import is_advice_request

    asked = ["Should we give this customer a bigger discount?", "Is this a good deal?",
             "Is it worth it for the customer?", "Do you recommend accepting the price?", "What should we offer?"]
    explained = ["Why is the pocket price lower than the invoice price?", "How good is the engine's confidence?",
                 "Which deductions are not shown on the invoice?", "Is the elasticity estimate reliable enough to use?"]
    assert all(is_advice_request(q) for q in asked)
    assert not any(is_advice_request(q) for q in explained)

    generator, llm = generator_for("The list price is 103,125.00 SAR [OrderLine:OL000248].")
    result = generator.explain(run.cases[0].pack, question=asked[0])
    assert result.abstained and "advice or a judgement" in result.reason
    assert llm.prompts == []
