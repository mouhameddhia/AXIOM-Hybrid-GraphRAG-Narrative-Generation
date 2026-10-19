"""
Tests for the narrative evaluation harness (D6).

The harness is the instrument the deliverable's claims rest on, so these
tests are mostly about the instrument lying. A metric that reads 1.00 because
it compared something to itself, or a gold set that silently drifted from the
payloads it describes, is worse than no measurement: it produces a confident
number with nothing behind it.

The model is scripted throughout, so the whole suite runs offline in under a
second and can sit in the regression check.
"""
from __future__ import annotations

import json

import pytest

from src.config.settings import NarrativeConfig, ReliabilityConfig
from src.evaluation.narrative_harness import (
    SERVED_WRONG_TARGET,
    NarrativeGoldSet,
    NarrativeGoldSetError,
    evaluate_narratives,
)
from src.narrative.evidence import EvidenceIndex, ResolvedRecord
from src.narrative.generator import NarrativeGenerator
from src.narrative.contract import PriceRecommendation
from src.narrative.facts import build_facts
from src.reliability.narrator import ReliableNarrator
from src.config.settings import PROJECT_ROOT

GOLD_SET = PROJECT_ROOT / "data" / "narrative_gold_set.json"


# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------
class ScriptedLLM:
    """Returns a fixed response, or one keyed to the subject under test."""

    def __init__(self, response: str = "", by_value: bool = False) -> None:
        self.response = response
        self.by_value = by_value
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        self.prompts.append(prompt)
        if not self.by_value:
            return self.response
        return _faithful_from_prompt(prompt)


def _faithful_from_prompt(prompt: str) -> str:
    """
    Build a correct narrative by reading the fact table out of the prompt.

    This makes a scripted model that is faithful for *any* case, which is
    what lets the harness be exercised across the whole gold set without a
    real model.
    """
    import re

    wanted = ("list price:", "invoice price:", "pocket price actually kept:")
    # Distinct citations, rotated across sentences. Reusing one handle
    # everywhere satisfies coverage while telling a reader nothing, and the
    # gold set's citation floor rejects it -- which this double was doing
    # until the floor was implemented and caught it.
    handles = re.findall(r"^\s*-\s+([^\s\[]\S*(?::\S+)?)\s*$", prompt, re.M)
    handles = [h for h in handles if ":" in h or h.startswith("trackA")]
    if not handles:
        handles = ["trackA-provisional-0.1"]

    sentences = []
    for line in prompt.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        for label in wanted:
            if line[2:].lower().startswith(label):
                value = line.split(":", 1)[1].split("[")[0].strip()
                handle = handles[len(sentences) % len(handles)]
                sentences.append(f"The {label[:-1]} is {value} [{handle}].")
    if not sentences:
        sentences = [f"No figures were available [{handles[0]}]."]
    return " ".join(sentences)


@pytest.fixture(scope="module")
def index():
    """
    Every handle the gold set's own packs cite, so evidence resolution is
    satisfied without loading the graph. Hand-listing handles here meant the
    generator abstained on records the packs really do cite.
    """
    handles: set[str] = set()
    for case in NarrativeGoldSet.load(GOLD_SET).cases:
        handles.update(build_facts(case.load_recommendation()).citations)
    return EvidenceIndex(
        {h: ResolvedRecord(h, h.split(":")[0], "...") for h in handles}
    )


@pytest.fixture
def configs():
    narrative = NarrativeConfig(
        enabled=True, max_retries=0, max_words=120,
        absolute_tolerance=0.01, relative_tolerance=0.005,
        min_citation_coverage=1.0, retry_temperature_step=0.2,
    )
    reliability = ReliabilityConfig(
        enabled=True, samples=1, min_verified=1, min_agreement=0.0,
        sample_temperature=0.7, enforce_attribution=True, corrective_enabled=False,
        min_support_ratio=0.75, corrective_top_k=6, corrective_max_added=4,
    )
    return narrative, reliability


def narrator_for(llm, configs, index=None):
    narrative, reliability = configs
    return ReliableNarrator(
        NarrativeGenerator(llm, narrative, base_temperature=0.7),
        reliability,
        narrative,
        evidence=index,
        finder=None,
    )


@pytest.fixture
def one_case_gold(tmp_path):
    """A single real case, so scoring can be asserted exactly."""
    full = NarrativeGoldSet.load(GOLD_SET)
    case = full.cases[0]
    path = tmp_path / "one.json"
    path.write_text(
        json.dumps(
            {
                "description": "single case",
                "language": "en",
                "version": "test",
                "cases": [
                    {
                        "case_id": case.case_id,
                        "recommendation_id": case.recommendation_id,
                        "payloads": str(case.payloads),
                        "subject": case.subject,
                        "required_facts": list(case.required_facts),
                        "min_citations": case.min_citations,
                        "expected_values": dict(case.expected_values),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return NarrativeGoldSet.load(path)


# --------------------------------------------------------------------------
# the gold set itself
# --------------------------------------------------------------------------
class TestGoldSet:
    def test_the_shipped_gold_set_loads(self):
        gold = NarrativeGoldSet.load(GOLD_SET)

        assert gold.cases
        assert gold.language == "en", "Arabic is out of scope for this phase"
        assert gold.version

    def test_every_case_references_a_loadable_payload(self):
        """A gold case pointing at a missing payload measures nothing."""
        for case in NarrativeGoldSet.load(GOLD_SET).cases:
            recommendation = case.load_recommendation()

            assert recommendation.recommendation_id == case.recommendation_id

    def test_gold_values_match_the_fact_pack_for_every_case(self):
        """
        The independence check, asserted directly. The gold set and the fact
        pack are built from the same payload by different code; if they ever
        disagree, every metric in a run is measuring the wrong thing.
        """
        from src.evaluation.narrative_harness import _compare_pack_to_gold

        for case in NarrativeGoldSet.load(GOLD_SET).cases:
            pack = build_facts(case.load_recommendation())

            matches, differences = _compare_pack_to_gold(case, pack)

            assert matches, f"{case.case_id} drifted: {differences}"

    def test_every_required_fact_is_reachable(self):
        for case in NarrativeGoldSet.load(GOLD_SET).cases:
            for key in case.required_facts:
                assert key in case.expected_values

    def test_a_case_with_no_expected_values_is_refused(self, tmp_path):
        """It would score a perfect faithfulness against an empty truth set."""
        path = tmp_path / "gold.json"
        path.write_text(
            json.dumps(
                {"cases": [{"case_id": "X", "payloads": "p.json",
                            "required_facts": [], "expected_values": {}}]}
            ),
            encoding="utf-8",
        )

        with pytest.raises(NarrativeGoldSetError, match="no expected_values"):
            NarrativeGoldSet.load(path)

    def test_a_required_fact_with_no_expected_value_is_refused(self, tmp_path):
        path = tmp_path / "gold.json"
        path.write_text(
            json.dumps(
                {"cases": [{"case_id": "X", "payloads": "p.json",
                            "required_facts": ["pocket_price"],
                            "expected_values": {"list_price": 1.0}}]}
            ),
            encoding="utf-8",
        )

        with pytest.raises(NarrativeGoldSetError, match="could never be scored"):
            NarrativeGoldSet.load(path)

    def test_a_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(NarrativeGoldSetError, match="not found"):
            NarrativeGoldSet.load(tmp_path / "absent.json")

    def test_malformed_json_names_the_line(self, tmp_path):
        path = tmp_path / "gold.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(NarrativeGoldSetError, match="not valid JSON"):
            NarrativeGoldSet.load(path)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
class TestScoring:
    def test_a_faithful_narrative_scores_one(self, one_case_gold, configs, index):
        report = evaluate_narratives(
            one_case_gold, narrator_for(ScriptedLLM(by_value=True), configs, index)
        )

        assert report.service_rate == 1.0
        assert report.faithfulness == 1.0
        assert report.served_wrong_rate == 0.0
        assert report.trustworthy

    def test_relevance_falls_when_a_required_figure_is_missing(
        self, one_case_gold, configs, index
    ):
        """
        The metric that catches a narrative which is faithful and useless.
        This one states only the list price, omitting invoice and pocket.
        """
        case = one_case_gold.cases[0]
        handle = build_facts(case.load_recommendation()).citations[0]
        text = (
            f"The list price is "
            f"{case.expected_values['list_price']:,.2f} SAR [{handle}]."
        )

        report = evaluate_narratives(
            one_case_gold, narrator_for(ScriptedLLM(text), configs, index)
        )

        assert report.faithfulness == 1.0, "every figure it states iscorrect"
        assert report.answer_relevance < 1.0
        assert "invoice_price" in report.results[0].missing_required

    def test_an_abstention_is_not_counted_as_a_wrong_answer(
        self, one_case_gold, configs, index
    ):
        """
        Withholding is the designed outcome when checks fail. Counting it as
        served-wrong would make the headline metric punish the behaviour the
        reliability layer exists to produce.
        """
        report = evaluate_narratives(
            one_case_gold,
            narrator_for(ScriptedLLM("The margin is 7,890.16 SAR [Nope:X]."), configs, index),
        )

        assert report.service_rate == 0.0
        assert report.served_wrong_rate == 0.0
        assert not report.results[0].served_wrong

    def test_a_single_repeated_citation_fails_the_floor(self, one_case_gold, configs, index):
        """
        Coverage alone is satisfiable by repeating one record after every
        sentence, which tells a reader nothing about which figure rests on
        what. The gold set's per-case floor is what catches it.
        """
        case = one_case_gold.cases[0]
        handle = build_facts(case.load_recommendation()).citations[0]
        text = " ".join(
            f"The {name.replace('_', ' ')} is {case.expected_values[name]:,.2f} SAR [{handle}]."
            for name in ("list_price", "invoice_price", "pocket_price")
        )

        report = evaluate_narratives(
            one_case_gold, narrator_for(ScriptedLLM(text), configs, index)
        )

        row = report.results[0]
        assert row.served
        assert row.citation_coverage == 1.0, "every claim is cited"
        assert row.citations == 1
        assert not row.meets_citation_floor
        assert row.weak_citation
        assert not row.served_wrong, (
            "nothing stated is false -- weak citation is a separate axis, and "
            "folding it into served-wrong made one number mean two failures"
        )
        assert report.weak_citation_rate == 1.0
        assert report.served_wrong_rate == 0.0

    def test_the_served_wrong_denominator_is_the_served_set(self):
        """"If it answers, how often is it wrong" -- not "of all attempts"."""
        from src.evaluation.narrative_harness import (
            NarrativeCaseResult,
            NarrativeEvaluationReport,
        )

        def row(case_id, served, attribution_ok):
            return NarrativeCaseResult(
                case_id=case_id, recommendation_id="R", served=served, text="t",
                unfaithful=(), faithfulness=1.0, answer_relevance=1.0,
                missing_required=(), citation_coverage=1.0, citations=2,
                min_citations=2, attribution_ok=attribution_ok, misattributed=(), agreement=1.0,
                samples_usable=1, samples_total=1, corrective_added=0,
                pack_matches_gold=True, pack_gold_diff=(),
            )

        report = NarrativeEvaluationReport(
            results=(row("a", True, True), row("b", True, False), row("c", False, True)),
            gold_set=GOLD_SET,
        )

        assert report.service_rate == pytest.approx(2 / 3)
        assert report.served_wrong_rate == pytest.approx(1 / 2)

    def test_gold_drift_marks_the_run_untrustworthy(
        self, one_case_gold, configs, index, tmp_path
    ):
        """
        A gold set out of step with its payloads measures the wrong thing, so
        the report says so instead of publishing a clean-looking number.
        """
        case = one_case_gold.cases[0]
        corrupted = tmp_path / "drift.json"
        values = dict(case.expected_values)
        values["list_price"] = values["list_price"] + 1000.0
        corrupted.write_text(
            json.dumps(
                {
                    "language": "en",
                    "cases": [
                        {
                            "case_id": case.case_id,
                            "recommendation_id": case.recommendation_id,
                            "payloads": str(case.payloads),
                            "required_facts": list(case.required_facts),
                            "expected_values": values,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        report = evaluate_narratives(
            NarrativeGoldSet.load(corrupted),
            narrator_for(ScriptedLLM(by_value=True), configs, index),
        )

        assert not report.trustworthy
        assert report.gold_drift
        assert "GOLD DRIFT" in report.summary()

    def test_the_report_serialises(self, one_case_gold, configs, index):
        report = evaluate_narratives(
            one_case_gold,
            narrator_for(ScriptedLLM(by_value=True), configs, index),
            label="unit",
            config_summary={"samples": 1},
        )

        payload = json.loads(json.dumps(report.to_dict()))

        assert payload["label"] == "unit"
        assert payload["served_wrong_target"] == SERVED_WRONG_TARGET
        assert len(payload["results"]) == 1


# --------------------------------------------------------------------------
# regression check
# --------------------------------------------------------------------------
class TestRegression:
    """
    The wiring the plan asks for: "run it as a regression check".

    A faithful model over the whole frozen gold set must clear the target.
    This runs offline against a scripted model, so it guards the harness and
    the pipeline wiring on every commit; the real-model numbers come from
    ``evaluate_narratives.py`` and are recorded in the deliverable document.
    """

    def test_a_faithful_model_clears_the_served_wrong_target(self, configs, index):
        gold = NarrativeGoldSet.load(GOLD_SET)

        report = evaluate_narratives(
            gold, narrator_for(ScriptedLLM(by_value=True), configs, index),
            label="regression",
        )

        assert report.trustworthy, f"gold drift: {report.gold_drift}"
        assert report.served_wrong_rate <= SERVED_WRONG_TARGET
        assert report.meets_served_wrong_target
        assert report.faithfulness == 1.0
        assert report.citation_coverage == 1.0

    def test_the_attribution_gate_is_what_separates_d5_from_d4(
        self, one_case_gold, configs, index
    ):
        """
        The comparison the deliverable rests on, made honest.

        An earlier version of the baseline still ran D5's attribution gate,
        so both configurations scored 0% served-wrong and the layer looked
        pointless. With enforcement off, the misattributed narrative is
        served and counted wrong; with it on, the same narrative is withheld.
        That difference is what D5 buys.
        """
        from dataclasses import replace

        case = one_case_gold.cases[0]
        pack = build_facts(case.load_recommendation())
        handle = pack.citations[0]
        # Verification-clean: both figures are in the pack, the citation
        # resolves, every claim is cited. Only the *nouns* are wrong.
        text = (
            f"The list price is {case.expected_values['list_price']:,.2f} SAR "
            f"[{handle}]. The invoice price is "
            f"{case.expected_values['pocket_price']:,.2f} SAR [{handle}]."
        )
        narrative_config, reliability = configs

        baseline = evaluate_narratives(
            one_case_gold,
            narrator_for(
                ScriptedLLM(text),
                (narrative_config, replace(reliability, enforce_attribution=False)),
                index,
            ),
        )
        guarded = evaluate_narratives(
            one_case_gold,
            narrator_for(
                ScriptedLLM(text),
                (narrative_config, replace(reliability, enforce_attribution=True)),
                index,
            ),
        )

        assert baseline.service_rate == 1.0
        assert baseline.served_wrong_rate == 1.0
        assert not baseline.meets_served_wrong_target

        assert guarded.service_rate == 0.0
        assert guarded.served_wrong_rate == 0.0
        assert guarded.meets_served_wrong_target

    def test_the_whole_gold_set_is_exercised(self, configs, index):
        gold = NarrativeGoldSet.load(GOLD_SET)

        report = evaluate_narratives(
            gold, narrator_for(ScriptedLLM(by_value=True), configs, index)
        )

        assert report.cases == len(gold.cases)
        assert report.cases >= 24, "the set was doubled to make the rate meaningful"
