"""
Tests for the evaluation harness.

Metrics are asserted against hand-computed values rather than against the
implementation's own output. A metric that silently drifts would invalidate
every quality claim made from it, so these are checked from the definitions.
"""
from __future__ import annotations

import json
import math

import pytest

from src.evaluation.harness import (
    PRIMARY_GAIN,
    SUPPORTING_GAIN,
    GoldCase,
    GoldSet,
    GoldSetError,
    evaluate,
)
from src.evaluation.metrics import (
    average_precision,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# -- metrics ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("ranked", "expected"),
    [
        (["a", "x", "y"], 1.0),
        (["x", "a", "y"], 0.5),
        (["x", "y", "a"], 1 / 3),
        (["x", "y", "z"], 0.0),
        ([], 0.0),
    ],
)
def test_reciprocal_rank(ranked, expected):
    assert reciprocal_rank(ranked, frozenset({"a"})) == pytest.approx(expected)


def test_recall_counts_only_the_top_k():
    ranked = ["a", "x", "b", "y", "c"]
    relevant = frozenset({"a", "b", "c"})

    assert recall_at_k(ranked, relevant, 1) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, relevant, 5) == pytest.approx(1.0)


def test_recall_of_an_empty_relevant_set_is_zero_not_an_error():
    assert recall_at_k(["a"], frozenset(), 5) == 0.0


def test_precision_at_k():
    ranked = ["a", "x", "b"]
    assert precision_at_k(ranked, frozenset({"a", "b"}), 3) == pytest.approx(2 / 3)
    assert precision_at_k(ranked, frozenset({"a", "b"}), 1) == pytest.approx(1.0)
    assert precision_at_k([], frozenset({"a"}), 3) == 0.0


def test_average_precision_rewards_front_loading():
    """Both rankings contain the same hits; the earlier one must score higher."""
    relevant = frozenset({"a", "b"})
    early = average_precision(["a", "b", "x", "y"], relevant)
    late = average_precision(["x", "y", "a", "b"], relevant)

    # (1/1 + 2/2) / 2 == 1.0
    assert early == pytest.approx(1.0)
    # (1/3 + 2/4) / 2
    assert late == pytest.approx((1 / 3 + 2 / 4) / 2)
    assert early > late


def test_hit_at_k():
    assert hit_at_k(["x", "a"], frozenset({"a"}), 2) == 1.0
    assert hit_at_k(["x", "a"], frozenset({"a"}), 1) == 0.0


def test_ndcg_is_one_for_the_ideal_ordering():
    gains = {"a": 3.0, "b": 1.0}
    assert ndcg_at_k(["a", "b", "x"], gains, 3) == pytest.approx(1.0)


def test_ndcg_penalises_context_above_the_answer():
    """
    Graded relevance is the point: both rankings retrieve the same two records,
    but the one putting the answering record first must score higher.
    """
    gains = {"answer": PRIMARY_GAIN, "context": SUPPORTING_GAIN}

    answer_first = ndcg_at_k(["answer", "context"], gains, 2)
    context_first = ndcg_at_k(["context", "answer"], gains, 2)

    assert answer_first == pytest.approx(1.0)
    assert context_first < answer_first
    # DCG = 1/log2(2) + 3/log2(3); ideal = 3/log2(2) + 1/log2(3)
    expected = (1.0 + 3.0 / math.log2(3)) / (3.0 + 1.0 / math.log2(3))
    assert context_first == pytest.approx(expected)


def test_ndcg_of_an_empty_ranking_is_zero():
    assert ndcg_at_k([], {"a": 3.0}, 5) == 0.0
    assert ndcg_at_k(["a"], {}, 5) == 0.0


# -- gold set ---------------------------------------------------------------


def test_project_gold_set_loads_and_validates(corpus):
    """
    The committed gold set must match the corpus the pipeline actually builds.

    This is the guard that keeps a stale label from looking like a retrieval
    regression.
    """
    from src.config.settings import PROJECT_ROOT

    path = PROJECT_ROOT / "data" / "gold_set.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("gold set not present")

    # Passing the corpus validates every referenced record id against the real
    # graph. This became a meaningful check once the suite moved onto the PoC
    # export: against the older sample extract, the ids the gold set names did
    # not exist, so validation had to be skipped and a stale label could
    # survive unnoticed.
    gold_set = GoldSet.load(path, corpus)

    assert gold_set.cases
    for case in gold_set.cases:
        assert case.primary
        assert case.questions["en"]


def test_unknown_record_id_is_rejected(tmp_path, corpus):
    """A label naming a nonexistent record must fail the run, not score zero."""
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "bogus",
                        "question_en": "anything",
                        "primary": "NoSuchLabel:NOPE",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GoldSetError, match="absent from the corpus"):
        GoldSet.load(path, corpus)


def test_case_without_a_question_is_rejected(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps({"cases": [{"id": "x", "primary": "A:1"}]}), encoding="utf-8"
    )
    with pytest.raises(GoldSetError, match="no question"):
        GoldSet.load(path)


def test_case_without_a_primary_is_rejected(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps({"cases": [{"id": "x", "question_en": "q"}]}), encoding="utf-8"
    )
    with pytest.raises(GoldSetError, match="no 'primary'"):
        GoldSet.load(path)


def test_missing_gold_set_file_is_reported(tmp_path):
    with pytest.raises(GoldSetError, match="not found"):
        GoldSet.load(tmp_path / "absent.json")


def test_gold_case_grades_primary_above_supporting():
    case = GoldCase(
        case_id="c",
        questions={"en": "q"},
        primary="A:1",
        supporting=("B:2",),
    )

    assert case.relevant == frozenset({"A:1", "B:2"})
    assert case.gains["A:1"] > case.gains["B:2"]


# -- end to end -------------------------------------------------------------


def test_evaluate_scores_a_real_pipeline(settings, encoder, cross_encoder, corpus):
    """The harness must run end to end against the real stack."""
    from src.pipeline.factory import build_retrieval_stack

    stack = build_retrieval_stack(settings, encoder=encoder, cross_encoder=cross_encoder)
    record = stack.corpus[0]
    gold_set = GoldSet(
        cases=(
            GoldCase(
                case_id="synthetic",
                questions={"en": record.text[:80]},
                primary=record.record_id,
                supporting=(),
            ),
        ),
        source=settings.corpus.json_path,
    )

    report = evaluate(stack.pipeline, gold_set, k=5, label="test")

    assert len(report.results) == 1
    assert 0.0 <= report.mrr <= 1.0
    assert 0.0 <= report.ndcg_at_k <= 1.0
    assert report.as_dict()["label"] == "test"


def test_report_splits_by_language(settings, encoder, cross_encoder):
    from src.pipeline.factory import build_retrieval_stack

    stack = build_retrieval_stack(settings, encoder=encoder, cross_encoder=cross_encoder)
    record = stack.corpus[0]
    gold_set = GoldSet(
        cases=(
            GoldCase(
                case_id="bilingual",
                questions={"en": record.text[:60], "ar": "ما هو السعر"},
                primary=record.record_id,
                supporting=(),
            ),
        ),
        source=settings.corpus.json_path,
    )

    report = evaluate(stack.pipeline, gold_set, k=5)

    assert set(report.by_language()) == {"en", "ar"}
    assert len(report.results) == 2


# -- question bank ----------------------------------------------------------


def test_project_question_bank_validates_against_the_real_corpus(corpus):
    """
    Every record id the question bank expects must exist in the graph.

    Like the gold set above, this only became checkable once the suite moved
    onto the PoC export. A bank asserting ids that do not exist would report
    retrieval failures that were really labelling errors.
    """
    from src.config.settings import PROJECT_ROOT
    from src.evaluation.question_bank import QuestionBank

    path = PROJECT_ROOT / "data" / "question_bank.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("question bank not present")

    bank = QuestionBank.load(path, corpus)

    assert bank.categories
    assert set(bank.languages) == {"en", "ar"}
    for probe in bank.all_probes:
        assert probe.questions.get("en") and probe.questions.get("ar")


def test_question_bank_rejects_an_unknown_record(corpus, tmp_path):
    from src.evaluation.question_bank import QuestionBank, QuestionBankError

    path = tmp_path / "bank.json"
    path.write_text(
        json.dumps(
            {
                "categories": [
                    {
                        "id": "c",
                        "questions": [
                            {
                                "id": "q",
                                "en": "x",
                                "ar": "x",
                                "expect_records": ["NoSuchLabel:NOPE"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(QuestionBankError, match="absent from the corpus"):
        QuestionBank.load(path, corpus)
