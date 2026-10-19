"""
One question through the whole architecture, live.

``QueryRunner.run`` takes what the user typed and drives every stage, emitting an
event as each finishes so the page can show the real output of each component:

    1 understand   identifiers, the Track A result the question is about (or the
                   exact-value path), and why
    2 retrieve     hybrid retrieval: ranked evidence with the stream trace
    3 track_a      the Track A result: structured view and closed fact pack
    4 exact        text-to-query: plan, Cypher, the value the graph computes
    5 pack         the closed set of figures and statements an answer may use
    6 generate     k samples, each generated then verified (numbers, citations,
                   wrong labels)
    7 reliability  evidence check, corrective retrieval, quorum, agreement, decision
    8 evaluate     the outcome scored with the project's own evaluation code

Routing is deterministic and visible: identifiers in the question select a Track A
result (a result the user picked beside the question is used when the question
names none); a factual question that names no result goes to text-to-query; a
question about a price that names nothing asks the user to choose. Nothing here
decides what an answer says -- that stays with the generator, verifier and
reliability layer.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from demo_app.events import fact_rows, plan_dict, serialise
from demo_app.runs import DemoCase, DemoRun
from src.config.settings import PROJECT_ROOT
from src.evaluation.harness import GoldSet
from src.evaluation.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from src.evaluation.narrative_harness import NarrativeGoldSet, score_case
from src.narrative.engine_draft import EngineSubject
from src.narrative.facts import Fact, FactPack
from src.narrative.verifier import verify
from src.reliability.attribution import check_attribution

__all__ = ["QueryRunner", "extract_identifiers", "is_overview"]

DATA = PROJECT_ROOT / "data"
Emit = Callable[[dict[str, Any]], None]

_IDENT = re.compile(r"\b([A-Za-z]{2,5}-?\d{2,})\b")
#: A question that asks for a value stored in the graph.
_EXACT_INTENT = re.compile(
    r"\b(how many|number of|count of|average|maximum|minimum|total|sum of|highest|lowest|"
    r"unit cost|list price|credit terms|shelf life|which agent)\b",
    re.IGNORECASE,
)
#: A question about what the pricing engine produced.
_RESULT_INTENT = re.compile(
    r"\b(recommend\w*|pocket|invoice|waterfall|erosion|margin|elasticity|segment|confidence|"
    r"acceptance|explain|why|deduction\w*|rebate|freight|discount|price range)\b",
    re.IGNORECASE,
)
_OVERVIEW = re.compile(r"^\s*(please\s+)?(explain|describe|summari[sz]e|walk me through)\b", re.IGNORECASE)
_SPECIFIC = re.compile(r"\b(why|how|which|whether|reliable|difference|lower|higher|than|compare)\b", re.IGNORECASE)


def extract_identifiers(question: str) -> list[str]:
    """Identifier-like tokens in the question (SKU0026, CUST0009, OL000248, REC-0005 ...)."""
    return list(dict.fromkeys(token.upper() for token in _IDENT.findall(question)))


def is_overview(question: str) -> bool:
    """A request to explain the whole result rather than to answer a specific question."""
    return bool(_OVERVIEW.search(question)) and not _SPECIFIC.search(question)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def _case_identifiers(case: DemoCase) -> dict[str, int]:
    """Identifiers a question can use to name this result, with how specific each is."""
    ids: dict[str, int] = {case.id.upper(): 3}
    subject = case.subject
    for attr in ("order_line_id", "product_id", "customer_id"):
        value = getattr(subject, attr, None)
        if value:
            ids[str(value).upper()] = 2
    if case.pack is not None:
        for handle in case.pack.citations:
            ids.setdefault(handle.split(":", 1)[-1].upper(), 1)
    return ids


class QueryRunner:
    """Runs one question end to end. Holds no per-question state."""

    def __init__(self, state) -> None:
        self.state = state
        self._narrative_gold = {
            c.recommendation_id: c for c in NarrativeGoldSet.load(DATA / "narrative_gold_set.json").cases
        }
        self._retrieval_gold = self._load_retrieval_gold()
        self._exact_gold = self._load_exact_gold()

    # ------------------------------------------------------------------ reference data
    @staticmethod
    def _load_retrieval_gold() -> dict[str, dict[str, Any]]:
        references: dict[str, dict[str, Any]] = {}
        try:
            for case in GoldSet.load(DATA / "gold_set.json").cases:
                for text in case.questions.values():
                    references[_normalise(text)] = {
                        "source": f"gold set, case {case.case_id}",
                        "primary": case.primary,
                        "relevant": case.relevant,
                        "gains": dict(case.gains),
                    }
            bank = json.loads((DATA / "question_bank.json").read_text(encoding="utf-8"))
            for category in bank["categories"]:
                for q in category["questions"]:
                    expected = frozenset(q.get("expect_records") or ())
                    if expected and q.get("en"):
                        references.setdefault(
                            _normalise(q["en"]),
                            {"source": f"question bank, {q['id']}", "primary": None, "relevant": expected, "gains": {}},
                        )
        except (OSError, ValueError, KeyError):
            pass
        return references

    @staticmethod
    def _load_exact_gold() -> dict[str, dict[str, Any]]:
        try:
            cases = json.loads((DATA / "cypher_gold_set.json").read_text(encoding="utf-8"))["cases"]
        except (OSError, ValueError, KeyError):
            return {}
        return {_normalise(c["question"]): c for c in cases}

    # ------------------------------------------------------------------ routing
    def _match(self, ids: list[str], selected: tuple[str, str] | None):
        """Track A results the question names, best first, with the identifiers that matched."""
        scored = []
        wanted = set(ids)
        for run in self.state.runs.values():
            for case in run.cases.values():
                if case.error or case.pack is None:
                    continue
                known = _case_identifiers(case)
                hits = [i for i in ids if i in known]
                if not hits:
                    continue
                scored.append((sum(known[i] for i in hits), len(hits), run, case, hits))
        scored.sort(key=lambda row: (-row[0], -row[1]))
        if selected:
            for row in scored:
                if (row[2].id, row[3].id) == selected:
                    scored.remove(row)
                    scored.insert(0, row)
                    break
        return scored, wanted

    def run(self, question: str, selected: tuple[str, str] | None, safeguards: bool, emit: Emit) -> None:
        state = self.state
        started = time.time()

        def send(stage: str, **payload: Any) -> None:
            emit({"stage": stage, "t": int((time.time() - started) * 1000), **payload})

        ids = extract_identifiers(question)
        anchors = state.stack.anchor_resolver.resolve(question)
        scored, _ = self._match(ids, selected)

        match = None
        candidates: list[dict[str, Any]] = []
        if scored:
            top = scored[0]
            match = top
            tied = [row for row in scored if row[0] == top[0] and row[1] == top[1]]
            candidates = [{"run": r[2].id, "case": r[3].id, "title": r[3].title} for r in scored[:6]]
            if len(tied) > 1 and not (selected and (top[2].id, top[3].id) == selected):
                tie_note = f"{len(tied)} results match equally; showing the first"
            else:
                tie_note = None
        elif selected and not ids:
            run = state.runs.get(selected[0])
            case = run.cases.get(selected[1]) if run else None
            if case is not None and case.pack is not None:
                match = (1, 0, run, case, [])
            tie_note = None
        else:
            tie_note = None

        exact_intent = bool(_EXACT_INTENT.search(question))
        result_intent = bool(_RESULT_INTENT.search(question))
        asks_for_stored_value = match is not None and exact_intent and not result_intent
        if asks_for_stored_value:
            match = None
            candidates = []
            tie_note = None
        if match is not None:
            path = "explain"
            if match[4]:
                reason = f"the question names {', '.join(match[4])}, which identifies this Track A result"
            else:
                reason = "no identifier in the question, so the result selected beside it is used"
        elif result_intent and not ids:
            path = "need_result"
            reason = "the question is about a price but names no product, customer or order line"
        else:
            path = "exact"
            reason = (
                "the question asks for a value stored in the graph, so it is computed there"
                if asks_for_stored_value
                else "no Track A result is named, so the question is answered from the graph"
            )

        send(
            "route",
            question=question,
            identifiers=ids,
            anchors=[{"mention": a.mention, "record_id": a.record_id} for a in anchors][:8],
            path=path,
            reason=reason,
            note=tie_note,
            matched=(
                {"run": match[2].id, "case": match[3].id, "title": match[3].title, "matched_on": match[4]}
                if match
                else None
            ),
            candidates=candidates,
        )

        # ---- 2 retrieval (always)
        outcome = state.stack.pipeline.run(question, top_k=10)
        ranked = [r.record_id for r in outcome.final]
        send(
            "retrieval",
            streams=list(state.stack.pipeline.stream_names),
            results=[
                {
                    "rank": rank,
                    "record_id": r.record_id,
                    "record_type": r.record_type,
                    "text": r.text[:300],
                    "streams": dict(r.metadata.get("stream_ranks", {})) if r.metadata else {},
                }
                for rank, r in enumerate(outcome.final, start=1)
            ],
            timings_ms={k: round(v, 2) for k, v in outcome.timings_ms.items()},
        )

        case = match[3] if match else None
        run_of_case = match[2] if match else None
        question_for_generation: str | None = None if (path == "explain" and is_overview(question)) else question

        if path == "need_result":
            send("trackA", skipped=True, reason="Choose a result in the list, or name a product, customer or order line.")
            send("exact", skipped=True, reason="Not a request for a value computed from the graph.")
            self._finish(send, started, question, ranked, None, None, None, None)
            return

        # ---- 3 Track A result
        if path == "explain":
            send("trackA", **state.case_payload(run_of_case, case))
        else:
            send("trackA", skipped=True, reason="The question does not name a Track A result.")

        # ---- 4 exact facts
        exact_result = None
        pack: FactPack | None = case.pack if case is not None else None
        subject = case.subject if case is not None else None
        kind = "explain"
        if path == "exact":
            exact_result = state.engine.ask(question)
            send(
                "exact",
                success=exact_result.success,
                plan=plan_dict(exact_result.plan),
                cypher=exact_result.cypher,
                answer=dict(exact_result.answer),
                evidence=[
                    {"record_id": e.record_id, "record_type": e.record_type, "text": e.text[:240]}
                    for e in exact_result.evidence
                ],
                attempts=exact_result.attempts,
                error=exact_result.error,
            )
            if exact_result.success:
                pack = self._exact_pack(question, exact_result)
                subject = EngineSubject("", "", None)
                kind = "exact"
        else:
            send("exact", skipped=True, reason="The question is about a Track A result, so its figures come from the engine.")

        # ---- 5-7 generation and reliability
        outcome_obj = None
        if pack is None:
            if path == "exact":
                send("decision", served=False, text="", reason=(
                    "no value could be computed from the graph, so there is nothing to explain "
                    f"({exact_result.error if exact_result else 'no query'})"), question=question,
                    citations=[], allowed_citations=[], facts=[], context_values=[], agreement=0.0, samples_used=0, nogen=True)
        else:
            narrator = state.narrators[(kind, bool(safeguards))]
            k = state.settings.reliability.samples if safeguards else 1

            def observer(stage: str, data: Any) -> None:
                send(**serialise(state.settings, stage, data, k, question_for_generation))

            outcome_obj = narrator.explain(
                subject, pack=pack, observer=observer, question=question_for_generation or (question if kind == "exact" else None)
            )

        self._finish(send, started, question, ranked, case, run_of_case, exact_result, (pack, outcome_obj, question_for_generation))

    # ------------------------------------------------------------------ exact answers
    @staticmethod
    def _exact_pack(question: str, result) -> FactPack | None:
        value = result.value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or result.plan is None:
            return None
        plan = result.plan
        subject = plan.aggregate_property or "records"
        label = f"{plan.aggregate or 'value'} of {subject} over {plan.label} records"
        if plan.filters:
            label += " where " + " and ".join(f"{f.property} {f.op} {f.value}" for f in plan.filters)
        return FactPack(
            subject=question,
            kind="graph_answer",
            currency_code="",
            facts=(Fact("graph_value", float(value), "count" if plan.aggregate == "count" else "number", "graph-query", label),),
            citations=tuple(result.citations),
            engine_version="graph-query",
        )

    # ------------------------------------------------------------------ evaluation
    def _finish(self, send, started, question, ranked, case, run, exact_result, generated) -> None:
        evaluation: dict[str, Any] = {}
        try:
            evaluation["retrieval"] = self._eval_retrieval(question, ranked, case)
            if generated is not None and generated[1] is not None:
                evaluation["generation"] = self._eval_generation(run, case, *generated)
            if exact_result is not None and exact_result.success:
                evaluation["exact"] = self._eval_exact(question, exact_result)
        except Exception as exc:  # an evaluation fault must not hide the answer
            evaluation["error"] = str(exc)
        send("evaluation", **evaluation)
        send("done", elapsed_s=round(time.time() - started, 1))

    def _eval_retrieval(self, question: str, ranked: list[str], case: DemoCase | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if case is not None and case.pack is not None:
            wanted = list(case.pack.citations)
            if wanted:
                found = [h for h in wanted if h in ranked]
                out["trackA"] = {
                    "wanted": len(wanted),
                    "found": len(found),
                    "recall": round(len(found) / len(wanted), 3),
                    "first_rank": min((ranked.index(h) + 1 for h in found), default=None),
                    "missing": [h for h in wanted if h not in ranked],
                }
        reference = self._retrieval_gold.get(_normalise(question))
        if reference:
            relevant = reference["relevant"]
            out["reference"] = {
                "source": reference["source"],
                "mrr": round(reciprocal_rank(ranked, relevant), 3),
                "hit_at_10": hit_at_k(ranked, relevant, 10),
                "recall_at_10": round(recall_at_k(ranked, relevant, 10), 3),
                "ndcg_at_10": round(ndcg_at_k(ranked, reference["gains"], 10), 3) if reference["gains"] else None,
                "primary": reference["primary"],
                "primary_rank": (ranked.index(reference["primary"]) + 1) if reference["primary"] in ranked else None,
            }
        return out

    def _eval_generation(self, run, case, pack: FactPack, outcome, asked: str | None) -> dict[str, Any]:
        result: dict[str, Any] = {"served": outcome.served, "agreement": round(outcome.agreement, 3)}
        if not outcome.served or outcome.narrative is None:
            result["reason"] = outcome.reason
            return result
        text = outcome.narrative.text
        report = verify(text, pack)
        attribution = check_attribution(text, pack)
        result["reference_free"] = {
            "figures_stated": len(report.numbers_found),
            "figures_traced": len(report.numbers_found) - sum(1 for v in report.violations if v.kind == "unsupported-number"),
            "citation_coverage": round(report.citation_coverage, 3),
            "labels_correct": attribution.ok,
            "misattributed": [m.role_key for m in attribution.misattributions],
        }
        gold = None
        if run is not None and run.id == "simulated" and case is not None:
            gold = self._narrative_gold.get(case.id)
        if gold is not None:
            scored = score_case(gold, pack, outcome)
            result["reference"] = {
                "source": f"frozen gold case {gold.case_id} (expected figures recorded independently of the code)",
                "faithfulness": round(scored.faithfulness, 3),
                "answer_relevance": None if asked else round(scored.answer_relevance, 3),
                "unfaithful": list(scored.unfaithful),
                "missing_required": [] if asked else list(scored.missing_required),
                "citations": scored.citations,
                "min_citations": scored.min_citations,
                "served_wrong": scored.served_wrong,
            }
        return result

    def _eval_exact(self, question: str, result) -> dict[str, Any]:
        reference = self._exact_gold.get(_normalise(question))
        if reference is None:
            return {"reference": None}
        expected = reference["expect_value"]
        value = result.value
        if isinstance(expected, (int, float)) and isinstance(value, (int, float)):
            ok = abs(float(value) - float(expected)) <= max(0.01, abs(float(expected)) * 1e-6)
        else:
            ok = any(str(expected) in (e.text or "") for e in result.evidence)
        return {
            "reference": {"source": f"exact-query gold case {reference['id']}", "expected": expected, "got": value, "correct": ok}
        }
