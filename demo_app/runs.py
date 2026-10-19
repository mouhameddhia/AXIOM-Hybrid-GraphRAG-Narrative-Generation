"""
Pricing-engine output as the demo sees it: a *run* holding explainable *cases*.

Two formats are understood:

* **engine** -- the engine's own run object (``waterfalls``, ``segments``,
  ``recommendations``, ``elasticities``; draft 0.2.0), read by
  :mod:`src.narrative.engine_draft`;
* **flat** -- one self-contained result per order line, the format the explanation
  components were developed against (``data/track_a_payloads_eval.json``), read by
  :mod:`src.narrative.contract`.

Either way a case carries the closed fact pack the explanation may use, a
structured ``view`` for the page, and the questions it can answer. A result that
is malformed or does not reconcile becomes a case with an ``error``: the parser's
refusal, shown as it is, never repaired.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.narrative.contract import ContractError, PriceRecommendation
from src.narrative.engine_draft import (
    EngineFormatError,
    EngineSubject,
    is_engine_result,
    parse_engine_result,
    suggested_questions,
)
from src.narrative.facts import FactPack, build_facts

__all__ = ["DemoCase", "DemoRun", "RunFormatError", "build_run", "parse_text"]


class RunFormatError(ValueError):
    """The file is not usable pricing-engine output; the message says why."""


@dataclass(frozen=True)
class DemoCase:
    id: str
    title: str
    pack: FactPack | None
    subject: Any
    view: Mapping[str, Any] = field(default_factory=dict)
    questions: tuple[str, ...] = ()
    error: str | None = None


@dataclass
class DemoRun:
    id: str
    title: str
    format: str  # "engine" or "flat"
    cases: dict[str, DemoCase]
    raw: Any
    provenance: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    builtin: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "format": self.format,
            "cases": len(self.cases),
            "refused": sum(1 for c in self.cases.values() if c.error),
            "provenance": dict(self.provenance),
            "notes": list(self.notes),
            "builtin": self.builtin,
        }


def parse_text(text: str) -> Any:
    """Parse JSON text, naming the line and column of a syntax error."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RunFormatError(f"not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}") from exc


def build_run(run_id: str, title: str, data: Any, *, builtin: bool = False) -> DemoRun:
    """Turn parsed JSON into a run, detecting the format."""
    if isinstance(data, Mapping) and is_engine_result(data) and not _looks_flat(data):
        try:
            engine = parse_engine_result(data)
        except EngineFormatError as exc:
            raise RunFormatError(str(exc)) from exc
        cases = {
            c.case_id: DemoCase(c.case_id, c.title, c.pack, c.subject, c.view, c.questions)
            for c in engine.cases
        }
        provenance = {
            "contract_version": engine.contract_version,
            "engine_version": engine.engine_version,
            "computed_at": engine.computed_at,
        }
        return DemoRun(run_id, title, "engine", cases, data, provenance, engine.notes, builtin)

    entries = _flat_entries(data)
    cases: dict[str, DemoCase] = {}
    for index, entry in enumerate(entries, start=1):
        case_id = str(entry.get("recommendation_id") or f"result-{index}")
        cases[case_id] = _flat_case(case_id, entry)
    if not cases:
        raise RunFormatError("no pricing result found: expected a list under 'recommendations'")
    notes = tuple(str(data[k]) for k in ("_provenance", "_generated_by") if isinstance(data, Mapping) and data.get(k))
    return DemoRun(run_id, title, "flat", cases, data, {}, notes, builtin)


def _looks_flat(data: Mapping[str, Any]) -> bool:
    recs = data.get("recommendations")
    return (
        isinstance(recs, list)
        and bool(recs)
        and isinstance(recs[0], Mapping)
        and ("recommended_price" in recs[0] or "recommendation_id" in recs[0])
        and not any(k in data for k in ("waterfalls", "segments", "elasticities", "provenance"))
    )


def _flat_entries(data: Any) -> list[Mapping[str, Any]]:
    if isinstance(data, Mapping) and isinstance(data.get("recommendations"), list):
        entries = data["recommendations"]
    elif isinstance(data, Mapping) and "recommended_price" in data:
        entries = [data]
    elif isinstance(data, list):
        entries = data
    else:
        raise RunFormatError(
            "unrecognised format: expected the engine's run object (with 'recommendations', 'waterfalls', ...) "
            "or a flat result (or a list of them) with 'recommended_price'"
        )
    bad = [i for i, e in enumerate(entries) if not isinstance(e, Mapping)]
    if bad:
        raise RunFormatError(f"entry {bad[0] + 1} is not a JSON object")
    return list(entries)


def _flat_case(case_id: str, entry: Mapping[str, Any]) -> DemoCase:
    title = f"{case_id} · {entry.get('product_id', '?')} / {entry.get('customer_id', '?')}"
    try:
        recommendation = PriceRecommendation.from_dict(entry)
    except ContractError as exc:
        return DemoCase(case_id, title + " (refused)", None, None, error=str(exc))
    pack = build_facts(recommendation)
    view = _flat_view(entry)
    title += f" · {recommendation.recommended_price:,.2f} {recommendation.currency_code}"
    return DemoCase(case_id, title, pack, recommendation, view, suggested_questions(view))


def _flat_view(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The same structure the engine cases expose, built from a flat result."""
    currency = entry.get("currency_code", "")
    rng = entry.get("price_range") or {}
    view: dict[str, Any] = {
        "recommendation": {
            "status": "complete",
            "price_range": {"low": rng.get("low"), "mid": entry.get("recommended_price"), "high": rng.get("high")},
            "acceptance": round(float(entry.get("acceptance_probability", 0)) * 100, 1),
            "expected_margin_pct": entry.get("expected_margin_pct"),
            "confidence": {"score": entry.get("confidence"), "band": None, "reasons": []},
            "constraints": [],
        }
    }
    wf = entry.get("waterfall")
    if isinstance(wf, Mapping):
        levels = [
            {"key": "list_value", "label": "list price", "amount": wf.get("list_price"), "status": "complete", "note": None},
            {"key": "invoice_price", "label": "invoice price", "amount": wf.get("invoice_price"), "status": "complete", "note": None},
            {"key": "pocket_price", "label": "pocket price actually kept", "amount": wf.get("pocket_price"), "status": "complete", "note": None},
        ]
        view["waterfall"] = {
            "order_line_id": entry.get("order_line_id"),
            "levels": levels,
            "currency": currency,
            "steps": [
                {"name": s.get("name"), "amount": s.get("amount"), "on_invoice": s.get("on_invoice")}
                for s in wf.get("steps", [])
            ],
            "pct": {},
            "confidence": None,
            "status": "complete",
        }
    return view
