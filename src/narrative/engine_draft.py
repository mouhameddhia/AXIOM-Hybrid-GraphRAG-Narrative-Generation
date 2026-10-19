"""
Reader for the pricing engine's own result format (``EngineResult``, draft 0.2.0).

The engine emits one object per run with four parallel arrays -- ``waterfalls``,
``segments``, ``recommendations``, ``elasticities`` -- plus provenance and
data-quality flags. Every record carries its own confidence and evidence
identifiers, and every figure is a ``Money`` value with a status. This module
turns one such run into *explainable cases*: one per recommendation (with the
waterfall of the order line it rests on, the customer's segment, the product's
elasticity and any flag that applies) and one per waterfall that no
recommendation refers to.

**Experimental.** The engine's contract is a draft: itemised waterfall steps were
confirmed to be coming but their shape is not final, and the sample that exercises
this reader is hand-built (``data/track_a_real_schema_sample_output.json``). The
reader follows the same rules as :mod:`src.narrative.contract`:

* it *reads*, never computes. The only arithmetic is the reconciliation check
  (``gross - on-invoice = invoice``, ``invoice - off-invoice = pocket`` ...), whose
  result is accept or refuse; a figure that does not add up is refused with the
  mismatch named, never corrected;
* a ``Money`` value whose status is not ``complete`` contributes no figure, only a
  statement that it is unavailable (shown as unavailable, never as zero);
* qualitative content (confidence bands and reasons, notes, flags) becomes
  *statements* the model may restate but not strengthen; numbers appearing inside
  them are admitted because they came from the engine;
* an abstained recommendation is a case in its own right: the explanation is that
  the engine declined to recommend, and why.

The mapping from the engine's identifier lists to graph handles is the one place
that names labels (``ENGINE_ID_LABELS``); everything else is generic.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.narrative.facts import Fact, FactPack
from src.narrative.verifier import extract_figures

__all__ = [
    "ENGINE_ID_LABELS",
    "EngineCase",
    "EngineFormatError",
    "EngineRun",
    "EngineSubject",
    "is_engine_result",
    "parse_engine_result",
    "suggested_questions",
]

#: How the engine's identifier lists map to graph labels (the handle is
#: ``Label:identifier``). The single place that knows this.
ENGINE_ID_LABELS: Mapping[str, str] = {
    "transaction_ids": "OrderLine",
    "customer_ids": "TradeCustomer",
    "product_ids": "SellableProduct",
    "discount_ids": "PriceComponent",
}

#: Reconciliation slack, in currency minor units.
TOLERANCE = 0.02

_WATERFALL_LEVELS: tuple[tuple[str, str, str, bool], ...] = (
    # engine field, fact key, label, required
    ("list_value", "list_price", "list price", True),
    ("negotiated_off_list", "negotiated_off_list", "reduction negotiated before invoicing", False),
    ("gross_value", "gross_value", "gross value", False),
    ("on_invoice_discount", "on_invoice_discount", "total deductions shown on the invoice", True),
    ("invoice_price", "invoice_price", "invoice price", True),
    ("off_invoice_leakage", "off_invoice_leakage", "total deductions NOT shown on the invoice", True),
    ("pocket_price", "pocket_price", "pocket price actually kept", True),
    ("cogs", "cogs", "cost of goods sold", False),
    ("pocket_margin", "pocket_margin", "pocket margin (pocket price minus cost of goods sold)", False),
)

_STEP_LABELS = {
    "invoice_discount": "invoice discount",
    "payment_discount": "payment discount",
    "volume_rebate": "volume rebate",
    "freight": "freight",
    "promotional_discount": "promotional discount",
    "commission": "commission",
}


class EngineFormatError(ValueError):
    """Raised when engine output is malformed, incomplete or does not reconcile."""


@dataclass(frozen=True, slots=True)
class EngineSubject:
    """What corrective retrieval needs to find records about this case."""

    product_id: str
    customer_id: str
    order_line_id: str | None


@dataclass(frozen=True, slots=True)
class EngineCase:
    """One explainable unit of an engine run."""

    case_id: str
    title: str
    pack: FactPack
    subject: EngineSubject
    #: Structured content for a front end (waterfall, recommendation, segment ...).
    view: Mapping[str, Any]
    questions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EngineRun:
    contract_version: str
    engine_version: str
    computed_at: str
    cases: tuple[EngineCase, ...]
    notes: tuple[str, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class _Money:
    amount: float | None
    currency: str
    status: str
    note: str | None

    @property
    def usable(self) -> bool:
        return self.amount is not None and self.status in ("complete", "partial")


# ----------------------------------------------------------------------- detection
def is_engine_result(data: Any) -> bool:
    """Whether ``data`` looks like the engine's run object rather than a flat result."""
    if not isinstance(data, Mapping):
        return False
    if not isinstance(data.get("recommendations"), Sequence) or isinstance(data.get("recommendations"), (str, bytes)):
        # a run may hold only waterfalls
        return isinstance(data.get("waterfalls"), Sequence) and not isinstance(data.get("waterfalls"), (str, bytes))
    markers = ("waterfalls", "segments", "elasticities", "provenance")
    return any(marker in data for marker in markers)


# ----------------------------------------------------------------------- primitives
def _fail(path: str, message: str) -> EngineFormatError:
    return EngineFormatError(f"{path}: {message}")


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(path, f"needs a number, got {value!r}")
    return float(value)


def _money(raw: Any, path: str, *, required: bool = False) -> _Money | None:
    if raw is None:
        if required:
            raise _fail(path, "is required")
        return None
    if not isinstance(raw, Mapping):
        raise _fail(path, "must be a money object (amount, currency, status, note)")
    status = str(raw.get("status", "complete"))
    amount = raw.get("amount")
    if amount is None:
        if status in ("complete", "partial"):
            raise _fail(path, f"has status {status!r} but no amount")
        parsed = None
    else:
        parsed = _number(amount, path + ".amount")
    note = raw.get("note")
    return _Money(parsed, str(raw.get("currency") or ""), status, str(note) if note else None)


def _confidence(raw: Any, path: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise _fail(path, "must be an object (score, band, reasons)")
    score = _number(raw.get("score"), path + ".score")
    if not 0.0 <= score <= 1.0:
        raise _fail(path + ".score", f"must be a probability in [0, 1], got {score}")
    reasons = raw.get("reasons") or []
    if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes)):
        raise _fail(path + ".reasons", "must be a list of strings")
    return {"score": score, "band": str(raw.get("band") or "unknown"), "reasons": [str(r) for r in reasons]}


def _evidence_handles(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        raise _fail(path, "must be an object with identifier lists")
    handles: list[str] = []
    for field_name, label in ENGINE_ID_LABELS.items():
        ids = raw.get(field_name) or []
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
            raise _fail(f"{path}.{field_name}", "must be a list of identifiers")
        for identifier in ids:
            text = str(identifier).strip()
            if text:
                handle = f"{label}:{text}"
                if handle not in handles:
                    handles.append(handle)
    return handles


def _percent(score: float) -> float:
    return round(score * 100, 1)


# ----------------------------------------------------------------------- statements
class _Statements:
    """Collects qualitative statements and the figures they contain."""

    def __init__(self, default_citation: str) -> None:
        self.default = default_citation
        self.lines: list[str] = []
        self.values: list[float] = []

    def add(self, text: str, citation: str | None = None) -> None:
        cite = citation or self.default
        self.lines.append(f"{text} [{cite}]")
        self.values.extend(extract_figures(text))


def _confidence_statement(label: str, conf: dict[str, Any] | None, out: _Statements, citation: str | None = None) -> None:
    if conf is None:
        return
    text = f"{label}: confidence band {conf['band']}"
    if conf["reasons"]:
        text += "; reasons given by the engine: " + " | ".join(conf["reasons"])
    out.add(text, citation)


# ----------------------------------------------------------------------- waterfall
@dataclass(slots=True)
class _Waterfall:
    order_line_id: str | None
    levels: dict[str, _Money]
    steps: list[dict[str, Any]]
    pct: dict[str, float]
    confidence: dict[str, Any] | None
    handles: list[str]
    customer_id: str | None
    product_id: str | None
    status: str


def _parse_waterfall(raw: Any, index: int) -> _Waterfall:
    path = f"waterfalls[{index}]"
    if not isinstance(raw, Mapping):
        raise _fail(path, "must be an object")
    key = raw.get("key") or {}
    order_line = str(key.get("order_line_id")) if isinstance(key, Mapping) and key.get("order_line_id") else None
    levels: dict[str, _Money] = {}
    for engine_field, _, _, required in _WATERFALL_LEVELS:
        money = _money(raw.get(engine_field), f"{path}.{engine_field}", required=required)
        if money is not None:
            levels[engine_field] = money

    def amount(name: str) -> float | None:
        money = levels.get(name)
        return money.amount if money is not None and money.usable else None

    def check(label: str, left: float | None, right: float | None, expected: float | None) -> None:
        if left is None or right is None or expected is None:
            return
        if abs(left - right - expected) > TOLERANCE:
            raise EngineFormatError(
                f"{path} does not add up: {label} (stated {expected:,.2f}, but the levels give "
                f"{left - right:,.2f}). The result is refused rather than corrected."
            )

    gross = amount("gross_value")
    listed = amount("list_value")
    on_invoice = amount("on_invoice_discount")
    off_invoice = amount("off_invoice_leakage")
    abs_or_none = lambda v: abs(v) if v is not None else None  # noqa: E731 - deductions are magnitudes
    check("list value minus the negotiated reduction should equal gross value", listed, abs_or_none(amount("negotiated_off_list")), gross)
    check("gross value minus deductions shown on the invoice should equal the invoice price", gross if gross is not None else listed, abs_or_none(on_invoice), amount("invoice_price"))
    check("invoice price minus deductions not shown on the invoice should equal the pocket price", amount("invoice_price"), abs_or_none(off_invoice), amount("pocket_price"))
    check("pocket price minus cost of goods sold should equal the pocket margin", amount("pocket_price"), amount("cogs"), amount("pocket_margin"))

    steps: list[dict[str, Any]] = []
    for position, step in enumerate(raw.get("steps") or []):
        spath = f"{path}.steps[{position}]"
        if not isinstance(step, Mapping):
            raise _fail(spath, "must be an object")
        money = _money(step.get("amount"), spath + ".amount", required=True)
        on_flag = step.get("on_invoice")
        if not isinstance(on_flag, bool):
            raise _fail(spath + ".on_invoice", f"needs a boolean, got {on_flag!r}")
        handles = _evidence_handles(step.get("evidence"), spath + ".evidence")
        discount = next((h for h in handles if h.startswith("PriceComponent:")), "")
        steps.append({"name": str(step.get("name") or f"step {position + 1}"), "money": money, "on_invoice": on_flag, "handle": discount})
    for flag, total, label in ((True, on_invoice, "deductions shown on the invoice"), (False, off_invoice, "deductions not shown on the invoice")):
        group = [abs(s["money"].amount) for s in steps if s["on_invoice"] is flag and s["money"].usable]
        if steps and group and total is not None and abs(sum(group) - abs(total)) > TOLERANCE:
            raise EngineFormatError(
                f"{path}.steps do not add up: the {label} are stated as {abs(total):,.2f} but the itemised steps "
                f"sum to {sum(group):,.2f}. The result is refused rather than corrected."
            )

    pct = {}
    for name in ("pocket_margin_pct", "price_realisation_pct"):
        if raw.get(name) is not None:
            pct[name] = _number(raw[name], f"{path}.{name}")
    evidence = raw.get("evidence") or {}
    handles = _evidence_handles(evidence, path + ".evidence")
    customers = [h.split(":", 1)[1] for h in handles if h.startswith("TradeCustomer:")]
    products = [h.split(":", 1)[1] for h in handles if h.startswith("SellableProduct:")]
    return _Waterfall(
        order_line_id=order_line,
        levels=levels,
        steps=steps,
        pct=pct,
        confidence=_confidence(raw.get("confidence"), path + ".confidence"),
        handles=handles,
        customer_id=customers[0] if customers else None,
        product_id=products[0] if products else None,
        status=str(raw.get("status") or "complete"),
    )


# ----------------------------------------------------------------------- parsing
def parse_engine_result(data: Mapping[str, Any]) -> EngineRun:
    """Turn one engine run into explainable cases, or raise :class:`EngineFormatError`."""
    if not is_engine_result(data):
        raise EngineFormatError("this is not an engine run: expected 'recommendations' and/or 'waterfalls' arrays")
    provenance = data.get("provenance") or {}
    contract_version = str(provenance.get("contract_version") or "unknown")
    engine_version = str(provenance.get("engine_version") or "unknown")
    computed_at = str(provenance.get("computed_at") or "")
    cite = f"engine-{engine_version}"

    waterfalls = [_parse_waterfall(w, i) for i, w in enumerate(data.get("waterfalls") or [])]
    recommendations = list(data.get("recommendations") or [])
    segments = list(data.get("segments") or [])
    elasticities = list(data.get("elasticities") or [])
    flags = list(data.get("data_quality_flags") or [])
    for i, r in enumerate(recommendations):
        if not isinstance(r, Mapping) or not r.get("customer_id") or not r.get("product_id"):
            raise _fail(f"recommendations[{i}]", "needs customer_id and product_id")

    used_waterfalls: set[int] = set()
    cases: list[EngineCase] = []

    def find_waterfall(rec_handles: list[str]) -> int | None:
        for i, w in enumerate(waterfalls):
            if w.order_line_id and f"OrderLine:{w.order_line_id}" in rec_handles:
                return i
        return None

    for i, rec in enumerate(recommendations):
        wf_index = find_waterfall(_evidence_handles(rec.get("evidence"), f"recommendations[{i}].evidence"))
        if wf_index is not None:
            used_waterfalls.add(wf_index)
        cases.append(_build_case(
            cite, f"rec-{i + 1}", rec, i, waterfalls[wf_index] if wf_index is not None else None,
            segments, elasticities, flags,
        ))
    for i, w in enumerate(waterfalls):
        if i in used_waterfalls:
            continue
        cases.append(_build_case(cite, f"line-{w.order_line_id or i + 1}", None, None, w, segments, elasticities, flags))

    if not cases:
        raise EngineFormatError("the run holds nothing to explain (no recommendations and no waterfalls)")
    return EngineRun(contract_version, engine_version, computed_at, tuple(cases),
                     notes=tuple(str(data[k]) for k in ("_README", "_steps_field_disclaimer") if data.get(k)))


def _scope_matches(scope: Any, product: str | None, customer: str | None) -> bool:
    if not isinstance(scope, Mapping):
        return False
    if scope.get("product_id") and product and str(scope["product_id"]) == product:
        return True
    return bool(scope.get("customer_id") and customer and str(scope["customer_id"]) == customer)


def _build_case(cite, case_id, rec, rec_index, wf, segments, elasticities, flags) -> EngineCase:
    facts: list[Fact] = []
    handles: list[str] = []
    notes = _Statements(cite)
    view: dict[str, Any] = {}
    currency = ""

    def add_handles(new: Sequence[str]) -> None:
        for h in new:
            if h not in handles:
                handles.append(h)

    def fact(key: str, value: float, kind: str, citation: str, label: str) -> None:
        facts.append(Fact(key, value, kind, citation, label))

    product = rec.get("product_id") if rec else (wf.product_id if wf else None)
    customer = rec.get("customer_id") if rec else (wf.customer_id if wf else None)
    product, customer = (str(product) if product else None), (str(customer) if customer else None)

    # ---- recommendation
    if rec is not None:
        path = f"recommendations[{rec_index}]"
        rec_handles = _evidence_handles(rec.get("evidence"), path + ".evidence")
        add_handles(rec_handles)
        conf = _confidence(rec.get("confidence"), path + ".confidence")
        status = str(rec.get("status") or "complete")
        pr = rec.get("price_range")
        margin = _money(rec.get("expected_margin"), path + ".expected_margin")
        deals = rec.get("comparable_deals_count")
        recv: dict[str, Any] = {"status": status, "confidence": conf, "constraints": list(rec.get("constraints_applied") or [])}
        if margin is not None and margin.currency:
            currency = currency or margin.currency
        if isinstance(pr, Mapping):
            low, high = _number(pr.get("low"), path + ".price_range.low"), _number(pr.get("high"), path + ".price_range.high")
            if low > high:
                raise _fail(path + ".price_range", f"low ({low}) is above high ({high})")
            mid = pr.get("mid")
            mid = _number(mid, path + ".price_range.mid") if mid is not None else None
            recv["price_range"] = {"low": low, "mid": mid, "high": high}
            fact("price_range_low", low, "money", cite, "bottom of the recommended range")
            if mid is not None:
                fact("recommended_price", mid, "money", cite, "recommended price (middle of the engine's range)")
            fact("price_range_high", high, "money", cite, "top of the recommended range")
        else:
            notes.add(f"The engine did not recommend a price for {customer} and {product} (status: {status})")
        if margin is not None and margin.usable:
            fact("expected_margin", margin.amount, "money", cite, "expected margin (forward-looking amount)")
            recv["expected_margin"] = margin.amount
        elif margin is not None:
            notes.add(f"Expected margin is unavailable (status: {margin.status})" + (f"; engine note: {margin.note}" if margin.note else ""))
        if rec.get("acceptance_probability") is not None:
            prob = _number(rec["acceptance_probability"], path + ".acceptance_probability")
            if not 0.0 <= prob <= 1.0:
                raise _fail(path + ".acceptance_probability", f"must be a probability in [0, 1], got {prob}")
            fact("acceptance_probability", _percent(prob), "percent", cite, "probability the customer accepts")
            recv["acceptance"] = _percent(prob)
        if deals is not None:
            count = _number(deals, path + ".comparable_deals_count")
            fact("comparable_deals", count, "count", cite, "number of comparable past deals")
            recv["comparable_deals"] = int(count)
        if conf is not None:
            fact("recommendation_confidence", _percent(conf["score"]), "percent", cite, "confidence in this recommendation")
            _confidence_statement("Recommendation", conf, notes)
        if recv["constraints"]:
            notes.add("Constraints applied to the recommendation: " + ", ".join(recv["constraints"]))
        if margin is not None and margin.note and margin.usable:
            notes.add("Engine note on the expected margin: " + margin.note)
        view["recommendation"] = recv

    # ---- waterfall
    if wf is not None:
        add_handles(wf.handles)
        line = f"OrderLine:{wf.order_line_id}" if wf.order_line_id else (wf.handles[0] if wf.handles else cite)
        add_handles([line] if wf.order_line_id else [])
        levels_view = []
        for engine_field, key, label, _ in _WATERFALL_LEVELS:
            money = wf.levels.get(engine_field)
            if money is None:
                continue
            currency = currency or money.currency
            levels_view.append({"key": engine_field, "label": label, "amount": money.amount, "status": money.status, "note": money.note})
            if money.usable:
                fact(key, abs(money.amount), "money", line, label)
            else:
                notes.add(f"The {label} is unavailable (status: {money.status})" + (f"; engine note: {money.note}" if money.note else ""), line)
        for step in wf.steps:
            money = step["money"]
            stage = "shown on the invoice" if step["on_invoice"] else "NOT shown on the invoice"
            label = f"{_STEP_LABELS.get(step['name'], step['name'].replace('_', ' '))} ({stage})"
            if money.usable:
                fact(f"step.{step['name']}", abs(money.amount), "money", step["handle"] or line, label)
        for key, label in (("pocket_margin_pct", "pocket margin percentage"), ("price_realisation_pct", "price realisation percentage")):
            if key in wf.pct:
                fact(key, wf.pct[key], "percent", line, label)
        _confidence_statement("Waterfall", wf.confidence, notes, line)
        view["waterfall"] = {
            "order_line_id": wf.order_line_id, "levels": levels_view, "currency": currency,
            "steps": [{"name": s["name"], "amount": s["money"].amount, "on_invoice": s["on_invoice"]} for s in wf.steps],
            "pct": wf.pct, "confidence": wf.confidence, "status": wf.status,
        }

    # ---- segment
    for seg in segments:
        if isinstance(seg, Mapping) and customer and str(seg.get("customer_id")) == customer:
            seg_handles = _evidence_handles(seg.get("evidence"), "segments[]")
            add_handles(seg_handles)
            conf = _confidence(seg.get("confidence"), "segments[].confidence")
            sv: dict[str, Any] = {"label": seg.get("segment_label"), "volume_tier": seg.get("volume_tier"),
                                  "features": list(seg.get("distinguishing_features") or []), "confidence": conf}
            if seg.get("segment_score") is not None:
                fact("segment_score", _percent(_number(seg["segment_score"], "segments[].segment_score")), "percent", cite, "customer segment score")
                sv["score"] = _percent(seg["segment_score"])
            if seg.get("price_sensitivity_index") is not None:
                fact("price_sensitivity_index", _number(seg["price_sensitivity_index"], "segments[].price_sensitivity_index"), "number", cite, "price sensitivity index")
                sv["sensitivity"] = seg["price_sensitivity_index"]
            contribution = _money(seg.get("margin_contribution"), "segments[].margin_contribution")
            if contribution is not None and contribution.usable:
                fact("segment_margin_contribution", abs(contribution.amount), "money", cite, "customer's margin contribution")
                sv["margin_contribution"] = contribution.amount
            text = f"Customer segment for {customer}: {seg.get('segment_label')}; volume tier {seg.get('volume_tier')}"
            if sv["features"]:
                text += "; distinguishing features: " + " | ".join(sv["features"])
            notes.add(text)
            _confidence_statement("Segment", conf, notes)
            view["segment"] = sv
            break

    # ---- elasticity
    for el in elasticities:
        if isinstance(el, Mapping) and _scope_matches(el.get("scope"), product, None):
            el_handles = _evidence_handles(el.get("evidence"), "elasticities[].evidence")
            add_handles(el_handles)
            conf = _confidence(el.get("confidence"), "elasticities[].confidence")
            ev: dict[str, Any] = {"status": el.get("status"), "confidence": conf, "fit_for_optimisation": el.get("fit_for_optimisation"),
                                  "shrunk_toward_prior": el.get("shrunk_toward_prior")}
            if el.get("beta") is not None:
                fact("elasticity_beta", _number(el["beta"], "elasticities[].beta"), "number", cite, "price elasticity (beta)")
                ev["beta"] = el["beta"]
            ci = el.get("credible_interval")
            if isinstance(ci, Sequence) and not isinstance(ci, (str, bytes)) and len(ci) == 2:
                fact("elasticity_low", _number(ci[0], "elasticities[].credible_interval[0]"), "number", cite, "lower end of the elasticity interval")
                fact("elasticity_high", _number(ci[1], "elasticities[].credible_interval[1]"), "number", cite, "upper end of the elasticity interval")
                ev["interval"] = [ci[0], ci[1]]
            if el.get("n_observations") is not None:
                fact("n_observations", _number(el["n_observations"], "elasticities[].n_observations"), "count", cite, "observations behind the elasticity estimate")
                ev["n_observations"] = el["n_observations"]
            if conf is not None:
                fact("elasticity_confidence", _percent(conf["score"]), "percent", cite, "confidence in the elasticity estimate")
            parts = [f"Elasticity estimate status: {el.get('status')}"]
            if el.get("fit_for_optimisation") is False:
                parts.append("the engine marks this estimate as NOT fit for price optimisation")
            if el.get("shrunk_toward_prior"):
                parts.append("the estimate was shrunk toward a prior")
            notes.add("; ".join(parts))
            _confidence_statement("Elasticity", conf, notes)
            view["elasticity"] = ev
            break

    # ---- data-quality flags
    flag_view = []
    for flag in flags:
        if isinstance(flag, Mapping) and (_scope_matches(flag.get("scope"), product, customer)):
            body = ", ".join(f"{k}: {v}" for k, v in flag.items() if k != "scope")
            notes.add(f"Data-quality flag raised by the engine ({body})")
            flag_view.append(dict(flag))
    if flag_view:
        view["flags"] = flag_view

    if not facts and not notes.lines:
        raise EngineFormatError(f"{case_id}: nothing in this record can be stated (no usable figure or statement)")

    title_bits = []
    if customer and product:
        title_bits.append(f"{customer} / {product}")
    if wf is not None and wf.order_line_id:
        title_bits.append(f"line {wf.order_line_id}")
    status = view.get("recommendation", {}).get("status")
    title = " · ".join(title_bits) or case_id
    if status and status != "complete":
        title += f" ({status})"
    pack = FactPack(
        subject=" / ".join(title_bits) or case_id,
        kind="engine_result",
        currency_code=currency,
        facts=tuple(facts),
        citations=tuple(handles),
        engine_version=cite,
        context=tuple(notes.lines),
        context_values=tuple(dict.fromkeys(notes.values)),
    )
    subject = EngineSubject(product_id=product or "", customer_id=customer or "", order_line_id=wf.order_line_id if wf else None)
    return EngineCase(case_id, title, pack, subject, view, suggested_questions(view))


# ----------------------------------------------------------------------- questions
def suggested_questions(view: Mapping[str, Any]) -> tuple[str, ...]:
    """Questions this case can actually answer, derived from the sections it holds."""
    out: list[str] = []
    rec = view.get("recommendation")
    if rec is not None:
        if rec.get("price_range"):
            out.append("What price range does the engine recommend and how likely is it to be accepted?")
        else:
            out.append("Why did the engine not recommend a price?")
        out.append("How confident is the engine in this recommendation, and why?")
    if "waterfall" in view:
        out.append("Why is the pocket price lower than the invoice price?")
        out.append("Which deductions are not shown on the invoice?")
        if any(level["key"] == "pocket_margin" for level in view["waterfall"]["levels"]):
            out.append("How much margin is left after the cost of goods sold?")
    if "elasticity" in view:
        out.append("Is the price elasticity estimate reliable enough to use?")
    if "segment" in view:
        out.append("Which customer segment is this customer in?")
    if "flags" in view:
        out.append("Did the engine raise any data-quality warning?")
    out.append("Explain this result in plain words.")
    return tuple(dict.fromkeys(out))
