"""
The fact pack: every number a narrative is permitted to state.

Each fact here was **read from Track A's payload**, never derived. This module
selects, labels and formats; it does not calculate. The one transformation it
performs is presentational -- a probability of ``0.78`` is published as a
single percentage fact of ``78.0`` so the quantity has exactly one numeric
form, because every accepted form is a form a model could reach by accident.

This is the mechanism behind the track's defining rule -- *explain, do not
compute; ground, do not guess*. Rather than instructing a model not to invent
figures and hoping, the generator builds a closed set of facts first, shows
the model only that set, and afterwards refuses any sentence containing a
number outside it (see :mod:`src.narrative.verifier`).

Every fact carries its attribution. A waterfall step keeps the handle of the
``PriceComponent`` record Track A cited for it, in the same ``Label:value``
form retrieval and text-to-query emit. A figure the engine produced -- the
recommended price, the confidence -- keeps the engine version instead,
because no single graph record backs a model's output.

Labels are bilingual. Arabic is deferred for this phase (English is the
current focus), but the labels are kept beside the English ones so the
Arabic path re-enables without a second, divergent set of numbers appearing.
Step names come from the payload, so their display translations are a lookup
with a fallback to the raw name: an unknown step degrades to its identifier
rather than dropping out of the explanation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.narrative.contract import PriceRecommendation

__all__ = ["Fact", "FactPack", "build_facts"]

#: Display names for the waterfall step names present in the AXIOM data.
#: Unknown steps fall back to their raw name from the payload.
_STEP_LABELS_EN: Mapping[str, str] = {
    "list_price": "list price",
    "invoice_discount": "invoice discount",
    "payment_discount": "payment discount",
    "volume_rebate": "volume rebate",
    "freight": "freight",
    "promotional_discount": "promotional discount",
    "commission": "commission",
}

_STEP_LABELS_AR: Mapping[str, str] = {
    "list_price": "سعر القائمة",
    "invoice_discount": "خصم الفاتورة",
    "payment_discount": "خصم السداد",
    "volume_rebate": "حسم الكمية",
    "freight": "تكاليف الشحن",
    "promotional_discount": "الخصم الترويجي",
    "commission": "العمولة",
}


@dataclass(frozen=True, slots=True)
class Fact:
    """One quantity a narrative may state, with what backs it."""

    key: str
    value: float
    #: ``money``, ``percent``, ``count`` or ``number`` (a plain figure with two
    #: decimals, such as an elasticity) -- drives display formatting only.
    kind: str
    citation: str
    label_en: str
    label_ar: str = ""

    def label(self, language: str = "en") -> str:
        return self.label_ar if language == "ar" and self.label_ar else self.label_en

    def formatted(self, currency_code: str = "") -> str:
        """The exact string the model is told to copy."""
        if self.kind == "money":
            return f"{self.value:,.2f} {currency_code}".strip()
        if self.kind == "percent":
            return f"{self.value:.1f}%"
        if self.kind == "number":
            return f"{self.value:,.2f}"
        return f"{self.value:,.0f}"


@dataclass(frozen=True, slots=True)
class FactPack:
    """
    The closed set of facts behind one narrative.

    ``facts`` is everything that may be quoted; ``citations`` is every record
    that may be cited.
    """

    subject: str
    kind: str
    currency_code: str
    facts: tuple[Fact, ...]
    citations: tuple[str, ...]
    engine_version: str = ""
    #: Qualitative statements the engine sent (confidence bands and reasons,
    #: notes, flags), each already ending in the citation that backs it. The
    #: model may restate them; they carry no figures of their own.
    context: tuple[str, ...] = ()
    #: Numbers that appear inside those statements. They came from the engine,
    #: so quoting them is allowed, and they are kept apart from ``facts`` so
    #: the figure table stays readable.
    context_values: tuple[float, ...] = ()

    @property
    def allowed_values(self) -> tuple[float, ...]:
        """Every number the verifier will admit."""
        return tuple(fact.value for fact in self.facts) + self.context_values

    def extended(self, extra: "tuple[Fact, ...] | list[Fact]") -> "FactPack":
        """
        A copy admitting additional facts and their citations.

        Used to fold graph-computed context (from text-to-query) in beside
        Track A's figures. Facts already present by key are kept, so an
        extension cannot silently redefine a number the engine sent.
        """
        by_key = {fact.key: fact for fact in self.facts}
        merged = list(self.facts)
        citations = list(self.citations)
        for fact in extra:
            if fact.key in by_key:
                continue
            merged.append(fact)
            if fact.citation and fact.citation not in citations:
                citations.append(fact.citation)
        return FactPack(
            subject=self.subject,
            kind=self.kind,
            currency_code=self.currency_code,
            facts=tuple(merged),
            citations=tuple(citations),
            engine_version=self.engine_version,
            context=self.context,
            context_values=self.context_values,
        )

    def render_context(self) -> str:
        """The qualitative statements shown to the model, one per line."""
        return "\n".join(f"- {line}" for line in self.context)

    def render(self, language: str = "en") -> str:
        """The fact table shown to the model."""
        return "\n".join(
            f"- {fact.label(language)}: {fact.formatted(self.currency_code)}"
            + (f"  [{fact.citation}]" if fact.citation else "")
            for fact in self.facts
        )


def build_facts(recommendation: PriceRecommendation) -> FactPack:
    """
    Turn one Track A payload into a closed fact set.

    Includes the waterfall when the payload carries one. Every value is
    Track A's own; the only change is ratios published as percentages, so a
    quantity cannot appear under two different numbers.
    """
    rec = recommendation
    engine = rec.engine_version
    facts: list[Fact] = [
        Fact("recommended_price", rec.recommended_price, "money", engine,
             "recommended price", "السعر الموصى به"),
        Fact("price_range_low", rec.price_range.low, "money", engine,
             "bottom of the recommended range", "الحد الأدنى للنطاق الموصى به"),
        Fact("price_range_high", rec.price_range.high, "money", engine,
             "top of the recommended range", "الحد الأعلى للنطاق الموصى به"),
        Fact("confidence", round(rec.confidence * 100, 1), "percent", engine,
             "engine confidence", "درجة ثقة المحرّك"),
        Fact("expected_margin_pct", rec.expected_margin_pct, "percent", engine,
             "expected margin", "الهامش المتوقع"),
        Fact("acceptance_probability", round(rec.acceptance_probability * 100, 1),
             "percent", engine, "probability the customer accepts",
             "احتمالية قبول العميل"),
    ]

    subject = f"{rec.product_id} / {rec.customer_id}"
    kind = "price_recommendation"

    if rec.waterfall is not None:
        facts.extend(_waterfall_facts(rec.waterfall))
        kind = "recommendation_and_waterfall"
        if rec.order_line_id:
            subject = f"{rec.product_id} / {rec.customer_id} / line {rec.order_line_id}"

    citations = list(rec.evidence_record_ids)
    if rec.waterfall is not None:
        for handle in rec.waterfall.record_ids:
            if handle not in citations:
                citations.append(handle)

    return FactPack(
        subject=subject,
        kind=kind,
        currency_code=rec.currency_code,
        facts=tuple(facts),
        citations=tuple(citations),
        engine_version=engine,
    )


def _waterfall_facts(waterfall) -> list[Fact]:
    """Publish Track A's walk, with each step's stage on its label."""
    facts = [
        Fact("list_price", waterfall.list_price, "money",
             waterfall.list_price_record_id, "list price", "سعر القائمة")
    ]
    for step in waterfall.steps:
        # The stage qualifier is part of the label, not a prompt instruction.
        # Without it the model cannot tell which deductions reach the invoice
        # price and which only reach the pocket price, and it reliably files
        # an off-invoice rebate under the invoice stage -- every number
        # traceable, the attribution wrong.
        qualifier_en = (
            " (shown on the invoice)" if step.on_invoice else " (NOT shown on the invoice)"
        )
        qualifier_ar = (
            " (ظاهر على الفاتورة)" if step.on_invoice else " (غير ظاهر على الفاتورة)"
        )
        facts.append(
            Fact(
                f"step.{step.name}",
                step.magnitude,
                "money",
                step.record_id,
                _STEP_LABELS_EN.get(step.name, step.name) + qualifier_en,
                _STEP_LABELS_AR.get(step.name, step.name) + qualifier_ar,
            )
        )
    facts.extend(
        [
            Fact("invoice_price", waterfall.invoice_price, "money", "",
                 "invoice price", "سعر الفاتورة"),
            Fact("pocket_price", waterfall.pocket_price, "money", "",
                 "pocket price actually kept", "السعر الصافي المحصَّل فعليًا"),
            Fact("total_deductions", round(abs(waterfall.total_deductions), 2), "money",
                 "", "total erosion from list to pocket",
                 "إجمالي التآكل من سعر القائمة إلى الصافي"),
            Fact("erosion_pct", waterfall.erosion_pct, "percent", "",
                 "erosion as a share of list price", "نسبة التآكل من سعر القائمة"),
            Fact("deduction_count", float(len(waterfall.steps)), "count", "",
                 "number of deduction steps", "عدد خطوات الخصم"),
        ]
    )
    return facts
