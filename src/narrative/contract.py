"""
The Track A result object this track consumes.

Track A (Pricing Engine & MLOps) produces the numbers. Track B explains them.
The onboarding plan is explicit about the direction of that dependency:

    "Track A emits a structured result object (for example, a Price
    Recommendation with its price range, confidence, expected margin,
    acceptance probability, and the identifiers of the records it used).
    You *consume* that object."

So **every figure in a narrative arrives through this module**. Nothing
downstream derives a subtotal, a percentage, or a difference. The waterfall
is not recomputed from the graph here: Track A sends the walk already
calculated -- list price, each step, the invoice and pocket levels -- and this
layer reads it. If Track A did not send a number, the narrative does not have
it and cannot state it.

**This contract is provisional.** Section 7 of the plan says the shape is
agreed in Week 2 and freezes in Week 7; Track A's real payloads do not exist
yet. This module encodes the shape §7 describes, parses it defensively, and
is deliberately the *only* place that knows the field names. When Nourjihen's
real payload lands, :meth:`PriceRecommendation.from_dict` is the single
function that changes -- everything downstream speaks in these dataclasses,
never in raw JSON.

Validation, not calculation
---------------------------
This module does do arithmetic, in exactly one place and for one purpose:
:meth:`Waterfall.consistency_error` checks that the numbers Track A sent
agree with each other -- that the steps actually walk from the list price to
the stated pocket price. That is *verification of an input*, and its only
possible outcomes are "accept the payload" or "refuse it". It never produces
a figure the narrative can quote: a value computed here is compared and then
discarded. The distinction matters, because a layer that silently corrects an
upstream number has started owning that number.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ContractError",
    "PriceRange",
    "PriceRecommendation",
    "Waterfall",
    "WaterfallStep",
]

#: Money is compared to minor units. Track A sends rounded figures; this is
#: the slack allowed when checking that they add up.
CONSISTENCY_TOLERANCE = 0.02


class ContractError(ValueError):
    """Raised when a Track A payload is malformed, incomplete, or inconsistent."""


@dataclass(frozen=True, slots=True)
class PriceRange:
    """The recommended price band, as Track A reported it."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ContractError(
                f"price_range.low ({self.low}) is above price_range.high ({self.high})"
            )


@dataclass(frozen=True, slots=True)
class WaterfallStep:
    """One deduction in the walk from list price to pocket price."""

    name: str
    amount: float
    on_invoice: bool
    record_id: str

    @property
    def magnitude(self) -> float:
        """The size of the deduction, sign removed, for display."""
        return round(abs(self.amount), 2)


@dataclass(frozen=True, slots=True)
class Waterfall:
    """
    The list-to-pocket walk, **as computed by Track A**.

    Every field is read from the payload. The subtotals are Track A's own; the
    narrative layer quotes them and never recalculates them.
    """

    list_price: float
    list_price_record_id: str
    steps: tuple[WaterfallStep, ...]
    invoice_price: float
    pocket_price: float
    total_deductions: float
    erosion_pct: float

    @property
    def on_invoice_steps(self) -> tuple[WaterfallStep, ...]:
        return tuple(step for step in self.steps if step.on_invoice)

    @property
    def off_invoice_steps(self) -> tuple[WaterfallStep, ...]:
        return tuple(step for step in self.steps if not step.on_invoice)

    @property
    def record_ids(self) -> tuple[str, ...]:
        """Every record backing this walk, in walk order."""
        ids = [self.list_price_record_id] if self.list_price_record_id else []
        ids.extend(step.record_id for step in self.steps if step.record_id)
        return tuple(ids)

    def consistency_error(self) -> str | None:
        """
        Check Track A's figures against each other. Never used as a source.

        Returns a description of the first disagreement found, or ``None``.
        The computed values here exist only to be compared; none of them
        reaches a narrative. A payload whose own numbers do not add up is
        refused rather than repaired -- correcting it would mean this layer
        had taken ownership of a pricing figure.
        """
        on_invoice = sum(step.amount for step in self.on_invoice_steps)
        expected_invoice = self.list_price + on_invoice
        if abs(expected_invoice - self.invoice_price) > CONSISTENCY_TOLERANCE:
            return (
                f"invoice_price is {self.invoice_price:,.2f}, but list_price "
                f"({self.list_price:,.2f}) plus the on-invoice steps "
                f"({on_invoice:,.2f}) comes to {expected_invoice:,.2f}"
            )

        off_invoice = sum(step.amount for step in self.off_invoice_steps)
        expected_pocket = self.invoice_price + off_invoice
        if abs(expected_pocket - self.pocket_price) > CONSISTENCY_TOLERANCE:
            return (
                f"pocket_price is {self.pocket_price:,.2f}, but invoice_price "
                f"({self.invoice_price:,.2f}) plus the off-invoice steps "
                f"({off_invoice:,.2f}) comes to {expected_pocket:,.2f}"
            )

        expected_total = sum(step.amount for step in self.steps)
        if abs(expected_total - self.total_deductions) > CONSISTENCY_TOLERANCE:
            return (
                f"total_deductions is {self.total_deductions:,.2f}, but the steps "
                f"sum to {expected_total:,.2f}"
            )
        return None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Waterfall":
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, Sequence) or not raw_steps:
            raise ContractError("waterfall needs a non-empty 'steps' array")

        steps = tuple(_parse_step(entry, index) for index, entry in enumerate(raw_steps))
        waterfall = cls(
            list_price=_require_float(payload, "list_price"),
            list_price_record_id=str(payload.get("list_price_record_id", "")),
            steps=steps,
            invoice_price=_require_float(payload, "invoice_price"),
            pocket_price=_require_float(payload, "pocket_price"),
            total_deductions=_require_float(payload, "total_deductions"),
            erosion_pct=_require_float(payload, "erosion_pct"),
        )
        problem = waterfall.consistency_error()
        if problem:
            raise ContractError(
                f"Track A's waterfall does not add up: {problem}. The payload is "
                f"refused rather than corrected -- this layer does not own pricing "
                f"figures."
            )
        return waterfall


@dataclass(frozen=True, slots=True)
class PriceRecommendation:
    """
    One price recommendation produced by Track A's engine.

    Every field here is a number Track A computed. The narrative layer may
    quote these and nothing else.
    """

    recommendation_id: str
    product_id: str
    customer_id: str
    currency_code: str
    recommended_price: float
    price_range: PriceRange
    #: Model confidence in [0, 1]. Presented as-is; never rescaled here.
    confidence: float
    #: Expected margin, as a percentage, exactly as the engine reported it.
    expected_margin_pct: float
    #: Probability the customer accepts, in [0, 1].
    acceptance_probability: float
    #: Citation handles for the graph records the engine used, in the same
    #: ``Label:value`` form the retrieval and text-to-query layers emit.
    evidence_record_ids: tuple[str, ...]
    engine_version: str = "unspecified"
    #: Optional: present when Track A also sent the list-to-pocket walk.
    waterfall: Waterfall | None = None
    order_line_id: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PriceRecommendation":
        """
        Parse and validate one Track A payload.

        Raises:
            ContractError: naming the specific field at fault, so contract
                drift between the two tracks surfaces as a clear message
                rather than as a narrative quoting a missing number.
        """
        if not isinstance(payload, Mapping):
            raise ContractError(
                f"Payload must be a JSON object, got {type(payload).__name__}"
            )

        raw_range = payload.get("price_range")
        if not isinstance(raw_range, Mapping):
            raise ContractError("Payload needs a 'price_range' object with low and high")

        evidence = payload.get("evidence_record_ids")
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            raise ContractError(
                "Payload needs 'evidence_record_ids' as a list of citation handles"
            )
        handles = tuple(str(item) for item in evidence if str(item).strip())
        if not handles:
            raise ContractError(
                "Payload carries no evidence_record_ids. A recommendation that "
                "cites nothing cannot be explained with citations, which is the "
                "whole point of this deliverable -- refusing it here is correct."
            )

        raw_waterfall = payload.get("waterfall")
        waterfall = None
        if raw_waterfall is not None:
            if not isinstance(raw_waterfall, Mapping):
                raise ContractError("'waterfall' must be an object when present")
            waterfall = Waterfall.from_dict(raw_waterfall)

        return cls(
            recommendation_id=_require_str(payload, "recommendation_id"),
            product_id=_require_str(payload, "product_id"),
            customer_id=_require_str(payload, "customer_id"),
            currency_code=_require_str(payload, "currency_code"),
            recommended_price=_require_float(payload, "recommended_price"),
            price_range=PriceRange(
                low=_require_float(raw_range, "low"),
                high=_require_float(raw_range, "high"),
            ),
            confidence=_require_ratio(payload, "confidence"),
            expected_margin_pct=_require_float(payload, "expected_margin_pct"),
            acceptance_probability=_require_ratio(payload, "acceptance_probability"),
            evidence_record_ids=handles,
            engine_version=str(payload.get("engine_version", "unspecified")),
            waterfall=waterfall,
            order_line_id=str(payload.get("order_line_id", "")),
        )

    @classmethod
    def load(
        cls, path: Path, recommendation_id: str | None = None
    ) -> "PriceRecommendation":
        """
        Read one recommendation from a JSON file.

        The file may hold a single payload, or a ``{"recommendations": [...]}``
        collection -- which is how the provisional sample payloads are stored
        while Track A's real output does not exist yet.
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ContractError(f"Payload file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"{path} is not valid JSON (line {exc.lineno}): {exc.msg}"
            ) from exc

        if isinstance(data, Mapping) and "recommendations" in data:
            entries = data["recommendations"]
            if not isinstance(entries, Sequence) or not entries:
                raise ContractError(f"{path} has an empty 'recommendations' array")
            if recommendation_id is None:
                return cls.from_dict(entries[0])
            for entry in entries:
                if (
                    isinstance(entry, Mapping)
                    and entry.get("recommendation_id") == recommendation_id
                ):
                    return cls.from_dict(entry)
            available = ", ".join(
                str(e.get("recommendation_id")) for e in entries if isinstance(e, Mapping)
            )
            raise ContractError(
                f"No recommendation {recommendation_id!r} in {path}. Available: {available}"
            )
        return cls.from_dict(data)


def _parse_step(entry: object, index: int) -> WaterfallStep:
    if not isinstance(entry, Mapping):
        raise ContractError(f"waterfall step {index} is not an object")
    return WaterfallStep(
        name=_require_str(entry, "name"),
        amount=_require_float(entry, "amount"),
        on_invoice=_require_bool(entry, "on_invoice"),
        record_id=str(entry.get("record_id", "")),
    )


def _require_str(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"Payload needs a non-empty string {field!r}")
    return value.strip()


def _require_float(payload: Mapping[str, Any], field: str) -> float:
    value = payload.get(field)
    # bool is an int subclass; True must never be read as the number 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"Payload needs a numeric {field!r}, got {value!r}")
    return float(value)


def _require_bool(payload: Mapping[str, Any], field: str) -> bool:
    """
    Accept a real boolean or the string form.

    The JSON graph export writes ``"true"`` while Neo4j stores a real
    boolean; a payload assembled from either source may carry either form,
    and assuming one silently mis-stages a deduction.
    """
    value = payload.get(field)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ContractError(f"Payload needs a boolean {field!r}, got {value!r}")


def _require_ratio(payload: Mapping[str, Any], field: str) -> float:
    value = _require_float(payload, field)
    if not 0.0 <= value <= 1.0:
        raise ContractError(f"{field!r} must be a probability in [0, 1], got {value}")
    return value
