"""Serialisation of pipeline objects into the JSON events the page consumes."""
from __future__ import annotations

import dataclasses
from typing import Any

from src.narrative.facts import FactPack
from src.reliability.attribution import check_attribution

__all__ = ["assessment_dict", "fact_rows", "plan_dict", "serialise"]


def plan_dict(plan) -> dict[str, Any] | None:
    if plan is None:
        return None
    raw = dataclasses.asdict(plan)
    return {key: value for key, value in raw.items() if value not in (None, [], False)}


def fact_rows(pack: FactPack) -> list[dict[str, Any]]:
    return [
        {
            "key": fact.key,
            "label": fact.label("en"),
            "value": fact.value,
            "formatted": fact.formatted(pack.currency_code),
            "kind": fact.kind,
            "citation": fact.citation,
        }
        for fact in pack.facts
    ]


def assessment_dict(assessment) -> dict[str, Any]:
    return {
        "figures": assessment.facts_total,
        "records": assessment.citations_total,
        "with_citation": assessment.facts_with_citation,
        "support": round(assessment.support_ratio, 3),
        "density": round(assessment.citation_density, 3),
        "floor": assessment.minimum_support,
        "weak": assessment.weak,
        "unresolvable": list(assessment.unresolvable),
        "text": assessment.describe(),
    }


def _sample(settings, data: dict[str, Any]) -> dict[str, Any]:
    narrative = data["narrative"]
    pack: FactPack = data["pack"]
    attribution: dict[str, Any] | None = None
    if narrative.success:
        report = check_attribution(
            narrative.text,
            pack,
            absolute_tolerance=settings.narrative.absolute_tolerance,
            relative_tolerance=settings.narrative.relative_tolerance,
        )
        attribution = {
            "ok": report.ok,
            "problems": [problem.describe() for problem in report.misattributions],
            "roles_checked": list(report.roles_checked),
        }
    verification = narrative.verification
    return {
        "stage": "sample",
        "index": data["index"],
        "verified": narrative.success,
        "attempts": narrative.attempts,
        "text": narrative.text if narrative.success else "",
        "reason": narrative.reason,
        "violations": (
            [f"{v.kind}: {v.detail}" for v in verification.violations]
            if verification is not None and not narrative.success
            else []
        ),
        "figures": len(verification.numbers_found) if verification is not None else 0,
        "citations": list(narrative.citations),
        "attribution": attribution,
    }


def serialise(settings, stage: str, data: Any, k: int, question: str | None) -> dict[str, Any]:
    """One observer callback from the reliability layer, as an event."""
    if stage == "pack":
        return {
            "stage": "pack",
            "subject": data.subject,
            "figures": len(data.facts),
            "statements": len(data.context),
            "records": len(data.citations),
            "k": k,
            "question": question,
            "facts": fact_rows(data),
            "statement_lines": list(data.context),
            "citations": list(data.citations),
            "engine_version": data.engine_version,
        }
    if stage == "assessment":
        return {"stage": "assessment", **assessment_dict(data)}
    if stage == "correction":
        return {
            "stage": "correction",
            "added": list(data.added),
            "rejected": list(data.rejected),
            "queries": list(data.queries_tried),
            "support_before": round(data.before.support_ratio, 3),
            "support_after": round(data.after.support_ratio, 3),
            "text": data.describe(),
        }
    if stage == "sample":
        return _sample(settings, data)
    if stage == "consistency":
        return {
            "stage": "consistency",
            "k": data.k,
            "usable": len(data.usable),
            "core": len(data.core),
            "union": len(data.union),
            "agreement": round(data.agreement, 3),
            "agreement_kind": data.agreement_kind,
            "abstained": data.abstained,
            "reason": data.reason,
            "elected": data.consensus.index if data.consensus is not None else None,
            "usable_indexes": [sample.index for sample in data.usable],
            "text": data.describe(),
        }
    if stage == "decision":
        return {
            "stage": "decision",
            "served": data.served,
            "text": data.text,
            "reason": data.reason,
            "citations": list(data.citations),
            "allowed_citations": list(data.pack.citations)
            + ([data.pack.engine_version] if data.pack.engine_version else []),
            "facts": fact_rows(data.pack),
            "context_values": list(data.pack.context_values),
            "agreement": round(data.agreement, 3),
            "samples_used": data.samples_used,
            "question": question,
        }
    return {"stage": stage}
