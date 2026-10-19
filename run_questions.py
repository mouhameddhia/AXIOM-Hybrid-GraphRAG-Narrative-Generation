"""
Run the diverse question bank against the hybrid retriever.

    python run_questions.py                       # everything, EN + AR
    python run_questions.py --list                # show categories, run nothing
    python run_questions.py -c win_loss           # one category
    python run_questions.py -c reference_price -c applicable_policies
    python run_questions.py --lang en             # English only
    python run_questions.py --detail              # show retrieved text
    python run_questions.py --failures            # only what did not meet expectations
    python run_questions.py --isolate             # hybrid vs each single retriever
    python run_questions.py --json out.json       # machine-readable results

Questions live in ``data/question_bank.json``, grouped by the evidence types the
onboarding plan names. Every expected record id is validated against the live
corpus before anything runs, so a stale expectation fails loudly instead of
looking like a retrieval miss.

``--isolate`` addresses the acceptance criterion directly: *"hybrid retrieval
beats any single retriever on the gold set"*. It runs the same questions through
BM25 alone, graph alone, and the fused hybrid, and tabulates the difference.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from src.config.settings import PROJECT_ROOT, Settings, get_settings
from src.evaluation.question_bank import (
    ProbeOutcome,
    QuestionBank,
    QuestionBankError,
    run_bank,
)
from src.graph.sources import GraphSourceError
from src.pipeline.factory import RetrievalStack, build_retrieval_stack
from src.pipeline.retriever_pipeline import HybridRetrievalPipeline
from src.retrieval.fusion import ReciprocalRankFuser
from src.utils.helpers import setup_logging

DEFAULT_BANK = PROJECT_ROOT / "data" / "question_bank.json"
WIDTH = 100
RULE = "=" * WIDTH
THIN = "-" * WIDTH

PASS = "PASS"
FAIL = "FAIL"
INFO = "--"


def build_stack(settings: Settings, *, offline: bool) -> RetrievalStack:
    """Assemble the stack, optionally without the transformer modalities."""
    if offline:
        settings = replace(settings, reranker=replace(settings.reranker, enabled=False))
    return build_retrieval_stack(settings, enable_dense=not offline)


def status_of(result: ProbeOutcome) -> str:
    """PASS / FAIL for a checkable probe, or INFO for an exploratory one."""
    if not result.checked:
        return INFO
    return PASS if result.passed else FAIL


def print_bank_contents(bank: QuestionBank, *, category_ids: Sequence[str] | None = None) -> None:
    """
    Print every question's full text, grouped by category.

    This is the answer to "where are the questions": a read-only view of
    ``data/question_bank.json`` with no retrieval run, so it works even before
    the corpus is built and costs nothing to call repeatedly while studying it.
    """
    wanted = set(category_ids) if category_ids else None
    shown = [c for c in bank.categories if wanted is None or c.category_id in wanted]

    print(f"{bank.source.name}: {len(bank.all_probes)} questions in "
          f"{len(bank.categories)} categories\n")

    for category in shown:
        print(RULE)
        print(f"{category.title.upper()}   [{category.category_id}]  "
              f"({len(category.probes)} questions)")
        if category.onboarding_ref:
            print(f"  onboarding: {category.onboarding_ref}")
        if category.why:
            print(f"  why       : {category.why}")
        print(RULE)
        for probe in category.probes:
            print(f"\n  {probe.probe_id}")
            for code in ("en", "ar"):
                if code in probe.questions:
                    print(f"    [{code}] {probe.questions[code]}")
            if probe.probes:
                print(f"    probes: {probe.probes}")
            expectation = []
            if probe.expect_records:
                expectation.append(f"expects {', '.join(probe.expect_records)}")
            if probe.expect_types:
                expectation.append(f"expects type in {', '.join(probe.expect_types)}")
            if probe.expect_no_anchor:
                expectation.append("expects no anchor grounded")
            if expectation:
                print(f"    check : {'; '.join(expectation)}")
        print()

    print(RULE)
    print(f"Run one category:   python run_questions.py -c {shown[0].category_id}")
    print(f"Run everything:     python run_questions.py --offline")


def print_header(stack: RetrievalStack, bank: QuestionBank, languages: Sequence[str]) -> None:
    print(RULE)
    print("AXIOM TRACK B - RETRIEVAL QUESTION BANK")
    print(RULE)
    print(f"  corpus     : {stack.describe()}")
    print(f"  bank       : {bank.source.name} - {len(bank.categories)} categories, "
          f"{len(bank.all_probes)} questions")
    print(f"  languages  : {', '.join(languages)}")
    print(f"  anchors    : {stack.anchor_resolver.indexed_value_count} resolvable values")


def print_category(
    category_id: str,
    title: str,
    onboarding_ref: str,
    why: str,
    results: Sequence[ProbeOutcome],
    *,
    detail: bool,
    top_k: int,
) -> None:
    """Render one category's results."""
    print(f"\n{RULE}")
    print(f"{title.upper()}   [{category_id}]")
    if onboarding_ref:
        print(f"  onboarding: {onboarding_ref}")
    if why:
        print(f"  why       : {why}")
    print(RULE)

    for result in results:
        marker = status_of(result)
        print(f"\n  [{marker}] {result.probe.probe_id}  ({result.language})")
        print(f"       Q: {result.question}")
        if result.probe.probes:
            print(f"       probes: {result.probe.probes}")

        anchors = result.outcome.anchors
        anchor_text = (
            ", ".join(f"{a.mention}->{a.record_id}" for a in anchors[:3]) if anchors else "none"
        )
        print(f"       anchors: {anchor_text}")

        if not result.ranked:
            print("       (no evidence returned)")
        for position, record in enumerate(result.outcome.final[:top_k], start=1):
            ranks = record.metadata.get("stream_ranks", {})
            provenance = ",".join(f"{name}#{rank}" for name, rank in sorted(ranks.items()))
            hit = " <-- expected" if record.record_id in result.probe.expect_records else ""
            print(f"         {position}. {record.record_id:<44} [{provenance}]{hit}")
            if detail:
                print(f"            {_preview(record.text)}")

        for note in result.notes:
            print(f"       ! {note}")
        if result.missing_records:
            print(f"       ! expected but missing: {', '.join(result.missing_records)}")
        if result.missing_types:
            print(f"       ! expected type missing: {', '.join(result.missing_types)}")


def _preview(text: str, width: int = 118) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def print_summary(results: Sequence[ProbeOutcome], bank: QuestionBank) -> None:
    """Per-category and per-language pass rates."""
    print(f"\n{RULE}")
    print("SUMMARY")
    print(RULE)

    titles = {c.category_id: c.title for c in bank.categories}
    by_category: dict[str, list[ProbeOutcome]] = {}
    for result in results:
        by_category.setdefault(result.probe.category_id, []).append(result)

    print(f"\n  {'category':<24} {'checked':>8} {'passed':>7} {'rate':>7}   {'title'}")
    print("  " + THIN[:WIDTH - 4])
    for category_id, rows in by_category.items():
        checked = [r for r in rows if r.checked]
        passed = [r for r in checked if r.passed]
        rate = f"{len(passed) / len(checked):.0%}" if checked else "  n/a"
        print(
            f"  {category_id:<24} {len(checked):>8} {len(passed):>7} {rate:>7}   "
            f"{titles.get(category_id, '')}"
        )

    checked_all = [r for r in results if r.checked]
    passed_all = [r for r in checked_all if r.passed]

    print(f"\n  {'language':<24} {'checked':>8} {'passed':>7} {'rate':>7}")
    print("  " + THIN[:WIDTH - 4])
    for language in sorted({r.language for r in results}):
        rows = [r for r in checked_all if r.language == language]
        hits = [r for r in rows if r.passed]
        rate = f"{len(hits) / len(rows):.0%}" if rows else "  n/a"
        print(f"  {language:<24} {len(rows):>8} {len(hits):>7} {rate:>7}")

    exploratory = len(results) - len(checked_all)
    latencies = [r.outcome.timings_ms.get("total", 0.0) for r in results]
    print(f"\n  total questions run : {len(results)}")
    print(f"  automatically checked: {len(checked_all)}  ({exploratory} exploratory, read by eye)")
    if checked_all:
        print(f"  passed               : {len(passed_all)}  ({len(passed_all) / len(checked_all):.1%})")
    if latencies:
        print(f"  median latency       : {sorted(latencies)[len(latencies) // 2]:.1f} ms")

    failures = [r for r in checked_all if not r.passed]
    if failures:
        print(f"\n  Not meeting expectations ({len(failures)}):")
        for result in failures:
            reason = []
            if result.missing_records:
                reason.append(f"missing {', '.join(result.missing_records)}")
            if result.missing_types:
                reason.append(f"no {', '.join(result.missing_types)} returned")
            if result.anchor_violation:
                reason.append("grounded an anchor it should not have")
            print(f"    - {result.probe.probe_id} ({result.language}): {'; '.join(reason)}")


def print_isolation(stack: RetrievalStack, settings: Settings, results: Sequence[ProbeOutcome], top_k: int) -> None:
    """
    Hybrid versus each single retriever.

    This is the onboarding's retrieval-quality acceptance criterion stated as a
    measurement: hybrid retrieval must beat any single retriever.
    """
    print(f"\n{RULE}")
    print("SINGLE RETRIEVER vs HYBRID")
    print("  acceptance criterion: hybrid retrieval beats any single retriever")
    print(RULE)

    checkable = [r for r in results if r.probe.expect_records]
    if not checkable:
        print("\n  No probes with exact expected records; nothing to isolate.")
        return

    variants: dict[str, HybridRetrievalPipeline] = {}
    for retriever in stack.retrievers:
        variants[f"{retriever.name} only"] = HybridRetrievalPipeline(
            retrievers=[retriever],
            fuser=ReciprocalRankFuser(settings.fusion),
            config=settings.retrieval,
            anchor_resolver=stack.anchor_resolver,
        )
    variants["hybrid (fused)"] = stack.pipeline

    print(f"\n  {len(checkable)} questions with an exact expected record\n")
    print(f"  {'configuration':<22} {'found':>8} {'rate':>7} {'MRR':>7}")
    print("  " + THIN[:48])

    scores: dict[str, float] = {}
    for name, pipeline in variants.items():
        found = 0
        reciprocal = 0.0
        for result in checkable:
            outcome = pipeline.run(result.question, top_k=top_k)
            ranked = outcome.record_ids
            best: int | None = None
            for expected in result.probe.expect_records:
                if expected in ranked:
                    position = ranked.index(expected) + 1
                    best = position if best is None else min(best, position)
            if best is not None:
                found += 1
                reciprocal += 1.0 / best
        mrr = reciprocal / len(checkable)
        scores[name] = mrr
        print(f"  {name:<22} {found:>8} {found / len(checkable):>6.0%} {mrr:>7.3f}")

    best_single = max(
        (name for name in scores if name != "hybrid (fused)"), key=lambda n: scores[n]
    )
    hybrid = scores["hybrid (fused)"]
    delta = hybrid - scores[best_single]
    print(f"\n  best single retriever : {best_single} (MRR {scores[best_single]:.3f})")
    print(f"  hybrid                : MRR {hybrid:.3f}")
    if delta > 0:
        print(f"  VERDICT: hybrid beats the best single retriever by {delta:+.3f} MRR")
    elif delta == 0:
        print("  VERDICT: hybrid matches the best single retriever; fusion adds no lift here")
    else:
        print(f"  VERDICT: hybrid is BEHIND the best single retriever by {delta:.3f} MRR")


def to_json(results: Sequence[ProbeOutcome]) -> list[dict[str, object]]:
    """Machine-readable results, for committing alongside a report."""
    return [
        {
            "category": r.probe.category_id,
            "id": r.probe.probe_id,
            "language": r.language,
            "question": r.question,
            "status": status_of(r),
            "anchors": [a.record_id for a in r.outcome.anchors],
            "ranked": list(r.ranked),
            "expect_records": list(r.probe.expect_records),
            "missing_records": list(r.missing_records),
            "missing_types": list(r.missing_types),
            "latency_ms": round(r.outcome.timings_ms.get("total", 0.0), 2),
        }
        for r in results
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("-c", "--category", action="append", help="Run only this category (repeatable)")
    parser.add_argument("--lang", action="append", choices=("en", "ar"))
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--offline", action="store_true", help="BM25 + graph only, no model downloads")
    parser.add_argument("--detail", action="store_true", help="Show the retrieved record text")
    parser.add_argument("--failures", action="store_true", help="Show only unmet expectations")
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip the per-question detail; print only the header, the summary "
             "table, and (with --isolate) the acceptance-criterion verdict. "
             "This is the presentation view.",
    )
    parser.add_argument("--isolate", action="store_true", help="Compare hybrid against each single retriever")
    parser.add_argument(
        "--list", action="store_true",
        help="Print every question's text (no retrieval, no corpus load) and exit",
    )
    parser.add_argument("--json", type=Path, help="Write machine-readable results here")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(logging.INFO if args.verbose else logging.WARNING)

    if args.list:
        # Reads the bank file only -- no graph load, no BM25 index, no model.
        # This is deliberately the cheapest possible way to read the questions.
        try:
            bank = QuestionBank.load(args.bank)
        except QuestionBankError as exc:
            print(f"Could not read the question bank: {exc}", file=sys.stderr)
            return 2
        print_bank_contents(bank, category_ids=args.category)
        return 0

    try:
        settings = get_settings()
        stack = build_stack(settings, offline=args.offline)
        bank = QuestionBank.load(args.bank, stack.corpus)
    except (QuestionBankError, GraphSourceError, ValueError) as exc:
        print(f"Could not start: {exc}", file=sys.stderr)
        return 2

    languages = tuple(args.lang) if args.lang else bank.languages
    print_header(stack, bank, languages)

    results = run_bank(
        stack.pipeline,
        bank,
        top_k=args.top_k,
        languages=languages,
        category_ids=args.category,
    )
    if not results:
        print("\nNo questions matched the selected filters.", file=sys.stderr)
        return 2

    if not args.summary_only:
        shown = [r for r in results if not r.passed] if args.failures else results
        by_category: dict[str, list[ProbeOutcome]] = {}
        for result in shown:
            by_category.setdefault(result.probe.category_id, []).append(result)

        for category in bank.categories:
            rows = by_category.get(category.category_id)
            if rows:
                print_category(
                    category.category_id,
                    category.title,
                    category.onboarding_ref,
                    category.why,
                    rows,
                    detail=args.detail,
                    top_k=args.top_k,
                )

    print_summary(results, bank)

    if args.isolate:
        print_isolation(stack, settings, results, args.top_k)

    if args.json:
        args.json.write_text(
            json.dumps(to_json(results), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nWrote machine-readable results to {args.json}")

    checked = [r for r in results if r.checked]
    return 0 if all(r.passed for r in checked) else 1


if __name__ == "__main__":
    raise SystemExit(main())
