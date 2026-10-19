"""
Retrieval evaluation against the English gold set.

    python evaluate_retrieval.py                  # full stack, English
    python evaluate_retrieval.py --offline        # BM25 + graph, no downloads
    python evaluate_retrieval.py --compare        # A/B the stream configurations
    python evaluate_retrieval.py --failures       # show only what did not work
    python evaluate_retrieval.py --json out.json  # machine-readable results

Arabic is out of scope for this phase and is not evaluated here.

Reports recall@k, MRR, nDCG@k, MAP and precision@k, with a per-case breakdown. ``--compare`` runs several configurations over the same
gold set and tabulates them side by side, which is how a weighting change is
justified with evidence rather than asserted.
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
from src.evaluation.harness import EvaluationReport, GoldSet, GoldSetError, evaluate
from src.graph.sources import GraphSourceError
from src.pipeline.factory import RetrievalStack, build_retrieval_stack
from src.utils.helpers import setup_logging

DEFAULT_GOLD_SET = PROJECT_ROOT / "data" / "gold_set.json"
EVALUATED_LANGUAGES = ("en",)
WIDTH = 96
RULE = "=" * WIDTH


def build_stack(settings: Settings, *, offline: bool) -> RetrievalStack:
    """Assemble the stack, optionally without the transformer modalities."""
    if offline:
        settings = replace(settings, reranker=replace(settings.reranker, enabled=False))
    return build_retrieval_stack(settings, enable_dense=not offline)


def configuration_variants(settings: Settings) -> list[tuple[str, Settings, bool]]:
    """
    The configurations compared by ``--compare``.

    Each is a named (settings, offline) pair. They isolate one variable at a
    time so a difference in the table can be attributed to a single change.
    """
    fusion = settings.fusion
    return [
        (
            "lexical+graph",
            replace(settings, reranker=replace(settings.reranker, enabled=False)),
            True,
        ),
        (
            "all streams, equal weight",
            replace(settings, reranker=replace(settings.reranker, enabled=False)),
            False,
        ),
        (
            "all streams, dense down-weighted",
            replace(
                settings,
                fusion=replace(fusion, weights={"bm25": 1.0, "graph": 1.0, "dense": 0.3}),
                reranker=replace(settings.reranker, enabled=False),
            ),
            False,
        ),
        (
            "all streams + rerank",
            settings,
            False,
        ),
        (
            "dense down-weighted + rerank",
            replace(
                settings,
                fusion=replace(fusion, weights={"bm25": 1.0, "graph": 1.0, "dense": 0.3}),
            ),
            False,
        ),
    ]


def print_report(report: EvaluationReport, *, show_cases: bool, failures_only: bool) -> None:
    """Render one evaluation report."""
    k = report.k
    print(f"\n{RULE}")
    print(f"RESULTS{f': {report.label}' if report.label else ''}")
    print(RULE)

    print(f"\n  Aggregate over {len(report.results)} question runs")
    print(f"    MRR (answering record) : {report.mrr:.3f}")
    print(f"    recall@{k:<2}              : {report.recall_at_k:.3f}")
    print(f"    nDCG@{k:<2}                : {report.ndcg_at_k:.3f}")
    print(f"    MAP                    : {report.map_score:.3f}")
    print(f"    precision@{k:<2}           : {report.precision_at_k:.3f}")
    print(f"    answer found in top-{k:<2} : {report.primary_found_rate:.1%}")
    print(f"    mean latency           : {report.mean_latency_ms:.1f} ms")

    per_language = report.by_language()
    if len(per_language) > 1:
        print("\n  By language")
        print(f"    {'lang':<6} {'MRR':>7} {'nDCG':>7} {'found':>8}")
        for language, sub in per_language.items():
            print(
                f"    {language:<6} {sub.mrr:>7.3f} {sub.ndcg_at_k:>7.3f} "
                f"{sub.primary_found_rate:>7.1%}"
            )

    rows = report.failures() if failures_only else report.results
    if show_cases and rows:
        heading = "Failures (answering record not retrieved)" if failures_only else "Per case"
        print(f"\n  {heading}")
        print(f"    {'case':<28} {'lang':<5} {'rank':>5} {'RR':>6} {'nDCG':>6}  answering record")
        for row in rows:
            rank = str(row.primary_rank) if row.primary_rank else "--"
            print(
                f"    {row.case_id:<28} {row.language:<5} {rank:>5} "
                f"{row.reciprocal_rank:>6.3f} {row.ndcg:>6.3f}  {row.primary}"
            )
            if failures_only:
                print(f"      q: {row.question}")
                print(f"      returned: {', '.join(row.ranked[:5]) or '(nothing)'}")


def print_comparison(reports: Sequence[EvaluationReport]) -> None:
    """Tabulate several configurations over the same gold set."""
    if not reports:
        return
    k = reports[0].k
    print(f"\n{RULE}")
    print("CONFIGURATION COMPARISON")
    print(RULE)
    print(
        f"\n  {'configuration':<34} {'MRR':>7} {f'recall@{k}':>10} "
        f"{f'nDCG@{k}':>9} {'found':>8} {'latency':>10}"
    )
    print("  " + "-" * (WIDTH - 4))
    best = max(reports, key=lambda report: report.mrr)
    for report in reports:
        marker = " *" if report is best else "  "
        print(
            f"  {report.label:<34} {report.mrr:>7.3f} {report.recall_at_k:>10.3f} "
            f"{report.ndcg_at_k:>9.3f} {report.primary_found_rate:>7.1%} "
            f"{report.mean_latency_ms:>8.1f}ms{marker}"
        )
    print(f"\n  * best MRR: {best.label}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--gold-set", type=Path, default=DEFAULT_GOLD_SET)
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Cut-off for @k metrics")
    parser.add_argument("--offline", action="store_true", help="BM25 + graph only")
    parser.add_argument("--compare", action="store_true", help="A/B the configurations")
    parser.add_argument("--cases", action="store_true", help="Show the per-case table")
    parser.add_argument("--failures", action="store_true", help="Show only failures, in detail")
    parser.add_argument("--json", type=Path, help="Write machine-readable results here")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(logging.INFO if args.verbose else logging.WARNING)
    settings = get_settings()

    try:
        if args.compare:
            reports = run_comparison(settings, args)
        else:
            reports = [run_single(settings, args)]
    except (GoldSetError, GraphSourceError, ValueError) as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = [report.as_dict() for report in reports]
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote machine-readable results to {args.json}")

    # Non-zero when the answering record was missed for any case, so this can
    # gate a pipeline without anyone reading the table.
    return 0 if all(report.primary_found_rate == 1.0 for report in reports) else 1


def run_single(settings: Settings, args: argparse.Namespace) -> EvaluationReport:
    """Evaluate one configuration."""
    stack = build_stack(settings, offline=args.offline)
    gold_set = GoldSet.load(args.gold_set, stack.corpus)

    print(RULE)
    print("AXIOM RETRIEVAL EVALUATION")
    print(RULE)
    print(f"  corpus   : {stack.describe()}")
    print(f"  gold set : {len(gold_set.cases)} cases, languages {', '.join(EVALUATED_LANGUAGES)}")
    print(f"  source   : {gold_set.source.name}")

    label = "lexical+graph" if args.offline else "full stack"
    report = evaluate(
        stack.pipeline,
        gold_set,
        k=args.top_k,
        languages=EVALUATED_LANGUAGES,
        label=label,
        config_summary={
            "rrf_k": settings.fusion.rrf_k,
            "streams": list(stack.pipeline.stream_names),
            "rerank": settings.reranker.enabled and not args.offline,
        },
    )
    print_report(report, show_cases=args.cases or args.failures, failures_only=args.failures)
    return report


def run_comparison(settings: Settings, args: argparse.Namespace) -> list[EvaluationReport]:
    """Evaluate every configuration variant over the same gold set."""
    reports: list[EvaluationReport] = []
    gold_set: GoldSet | None = None

    print(RULE)
    print("AXIOM RETRIEVAL EVALUATION - CONFIGURATION COMPARISON")
    print(RULE)

    for name, variant, offline in configuration_variants(settings):
        stack = build_stack(variant, offline=offline)
        if gold_set is None:
            gold_set = GoldSet.load(args.gold_set, stack.corpus)
            print(f"  corpus   : {stack.describe()}")
            print(
                f"  gold set : {len(gold_set.cases)} cases, "
                f"languages {', '.join(EVALUATED_LANGUAGES)}"
            )
        print(f"  running  : {name} ...")
        reports.append(
            evaluate(
                stack.pipeline,
                gold_set,
                k=args.top_k,
                languages=EVALUATED_LANGUAGES,
                label=name,
                config_summary={
                    "streams": list(stack.pipeline.stream_names),
                    "weights": dict(variant.fusion.weights),
                    "rerank": variant.reranker.enabled and not offline,
                },
            )
        )

    print_comparison(reports)
    if args.failures:
        for report in reports:
            print_report(report, show_cases=True, failures_only=True)
    return reports


if __name__ == "__main__":
    raise SystemExit(main())
