r"""
Grounded narrative generation (D4).

    Track A payload                      the graph
    (every number)                       (the cited records)
          |                                    |
    contract.parse                    evidence.resolve
    validate, never derive            does each citation exist?
          |                                    |
          +-------------> fact pack <----------+
                              |
                        prompt (English)
                              |
                             LLM          <- wording only
                              |
                         verification  --fail--> corrective retry
                              |                        |
                             pass                   exhausted
                              |                        |
                          Narrative                abstention

**This module computes no pricing figure, and neither does anything else in
this package.** Every quantity a narrative may state arrived in Track A's
payload and was published into a closed fact pack before the model was
called; every quantity the model actually wrote is checked against that pack
afterwards. Between those two points a number cannot enter the text
undetected, which is what the track's rule demands: **explain, do not
compute; ground, do not guess.**

The graph is consulted for one thing only -- confirming the records Track A
cited actually exist, and carrying their text alongside the narrative so a
reviewer can check a claim against the record. That is retrieval, not
calculation.

Abstention is a first-class outcome, not an error path. The plan asks the
system to "abstain when evidence is insufficient", so a narrative that cannot
be verified is never downgraded into a best-effort answer with a warning
attached -- it is withheld, with the reason. Four things trigger it: a fact
pack with nothing to cite, a citation that resolves to no record, a
language-model backend that is unreachable, and a narrative that still fails
verification after its corrective attempts.

Language: English is the current focus. The Arabic path (prompt, labels and
the bidirectional handling in :mod:`src.narrative.rtl`) is implemented and
tested but not part of this phase's claims.

Note on scope: the corrective retry here is the same single-track self-heal
the text-to-query layer uses -- generate, check, feed the exact failure back,
try again. The k=3 self-consistency sampling and abstain-on-disagreement
mechanism is D5, Week 6, and is deliberately not built here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config.settings import NarrativeConfig
from src.narrative.evidence import EvidenceIndex
from src.narrative.facts import FactPack
from src.narrative.relevance import question_problem
from src.narrative.rtl import apply_rtl, direction_for, strip_bidi_controls
from src.narrative.verifier import VerificationReport, verify
from src.text_to_query.llm_client import LLMClient, LLMRequestError

__all__ = ["Narrative", "NarrativeGenerator"]

logger = logging.getLogger(__name__)

_LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}

_PROMPT = """You are a pricing analyst writing a short, factual explanation for a
business reader. Write in {language_name}.

What you are explaining: {what}
Subject: {subject}
{structure}
{question_block}These are the ONLY figures you may state. Each is already calculated. Copy
each one exactly as written, including its separators and currency:

{facts}
{context_block}
Records you may cite:
{citations}

Rules you must follow:
1. Never state a number that is not in the list above. Do not add, subtract,
   average, or convert anything -- all the arithmetic is already done.
2. Every sentence that states a figure must cite the record that figure
   rests on, in square brackets, like [{example_citation}]. One citation at
   the end of the paragraph is not enough -- each such sentence needs its own.
   Use the record shown beside that figure in the table above. Only when a
   figure has no record of its own may you cite another from the list, and
   then pick the closest one. Citing the same record after every sentence is
   wrong even when that record is valid: it tells the reader nothing about
   which figure rests on what.
3. If something is not in the list above, do not mention it. Do not speculate
   about causes, and do not recommend actions.
4. Write {max_words} words or fewer, as flowing prose. No headings, no bullet
   points, no markdown.
{statement_rule}{language_rule}
{example_block}
Write the explanation now, and nothing else."""

_EXAMPLE_FAKE = """This is the citation density required. Note that EVERY sentence below states
a figure and EVERY one of them ends with a record in brackets:

    The list price is 48,123.45 SAR [PriceComponent:PC0000000]. An invoice
    discount of 1,234.56 SAR leaves an invoice price of 46,888.89 SAR
    [PriceComponent:PC0000001]. A volume rebate of 987.65 SAR, which does not
    appear on the invoice, brings the pocket price to 45,901.24 SAR
    [PriceComponent:PC0000002].

Note that the three sentences cite three *different* records -- each figure's
own. Every figure and record id in that example is fake and appears nowhere in
your fact table. Copy the sentence pattern, the citation density, and the use
of a distinct record per figure; never the values.
"""

_EXAMPLE_REAL = """This is the citation density required. Note that EVERY sentence below states
a figure and ends with the record shown beside it in the table. Here is the
pattern, using the first figures of your own table (you may reuse them):

{lines}
"""

_RETRY = """

Your previous attempt was rejected by an automatic check:

{violations}

Previous attempt:
{previous}

Write it again, corrected. Use only the listed figures and the listed records."""

_WHAT = {
    "graph_answer": "a value computed exactly by the knowledge graph in answer to the reader's question",
    "engine_result": (
        "a result produced by the pricing engine: how an order line moves from list "
        "price to the amount kept, the engine's price recommendation, and how far each "
        "part can be trusted"
    ),
    "price_recommendation": "a recommended price produced by the pricing engine",
    "recommendation_and_waterfall": (
        "a recommended price, and how an existing order line moves from list "
        "price down to the amount actually kept"
    ),
}

#: Structural guidance per explanation type. This shapes the *order of the
#: argument*, never its arithmetic -- a waterfall read in the wrong order
#: implies the invoice price already includes off-invoice leakage, which is
#: precisely the confusion the waterfall exists to dispel.
_STRUCTURE = {
    "engine_result": (
        "\nCover, in this order and only where the data below supports it: the "
        "recommended range and how likely it is to be accepted; the walk from list "
        "price to pocket price, keeping the deductions shown on the invoice apart "
        "from those that are not; then how confident the engine is and any "
        "warning it raised.\n"
    ),
    "recommendation_and_waterfall": (
        "\nCover the recommended price and its range first. Then walk the "
        "existing line in this order, keeping the two stages distinct:\n"
        "  list price\n"
        "    minus the deductions that appear on the invoice -> invoice price\n"
        "    minus the deductions that do not appear on the invoice -> pocket price\n"
        "Make clear that the second group never shows on the invoice, which is\n"
        "why the pocket price is lower than the invoice price.\n"
    ),
}


_QUESTION = (
    "The reader asked: \"{question}\"\n"
    "Answer that question directly, in the fewest sentences that fully answer it, using\n"
    "only the figures and statements below. If they do not contain what is needed,\n"
    "or the question is not about this pricing result, reply with exactly the single\n"
    "word INSUFFICIENT and nothing else.\n\n"
)

_STATEMENT_RULE = (
    "   Statements from the engine may be restated in your own words, but never\n"
    "   strengthened: if the engine says confidence is low, do not call it reliable.\n"
    "   Cite the record shown beside a statement when you use it.\n"
)

#: Length allowance for the general explanation of a rich engine result.
_OVERVIEW_WORDS = {"engine_result": 190}

_LANGUAGE_RULE_AR = (
    "5. Write in Modern Standard Arabic. Keep record identifiers, product codes\n"
    "   and currency codes in Latin script exactly as given.\n"
)


@dataclass(frozen=True, slots=True)
class Narrative:
    """One generated explanation, or a reasoned refusal to produce one."""

    subject: str
    language: str
    direction: str
    #: Display-ready text: for Arabic this carries the bidirectional controls.
    text: str
    success: bool
    abstained: bool
    attempts: int
    citations: tuple[str, ...] = ()
    reason: str | None = None
    verification: VerificationReport | None = None

    @property
    def plain_text(self) -> str:
        """The text with directional controls removed, for logs and tests."""
        return strip_bidi_controls(self.text)

    def as_html(self) -> str:
        """The narrative in a correctly-directed block, for a web surface."""
        return f'<p dir="{self.direction}" lang="{self.language}">{self.plain_text}</p>'


class NarrativeGenerator:
    """
    Turns a verified fact pack into a cited explanation, or abstains.

    Usage::

        generator = NarrativeGenerator(llm, settings.narrative)
        narrative = generator.explain(pack, language="ar")
        if narrative.success:
            print(narrative.text)
    """

    __slots__ = ("_llm", "_config", "_base_temperature")

    def __init__(
        self,
        llm_client: LLMClient,
        config: NarrativeConfig,
        *,
        base_temperature: float = 0.0,
    ) -> None:
        self._llm = llm_client
        self._config = config
        self._base_temperature = base_temperature

    def explain(
        self,
        pack: FactPack,
        *,
        language: str = "en",
        evidence: EvidenceIndex | None = None,
        question: str | None = None,
    ) -> Narrative:
        """
        Write one explanation of ``pack`` in ``language``.

        Args:
            pack: the closed fact set. Nothing outside it can be stated.
            language: ``"en"`` (the current focus) or ``"ar"``.
            evidence: when supplied, every handle the pack cites must resolve
                against the graph before the model is called.
            question: when given, the model answers that question from the
                fact pack instead of writing the general explanation. It may
                decline with ``INSUFFICIENT`` when the pack cannot answer, which
                is reported as an abstention. The same verification applies.

        Returns:
            A :class:`Narrative`. Check ``success`` before using ``text`` --
            an abstention carries its reason instead of an explanation.
        """
        if language not in _LANGUAGE_NAMES:
            raise ValueError(
                f"language must be one of {sorted(_LANGUAGE_NAMES)}, got {language!r}"
            )

        insufficient = self._insufficient_evidence(pack, evidence)
        if insufficient:
            return self._abstain(pack, language, insufficient, attempts=0)

        question = (question or "").strip() or None
        problem = question_problem(pack, question) if question else None
        if problem:
            return self._abstain(pack, language, problem + ", so nothing was generated", attempts=0)
        prompt_base = self._build_prompt(pack, language, question)
        previous: str | None = None
        report: VerificationReport | None = None
        total_attempts = self._config.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            prompt = prompt_base
            if previous is not None and report is not None:
                prompt += _RETRY.format(
                    violations=report.summary(), previous=previous
                )

            temperature = self._config.temperature_for(attempt, self._base_temperature)
            try:
                raw = self._llm.complete(prompt, temperature=temperature)
            except LLMRequestError as exc:
                logger.error("Narrative generation failed: %s", exc)
                return self._abstain(
                    pack, language, f"the language model was unreachable: {exc}", attempt
                )

            candidate = _clean(raw)
            if question and candidate.upper().startswith("INSUFF"):  # tolerate a misspelt refusal
                return self._abstain(
                    pack,
                    language,
                    "the available figures and statements cannot answer this question",
                    attempt,
                )
            report = verify(
                candidate,
                pack,
                absolute_tolerance=self._config.absolute_tolerance,
                relative_tolerance=self._config.relative_tolerance,
                min_citation_coverage=self._config.min_citation_coverage,
            )
            if report.ok:
                logger.info(
                    "Narrative verified on attempt %d/%d (%s, %d citation(s))",
                    attempt, total_attempts, language, len(report.citations_found),
                )
                return Narrative(
                    subject=pack.subject,
                    language=language,
                    direction=direction_for(language),
                    text=apply_rtl(candidate) if language == "ar" else candidate,
                    success=True,
                    abstained=False,
                    attempts=attempt,
                    citations=report.citations_found,
                    verification=report,
                )

            logger.warning(
                "Attempt %d/%d rejected: %s",
                attempt, total_attempts, "; ".join(v.kind for v in report.violations),
            )
            previous = candidate

        return self._abstain(
            pack,
            language,
            "the explanation could not be verified against the evidence",
            total_attempts,
            report,
        )

    def _insufficient_evidence(
        self, pack: FactPack, evidence: EvidenceIndex | None
    ) -> str | None:
        """
        Whether this pack can support a cited narrative at all.

        When an index is supplied, every handle Track A cited must resolve to
        a real record. A citation pointing at a record the graph does not
        contain is worse than no citation: it looks checkable and is not.
        """
        if not pack.facts:
            return "there are no figures to explain"
        if not pack.citations and not pack.engine_version:
            return "there are no records to cite"
        if evidence is not None:
            missing = evidence.missing(pack.citations)
            if missing:
                return (
                    f"{len(missing)} cited record(s) do not exist in the graph: "
                    f"{', '.join(missing)}"
                )
        return None

    def _abstain(
        self,
        pack: FactPack,
        language: str,
        reason: str,
        attempts: int,
        report: VerificationReport | None = None,
    ) -> Narrative:
        return Narrative(
            subject=pack.subject,
            language=language,
            direction=direction_for(language),
            text="",
            success=False,
            abstained=True,
            attempts=attempts,
            reason=reason,
            verification=report,
        )

    def _build_prompt(self, pack: FactPack, language: str, question: str | None = None) -> str:
        if question:
            question_block = _QUESTION.format(question=question)
            structure = ""
            max_words = min(self._config.max_words, 90)
        else:
            question_block = ""
            structure = (
                _engine_structure(pack) if pack.kind == "engine_result" else _STRUCTURE.get(pack.kind, "")
            )
            max_words = _OVERVIEW_WORDS.get(pack.kind, self._config.max_words)
        context = pack.render_context()
        context_block = (
            "\nStatements from the engine you may restate (copy the wording; add nothing; "
            "each carries its record):\n" + context + "\n"
            if context
            else ""
        )
        return _PROMPT.format(
            language_name=_LANGUAGE_NAMES[language],
            what=_WHAT.get(pack.kind, "a pricing result"),
            subject=pack.subject,
            structure=structure,
            question_block=question_block,
            facts=pack.render(language),
            context_block=context_block,
            citations="\n".join(f"- {c}" for c in (pack.citations or (pack.engine_version,))),
            example_citation=(pack.citations or (pack.engine_version,))[0],
            max_words=max_words,
            statement_rule=_STATEMENT_RULE if context else "",
            language_rule=_LANGUAGE_RULE_AR if language == "ar" else "",
            example_block="" if question else _example_block(pack),
        )


def _engine_structure(pack: FactPack) -> str:
    """
    The topics an overview of an engine result may cover: only those the result
    actually holds. Asked to cover a waterfall that is not there, a small model
    invents a sentence about one (observed on an abstained recommendation).
    """
    keys = {fact.key for fact in pack.facts}
    topics: list[str] = []
    if keys & {"price_range_low", "recommended_price", "price_range_high"}:
        topics.append("the recommended price range and how likely the customer is to accept it")
    elif "comparable_deals" in keys:
        topics.append(
            "that the engine did not recommend a price, and the reasons it gives "
            "(the number of comparable past deals and its own statements)"
        )
    if "list_price" in keys:
        topics.append(
            "the walk from list price to pocket price, keeping the deductions shown on "
            "the invoice apart from those that are not"
        )
    if any(key.startswith("elasticity") for key in keys):
        topics.append("how far the elasticity estimate can be trusted")
    topics.append("how confident the engine is, and any warning it raised")
    listed = "\n".join(f"  {i}. {topic}" for i, topic in enumerate(topics, start=1))
    return (
        "\nCover only these topics, in this order, and nothing else. Do not describe "
        "anything that is not in the data below:\n" + listed + "\n"
    )


def _example_block(pack: FactPack) -> str:
    """
    The worked example of citation density.

    Results in the original flat format keep the example with made-up values the
    measured runs used. A result read from the engine's own format may hold no
    waterfall, and a model given fake figures it cannot match to anything copies
    them (observed: an abstained recommendation was answered with the example's
    invoice figures). For those the example is built from the pack's own first
    figures, so imitating it is harmless.
    """
    if pack.kind != "engine_result":
        return _EXAMPLE_FAKE
    lines = []
    for fact in pack.facts:
        if len(lines) == 3:
            break
        cite = fact.citation or pack.engine_version
        lines.append(f"    The {fact.label_en} is {fact.formatted(pack.currency_code)} [{cite}].")
    if not lines:
        return _EXAMPLE_FAKE
    return _EXAMPLE_REAL.format(lines="\n".join(lines))


def _clean(raw: str) -> str:
    """
    Strip the wrapping models habitually add around prose.

    Code fences and a leading "Explanation:" label are not content, and
    letting them through would cost a verification retry to remove.
    """
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            fenced = parts[1]
            first_line, _, rest = fenced.partition("\n")
            text = (rest if first_line.strip().isalpha() else fenced).strip()
    for label in ("explanation:", "narrative:", "answer:"):
        if text.lower().startswith(label):
            text = text[len(label) :].strip()
            break
    return text
