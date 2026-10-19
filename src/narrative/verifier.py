"""
Verification of a generated narrative against its fact pack.

The onboarding plan sets the bar: *"narratives explain only engine-computed
numbers; unsupported statements do not appear."* A prompt asking a model to
obey that is not a control -- it is a request. This module is the control.
Every narrative is read back after generation, every number in it is extracted
and matched against the closed fact set, every citation is checked against the
evidence that fact set carries, and a narrative that fails is never returned
as an answer.

Reading a narrative the way a human reads it
--------------------------------------------
Extraction has to see exactly what a reader sees, which takes more than a
digit regex:

*Bidirectional controls are stripped first.* An isolate character sitting
between a digit and its decimal point would split one number into two and
hide a fabricated figure from the check.

*Arabic-Indic digits are normalised.* ``١٠٣٬١٢٥٫٠٠`` is the same number as
``103,125.00`` and must be verified identically, or the Arabic path would
silently enforce nothing -- the precise failure the plan warns about when it
says Arabic faithfulness must be tested explicitly rather than assumed.

*Identifiers are masked before numbers are read.* ``OL000248`` and
``PriceComponent:PC0001166`` are full of digits that are not quantities.
Reading them as numbers would produce constant false violations, and the
usual response to a noisy check is to weaken it.

What counts as a violation
--------------------------
A number that matches no fact, a citation naming a record outside the
evidence set, or a narrative citing nothing at all. Each is reported with
enough detail to be fed back to the model for one corrective attempt -- the
same self-heal shape the text-to-query layer uses -- before the generator
abstains.

Two documented tolerances keep the check honest rather than merely strict:
rounding within :data:`DEFAULT_RELATIVE_TOLERANCE` is accepted, because a
narrative saying 13.2% when the fact is 13.2 should not fail on float
representation; and four-digit years are ignored, because a year is not a
pricing quantity and masking them is cheaper than the alternative of
forbidding the model from ever writing a date.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from src.narrative.facts import FactPack
from src.narrative.rtl import strip_bidi_controls

__all__ = [
    "extract_figures",
    "DEFAULT_ABSOLUTE_TOLERANCE",
    "DEFAULT_MIN_CITATION_COVERAGE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "VerificationReport",
    "Violation",
    "verify",
]

#: Absolute slack, in currency minor units.
DEFAULT_ABSOLUTE_TOLERANCE = 0.01
#: Relative slack, for a narrative that rounds a figure for readability.
DEFAULT_RELATIVE_TOLERANCE = 0.005
#: Fraction of numeric claims that must carry a citation. The acceptance
#: criterion reads "every narrative claim cites a specific record", so the
#: faithful default is 1.0 -- a sentence asserting a figure with nothing
#: attached is exactly the unsupported statement the criterion forbids.
DEFAULT_MIN_CITATION_COVERAGE = 1.0

_ARABIC_INDIC = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
#: Arabic decimal separator, thousands separator, and percent sign.
_ARABIC_PUNCT = str.maketrans({"٫": ".", "٬": ",", "٪": "%"})

#: A bracketed attribution, or a bare ``Label:value`` handle.
#:
#: The bracketed form deliberately accepts any non-space token rather than
#: requiring a colon: figures that came from Track A's engine are attributed
#: to its version string, which carries no colon, and a narrative resting
#: entirely on engine figures would otherwise read as citing nothing at all.
#: Anything bracketed that is not in the evidence set still fails as an
#: unknown citation, so widening what counts as an attribution does not
#: widen what counts as *valid*.
_CITATION = re.compile(r"\[([^\[\]\s]+)\]|\b([A-Z][A-Za-z0-9_]*:[A-Za-z0-9_\-#.]+)")
#: Any token mixing letters and digits: an identifier, not a quantity.
_IDENTIFIER = re.compile(r"\b(?=[A-Za-z0-9_\-.]*[A-Za-z])(?=[A-Za-z0-9_\-.]*\d)[A-Za-z0-9_\-.]+\b")
#: ISO dates, and years standing alone in prose.
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
#: A quantity, with optional thousands separators and decimals.
_NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
#: Sentence boundary: terminal punctuation followed by whitespace. A decimal
#: point is followed by a digit, not whitespace, so "103,125.00 SAR" does not
#: split here -- which is why the simple pattern is safe on money-dense prose.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class Violation:
    """One reason a narrative was not accepted."""

    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """The outcome of checking one narrative."""

    ok: bool
    violations: tuple[Violation, ...]
    numbers_found: tuple[float, ...]
    citations_found: tuple[str, ...]
    #: Sentences that assert at least one figure. These are the "claims" the
    #: acceptance criterion talks about.
    claims_total: int = 0
    #: Of those, how many carry a citation.
    claims_cited: int = 0

    @property
    def citation_coverage(self) -> float:
        """
        Fraction of numeric claims carrying a citation.

        A narrative with no numeric claims scores 1.0 rather than 0.0: it has
        no unsupported statements, which is what the metric measures. Scoring
        it zero would punish a correct abstention message.
        """
        if self.claims_total == 0:
            return 1.0
        return self.claims_cited / self.claims_total

    def summary(self) -> str:
        """The exact text fed back to the model on a corrective attempt."""
        return "\n".join(f"- {v.kind}: {v.detail}" for v in self.violations)


def verify(
    narrative: str,
    pack: FactPack,
    *,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    require_citation: bool = True,
    min_citation_coverage: float = DEFAULT_MIN_CITATION_COVERAGE,
) -> VerificationReport:
    """
    Check a narrative against the only facts it was allowed to use.

    Args:
        narrative: the generated text, in either language.
        pack: the closed fact set it was generated from.
        absolute_tolerance: slack in currency minor units.
        relative_tolerance: slack for a figure rounded for readability.
        require_citation: whether a narrative with no citation at all is a
            violation. It is, for this deliverable -- the argument for making
            it an option is that a single abstention message legitimately
            cites nothing.
        min_citation_coverage: the fraction of numeric claims that must carry
            a citation. At 1.0 every sentence asserting a figure must say
            which record it rests on.

    Returns:
        A :class:`VerificationReport`. ``ok`` is True only when every number
        traces to a fact and every citation to a record in the evidence set.
    """
    plain = strip_bidi_controls(narrative)
    normalised = plain.translate(_ARABIC_INDIC).translate(_ARABIC_PUNCT)

    citations = _extract_citations(normalised)
    numbers = _extract_numbers(normalised)

    violations: list[Violation] = []

    allowed = pack.allowed_values
    for number in numbers:
        if not _matches_any(number, allowed, absolute_tolerance, relative_tolerance):
            violations.append(
                Violation(
                    kind="unsupported-number",
                    detail=(
                        f"The narrative states {number:,.2f}, which is not one of the "
                        f"figures provided. Use only the numbers in the fact table, "
                        f"copied exactly."
                    ),
                )
            )

    known = set(pack.citations)
    if pack.engine_version:
        known.add(pack.engine_version)
    for citation in citations:
        if citation not in known:
            violations.append(
                Violation(
                    kind="unknown-citation",
                    detail=(
                        f"The narrative cites [{citation}], which is not in the "
                        f"evidence for this explanation."
                    ),
                )
            )

    if require_citation and not citations:
        violations.append(
            Violation(
                kind="missing-citation",
                detail=(
                    "The narrative cites no record at all. Every factual claim "
                    "must carry the record id it rests on, in square brackets."
                ),
            )
        )

    claims_total, claims_cited, uncited = _claim_coverage(normalised)
    if claims_total and claims_cited / claims_total < min_citation_coverage:
        violations.append(
            Violation(
                kind="uncited-claim",
                detail=(
                    f"{claims_total - claims_cited} of {claims_total} sentence(s) "
                    f"state a figure without citing a record. Add the record id in "
                    f"square brackets to each. First uncited: {uncited[0]!r}"
                ),
            )
        )

    return VerificationReport(
        ok=not violations,
        violations=tuple(violations),
        numbers_found=tuple(numbers),
        citations_found=tuple(citations),
        claims_total=claims_total,
        claims_cited=claims_cited,
    )


def _claim_coverage(text: str) -> tuple[int, int, list[str]]:
    """
    Count sentences that assert a figure, and how many cite a record.

    This is what turns "every narrative claim cites a specific record" from a
    prompt instruction into a measurable property. A *claim* is a sentence
    stating at least one quantity -- identifiers and dates masked out, so
    "line OL000248 was reviewed" is not a claim and needs no citation. A claim
    counts as cited when a record handle appears in that same sentence.

    Attribution is scoped to the sentence rather than the paragraph on
    purpose: a single trailing citation after four sentences of figures tells
    a reader nothing about which record supports which number, and that
    ambiguity is precisely what the criterion exists to remove.

    Returns:
        ``(claims_total, claims_cited, uncited_sentences)``.
    """
    total = 0
    cited = 0
    uncited: list[str] = []
    for sentence in _SENTENCE_BREAK.split(text):
        stripped = sentence.strip()
        if not stripped:
            continue
        has_citation = bool(_CITATION.search(stripped))
        # Mask before counting, exactly as _extract_numbers does, so a record
        # id or a date never makes a sentence look like a numeric claim.
        masked = _CITATION.sub(" ", stripped)
        masked = _ISO_DATE.sub(" ", masked)
        masked = _IDENTIFIER.sub(" ", masked)
        masked = _YEAR.sub(" ", masked)
        if not _NUMBER.search(masked):
            continue
        total += 1
        if has_citation:
            cited += 1
        else:
            uncited.append(" ".join(stripped.split())[:90])
    return total, cited, uncited


def _extract_citations(text: str) -> list[str]:
    """Every record handle the narrative cites, in first-seen order."""
    found: list[str] = []
    for match in _CITATION.finditer(text):
        handle = match.group(1) or match.group(2)
        if handle and handle not in found:
            found.append(handle)
    return found


def _extract_numbers(text: str) -> list[float]:
    """
    Every quantity the narrative states.

    Citations, identifiers and dates are blanked first, so their digits are
    never mistaken for figures.
    """
    masked = _CITATION.sub(" ", text)
    masked = _ISO_DATE.sub(" ", masked)
    masked = _IDENTIFIER.sub(" ", masked)
    masked = _YEAR.sub(" ", masked)

    numbers: list[float] = []
    for raw in _NUMBER.findall(masked):
        try:
            numbers.append(abs(float(raw.replace(",", ""))))
        except ValueError:  # pragma: no cover - the pattern cannot produce this
            continue
    return numbers


def _matches_any(
    number: float,
    allowed: Sequence[float],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    """Whether one stated number traces to a published fact."""
    for value in allowed:
        tolerance = max(absolute_tolerance, abs(value) * relative_tolerance)
        if abs(number - abs(value)) <= tolerance:
            return True
    return False


def extract_figures(text: str) -> tuple[float, ...]:
    """
    The quantities a reader would see in ``text`` (absolute values), with
    identifiers, dates and years masked out exactly as :func:`verify` does.

    Used to admit the numbers that appear inside the engine's own notes: they
    came from the engine, so quoting them is allowed.
    """
    plain = strip_bidi_controls(text)
    return tuple(_extract_numbers(plain.translate(_ARABIC_INDIC).translate(_ARABIC_PUNCT)))
