"""
Attribution checking: does the sentence naming a figure state *that* figure?

This closes the gap the D4 verifier explicitly left open. Verification proves
every number in a narrative came from the fact pack, and every citation from
the evidence set. It cannot prove the numbers landed on the right nouns. A
sentence reading

    "...leaving an invoice price of 37,500.00 SAR [PriceComponent:PC0000035]."

passes verification cleanly when 37,500.00 is the *list* price: the figure is
attested, the citation resolves, and the claim is still false. That is the
residual served-wrong risk after D4, and the onboarding plan files the remedy
under D5 as "hallucination checks".

How it works
------------
Each fact in the pack contributes a *role phrase* -- the words a narrative
would use to name it ("invoice price", "volume rebate"). Wherever a role is
**asserted**, the sentence making the assertion must contain that role's own
value. If it states figures but not the right one, the narrative has put a
number on the wrong noun.

Three rules make this precise rather than merely strict. Each was forced by a
false positive on real generated output, not anticipated:

**A role with no figure after it is named, not valued.** This is the gate
that does most of the work, and it was arrived at the slow way. English puts
an asserted value *after* its noun -- "a list price of 37,500.00" -- so a role
with nothing quantitative following it is being referred to: "...which
results in the final pocket price." Every false positive this check produced
on real output had its figures only in front of the role.

**Assertion, not reference.** "13.2% *of the* list price" and "5,019.54 SAR
*below the* list price" name the list price without claiming a value for it.
A role preceded by a referential preposition is skipped; requiring a figure
there flagged three correct sentences.

``to`` belongs in that list and its absence was a live false positive worth
recording: "total erosion **from the list price to the pocket price** is
6,886.88 SAR" names both endpoints of a range. ``from the`` was covered and
``to the`` was not, so the pocket price read as an assertion, the sentence
held no pocket figure, and a correct narrative was reported as served-wrong.
It inflated the measured baseline error rate until an audit of the flagged
text caught it -- which is the argument for recording the offending narrative
beside every rate rather than trusting the number.

**The sentence is the unit, not the adjacent word.** An earlier version read
forward to the first number, which misread "The invoice price after a
2,052.62 SAR invoice discount is 67,752.74 SAR" -- the intervening clause
supplies a number before the real one arrives. Real prose interleaves roles
and values freely in both directions, so the check asks whether the sentence
*contains* the right value rather than where it sits.

**A transformed role is a source, not a claim.** "The list price is reduced to
29,516.90 SAR" names the list price and states a different figure, and it is
perfectly correct prose: the number is the *result* of the transformation, not
the list price. Roles adjacent to a transformation verb are therefore skipped,
in either direction ("reduces the list price to ..." reads the same way).
Without this, correct waterfall narratives -- whose whole subject is one
figure being transformed into another -- were flagged.

**A comparative states a difference, not a value.** "...with the final pocket
price being 44.33 SAR lower than the invoice price" is both correct English
and a true statement, and the figure in it is the gap, not the pocket price.

**Identifiers are masked.** ``SKU0057`` ends in digits, and reading it as the
number 57 turned a correct recommended-price sentence into a false positive.

Scope, stated honestly
----------------------
Five linguistic exemptions were needed before this check stopped firing on
correct prose: referential, transformational and comparative constructions,
range endpoints, and identifiers. Each is a real grammatical category rather
than a string patch, and the 144-swap sweep in the test suite guards against
the exemptions quietly eating recall. But the count is itself the finding:
deciding whether a sentence *asserts* a value is a natural-language inference
problem, and regular expressions approximate it. A claim-level entailment
model is the principled replacement, and that is D6 territory rather than a
sixth pattern here.

The check is one-directional: it can prove a misattribution, but a narrative
it passes is not thereby proven faithful. Two known limits follow from the
sentence-scoped design -- a sentence that legitimately states two roles'
values satisfies both roles even if they were swapped, and a role whose
phrase is not in the table is never inspected. Skipping transformed roles costs
recall too: "the invoice discount reduces the list price to X" is never
checked. It catches the dominant error mode, where a role is asserted
outright with a value that is not its own, and it is silent on all twelve
narratives of the evaluation set.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from src.narrative.facts import Fact, FactPack
from src.narrative.rtl import strip_bidi_controls

__all__ = [
    "AttributionReport",
    "Misattribution",
    "check_attribution",
    "role_phrases",
]

#: Role phrases for the fact keys the contract defines. Derived from the fact
#: labels but trimmed to the words prose actually uses -- a narrative writes
#: "pocket price", never "pocket price actually kept". A fact with no phrase
#: is not inspected: silence beats a guessed phrase that fires falsely.
_ROLE_PHRASES: Mapping[str, tuple[str, ...]] = {
    "list_price": ("list price",),
    "invoice_price": ("invoice price",),
    "pocket_price": ("pocket price",),
    "recommended_price": ("recommended price",),
    "total_deductions": ("total erosion", "total deductions"),
    "expected_margin_pct": ("expected margin",),
    "acceptance_probability": ("acceptance probability",),
    "confidence": ("engine confidence",),
    # Roles that appear in results read from the engine's own output format.
    "pocket_margin": ("pocket margin",),
    "cogs": ("cost of goods sold",),
    "gross_value": ("gross value",),
    "expected_margin": ("expected margin",),
    # Phrased so that the recommendation's confidence cannot be confused with the
    # confidence in an elasticity or segment estimate in the same result.
    "recommendation_confidence": (
        "confidence in this recommendation",
        "confidence in the recommendation",
        "recommendation confidence",
    ),
}

#: Step names map to the words a narrative uses for them.
_STEP_PHRASES: Mapping[str, tuple[str, ...]] = {
    "list_price": ("list price",),
    "invoice_discount": ("invoice discount",),
    "payment_discount": ("payment discount",),
    "volume_rebate": ("volume rebate",),
    "freight": ("freight",),
    "promotional_discount": ("promotional discount",),
    "commission": ("commission",),
}

#: A role preceded by one of these is being referred to, not asserted.
_REFERENTIAL = re.compile(
    r"(?:of|below|above|than|from|against|versus|vs\.?|under|over|beneath|"
    r"compared to|relative to|off|to|into|toward|towards|down to|up to|"
    # "the margin left AFTER the cost of goods sold is 25,552.48": the figure is the
    # margin, and the cost is only the thing it is measured after (found on a correct
    # answer to a question about margin).
    r"after|before|minus|less|net of|excluding)"
    r"\s+(?:the\s+|a\s+|an\s+|its\s+)?$"
)

#: A transformation verb taking the role as its object: "reduces the list
#: price to ...". The figure that follows is the result, not the role.
_TRANSFORM_BEFORE = re.compile(
    r"\b(?:reduc\w+|lower\w+|cuts?|cutting|discount\w*|decreas\w+|drop\w+|"
    r"bring\w*|brought|erod\w+|trim\w+|tak\w+|adjust\w+|scal\w+)\s+"
    r"(?:the\s+|a\s+|an\s+|its\s+)?$"
)
#: A comparative following the role: "the pocket price being 44.33 SAR lower
#: than the invoice price". That states a *difference*, not the role's value,
#: and it is usually true -- 2,484.50 - 2,440.17 really is 44.33.
_COMPARATIVE_AFTER = re.compile(
    r"^\s*(?:being|is|was|are|were|,)?\s*(?:[\d,.]+\s*\w{0,4}\s*,?\s*)?"
    r"(?:lower|higher|less|more|below|above|under|over|greater|smaller|cheaper)\b"
)
#: The same verb on the other side: "the list price is reduced to ...".
_TRANSFORM_AFTER = re.compile(
    r"^\s*(?:,[^,]{0,40},)?\s*(?:is|was|are|were|gets?|got|then)?\s*"
    r"(?:reduc\w+|lower\w+|cut|discount\w*|decreas\w+|dropp?\w*|brought|"
    r"erod\w+|trimm?\w*|taken|adjust\w+|scal\w+|down)\b"
)

#: "the recommended price range is 362.06 to 394.94": the figures belong to the range,
#: not to the role phrase that precedes the word "range".
_LEADING_CITATION = re.compile(r"^\s*\[[^\[\]]*\]")
_RANGE_AFTER = re.compile(r"^\s*(?:range|band|interval)\b")
_NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_CITATION = re.compile(r"\[[^\[\]]*\]")
#: Any token mixing letters and digits -- SKU0057, OL000248, PC0000035. These
#: are identifiers, not quantities, and must never be read as asserted values.
_IDENTIFIER = re.compile(
    r"\b(?=[A-Za-z0-9_\-.]*[A-Za-z])(?=[A-Za-z0-9_\-.]*\d)[A-Za-z0-9_\-.]+\b"
)
#: Terminal punctuation followed by whitespace. A decimal point is followed by
#: a digit, so "103,125.00 SAR" does not split here.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class Misattribution:
    """One role asserted in a sentence that does not contain its value."""

    role_key: str
    role_phrase: str
    expected: float
    stated: tuple[float, ...]
    context: str

    def describe(self) -> str:
        stated = ", ".join(f"{value:,.2f}" for value in self.stated) or "no figure"
        return (
            f"The sentence asserting {self.role_phrase!r} states {stated} but not "
            f"its actual value {self.expected:,.2f}. Context: {self.context!r}"
        )


@dataclass(frozen=True, slots=True)
class AttributionReport:
    """The outcome of checking which numbers landed on which nouns."""

    ok: bool
    misattributions: tuple[Misattribution, ...]
    #: Roles found asserted and confirmed to carry their own value.
    roles_checked: tuple[str, ...]

    def summary(self) -> str:
        return "\n".join(f"- misattributed: {m.describe()}" for m in self.misattributions)


def role_phrases(fact: Fact) -> tuple[str, ...]:
    """The words a narrative would use to name this fact."""
    if fact.key.startswith("step."):
        return _STEP_PHRASES.get(fact.key[len("step.") :], ())
    return _ROLE_PHRASES.get(fact.key, ())


def check_attribution(
    narrative: str,
    pack: FactPack,
    *,
    absolute_tolerance: float = 0.01,
    relative_tolerance: float = 0.005,
) -> AttributionReport:
    """
    Check that each asserted role carries its own value.

    Args:
        narrative: the generated text.
        pack: the fact pack it was generated from.
        absolute_tolerance: slack in currency minor units.
        relative_tolerance: slack for a figure rounded for readability.

    Returns:
        An :class:`AttributionReport`. ``ok`` is False when a sentence
        asserts a role while stating figures that do not include its value.
    """
    text = strip_bidi_controls(narrative)
    by_key = {fact.key: fact for fact in pack.facts}

    problems: list[Misattribution] = []
    checked: list[str] = []

    for sentence in _SENTENCE_BREAK.split(text):
        stripped = sentence.strip()
        if not stripped:
            continue
        numbers = _numbers_in(stripped)
        if not numbers:
            # Nothing is asserted here, so nothing can be misattributed.
            continue

        for start, end, key in _asserted_roles(stripped, pack):
            expected = by_key[key].value
            checked.append(key)
            if any(
                _close(value, expected, absolute_tolerance, relative_tolerance)
                for value in numbers
            ):
                continue
            problems.append(
                Misattribution(
                    role_key=key,
                    role_phrase=stripped[start:end].lower(),
                    expected=expected,
                    stated=numbers,
                    context=" ".join(stripped.split())[:110],
                )
            )

    return AttributionReport(
        ok=not problems,
        misattributions=tuple(problems),
        roles_checked=tuple(dict.fromkeys(checked)),
    )


def _asserted_roles(sentence: str, pack: FactPack) -> list[tuple[int, int, str]]:
    """
    Role occurrences in this sentence that assert a value.

    Overlaps resolve in favour of the longer phrase, so "invoice discount" is
    not also counted as a bare "invoice price" neighbour. Referential mentions
    are dropped.
    """
    lowered = sentence.lower()
    found: list[tuple[int, int, str]] = []
    for fact in pack.facts:
        for phrase in role_phrases(fact):
            for match in re.finditer(re.escape(phrase), lowered):
                before = lowered[: match.start()]
                after = lowered[match.end() :]
                if not _numbers_in(after):
                    # Nothing quantitative follows, so the role is being
                    # named rather than valued: "...results in the final
                    # pocket price." English puts an asserted value after
                    # its noun, and every false positive this check produced
                    # on real output had its figures only in front.
                    continue
                if _REFERENTIAL.search(before):
                    continue  # "13.2% of the list price" -- named, not claimed
                # A citation may stand between the role and the word that tells how it
                # is used ("the pocket price [OrderLine:OL000248] is lower than ...").
                after_words = _LEADING_CITATION.sub("", after)
                if _TRANSFORM_BEFORE.search(before) or _TRANSFORM_AFTER.match(after_words):
                    continue  # "the list price is reduced to ..." -- a source
                if _COMPARATIVE_AFTER.match(after_words):
                    continue  # "... 44.33 SAR lower than ..." -- a difference
                if _RANGE_AFTER.match(after_words):
                    continue  # "recommended price range is A to B" -- the range's ends
                found.append((match.start(), match.end(), fact.key))

    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    kept: list[tuple[int, int, str]] = []
    for item in found:
        if kept and item[0] < kept[-1][1]:
            continue
        kept.append(item)
    return kept


def _numbers_in(sentence: str) -> tuple[float, ...]:
    """Every quantity asserted in a sentence, identifiers and citations aside."""
    masked = _IDENTIFIER.sub(" ", _CITATION.sub(" ", sentence))
    values: list[float] = []
    for raw in _NUMBER.findall(masked):
        try:
            values.append(abs(float(raw.replace(",", ""))))
        except ValueError:  # pragma: no cover - the pattern cannot produce this
            continue
    return tuple(values)


def _close(
    stated: float, expected: float, absolute_tolerance: float, relative_tolerance: float
) -> bool:
    tolerance = max(absolute_tolerance, abs(expected) * relative_tolerance)
    return abs(stated - abs(expected)) <= tolerance
