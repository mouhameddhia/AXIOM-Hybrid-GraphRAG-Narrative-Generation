"""
Is a question about this result at all?

Question mode answers from a closed fact pack. A small model asked something
unrelated ("What is the capital of France?") does not always say so: given a rich
pack it recites figures, and because all samples recite the same ones they agree,
so every downstream check passes. Measured on the engine sample, four unrelated
questions out of sixteen were answered this way, one with agreement 0.71. A fact
that is true and cited but does not answer the question is still a failed answer,
and neither the verifier nor the wrong-label check can see it.

This module closes the cheapest part of that gap *before* the model is called,
deterministically and auditably: a question is on-topic only if at least one of its
content words (stop words removed, light suffix stripping) occurs in the
vocabulary of the result -- its figure labels, the engine's statements, the subject
-- or in a short glossary of pricing terms. It will not judge whether an answer
is relevant to a question that *is* on-topic; that stays unchecked and is listed as
a limit. It also errs towards declining: a question phrased with none of these
words is declined with the reason, and can be rephrased.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from src.narrative.facts import FactPack

__all__ = ["is_advice_request", "is_on_topic", "question_problem", "question_terms"]

_WORD = re.compile(r"[A-Za-z]{3,}")

_STOP = frozenset(
    """
    the and for are but not you all any can had her was one our out has his how its may new now old
    see two way who did get let put say she too use what when where which while why with would
    this that these those there their them then than from into over under about after before
    does done have having been being were will shall should could might must also just very much
    many some such only own same other another each both few more most less least here your
    tell show give explain describe please need want know like make made take took
    """.split()
)

#: Terms that mark a question as being about pricing and its explanation, whether or
#: not the particular result uses the word.
_GLOSSARY = frozenset(
    """
    price prices pricing cost costs margin margins discount discounts deduction deductions rebate rebates
    freight invoice invoiced pocket list waterfall erosion leakage gross net profit profitable
    recommend recommended recommendation range band accept accepted acceptance probability likely likelihood
    elasticity elastic demand sensitivity sensitive segment segments customer customers product products
    order orders line deal deals comparable engine confident confidence reliable reliability uncertain
    uncertainty warning warnings flag flags quality data estimate estimates observation observations
    constraint constraints floor credit terms volume tier abstain abstained decline declined
    keep kept paid pay charge charged amount total sum cheap expensive high low increase decrease
    realisation realization negotiated negotiation off commission promotional
    """.split()
)


def _stem(word: str) -> str:
    word = word.lower()
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ied", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def question_terms(question: str) -> tuple[str, ...]:
    """The content words of a question, stemmed."""
    return tuple(
        dict.fromkeys(
            _stem(word)
            for word in _WORD.findall(question)
            if word.lower() not in _STOP
        )
    )


def _vocabulary(pack: FactPack) -> set[str]:
    texts: list[str] = [pack.subject]
    texts += [fact.label_en for fact in pack.facts]
    texts += list(pack.context)
    words: Iterable[str] = (w for text in texts for w in _WORD.findall(text))
    vocab = {_stem(w) for w in words if w.lower() not in _STOP}
    return vocab | {_stem(w) for w in _GLOSSARY}


#: A request for advice or a judgement rather than an explanation of a figure. The system
#: explains what the engine computed and does not recommend actions (rule 3 of the prompt),
#: and a model asked "is this a good deal?" recites true figures that answer nothing.
_ADVICE = re.compile(
    r"\b(should (we|i|you|they)|is (it|this|that) (a )?(good|bad|great|fair|worth)|worth it|"
    r"do you (recommend|suggest|think)|would you|what should|can we afford|ought to)\b",
    re.IGNORECASE,
)


def is_advice_request(question: str) -> bool:
    """Whether the question asks what to do or whether something is good, not what the figures are."""
    return bool(_ADVICE.search(question))


def question_problem(pack: FactPack, question: str) -> str | None:
    """
    Why this question will not be answered, or ``None`` when it may be. Checked before
    any model call, so a declined question costs nothing.
    """
    if is_advice_request(question):
        return (
            "the question asks for advice or a judgement; this system explains the figures the "
            "engine computed and does not recommend actions"
        )
    if not is_on_topic(pack, question):
        return (
            "the question does not relate to this result (none of its words appears in "
            "the figures, statements or pricing terms)"
        )
    return None


def is_on_topic(pack: FactPack, question: str) -> bool:
    """
    Whether at least one content word of ``question`` belongs to this result's
    vocabulary. A question with no content words at all is not on-topic.
    """
    terms = question_terms(question)
    if not terms:
        return False
    vocabulary = _vocabulary(pack)
    return any(term in vocabulary for term in terms)
