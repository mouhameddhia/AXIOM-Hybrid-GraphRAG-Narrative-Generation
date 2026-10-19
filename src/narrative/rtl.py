"""
Right-to-left rendering for Arabic narratives.

The deliverable asks for "correct right-to-left handling", and the failure it
guards against is specific. An Arabic narrative in this system is never purely
Arabic: it is full of Latin-script identifiers -- ``OL000248``, ``SAR``,
``PriceComponent:PC0001166`` -- because citation handles are the whole point
of the output. Dropping those into an RTL paragraph without isolating them
lets the Unicode bidirectional algorithm reorder their boundaries against
neighbouring punctuation, so a citation renders as ``[PC0001166:PriceComponent``
or a bracket jumps to the wrong end of the token.

That is not a font problem or a display-layer problem to push downstream: it
is a property of the *string*, and it travels with the string into any
terminal, PDF, or web page that receives it. So it is fixed here, at
generation time, with the Unicode isolate characters designed for exactly
this: ``U+2066`` LEFT-TO-RIGHT ISOLATE opens a run whose internal direction is
LTR and whose *boundaries* do not affect the surrounding RTL text, and
``U+2069`` POP DIRECTIONAL ISOLATE closes it.

The paragraph itself is marked RTL with ``U+200F`` RIGHT-TO-LEFT MARK, so a
narrative that happens to begin with a Latin identifier still lays out
right-to-left rather than inheriting direction from its first character.

:func:`strip_bidi_controls` exists because these characters must never reach
the verifier: an isolate sitting between a digit and its decimal point would
break number extraction, and the verifier's job is to read exactly what a
human will read.
"""
from __future__ import annotations

import re

__all__ = [
    "LRI",
    "PDI",
    "RLM",
    "apply_rtl",
    "contains_rtl",
    "direction_for",
    "isolate_ltr_runs",
    "strip_bidi_controls",
]

#: LEFT-TO-RIGHT ISOLATE -- opens an LTR run that cannot disturb its neighbours.
LRI = "⁦"
#: POP DIRECTIONAL ISOLATE -- closes the innermost isolate.
PDI = "⁩"
#: RIGHT-TO-LEFT MARK -- pins the paragraph's base direction to RTL.
RLM = "‏"

_BIDI_CONTROLS = re.compile(f"[{LRI}{PDI}{RLM}⁧⁨‎‪-‮]")

#: Arabic, Arabic Supplement, Arabic Extended-A, and the presentation forms.
_RTL_RANGE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")

#: A bracketed citation, or any token containing a Latin letter. Matched as
#: one unit so the brackets travel inside the isolate with their contents
#: rather than being reordered away from them.
_LTR_RUN = re.compile(r"\[[^\[\]]*[A-Za-z][^\[\]]*\]|[A-Za-z][A-Za-z0-9_:.\-#/]*")


def contains_rtl(text: str) -> bool:
    """Whether the text contains any right-to-left script."""
    return bool(_RTL_RANGE.search(text))


def direction_for(language: str) -> str:
    """The base direction for a language code: ``rtl`` for Arabic."""
    return "rtl" if language.lower().startswith("ar") else "ltr"


def isolate_ltr_runs(text: str) -> str:
    """
    Wrap Latin-script runs in bidirectional isolates.

    Already-isolated runs are left alone, so applying this twice is safe --
    which matters because a narrative may be post-processed more than once.
    """
    if LRI in text:
        return text

    def wrap(match: re.Match[str]) -> str:
        return f"{LRI}{match.group(0)}{PDI}"

    return _LTR_RUN.sub(wrap, text)


def apply_rtl(text: str) -> str:
    """
    Prepare a narrative for right-to-left display.

    Pins the paragraph direction and isolates every Latin run inside it. Text
    with no RTL content is returned untouched, so calling this on an English
    narrative is a no-op rather than an error, and already-processed text is
    returned unchanged so a narrative passed through more than once does not
    accumulate direction marks.
    """
    if not contains_rtl(text) or text.startswith(RLM):
        return text
    return RLM + isolate_ltr_runs(text)


def strip_bidi_controls(text: str) -> str:
    """
    Remove every directional control character.

    Used before verification and in tests: the verifier must read the same
    sequence of digits a human reads, and an isolate between a digit and a
    decimal separator would hide a fabricated number from it.
    """
    return _BIDI_CONTROLS.sub("", text)
