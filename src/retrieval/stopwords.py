"""
Bilingual stopword lists for lexical retrieval.

Why this exists is specific to the corpus rather than generic hygiene. The
AXIOM corpus is almost entirely key-value records, so ordinary English function
words are *rare* in it: ``"the"`` occurs in 80 of 5,794 records (1.4%), only
inside the prose ``SellableProduct.description`` field. BM25 therefore assigns
stopwords a **high IDF** and treats them as strongly discriminative.

The consequence, measured on the question bank: the natural-language question

    "Give me the full picture of opportunity OPP00252 including the customer,
     the quotes and the outcome."

scored 40 ``SellableProduct`` records at an identical 20.26 on the strength of
``the`` (x4) and ``of`` alone, while ``DealOutcome:OPP00252`` -- which matches
the rare, genuinely discriminative ``opp00252`` -- did not make the list.

Removing these terms before scoring restores the intended behaviour: the
identifier dominates, and prose descriptions stop hijacking long questions.

Both lists are deliberately short. Aggressive stemming or a large list would
risk removing pricing vocabulary that carries real meaning here (``price``,
``cost``, ``discount``), and everything is overridable through
``BM25_STOPWORDS`` so the behaviour stays configuration-driven.
"""
from __future__ import annotations

from typing import Final

__all__ = ["ARABIC_STOPWORDS", "DEFAULT_STOPWORDS", "ENGLISH_STOPWORDS"]

#: Common English function words. Restricted to words with no domain meaning:
#: notably absent are "for", "by" and "per", which appear inside record text in
#: ways worth matching, and anything pricing-related.
ENGLISH_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a", "about", "all", "am", "an", "and", "any", "are", "as", "at",
        "be", "been", "being", "but", "can", "could", "did", "do", "does",
        "doing", "done", "from", "get", "give", "had", "has", "have", "he",
        "her", "here", "him", "his", "how", "i", "if", "in", "into", "is",
        "it", "its", "just", "me", "more", "most", "much", "must", "my",
        "of", "on", "only", "or", "other", "our", "out", "over", "own",
        "please", "same", "she", "should", "show", "so", "some", "such",
        "tell", "than", "that", "the", "their", "them", "then", "there",
        "these", "they", "this", "those", "to", "up", "us", "was", "we",
        "were", "what", "when", "where", "which", "while", "who", "whom",
        "why", "will", "with", "would", "you", "your",
    }
)

#: Common Arabic function words, in the undiacritised form the tokenizer
#: produces after stripping tashkeel. Kept small for the same reason as above.
ARABIC_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "أن", "ان", "إن", "أو", "او", "إلى", "الى", "التي", "الذي", "الذين",
        "أي", "اي", "أين", "اين", "با", "بعد", "بين", "ثم", "حتى", "عن",
        "على", "عليه", "في", "فى", "قد", "كان", "كانت", "كل", "كما", "لا",
        "لدى", "لم", "لن", "له", "لها", "ما", "مع", "من", "هذا", "هذه",
        "هل", "هم", "هي", "هو", "و", "ولا", "ولكن", "يكون", "كيف", "متى",
        "ماذا", "لماذا", "أعطني", "اعطني", "أخبرني", "اخبرني", "لي", "بها",
    }
)

#: The default applied when ``BM25_STOPWORDS`` is not set.
DEFAULT_STOPWORDS: Final[frozenset[str]] = ENGLISH_STOPWORDS | ARABIC_STOPWORDS
