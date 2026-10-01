"""Small, local keyword matching for optional context; no model request is needed."""

from __future__ import annotations

import re

_COMMON_WORDS = (
    "a an and are as at be by can could do for from have how in is it its me my of on or "
    "please that the their then this to use was we what when with would you your "
    "add change create fix help make task "
    "aby albo ale co czy dla do i jak jest mi na nie o oraz po proszę się ten to w z za "
    "dodaj popraw zrób zadanie"
)
_STOP_WORDS = frozenset(_COMMON_WORDS.split())


def keywords(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[^\W_]+", text.casefold())
        if len(word) >= 2 and not word.isdecimal() and word not in _STOP_WORDS
    }


def is_follow_up(goal: str) -> bool:
    """Keep task context for short, explicit continuation/approval messages."""
    return len(goal) <= 120 and bool(
        re.match(
            r"\s*(continue|go ahead|do it|yes|ok|okay|kontynuuj|dalej|tak|zrób to|zrob to)\b",
            goal,
            re.IGNORECASE,
        )
    )
