"""Reference tokenizer for context budgets. Pinned and versioned so budgets are comparable across systems."""

from __future__ import annotations

import re

TOKENIZER_NAME = "strata-ref-v1"
_TOK = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    return len(_TOK.findall(text))


def truncate_to_tokens(text: str, n: int) -> str:
    if n <= 0:
        return ""
    count = 0
    for m in _TOK.finditer(text):
        count += 1
        if count == n:
            return text[: m.end()]
    return text


def enforce_budget(items, budget: int):
    """Harness-side truncation: keep items in order until B tokens; truncate the item that crosses B.

    Returns (kept_items, info) where info records whether truncation happened and the tokens used."""
    kept, used, truncated, dropped = [], 0, False, 0
    for it in items:
        n = count_tokens(it.text)
        if used + n <= budget:
            kept.append(it)
            used += n
        elif used < budget:
            kept.append(it.model_copy(update={"text": truncate_to_tokens(it.text, budget - used)}))
            used = budget
            truncated = True
        else:
            dropped += 1
            truncated = True
    return kept, {"tokens": used, "truncated": truncated, "dropped_items": dropped, "tokenizer": TOKENIZER_NAME}
