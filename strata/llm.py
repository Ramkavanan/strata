"""Optional LLM pieces: an OpenAI-compatible completion client and a reader for prose contexts.

Used when a system returns free-text memories or facts that the reference reader cannot parse. Evidence scores
are native only when the adapter supplies span ids; otherwise responses are flagged `attribution: aligned`.
Nothing here is used to compute ground truth.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Optional

from .schema import ContextItem, Response, Task, Usage
from .tokenizer import count_tokens

Complete = Callable[[str, str], str]  # (system, user) -> text


def openai_complete(model: Optional[str] = None, temperature: float = 0.0) -> Complete:
    from openai import OpenAI  # optional dependency: pip install strata-bench[llm]

    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
    model = model or os.environ["STRATA_READER_MODEL"]

    def run(system: str, user: str) -> str:
        r = client.chat.completions.create(
            model=model,
            temperature=temperature,
            seed=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return r.choices[0].message.content or ""

    return run


def default_complete() -> Complete:
    return openai_complete()


SYSTEM = (
    "You answer questions about an organization using ONLY the context items provided. Each item has an id. "
    "Decide which statements are still in force: a later statement by someone without authority over the topic does not "
    "override an earlier decision; statements in different authority domains that conflict are 'contested'; if nothing "
    "in the context establishes the answer, say 'undetermined' and set abstain true. Reply with one JSON object."
)
SCHEMA = (
    '{"answer": str, "state": {"values": [str], "status": "valid|contested|undetermined", "since": "YYYY-MM-DD|null"}, '
    '"evidence": [item ids], "abstain": bool, "staleness_flag": bool, "ids": [item ids]|null, "drifted": bool|null, '
    '"resolved": bool|null, "resolver": item id|null, '
    '"lineage": {"decider": [ids], "reasons": [ids], "trigger": [ids], "discussion": [ids], "follow_ups": [ids], "status": str}|null, '
    '"chain": [{"value": str, "since": "YYYY-MM-DD", "status": str, "evidence": [ids]}]|null}'
)


class LLMReader:
    def __init__(self, complete: Complete):
        self.complete = complete

    def describe(self) -> dict:
        return {"reader": "llm"}

    def answer(self, task: Task, context: list[ContextItem]) -> Response:
        ctx = "\n".join(f"[{(it.source_ids or ['?'])[0]}] {it.text}" for it in context)
        user = (
            f"Query time: {task.query_time.isoformat()}\nQuestion ({task.family}): {task.question}\n"
            f"Params: {json.dumps(task.params)}\n\nContext:\n{ctx or '(empty)'}\n\nJSON schema: {SCHEMA}"
        )
        raw = self.complete(SYSTEM, user)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return Response(
                item_id=task.item_id,
                abstain=True,
                answer=raw[:200],
                usage=Usage(input_tokens=count_tokens(SYSTEM + user), output_tokens=count_tokens(raw), model_calls=1),
                attribution="aligned",
            )
        data = json.loads(m.group(0))
        known = {sid for it in context for sid in it.source_ids}
        data["item_id"] = task.item_id
        data["attribution"] = "native" if known else "aligned"
        data["usage"] = Usage(
            input_tokens=count_tokens(SYSTEM + user), output_tokens=count_tokens(raw), model_calls=1
        ).model_dump()
        for key in ("evidence", "ids"):
            if isinstance(data.get(key), list):
                data[key] = [x for x in data[key] if x in known]
        try:
            return Response.model_validate(data)
        except Exception:
            return Response(item_id=task.item_id, abstain=True, answer=raw[:200], usage=Usage(model_calls=1))
