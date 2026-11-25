"""Evaluation loop: ingest, answer every item under a budget, score. Adapter-agnostic."""

from __future__ import annotations

import time
from typing import Callable, Optional

from . import operators
from .adapters.base import ContextSystem
from .questions import generate_items, growth_items
from .reader import ReferenceReader
from .scenario import Scenario
from .schema import ContextItem, Item, Response, Task, Usage
from .score import aggregate, growth_metrics, score_item
from .tokenizer import count_tokens, enforce_budget

BUDGETS = [2_000, 8_000, 32_000, 128_000]
BLOCKS = ("perturbation", "budget", "growth", "attribution")


def make_reader(adapter: ContextSystem, kind: Optional[str] = None):
    """The consumer that turns context into a structured response. `kind` overrides the adapter's default, which
    is how one context system is evaluated under several fixed readers (Section 6.2)."""
    explicit = kind is not None
    kind = kind or getattr(adapter, "reader", "recency")
    if kind == "llm":
        from .llm import LLMReader, default_complete

        return LLMReader(default_complete())
    overrides = {} if explicit else getattr(adapter, "reader_overrides", {})
    return ReferenceReader(kind, **overrides)


def align_context(context: list[ContextItem], artifacts) -> tuple[list[ContextItem], bool]:
    """Fallback provenance for systems that return text without span ids: best token-overlap span (flagged aligned)."""
    import re

    tok = lambda t: set(re.findall(r"[a-z0-9]+", t.lower()))
    spans = [(s.id, tok(s.text)) for a in artifacts for s in a.spans]
    out, aligned = [], False
    for it in context:
        if it.source_ids:
            out.append(it)
            continue
        t = tok(it.text)
        best, score = None, 0.0
        for sid, st in spans:
            j = len(t & st) / max(1, len(t | st))
            if j > score:
                best, score = sid, j
        aligned = True
        out.append(it.model_copy(update={"source_ids": [best] if best and score >= 0.3 else []}))
    return out, aligned


def _answer(adapter, reader, item: Item, budget: int, scn: Scenario) -> tuple[Response, dict]:
    resp, info = answer_task(adapter, reader, item.task(budget), scn.artifacts)
    resp.item_id = item.id
    return resp, info


def answer_task(adapter, reader, task: Task, artifacts) -> tuple[Response, dict]:
    """One task through the system and its consumer. Needs no ground truth, so it also serves hidden splits."""
    budget = task.budget
    before = adapter.usage()
    t0 = time.perf_counter()
    info = {"tokens": 0, "truncated": False}
    if getattr(adapter, "own_reader", False):
        resp = adapter.answer(task)
    else:
        ctx = adapter.context(task, budget)
        ctx, info = enforce_budget(ctx, budget)
        aligned = False
        if getattr(adapter, "provenance", "native") == "aligned":
            ctx, aligned = align_context(ctx, artifacts)
        resp = reader.answer(task, ctx)
        if aligned:
            resp = resp.model_copy(update={"attribution": "aligned"})
    after = adapter.usage()
    # Token accounting for cost: what the consumer read, what the system's own models read, what was written.
    reader_in, reader_out = resp.usage.input_tokens, resp.usage.output_tokens
    info["system_input_tokens"] = after.input_tokens - before.input_tokens
    info["consumer_input_tokens"] = reader_in or (info.get("tokens", 0) + count_tokens(task.question))
    info["output_tokens"] = reader_out or count_tokens(resp.answer or "")
    resp.usage = Usage(
        input_tokens=resp.usage.input_tokens + (after.input_tokens - before.input_tokens),
        output_tokens=resp.usage.output_tokens + (after.output_tokens - before.output_tokens),
        model_calls=resp.usage.model_calls + (after.model_calls - before.model_calls),
        tool_calls=after.tool_calls - before.tool_calls,
        retries=after.retries - before.retries,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    resp.item_id = task.item_id
    return resp, info


def _prepare(adapter, scn: Scenario, items: list[Item]) -> None:
    if getattr(adapter, "needs_ground_truth", False):
        adapter.attach(scn, items)


def evaluate_render(
    adapter: ContextSystem,
    scn: Scenario,
    budget: int = 128_000,
    repeats: int = 1,
    make_adapter: Optional[Callable[[], ContextSystem]] = None,
    reader=None,
) -> dict:
    """Ingest the render once, answer every item `repeats` times, return scored rows and raw responses."""
    items = generate_items(scn)
    adapter.reset()
    _prepare(adapter, scn, items)
    t0 = time.perf_counter()
    adapter.ingest(scn.public_artifacts(), checkpoint=len(scn.checkpoints) - 1)
    ingest_s = time.perf_counter() - t0
    ingest_usage = adapter.usage()
    reader = reader or make_reader(adapter)
    rows, responses = [], []
    for run in range(repeats):
        for it in items:
            resp, info = _answer(adapter, reader, it, budget, scn)
            row = score_item(it, resp, scn)
            row.update(
                {
                    "run": run,
                    "budget": budget,
                    "context_tokens": info.get("tokens", 0),
                    "truncated": info.get("truncated", False),
                    "consumer_input_tokens": info["consumer_input_tokens"],
                    "system_input_tokens": info["system_input_tokens"],
                    "answer_tokens": info["output_tokens"],
                    "tool_calls": resp.usage.tool_calls,
                }
            )
            rows.append(row)
            responses.append(resp.model_dump(mode="json"))
    solved_all = None
    if repeats > 1:
        by: dict[str, list[bool]] = {}
        for r in rows:
            by.setdefault(r["item_id"], []).append(r["correct"])
        solved_all = sum(all(v) for v in by.values()) / len(by)
    return {
        "render_id": scn.render_id,
        "scenario_id": scn.id,
        "budget": budget,
        "rows": rows,
        "responses": responses,
        "aggregate": aggregate(rows),
        "reliability_share_solved_all_runs": solved_all,
        "ingest_seconds": ingest_s,
        "ingest_usage": ingest_usage.model_dump(),
        "n_items": len(items),
    }


def evaluate_growth(adapter: ContextSystem, scn: Scenario, budget: int = 128_000, reader=None) -> dict:
    """Growing-state protocol: K increments by ingestion time, a fixed panel re-asked at every checkpoint."""
    adapter.reset()
    reader = reader or make_reader(adapter)
    cps = scn.checkpoints
    expected: dict[str, list[tuple]] = {}
    answered: dict[str, list[Optional[tuple]]] = {}
    rows, responses = [], []
    prev = None
    for k, cp in enumerate(cps):
        inc = [a for a in scn.public_artifacts() if a.ingest_date <= cp and (prev is None or a.ingest_date > prev)]
        panel = growth_items(scn, k)
        _prepare(adapter, scn, panel)
        adapter.ingest(inc, checkpoint=k)
        for it in panel:
            resp, info = _answer(adapter, reader, it, budget, scn)
            row = score_item(it, resp, scn)
            row.update({"checkpoint": k, "budget": budget})
            rows.append(row)
            responses.append({"checkpoint": k, "response": resp.model_dump(mode="json")})
            exp_vals = tuple(it.expected["values"])
            ans_vals = tuple(resp.state.values) if resp.state and resp.state.values else None
            expected.setdefault(it.variable, [()] * len(cps))
            answered.setdefault(it.variable, [None] * len(cps))
            expected[it.variable][k] = exp_vals
            answered[it.variable][k] = ans_vals
        prev = cp
    gm = growth_metrics(expected, answered)
    return {
        "render_id": scn.render_id,
        "scenario_id": scn.id,
        "growth": gm,
        "rows": rows,
        "responses": responses,
        "accuracy_by_checkpoint": [
            aggregate([r for r in rows if r["checkpoint"] == k])["accuracy_by_family"] for k in range(len(cps))
        ],
        "checkpoints": [c.isoformat() for c in cps],
    }


def run_block(
    adapter: ContextSystem,
    scenarios: list[Scenario],
    block: str,
    repeats: int = 1,
    operator_names: Optional[list[str]] = None,
    budgets: Optional[list[int]] = None,
    reader=None,
) -> dict:
    """The three blocks of Section 4.4, run over `scenarios` (clean renders). Same items across blocks."""
    out: dict = {"block": block, "evaluations": []}
    for base in scenarios:
        if block == "perturbation":
            for name in operator_names or operators.OPERATORS:
                scn = operators.apply(base, name)
                out["evaluations"].append(evaluate_render(adapter, scn, 128_000, repeats, reader=reader))
        elif block == "budget":
            for name in ("clean", "contradict_low"):
                scn = operators.apply(base, name)
                for b in budgets or BUDGETS:
                    out["evaluations"].append(evaluate_render(adapter, scn, b, repeats, reader=reader))
        elif block == "growth":
            for name in ("clean", "noisy"):
                scn = operators.apply(base, name)
                out["evaluations"].append(evaluate_growth(adapter, scn, reader=reader))
        elif block == "attribution":
            pass
        else:
            raise ValueError(block)
    if block == "attribution":
        return run_attribution(adapter, scenarios, reader, operator_names)
    return out


def run_attribution(
    adapter: ContextSystem,
    scenarios: list[Scenario],
    reader=None,
    operator_names: Optional[list[str]] = None,
    system_evals: Optional[list[dict]] = None,
) -> dict:
    """Section 6.1: run the same consumer three times, with no context, with the system under test, and with oracle
    context. The shortfall from a perfect outcome then decomposes exactly into context value, context loss and
    consumer gap. `system_evals` reuses perturbation-block runs of the system so a paid system is not run twice."""
    from .adapters.base import get_adapter

    if getattr(adapter, "own_reader", False):
        return {
            "block": "attribution",
            "evaluations": [],
            "skipped": "the system answers directly, so its consumer cannot be given other contexts",
        }
    reader = reader or make_reader(adapter)
    none, oracle = get_adapter("no_context"), get_adapter("oracle_context")
    have = {(e["scenario_id"], e["render_id"]): e for e in (system_evals or []) if e.get("budget") == 128_000}
    out: dict = {"block": "attribution", "evaluations": []}
    for base in scenarios:
        for name in operator_names or operators.OPERATORS:
            scn = operators.apply(base, name)
            for cond, ad in (("none", none), ("system", adapter), ("oracle", oracle)):
                ev = have.get((scn.id, scn.render_id)) if cond == "system" else None
                if ev is None:
                    ev = evaluate_render(ad, scn, 128_000, 1, reader=reader)
                else:
                    ev = {**ev, "rows": [r for r in ev["rows"] if r["run"] == 0], "responses": []}
                out["evaluations"].append({**ev, "condition": cond})
    return out


def budget_curve(rows_by_budget: dict[int, list[dict]], metric: str = "correct") -> dict[int, float]:
    return {b: sum(float(r[metric]) for r in rs) / len(rs) for b, rs in sorted(rows_by_budget.items()) if rs}
