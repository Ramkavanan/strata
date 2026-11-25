"""Hidden splits: answer without ground truth, then score with the maintainers' private answers.

A participant runs `strata answer` on the validation split and sends the responses file. A maintainer, who holds
the world models, runs `strata score`. The participant never sees an answer and the harness never trusts a score
it did not compute.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml

from . import operators
from .datasets import Dataset
from .questions import generate_items
from .results import summarize
from .runner import answer_task, make_reader
from .schema import Response, Task
from .score import score_item
from .templates import build_scenario


def answer_split(adapter, ds: Dataset, reader=None, budget: int = 128_000, limit: Optional[int] = None) -> list[dict]:
    """Ingest each condition of each scenario and answer its questions. Returns one record per response."""
    if getattr(adapter, "needs_ground_truth", False):
        raise ValueError("an oracle cannot run on a split that ships no ground truth")
    reader = reader or make_reader(adapter)
    by_cond: dict[tuple[str, str], list] = defaultdict(list)
    for g in ds.tasks():
        by_cond[(g.scenario_id, g.operator)].append(g)
    records = []
    conditions = ds.conditions()
    if limit:
        keep = sorted({sid for sid, _ in conditions})[:limit]
        conditions = [c for c in conditions if c[0] in keep]
    for sid, code in conditions:
        arts = ds.corpus(sid, code)
        adapter.reset()
        t0 = time.perf_counter()
        adapter.ingest(arts, checkpoint=0)
        ingest = {"seconds": time.perf_counter() - t0, "usage": adapter.usage().model_dump()}
        for g in by_cond[(sid, code)]:
            task = Task(
                item_id=g.id, family=g.family, question=g.input, query_time=g.query_time, params=g.params, budget=budget
            )
            resp, info = answer_task(adapter, reader, task, arts)
            records.append(
                {
                    "scenario_id": sid,
                    "condition": code,
                    "response": resp.model_dump(mode="json"),
                    "info": {
                        k: info.get(k, 0)
                        for k in (
                            "tokens",
                            "truncated",
                            "consumer_input_tokens",
                            "system_input_tokens",
                            "output_tokens",
                        )
                    },
                    "ingest": ingest,
                }
            )
    return records


def score_submission(
    records: list[dict],
    private_dir: str,
    dataset_name: str,
    adapter_desc: dict,
    extra: dict,
    prices: Optional[dict] = None,
) -> dict:
    """Score a responses file against the private world models. Returns the same structure as results.json."""
    priv = Path(private_dir) / dataset_name / "validation"
    mapping = json.loads((priv / "mapping.json").read_text())
    task_ids = json.loads((priv / "task_ids.json").read_text())
    templates = {p.stem: yaml.safe_load(p.read_text()) for p in (priv / "templates").glob("*.yaml")}
    renders: dict[tuple[str, str], tuple] = {}
    evals: dict[tuple[str, str], dict] = {}
    unknown = 0
    for rec in records:
        sid, code = rec["scenario_id"], rec["condition"]
        if sid not in mapping or code not in mapping[sid]:
            unknown += 1
            continue
        key = (sid, code)
        if key not in renders:
            base = build_scenario(templates[mapping[sid]["template"]], 0)
            scn = operators.apply(base, mapping[sid][code])
            renders[key] = (scn, {i.id: i for i in generate_items(scn)})
            evals[key] = {
                "render_id": scn.render_id,
                "scenario_id": scn.id,
                "budget": 128_000,
                "rows": [],
                "ingest_usage": rec["ingest"]["usage"],
                "reliability_share_solved_all_runs": None,
            }
        scn, items = renders[key]
        resp = Response.model_validate(rec["response"])
        real = task_ids.get(resp.item_id)
        if real is None or real not in items:
            unknown += 1
            continue
        row = score_item(items[real], resp.model_copy(update={"item_id": real}), scn)
        info = rec.get("info", {})
        row.update(
            {
                "run": 0,
                "budget": 128_000,
                "context_tokens": info.get("tokens", 0),
                "truncated": info.get("truncated", False),
                "consumer_input_tokens": info.get("consumer_input_tokens", 0),
                "system_input_tokens": info.get("system_input_tokens", 0),
                "answer_tokens": info.get("output_tokens", 0),
                "tool_calls": resp.usage.tool_calls,
            }
        )
        evals[key]["rows"].append(row)
    expected = sum(len(v[1]) for v in renders.values())
    answered = sum(len(e["rows"]) for e in evals.values())
    res = summarize(
        adapter_desc,
        {"perturbation": {"evaluations": list(evals.values())}},
        {
            **extra,
            "split": "validation",
            "unknown_records": unknown,
            "coverage": answered / expected if expected else 0.0,
            "n_templates": len({mapping[s]["template"] for s, _ in renders}),
        },
        prices,
    )
    res["rows"] = [r for e in evals.values() for r in e["rows"]]
    return res
