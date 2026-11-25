"""`strata verify`: re-derive ground truth from the scenario run and re-score every recorded response.

A published number is valid only if the render and item hashes match the scenario run and the recomputed aggregates
equal the recorded ones. Verification never trusts the results file for ground truth.
"""

from __future__ import annotations

import json
import os

from . import operators
from .questions import generate_items, growth_items
from .scenario import Scenario
from .schema import Response
from .score import aggregate, score_item


def load_render(scn_run: str, scenario_id: str, op: str) -> Scenario:
    with open(os.path.join(scn_run, "renders", scenario_id, f"{op}.json")) as f:
        return Scenario.from_dict(json.load(f))


def _close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, (int, float)):
        return abs(a - b) <= tol
    return a == b


def verify_evaluation(eval_dir: str, scn_run: str) -> list[str]:
    problems: list[str] = []
    with open(os.path.join(eval_dir, "results.json")) as f:
        results = json.load(f)
    with open(os.path.join(scn_run, "manifest.json")) as f:
        sman = json.load(f)
    hashes = sman.get("hashes", {})
    cache: dict[tuple, Scenario] = {}
    item_cache: dict[tuple, dict] = {}
    rescored: dict[tuple, list[dict]] = {}
    with open(os.path.join(eval_dir, "responses.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            sid, op = rec["scenario_id"], rec["op"]
            key = (sid, op)
            if key not in cache:
                scn = load_render(scn_run, sid, op)
                want = hashes.get(f"{sid}/{op}", {})
                if want and (scn.render_hash() != want.get("render") or scn.world_hash() != want.get("world")):
                    problems.append(f"{sid}/{op}: stored render does not match the scenario manifest hashes")
                regen = operators.apply(
                    Scenario.from_dict(json.load(open(os.path.join(scn_run, "renders", sid, "clean.json")))), op
                )
                if regen.render_hash() != scn.render_hash():
                    problems.append(f"{sid}/{op}: operator output no longer reproduces the stored render")
                cache[key] = scn
            scn = cache[key]
            if rec["block"] == "growth":
                ik = (sid, op, "g", rec["checkpoint"])
                if ik not in item_cache:
                    item_cache[ik] = {i.id: i for i in growth_items(scn, rec["checkpoint"])}
            else:
                ik = (sid, op, "s")
                if ik not in item_cache:
                    item_cache[ik] = {i.id: i for i in generate_items(scn)}
            item = item_cache[ik].get(rec["response"]["item_id"])
            if item is None:
                problems.append(f"{rec['response']['item_id']}: item not regenerated from the scenario")
                continue
            row = score_item(item, Response.model_validate(rec["response"]), scn)
            if rec.get("condition") in (None, "system"):
                rescored.setdefault((rec["block"], op, rec.get("budget")), []).append(row)
    pert = results.get("blocks", {}).get("perturbation", {}).get("operators", {})
    for op, agg in pert.items():
        rows = rescored.get(("perturbation", op, 128_000), [])
        if rows:
            if not _close(json.loads(json.dumps(aggregate(rows))), agg):
                problems.append(f"perturbation/{op}: recomputed aggregates differ from results.json")
        else:
            problems.append(f"perturbation/{op}: no responses recorded")
    return problems
