"""Golden data: deterministic regeneration of scenarios, renders and items with content hashes checked in CI.

The golden manifest pins, per template and operator, the world, render and item hashes. Any change to semantics,
renderer, operators or question generation shows up here and must be intentional (and, for the paper's own examples,
accompanied by an erratum).
"""

from __future__ import annotations

import json
import os

from . import operators
from .questions import generate_items
from .scenario import sha
from .templates import build_scenario, load_template, template_paths

GOLDEN_SEED = 0


def items_hash(items) -> str:
    return sha([i.model_dump(mode="json") for i in items])


def golden_manifest(templates_dir: str, seed: int = GOLDEN_SEED) -> dict:
    out: dict = {"seed": seed, "templates": {}}
    for path in template_paths(templates_dir):
        tpl = load_template(path)
        base = build_scenario(tpl, seed)
        ops = {}
        for name in operators.OPERATORS:
            r = operators.apply(base, name)
            items = generate_items(r)
            ops[name] = {
                "world": r.world_hash(),
                "render": r.render_hash(),
                "items": items_hash(items),
                "n_items": len(items),
                "artifacts": len(r.artifacts),
            }
        out["templates"][tpl["id"]] = {"scenario_id": base.id, "operators": ops}
    return out


def write_golden(templates_dir: str, out_dir: str, seed: int = GOLDEN_SEED) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    man = golden_manifest(templates_dir, seed)
    for path in template_paths(templates_dir):
        tpl = load_template(path)
        base = build_scenario(tpl, seed)
        d = os.path.join(out_dir, tpl["id"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "scenario.clean.json"), "w") as f:
            json.dump(base.to_dict(), f, indent=1, sort_keys=True, default=str)
            f.write("\n")
        with open(os.path.join(d, "items.clean.jsonl"), "w") as f:
            for it in generate_items(base):
                f.write(it.model_dump_json() + "\n")
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
        f.write("\n")
    return man


def check_golden(templates_dir: str, golden_dir: str) -> list[str]:
    path = os.path.join(golden_dir, "manifest.json")
    if not os.path.exists(path):
        return [f"{path} missing; run `strata golden --write`"]
    with open(path) as f:
        want = json.load(f)
    got = golden_manifest(templates_dir, want["seed"])
    diffs = []
    for tid, t in want["templates"].items():
        if tid not in got["templates"]:
            diffs.append(f"{tid}: template missing")
            continue
        for op, h in t["operators"].items():
            g = got["templates"][tid]["operators"].get(op)
            for k in ("world", "render", "items"):
                if g is None or g[k] != h[k]:
                    diffs.append(f"{tid}/{op}: {k} hash changed")
    for tid in got["templates"]:
        if tid not in want["templates"]:
            diffs.append(f"{tid}: not in golden manifest")
    return diffs
