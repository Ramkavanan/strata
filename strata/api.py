"""Python API: load a golden dataset and evaluate a system in a few lines.

from strata import evaluate, load_dataset
results = evaluate("bm25", reader="reference")            # development split of the default dataset
results = evaluate(MySystem(), operators=["clean", "contradict_low"], limit=3)
for g in load_dataset().goldens("clean")[:3]:
    print(g.input, "->", g.expected_output)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from . import __version__
from .adapters.base import ContextSystem, get_adapter
from .datasets import DEFAULT_DATASET, load_dataset
from .results import summarize
from .runner import BLOCKS, make_reader, run_attribution, run_block
from .scenario import Scenario

ROOT = Path(__file__).resolve().parent.parent


def evaluate(
    system: Union[str, ContextSystem],
    dataset: str = DEFAULT_DATASET,
    split: str = "dev",
    block: str = "perturbation",
    reader: Optional[str] = None,
    operators: Optional[list[str]] = None,
    budgets: Optional[list[int]] = None,
    repeats: int = 1,
    limit: Optional[int] = None,
    scenarios: Optional[list[Scenario]] = None,
    config: Optional[dict] = None,
    price_table: Optional[str] = None,
) -> dict:
    """Evaluate one context system with one reader, in memory. Returns the structure written to results.json,
    plus the scored rows under "rows". `limit` keeps the first N scenarios for a quick look."""
    adapter = get_adapter(system, **(config or {})) if isinstance(system, str) else system
    scns = scenarios if scenarios is not None else load_dataset(dataset, split).scenarios()
    if limit:
        scns = scns[:limit]
    rd = make_reader(adapter, reader)
    prices = json.loads(Path(price_table or ROOT / "config" / "prices" / "2026-11.json").read_text())
    blocks = BLOCKS if block == "all" else tuple(block.split(","))
    out: dict = {}
    rows = []
    for b in blocks:
        if b == "attribution":
            out[b] = run_attribution(
                adapter, scns, rd, operators, system_evals=out.get("perturbation", {}).get("evaluations")
            )
        else:
            out[b] = run_block(
                adapter,
                scns,
                b,
                repeats=repeats if b == "perturbation" else 1,
                budgets=budgets,
                reader=rd,
                operator_names=operators,
            )
        for ev in out[b]["evaluations"]:
            op = ev["render_id"].split("(")[0].split("@")[0]
            rows += [{**r, "block": b, "op": op, "condition": ev.get("condition")} for r in ev["rows"]]
    desc = getattr(rd, "describe", lambda: {})()
    name = adapter.describe().get("adapter", getattr(adapter, "name", "system"))
    res = summarize(
        adapter.describe(),
        out,
        {
            "harness_version": __version__,
            "git_sha": "",
            "scenario_run": None,
            "dataset": dataset if scenarios is None else None,
            "reader": desc,
            "system": f"{name}+{desc.get('preset') or desc.get('reader')}",
            "price_table": "2026-11.json",
            "prices": prices,
            "repeats": repeats,
            "blocks_run": list(blocks),
            "n_templates": len({s.template_id for s in scns}),
        },
        prices,
    )
    res["rows"] = [{**r, "system": res["system"]} for r in rows]
    return res
