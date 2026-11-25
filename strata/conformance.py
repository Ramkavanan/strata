"""`strata adapter-check NAME`: the conformance suite any vendor can run before submitting results."""

from __future__ import annotations

from pathlib import Path

from .adapters.base import ContextSystem, get_adapter
from .questions import generate_items
from .schema import ContextItem, Usage
from .templates import build_scenario, load_template
from .tokenizer import enforce_budget

ROOT = Path(__file__).resolve().parent.parent


def run_conformance(name: str, config: dict) -> int:
    checks: list[tuple[str, bool, str]] = []

    def ok(label, cond, detail=""):
        checks.append((label, bool(cond), detail))

    scn = build_scenario(load_template(str(ROOT / "templates" / "appendix_a_ex1.yaml")), 0)
    items = generate_items(scn)
    try:
        ad = get_adapter(name, **config)
    except Exception as e:
        print(f"FAIL cannot construct adapter: {e}")
        return 1
    proto = isinstance(ad, ContextSystem)
    ok("implements the ContextSystem protocol (reset, ingest, context, describe, usage)", proto)
    if not proto:
        missing = [m for m in ("reset", "ingest", "context", "describe", "usage") if not callable(getattr(ad, m, None))]
        print(f"FAIL implements the ContextSystem protocol: missing {missing}")
        return 1
    try:
        ad.reset()
        if getattr(ad, "needs_ground_truth", False):
            ad.attach(scn, items)
        ad.ingest(scn.public_artifacts(), checkpoint=0)
    except Exception as e:
        print(f"FAIL ingest raised {type(e).__name__}: {e}")
        return 1
    desc = ad.describe()
    ok("describe() returns a dict naming the adapter", isinstance(desc, dict) and "adapter" in desc)
    ok("describe() declares event_time_support and provenance", "event_time_support" in desc and "provenance" in desc)
    ok("usage() returns Usage", isinstance(ad.usage(), Usage))
    span_ids = {s.id for a in scn.artifacts for s in a.spans}
    t1 = [i for i in items if i.family == "T1"][:3]
    for it in t1:
        task = it.task(2_000)
        try:
            if getattr(ad, "own_reader", False):
                resp = ad.answer(task)
                ok(f"{it.id}: answer() returns a Response", resp.item_id == it.id)
                continue
            ctx = ad.context(task, 2_000)
        except Exception as e:
            ok(f"{it.id}: context() runs without raising", False, f"{type(e).__name__}: {e}")
            continue
        ok(f"{it.id}: context() returns ContextItem list", all(isinstance(c, ContextItem) for c in ctx))
        kept, info = enforce_budget(ctx, 2_000)
        ok(f"{it.id}: harness can enforce the 2k budget", info["tokens"] <= 2_000)
        if desc.get("provenance") == "native":
            ids = [s for c in ctx for s in c.source_ids]
            ok(f"{it.id}: native provenance ids are known span ids", all(s in span_ids for s in ids), f"{len(ids)} ids")
    width = max(len(c[0]) for c in checks)
    for label, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {label:<{width}}  {detail}")
    bad = sum(1 for c in checks if not c[1])
    print(f"\n{len(checks) - bad}/{len(checks)} checks passed for adapter {name!r}")
    return 1 if bad else 0
