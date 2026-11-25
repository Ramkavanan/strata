"""H2: ablate each mechanism of the modular reference system and classify its effect on robustness under every operator.

Robustness is the sensitivity (perturbed minus clean, on the same affected state items); the effect of a mechanism is the
difference between the full system's sensitivity and the ablated system's sensitivity (Section 9.1)."""

from __future__ import annotations

from collections import defaultdict

from . import operators
from .adapters.reference import ReferenceSystem
from .runner import evaluate_render
from .scenario import Scenario
from .schema import STATE_FAMILIES
from .stats import classify_effect, cluster_bootstrap, holm

MECHANISMS = ("temporal", "authority", "lineage", "validation")


def run_ablation(scenarios: list[Scenario], ops: list[str] = None, mechanisms=MECHANISMS) -> dict:
    ops = ops or operators.OPERATORS
    variants = {"full": ReferenceSystem()}
    for m in mechanisms:
        variants[f"no_{m}"] = ReferenceSystem(**{m: False})
    rows: dict[tuple, dict[str, dict]] = {}
    for vname, ad in variants.items():
        for base in scenarios:
            for op in ops:
                ev = evaluate_render(ad, operators.apply(base, op), 128_000)
                for r in ev["rows"]:
                    rows[(vname, op, r["item_id"])] = r
    out: dict = {"matrix": {}, "h2": {}}
    pvals = {}
    for m in mechanisms:
        for op in ops:
            if op == "clean":
                continue
            by_t: dict[str, list[float]] = defaultdict(list)
            for (v, o, iid), r in rows.items():
                if v != "full" or o != op or not r["affected"] or r["family"] not in STATE_FAMILIES:
                    continue
                ab = rows.get((f"no_{m}", op, iid))
                fc, ac = rows.get(("full", "clean", iid)), rows.get((f"no_{m}", "clean", iid))
                if ab is None or fc is None or ac is None:
                    continue
                # difference in sensitivity: (full_pi - full_clean) - (ablated_pi - ablated_clean)
                by_t[r["scenario_id"]].append(
                    (float(r["correct"]) - float(fc["correct"])) - (float(ab["correct"]) - float(ac["correct"]))
                )
            boot = cluster_bootstrap(by_t)
            out["matrix"][f"{m}|{op}"] = {**boot, "classification": classify_effect(boot)}
            if boot["p"] is not None:
                pvals[f"{m}|{op}"] = boot["p"]
    for k, p in holm(pvals).items():
        out["matrix"][k]["p_holm"] = p
    for m in mechanisms:
        cls = {
            op: out["matrix"][f"{m}|{op}"]["classification"]
            for op in ops
            if op != "clean" and f"{m}|{op}" in out["matrix"]
        }
        out["h2"][m] = {
            "improves_under": sorted(o for o, c in cls.items() if c == "improved"),
            "equivalent_under": sorted(o for o, c in cls.items() if c == "equivalent"),
        }
    out["h2_supported"] = any(v["improves_under"] and v["equivalent_under"] for v in out["h2"].values())
    out["note"] = (
        "H2 holds if at least one mechanism improves robustness under one operator and is equivalent under another; "
        "it is disconfirmed if every mechanism's effect is uniform across operators."
    )
    return out
