"""Assemble results.json from block outputs: pooled aggregates, sensitivity with bootstrap, failure attribution,
cost per correct outcome, budget curves, growth."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from .runner import budget_curve
from .schema import STATE_FAMILIES
from .score import aggregate, aubc, b90, balanced_accuracy, cost_usd, cpco, sensitivity
from .stats import classify_effect, cluster_bootstrap, holm
from .tokenizer import TOKENIZER_NAME

MIN_TEMPLATES = 10  # below this the bootstrap over templates is too coarse to trust an equivalence or degradation claim
CPCO_HORIZONS = (100, 1_000, 10_000)


def _op_name(render_id: str) -> str:
    return render_id.split("(")[0].split("@")[0]


def _paired(rows: list[dict], clean: list[dict], state_only: bool) -> dict[str, list[float]]:
    cl = {r["item_id"]: r for r in clean}
    by_t: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if not r["affected"] or r["item_id"] not in cl or (state_only and r["family"] not in STATE_FAMILIES):
            continue
        by_t[r["scenario_id"]].append(float(r["correct"]) - float(cl[r["item_id"]]["correct"]))
    return by_t


def _costs(evals: list[dict], prices: dict, ingest_key: Optional[str]) -> dict:
    rows = [r for ev in evals for r in ev["rows"]]
    if not rows:
        return {}
    q = [
        cost_usd(r["consumer_input_tokens"] + r["system_input_tokens"], r["answer_tokens"], prices, "reader-reference")
        for r in rows
    ]
    ingest = [
        cost_usd(ev["ingest_usage"]["input_tokens"], 0, prices, ingest_key) if ingest_key else 0.0 for ev in evals
    ]
    acc = sum(float(r["correct"]) for r in rows) / len(rows)
    mq, mi = sum(q) / len(q), sum(ingest) / len(ingest)
    return {
        "mean_query_cost_usd": mq,
        "mean_ingest_cost_usd": mi,
        "mean_context_tokens": sum(r["context_tokens"] for r in rows) / len(rows),
        "mean_system_input_tokens": sum(r["system_input_tokens"] for r in rows) / len(rows),
        "mean_tool_calls": sum(r["tool_calls"] for r in rows) / len(rows),
        "accuracy": acc,
        "cpco_usd": cpco(mq, mi, acc),
        "cpco_curve_usd": {str(h): cpco(mq, mi, acc, h) for h in CPCO_HORIZONS},
    }


def summarize(adapter_desc: dict, blocks: dict[str, dict], extra: dict, prices: Optional[dict] = None) -> dict:
    res: dict[str, Any] = {"adapter": adapter_desc, "tokenizer": TOKENIZER_NAME, **extra, "blocks": {}}
    prices = prices or {"models": {}}
    pert = blocks.get("perturbation")
    if pert:
        by_op: dict[str, list[dict]] = defaultdict(list)
        ev_by_op: dict[str, list[dict]] = defaultdict(list)
        for ev in pert["evaluations"]:
            by_op[_op_name(ev["render_id"])].extend(ev["rows"])
            ev_by_op[_op_name(ev["render_id"])].append(ev)
        out: dict[str, Any] = {"operators": {}, "sensitivity": {}, "cost": {}}
        clean = by_op.get("clean", [])
        pvals: dict[str, float] = {}
        for op, rows in by_op.items():
            out["operators"][op] = aggregate(rows)
            out["cost"][op] = _costs(ev_by_op[op], prices, adapter_desc.get("ingest_price_key"))
            if op != "clean" and clean:
                # primary: state accuracy on affected items (the H1 test); secondary: every affected item
                boot = cluster_bootstrap(_paired(rows, clean, state_only=True))
                allf = cluster_bootstrap(_paired(rows, clean, state_only=False))
                un = sensitivity(rows, clean, "correct", only_affected=False)
                out["sensitivity"][op] = {
                    **boot,
                    "n_pairs": sum(len(v) for v in _paired(rows, clean, True).values()),
                    "classification": classify_effect(boot),
                    "underpowered": boot["n_clusters"] < MIN_TEMPLATES,
                    "all_families": {**allf, "classification": classify_effect(allf)},
                    "delta_all_items": un["delta"],
                }
                if boot["p"] is not None:
                    pvals[op] = boot["p"]
        for op, a in holm(pvals).items():
            out["sensitivity"][op]["p_holm"] = a
        # the staleness flag is a paired test: 1 on the stale render, 0 on the clean render of the same scenario
        t13 = [r for op in ("clean", "stale") for r in by_op.get(op, []) if r["family"] == "T13"]
        out["staleness_flag_balanced_accuracy"] = balanced_accuracy(
            tp=sum(1 for r in t13 if r["flag_gold"] and r["flag_pred"]),
            fn=sum(1 for r in t13 if r["flag_gold"] and not r["flag_pred"]),
            tn=sum(1 for r in t13 if not r["flag_gold"] and not r["flag_pred"]),
            fp=sum(1 for r in t13 if not r["flag_gold"] and r["flag_pred"]),
        )
        out["reliability"] = [
            ev["reliability_share_solved_all_runs"]
            for ev in pert["evaluations"]
            if ev.get("reliability_share_solved_all_runs") is not None
        ]
        res["blocks"]["perturbation"] = out
    att = blocks.get("attribution")
    if att and att.get("evaluations"):
        cond: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
        for ev in att["evaluations"]:
            for r in ev["rows"]:
                cond[_op_name(ev["render_id"])][ev["condition"]][r["item_id"]] = r
        out = {}
        for op, c in cond.items():
            ids = sorted(set(c["none"]) & set(c["system"]) & set(c["oracle"]))
            if not ids:
                continue
            y = {k: sum(float(c[k][i]["correct"]) for i in ids) / len(ids) for k in ("none", "system", "oracle")}
            v_t: dict[str, list[float]] = defaultdict(list)
            l_t: dict[str, list[float]] = defaultdict(list)
            for i in ids:
                t = c["system"][i]["scenario_id"]
                v_t[t].append(float(c["system"][i]["correct"]) - float(c["none"][i]["correct"]))
                l_t[t].append(float(c["oracle"][i]["correct"]) - float(c["system"][i]["correct"]))
            vb, lb = cluster_bootstrap(v_t), cluster_bootstrap(l_t)
            out[op] = {
                "n": len(ids),
                "y_none": y["none"],
                "y_system": y["system"],
                "y_oracle": y["oracle"],
                "context_value": y["system"] - y["none"],
                "context_loss": y["oracle"] - y["system"],
                "consumer_gap": 1 - y["oracle"],
                "context_value_ci95": vb["ci95"],
                "context_loss_ci95": lb["ci95"],
                "context_value_class": classify_effect(vb),
            }
        res["blocks"]["attribution"] = out
    elif att:
        res["blocks"]["attribution"] = {"skipped": att.get("skipped")}
    bud = blocks.get("budget")
    if bud:
        out = {}
        groups: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for ev in bud["evaluations"]:
            groups[_op_name(ev["render_id"])][ev["budget"]].extend(ev["rows"])
        for op, byb in groups.items():
            curve = budget_curve(byb)
            out[op] = {
                "curve": {str(k): v for k, v in curve.items()},
                "aubc": aubc(curve),
                "b90": b90(curve),
                "truncated_share": {str(b): sum(r["truncated"] for r in rs) / len(rs) for b, rs in byb.items()},
            }
        res["blocks"]["budget"] = out
    gro = blocks.get("growth")
    if gro:
        res["blocks"]["growth"] = {
            ev["render_id"].split("@")[0] + "|" + ev["scenario_id"]: {
                "growth": ev["growth"],
                "checkpoints": ev["checkpoints"],
            }
            for ev in gro["evaluations"]
        }
        pooled: dict[str, dict[str, list]] = defaultdict(lambda: {"lat": [], "stale": [], "events": []})
        for ev in gro["evaluations"]:
            op = _op_name(ev["render_id"])
            g = ev["growth"]
            if g["update_latency_mean"] is not None:
                pooled[op]["lat"].append(g["update_latency_mean"])
            if g["stale_answer_rate"] is not None:
                pooled[op]["stale"].append(g["stale_answer_rate"])
            pooled[op]["events"].extend(g.get("events", []))
        res["blocks"]["growth_pooled"] = {
            op: {
                "update_latency_mean": (sum(v["lat"]) / len(v["lat"])) if v["lat"] else None,
                "stale_answer_rate": (sum(v["stale"]) / len(v["stale"])) if v["stale"] else None,
                "events": v["events"],
            }
            for op, v in pooled.items()
        }
    return res
