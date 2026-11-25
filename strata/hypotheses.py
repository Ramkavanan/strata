"""Pre-registered hypothesis report (Section 9.1), computed from evaluation runs of several systems.

Each hypothesis is reported as supported, disconfirmed or inconclusive, with the numbers behind the verdict. H6 to
H8 are exploratory and labelled so. A verdict is only as good as the systems and templates it rests on: every
section says how many of each it had.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Any, Optional

from .schema import STATE_FAMILIES
from .score import cost_usd, cpco
from .stats import SESOI, classify_effect, cluster_bootstrap, kendall_tau_b, permutation_interaction

UPPER_BOUNDS = {"oracle", "oracle_context"}
NO_MEMORY = {"no_context"}  # has nothing to degrade or update, so it is left out of H1 and H3
H5_BASELINES = ("no_context", "long_context", "vector")


def load_run(path: str) -> dict:
    with open(os.path.join(path, "results.json")) as f:
        results = json.load(f)
    rows = []
    rp = os.path.join(path, "rows.jsonl")
    if os.path.exists(rp):
        with open(rp) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return {"results": results, "rows": rows, "path": path}


def _pert_rows(run: dict) -> list[dict]:
    return [r for r in run["rows"] if r["block"] == "perturbation" and r.get("run", 0) == 0 and not r.get("condition")]


def _adapter(run: dict) -> str:
    return run["results"]["adapter"]["adapter"]


def _reader(run: dict) -> str:
    rd = run["results"].get("reader") or {}
    return rd.get("preset") or rd.get("reader") or "own"


def _system(run: dict) -> str:
    return run["results"].get("system") or f"{_adapter(run)}+{_reader(run)}"


def _row_cost(r: dict, prices: dict) -> float:
    return cost_usd(
        r.get("consumer_input_tokens", 0) + r.get("system_input_tokens", 0),
        r.get("answer_tokens", 0),
        prices,
        "reader-reference",
    )


def _ingest_cost(run: dict) -> float:
    cost = run["results"]["blocks"].get("perturbation", {}).get("cost", {})
    vals = [c["mean_ingest_cost_usd"] for c in cost.values() if c]
    return sum(vals) / len(vals) if vals else 0.0


def _paired_by_template(
    a_rows: list[dict], b_rows: list[dict], key=lambda r: (r["op"], r["item_id"])
) -> dict[str, list[float]]:
    b = {key(r): r for r in b_rows}
    out: dict[str, list[float]] = defaultdict(list)
    for r in a_rows:
        o = b.get(key(r))
        if o is not None:
            out[r["scenario_id"]].append(float(r["correct"]) - float(o["correct"]))
    return out


# --------------------------------------------------------------------------- H1
def h1(runs: list[dict]) -> dict:
    runs = [r for r in runs if _adapter(r) not in UPPER_BOUNDS | NO_MEMORY]
    degraded, deltas = {}, {}
    all_equivalent = True
    for run in runs:
        sens = run["results"]["blocks"].get("perturbation", {}).get("sensitivity", {})
        degraded[_system(run)] = sorted(op for op, s in sens.items() if s["classification"] == "degraded")
        all_equivalent &= all(s["classification"] == "equivalent" for s in sens.values())
        rows = _pert_rows(run)
        clean = {r["item_id"]: r for r in rows if r["op"] == "clean"}
        d: dict[str, dict[str, float]] = defaultdict(dict)
        for r in rows:
            if r["op"] != "clean" and r["affected"] and r["family"] in STATE_FAMILIES and r["item_id"] in clean:
                d[r["op"]][r["item_id"]] = float(r["correct"]) - float(clean[r["item_id"]]["correct"])
        deltas[_system(run)] = d
    inter = permutation_interaction(deltas) if len(runs) >= 2 else {"p": None}
    n = len(runs)
    k = sum(1 for v in degraded.values() if v)
    if n < 2 or inter["p"] is None:
        verdict = "not decidable"
    elif k * 2 >= n and inter["p"] < 0.05:
        verdict = "supported"
    elif all_equivalent or inter["p"] >= 0.05:
        verdict = "disconfirmed"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "systems": n,
        "systems_with_a_degrading_operator": k,
        "degraded_by_operator": degraded,
        "interaction": inter,
        "test": "state accuracy on affected items: at least one operator degraded for at least half of the systems, "
        "and an operator-by-system interaction (permutation test within item)",
    }


# --------------------------------------------------------------------------- H3
def h3(runs: list[dict], horizon: int = 5) -> dict:
    from .models import fit_hazard, median_latency

    runs = [r for r in runs if _adapter(r) not in UPPER_BOUNDS | NO_MEMORY]
    med, events_all = {}, []
    for run in runs:
        gp = run["results"]["blocks"].get("growth_pooled", {})
        ev = [e for op in gp.values() for e in op.get("events", [])]
        if ev:
            med[_system(run)] = median_latency(ev, horizon)
            events_all += [{**e, "system": _system(run), "template": str(e["item"]).split(".")[0]} for e in ev]
    sep = (max(med.values()) - min(med.values())) if len(med) >= 2 else None
    try:
        hazard = fit_hazard(events_all) if events_all else {"fitted": False, "reason": "no growth block"}
    except ImportError as e:
        hazard = {"fitted": False, "reason": str(e)}
    # accuracy under contradict_low on affected state items: best against worst system, paired by item
    acc, rows_by = {}, {}
    for run in runs:
        rs = [
            r
            for r in _pert_rows(run)
            if r["op"] == "contradict_low" and r["affected"] and r["family"] in STATE_FAMILIES
        ]
        if rs:
            acc[_system(run)] = sum(float(r["correct"]) for r in rs) / len(rs)
            rows_by[_system(run)] = rs
    contrast: dict[str, Any] = {}
    if len(acc) >= 2:
        best, worst = max(acc, key=acc.get), min(acc, key=acc.get)
        boot = cluster_bootstrap(_paired_by_template(rows_by[best], rows_by[worst]))
        contrast = {
            "best": best,
            "worst": worst,
            "difference": boot["mean"],
            "ci95": boot["ci95"],
            "classification": classify_effect(boot),
        }
    if sep is None or not contrast:
        verdict = "not decidable"
    elif sep >= 1 and contrast["classification"] == "improved":
        verdict = "supported"
    elif (sep == 0) and contrast["classification"] == "equivalent":
        verdict = "disconfirmed"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "median_update_latency": med,
        "latency_separation": sep,
        "hazard_model": hazard,
        "contradict_low_accuracy": acc,
        "contradict_low_contrast": contrast,
        "test": "median update latency of the most and least responsive systems at least one checkpoint apart, and "
        "accuracy under contradict_low differing across systems by at least the smallest effect of interest",
    }


# --------------------------------------------------------------------------- H4
def h4(runs: list[dict], replicates: int = 1000, seed: int = 0) -> dict:
    runs = [r for r in runs if _adapter(r) not in UPPER_BOUNDS]
    prices = runs[0]["results"].get("prices", {"models": {}}) if runs else {"models": {}}
    data = {}
    for run in runs:
        rows = _pert_rows(run)
        by_t: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            by_t[r["scenario_id"]].append((float(r["correct"]), _row_cost(r, prices)))
        if by_t:
            data[_system(run)] = (by_t, _ingest_cost(run))

    def point(templates: Optional[list[str]] = None) -> tuple[dict[str, float], dict[str, Optional[float]]]:
        acc, cp = {}, {}
        for s, (by_t, ingest) in data.items():
            ts = templates if templates is not None else list(by_t)
            pairs = [p for t in ts for p in by_t.get(t, [])]
            if not pairs:
                continue
            a = sum(p[0] for p in pairs) / len(pairs)
            q = sum(p[1] for p in pairs) / len(pairs)
            acc[s], cp[s] = a, cpco(q, ingest, a)
        return acc, cp

    acc, cp = point()
    names = sorted(s for s in acc if cp.get(s) is not None)
    if len(names) < 3:
        return {
            "verdict": "not decidable",
            "reason": "needs at least three systems with a defined cost per correct outcome",
            "accuracy": acc,
            "cpco_usd": cp,
        }
    tau = kendall_tau_b([acc[n] for n in names], [-cp[n] for n in names])
    templates = sorted({t for s in names for t in data[s][0]})
    rng = random.Random(seed)
    taus, diffs = [], defaultdict(list)
    for _ in range(replicates):
        pick = [templates[rng.randrange(len(templates))] for _ in templates]
        a, c = point(pick)
        ok = [n for n in names if c.get(n) is not None]
        if len(ok) >= 3:
            t = kendall_tau_b([a[n] for n in ok], [-c[n] for n in ok])
            if t is not None:
                taus.append(t)
        for i, x in enumerate(names):
            for y in names[i + 1 :]:
                if c.get(x) is not None and c.get(y) is not None:
                    diffs[(x, y)].append((a[x] - a[y], c[x] - c[y]))
    taus.sort()
    ci = [taus[int(0.025 * (len(taus) - 1))], taus[int(0.975 * (len(taus) - 1))]] if taus else None
    reversals = []
    for (x, y), ds in diffs.items():
        da, dc = sorted(d[0] for d in ds), sorted(d[1] for d in ds)
        lo_a, hi_a = da[int(0.025 * (len(da) - 1))], da[int(0.975 * (len(da) - 1))]
        lo_c, hi_c = dc[int(0.025 * (len(dc) - 1))], dc[int(0.975 * (len(dc) - 1))]
        # a reversal: one system is reliably more accurate and reliably more expensive per correct outcome
        if (lo_a > 0 and lo_c > 0) or (hi_a < 0 and hi_c < 0):
            reversals.append({"more_accurate_and_costlier": x if lo_a > 0 else y, "other": y if lo_a > 0 else x})
    if ci is None:
        verdict = "not decidable"
    elif ci[1] < 1 and reversals:
        verdict = "supported"
    elif ci[1] >= 1 and not reversals:
        verdict = "disconfirmed"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "systems": names,
        "accuracy": {n: acc[n] for n in names},
        "cpco_usd": {n: cp[n] for n in names},
        "kendall_tau_b": tau,
        "tau_ci95": ci,
        "significant_reversals": reversals,
        "note": "costs use the reference rates in the price table, not vendor prices",
        "test": "the 95% interval for Kendall tau-b between the accuracy ranking and the cost-per-correct-outcome ranking "
        "excludes 1, and at least one pair of systems reverses order with both differences outside their intervals",
    }


# --------------------------------------------------------------------------- H5
def h5(runs: list[dict]) -> dict:
    prices = runs[0]["results"].get("prices", {"models": {}}) if runs else {"models": {}}
    by = {(_adapter(r), _reader(r)): r for r in runs}

    def cp(run: dict) -> Optional[float]:
        rows = _pert_rows(run)
        if not rows:
            return None
        a = sum(float(r["correct"]) for r in rows) / len(rows)
        return cpco(sum(_row_cost(r, prices) for r in rows) / len(rows), _ingest_cost(run), a)

    out, any_holds, decidable = {}, False, False
    for (adapter, reader), run in sorted(by.items()):
        if adapter in UPPER_BOUNDS or adapter in H5_BASELINES:
            continue
        base = {b: by.get((b, reader)) for b in H5_BASELINES}
        if any(v is None for v in base.values()):
            out[_system(run)] = {
                "decidable": False,
                "reason": f"needs no_context, long_context and vector runs with the {reader} reader",
            }
            continue
        decidable = True
        boot = cluster_bootstrap(_paired_by_template(_pert_rows(run), _pert_rows(base["no_context"])))
        mine, long_, vec = cp(run), cp(base["long_context"]), cp(base["vector"])
        cheaper = mine is not None and all(x is None or mine < x for x in (long_, vec))
        holds = classify_effect(boot) == "improved" and cheaper
        any_holds |= holds
        out[_system(run)] = {
            "decidable": True,
            "holds": holds,
            "gain_over_no_context": boot["mean"],
            "gain_ci95": boot["ci95"],
            "gain_class": classify_effect(boot),
            "cpco_usd": mine,
            "cpco_long_context_usd": long_,
            "cpco_vector_usd": vec,
            "cheaper_than_both": cheaper,
        }
    verdict = (
        "not decidable"
        if not decidable
        else ("supported for at least one system" if any_holds else "disconfirmed for this track")
    )
    return {
        "verdict": verdict,
        "track": "question answering",
        "systems": out,
        "test": "outcome improved over no context, and cost per correct outcome below both the long-context and the "
        "vector-retrieval baselines, with the same reader",
    }


# --------------------------------------------------------------------------- exploratory
def h6(runs: list[dict]) -> dict:
    pts = {}
    for run in runs:
        b = run["results"]["blocks"].get("budget", {}).get("clean")
        if b and b.get("aubc") is not None and _adapter(run) not in UPPER_BOUNDS:
            top = b["curve"][max(b["curve"], key=int)]
            pts[_system(run)] = (b["aubc"], top, b.get("b90"))
    if len(pts) < 3:
        return {"exploratory": True, "decidable": False, "reason": "needs budget blocks from at least three systems"}
    names = sorted(pts)
    tau = kendall_tau_b([pts[n][0] for n in names], [pts[n][1] for n in names])
    return {
        "exploratory": True,
        "decidable": True,
        "aubc": {n: pts[n][0] for n in names},
        "accuracy_at_largest_budget": {n: pts[n][1] for n in names},
        "b90": {n: pts[n][2] for n in names},
        "kendall_tau_b": tau,
        "orderings_differ": tau is not None and tau < 1,
    }


def h7(runs: list[dict]) -> dict:
    pts = {}
    for run in runs:
        p = run["results"]["blocks"].get("perturbation")
        if not p or _adapter(run) in UPPER_BOUNDS:
            continue
        flag = p.get("staleness_flag_balanced_accuracy")
        rows = [r for r in _pert_rows(run) if r["op"] == "clean"]
        if flag is not None and rows:
            pts[_system(run)] = (flag, sum(float(r["correct"]) for r in rows) / len(rows))
    if len(pts) < 3:
        return {
            "exploratory": True,
            "decidable": False,
            "reason": "needs perturbation blocks from at least three systems",
        }
    names = sorted(pts)
    return {
        "exploratory": True,
        "decidable": True,
        "staleness_flag_balanced_accuracy": {n: pts[n][0] for n in names},
        "clean_accuracy": {n: pts[n][1] for n in names},
        "kendall_tau_b": kendall_tau_b([pts[n][0] for n in names], [pts[n][1] for n in names]),
    }


def hypothesis_report(runs: list[dict], ablation: Optional[dict] = None, fleet: Optional[dict] = None) -> dict:
    templates = {r["scenario_id"] for run in runs for r in run["rows"]}
    rep: dict[str, Any] = {
        "systems": sorted({_system(r) for r in runs}),
        "templates": len(templates),
        "smallest_effect_of_interest": SESOI,
        "H1": h1(runs),
        "H2": (
            {
                "verdict": "supported" if ablation["h2_supported"] else "not supported",
                "mechanisms": ablation["h2"],
                "note": ablation["note"],
            }
            if ablation
            else {"verdict": "not decidable", "reason": "run the ablation"}
        ),
        "H3": h3(runs),
        "H4": h4(runs),
        "H5": h5(runs),
        "H6": h6(runs),
        "H7": h7(runs),
        "H8": (
            {"exploratory": True, **fleet}
            if fleet
            else {"exploratory": True, "decidable": False, "reason": "run the fleet track"}
        ),
    }
    return rep


def render_hypotheses(rep: dict) -> str:
    L = [
        "# STRATA hypothesis report",
        "",
        f"{len(rep['systems'])} systems, {rep['templates']} templates, smallest effect of interest {rep['smallest_effect_of_interest']:.0%}.",
        "Systems: " + ", ".join(rep["systems"]) + ".",
        "",
    ]
    if rep["templates"] < 10:
        L += ["Fewer than 10 templates: treat every verdict below as a check of the machinery, not a finding.", ""]
    for h in ("H1", "H2", "H3", "H4", "H5"):
        v = rep[h]
        L += [f"## {h}: {v.get('verdict')}", ""]
        if v.get("test"):
            L += [f"Test: {v['test']}.", ""]
        if h == "H1" and v.get("degraded_by_operator"):
            L += ["| system | operators classified as degrading |", "|---|---|"]
            L += [f"| {s} | {', '.join(ops) or 'none'} |" for s, ops in v["degraded_by_operator"].items()]
            i = v["interaction"]
            L += ["", f"Operator-by-system interaction: p = {i.get('p')} over {i.get('items')} paired items.", ""]
        if h == "H2" and v.get("mechanisms"):
            L += ["| mechanism | improves robustness under | equivalent under |", "|---|---|---|"]
            L += [
                f"| {m} | {', '.join(x['improves_under']) or 'none'} | {', '.join(x['equivalent_under']) or 'none'} |"
                for m, x in v["mechanisms"].items()
            ]
            L.append("")
        if h == "H3":
            L += [
                f"Median update latency (checkpoints): {v.get('median_update_latency')}; separation {v.get('latency_separation')}.",
                f"Accuracy under contradict_low on affected state items: {_round(v.get('contradict_low_accuracy'))}.",
                f"Best against worst: {_round(v.get('contradict_low_contrast'))}.",
                f"Hazard model: {v['hazard_model'].get('reason') or v['hazard_model'].get('system_terms')}.",
                "",
            ]
        if h == "H4" and v.get("systems"):
            L += ["| system | accuracy | cost per correct outcome (USD, reference rates) |", "|---|---|---|"]
            L += [
                f"| {s} | {v['accuracy'][s]:.3f} | {v['cpco_usd'][s]:.6f} |"
                for s in sorted(v["systems"], key=lambda s: -v["accuracy"][s])
            ]
            L += [
                "",
                f"Kendall tau-b between the two rankings: {v['kendall_tau_b']:.2f}, 95% interval {_round(v['tau_ci95'])}. "
                f"Significant reversals: {len(v['significant_reversals'])}.",
                "",
            ]
        if h == "H5":
            rowsx = [(s, x) for s, x in v["systems"].items() if x.get("decidable")]
            if rowsx:
                L += [
                    "| system | gain over no context [95% CI] | CPCO | long context | vector | holds |",
                    "|---|---|---|---|---|---|",
                ]
                for s, x in rowsx:
                    L.append(
                        f"| {s} | {x['gain_over_no_context']:+.3f} [{x['gain_ci95'][0]:+.3f}, {x['gain_ci95'][1]:+.3f}] | {_f6(x['cpco_usd'])} | "
                        f"{_f6(x['cpco_long_context_usd'])} | {_f6(x['cpco_vector_usd'])} | {'yes' if x['holds'] else 'no'} |"
                    )
                L.append("")
    L += ["## Exploratory (H6 to H8, unadjusted)", ""]
    for h in ("H6", "H7", "H8"):
        L.append(f"- {h}: {_round({k: v for k, v in rep[h].items() if k != 'exploratory'})}")
    return "\n".join(L) + "\n"


def _f6(x) -> str:
    return "-" if x is None else f"{x:.6f}"


def _round(x):
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x
