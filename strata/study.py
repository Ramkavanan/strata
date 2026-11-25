"""A study: every block for every (system, reader) pair on a dataset, the ablation, the tracks, the power analysis
and the hypothesis report, written to one directory. This is the procedure the paper pre-registers, end to end."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from . import worldgen
from .ablation import run_ablation
from .adapters.base import get_adapter
from .datasets import SPEC, load_dataset
from .hypotheses import hypothesis_report, load_run, render_hypotheses
from .power import power_table, required_templates
from .reader import ReferenceReader
from .runner import evaluate_render
from .runs import dump_json, new_run
from .schema import STATE_FAMILIES
from .stats import icc_anova
from .templates import build_scenario

ROOT = Path(__file__).resolve().parent.parent
PILOT_INDICES = range(100, 105)  # five B2B templates, never part of a released split


def pilot_scenarios() -> list:
    return [
        build_scenario(
            worldgen.generate_template("b2b_product_team", i, SPEC["public_seed"], SPEC["render"], SPEC["panel_size"]),
            0,
        )
        for i in PILOT_INDICES
    ]


def pilot_power(system: str = "vector", reader: str = "reference", replicates: int = 300) -> dict:
    """Section 9.2: run the pilot, measure what the simulation assumes, then size the study by simulation."""
    from . import operators

    scns = pilot_scenarios()
    ad, rd = get_adapter(system), ReferenceReader(reader)
    measured = {}
    for op in ("contradict_low", "contradict_cross", "stale", "incomplete", "late", "noisy"):
        clean_rows, pert_rows = [], []
        for s in scns:
            clean_rows += evaluate_render(ad, operators.apply(s, "clean"), reader=rd)["rows"]
            pert_rows += evaluate_render(ad, operators.apply(s, op), reader=rd)["rows"]
        cl = {r["item_id"]: r for r in clean_rows}
        pairs = [
            (cl[r["item_id"]], r)
            for r in pert_rows
            if r["affected"] and r["family"] in STATE_FAMILIES and r["item_id"] in cl
        ]
        if not pairs:
            measured[op] = {"pairs": 0}
            continue
        by_t: dict[str, list[float]] = {}
        for c, _ in pairs:
            by_t.setdefault(c["scenario_id"], []).append(float(c["correct"]))
        pc = sum(float(c["correct"]) for c, _ in pairs) / len(pairs)
        pp = sum(float(p["correct"]) for _, p in pairs) / len(pairs)
        same = sum(c["correct"] == p["correct"] for c, p in pairs) / len(pairs)
        measured[op] = {
            "pairs": len(pairs),
            "pairs_per_template": len(pairs) / len(scns),
            "clean_accuracy": pc,
            "perturbed_accuracy": pp,
            "observed_delta": pp - pc,
            "concordance": same,
            "icc_estimate": icc_anova(by_t),
        }
    ref = measured.get("contradict_low", {})
    items = max(1, round(ref.get("pairs_per_template", 3)))
    p_clean = min(0.95, max(0.05, ref.get("clean_accuracy", 0.8)))
    table = power_table(items, p_clean, pairing=0.5, replicates=replicates)
    need = {
        f"icc={icc} effect={eff}": {
            "rule_as_written": required_templates(table, icc, eff, "power_beyond_sesoi"),
            "interval_excludes_zero": required_templates(table, icc, eff, "power_excludes_zero"),
        }
        for icc in (0.05, 0.15, 0.30)
        for eff in (0.05, 0.075, 0.10, 0.15)
    }
    L = [
        f"Pilot: {len(scns)} B2B templates excluded from every released split; system {system}, reader {reader}.",
        "",
        "| operator | affected state pairs | per template | clean acc | perturbed acc | delta | ICC estimate |",
        "|---|---|---|---|---|---|---|",
    ]
    for op, m in measured.items():
        if m["pairs"]:
            L.append(
                f"| {op} | {m['pairs']} | {m['pairs_per_template']:.1f} | {m['clean_accuracy']:.2f} | {m['perturbed_accuracy']:.2f} | "
                f"{m['observed_delta']:+.2f} | {m['icc_estimate'] if m['icc_estimate'] is None else round(m['icc_estimate'], 2)} |"
            )
        else:
            L.append(f"| {op} | 0 | 0 | - | - | - | - |")
    L += [
        "",
        f"Simulation inputs: {items} affected state items per template, clean accuracy {p_clean:.2f}, pairing 0.5.",
        "",
        "Power to classify a degradation as degraded (95% interval wholly beyond the 5-point bound), ICC 0.30:",
        "",
        "| true effect | " + " | ".join(str(n) for n in (10, 20, 30, 45, 60, 90, 120)) + " templates |",
        "|---|" + "---|" * 7,
    ]
    for eff in (0.05, 0.075, 0.10, 0.15):
        cells = [
            next(r for r in table if r["icc"] == 0.30 and r["true_effect"] == eff and r["templates"] == n)[
                "power_beyond_sesoi"
            ]
            for n in (10, 20, 30, 45, 60, 90, 120)
        ]
        L.append(f"| {eff:.3f} | " + " | ".join(f"{c:.2f}" for c in cells) + " |")
    L += [
        "",
        "Power to detect a degradation at all (95% interval wholly below zero), ICC 0.30:",
        "",
        "| true effect | " + " | ".join(str(n) for n in (10, 20, 30, 45, 60, 90, 120)) + " templates |",
        "|---|" + "---|" * 7,
    ]
    for eff in (0.05, 0.075, 0.10, 0.15):
        cells = [
            next(r for r in table if r["icc"] == 0.30 and r["true_effect"] == eff and r["templates"] == n)[
                "power_excludes_zero"
            ]
            for n in (10, 20, 30, 45, 60, 90, 120)
        ]
        L.append(f"| {eff:.3f} | " + " | ".join(f"{c:.2f}" for c in cells) + " |")
    L += [
        "",
        "Templates needed for 80% power at ICC 0.30: "
        + "; ".join(
            f"true effect {eff}: {need[f'icc=0.3 effect={eff}']['rule_as_written'] or 'more than 120'} (rule as written), "
            f"{need[f'icc=0.3 effect={eff}']['interval_excludes_zero'] or 'more than 120'} (interval excludes zero)"
            for eff in (0.05, 0.10, 0.15)
        )
        + ".",
        "",
        "A true effect equal to the 5-point bound cannot be classified as degraded with useful power at any sample size, "
        "because the rule asks the whole interval to lie beyond the bound.",
    ]
    return {
        "pilot_measured": measured,
        "inputs": {"items_per_template": items, "p_clean": p_clean, "pairing": 0.5},
        "table": table,
        "required_templates": need,
        "text": "\n".join(L),
    }


def run_study(
    root: str,
    systems: list[str],
    readers: list[str],
    dataset: str = "strata-core-v0",
    budgets: Optional[list[int]] = None,
    limit: Optional[int] = None,
    repeats: int = 1,
) -> str:
    from . import tracks as T
    from .cli import main as cli_main
    from .cli import run_evaluation
    from .runs import resolve_run

    t0 = time.time()
    cli_main(["--root", root, "generate", "--dataset", dataset])
    scn_run = resolve_run(root, "scenarios")
    if limit:  # keep only the first N scenarios of the run
        for sid in sorted(os.listdir(os.path.join(scn_run, "renders")))[limit:]:
            import shutil

            shutil.rmtree(os.path.join(scn_run, "renders", sid))
    scns = load_dataset(dataset).scenarios()[: limit or None]
    prices = json.loads((ROOT / "config" / "prices" / "2026-11.json").read_text())
    run_dirs, timings = [], {}
    for sname in systems:
        for rd in readers:
            t1 = time.time()
            d, _ = run_evaluation(root, scn_run, sname, {}, "all", repeats, budgets, rd)
            run_dirs.append(d)
            timings[f"{sname}+{rd}"] = round(time.time() - t1, 1)
    runs = [load_run(d) for d in run_dirs]
    ablation = run_ablation(scns)
    fleet = T.fleet_comparison(lambda: get_adapter("bm25"), scns, "reference", prices=prices)
    fleet_summary = {
        "decidable": True,
        "system": "bm25+reference",
        **{
            k: {
                m: fleet[k][m]
                for m in (
                    "cross_agent_consistency",
                    "all_agents_correct",
                    "propagation_lag_mean",
                    "changes_not_reaching_all_agents",
                    "fleet_cost_usd",
                )
            }
            for k in ("shared", "independent_same_cadence", "independent_staggered")
        },
    }
    rep = hypothesis_report(runs, ablation, fleet_summary)
    tracks = {}
    for sname, rd, agent in (
        ("reference", "reference", T.CarefulAgent()),
        ("bm25", "recency", T.CarefulAgent()),
        ("bm25", "recency", T.NaiveAgent()),
    ):
        if sname in systems:
            tracks[f"agents/{agent.name}/{sname}+{rd}"] = T.run_agent_track(
                get_adapter(sname), scns, agent, ReferenceReader(rd)
            )["summary"]
    for sname, rd in (("reference", "reference"), ("bm25", "recency"), ("bm25", "reference")):
        if sname in systems:
            tracks[f"business/{sname}+{rd}"] = T.run_business_track(get_adapter(sname), scns, ReferenceReader(rd))[
                "summary"
            ]
    try:
        from .models import fit_glmm

        rows = [
            r
            for run in runs
            for r in run["rows"]
            if r["block"] == "perturbation"
            and r.get("run", 0) == 0
            and run["results"]["adapter"]["adapter"] not in ("oracle", "oracle_context")
        ]
        glmm = fit_glmm(rows)
    except Exception as e:  # the mixed model is an optional extra and can fail to converge
        glmm = {"fitted": False, "reason": f"{type(e).__name__}: {e}"}
    out_dir, _ = new_run(root, "studies", {"systems": systems, "readers": readers, "dataset": dataset, "limit": limit})
    dump_json(
        os.path.join(out_dir, "study.json"),
        {
            "hypotheses": rep,
            "ablation": ablation,
            "fleet": fleet,
            "tracks": tracks,
            "glmm": glmm,
            "evaluation_runs": [os.path.basename(d) for d in run_dirs],
            "timings_seconds": timings,
            "total_seconds": round(time.time() - t0, 1),
        },
    )
    L = [render_hypotheses(rep)]
    L += ["## Mixed model (Section 9.2)", ""]
    if glmm.get("fitted"):
        L += [
            f"{glmm['n_rows']} rows, {len(glmm['systems'])} systems, {len(glmm['operators'])} operators. Template sd {glmm['sd_template']:.2f}, "
            f"item sd {glmm['sd_item']:.2f}, intra-template correlation {glmm['icc_template']:.3f}.",
            f"Joint test of the {glmm['interaction_df']} system-by-operator terms: chi-square {glmm['interaction_wald_chi2']:.1f}, "
            f"p = {glmm['interaction_p']:.3g} ({glmm['method']}).",
            "",
        ]
    else:
        L += [f"Not fitted: {glmm.get('reason')}.", ""]
    L += ["## Other consumer tracks (scripted consumers)", ""]
    for name, summ in tracks.items():
        if name.startswith("agents"):
            L += [
                f"### {name}",
                "",
                "| operator | execution success | final value correct | policy compliance | recovery | wrongly removed |",
                "|---|---|---|---|---|---|",
            ]
            L += [
                f"| {op} | {_f(v['execution_success'])} | {_f(v['final_value_correct'])} | {_f(v['policy_compliance'])} | {_f(v['recovery_rate'])} | {_f(v['wrongly_removed'])} |"
                for op, v in summ.items()
            ]
        else:
            L += [
                f"### {name}",
                "",
                "| operator | brief success | claim accuracy | superseded claims | unsupported claims | omissions |",
                "|---|---|---|---|---|---|",
            ]
            L += [
                f"| {op} | {_f(v['brief_success'])} | {_f(v['claim_accuracy'])} | {_f(v['superseded_claim_rate'])} | {_f(v['unsupported_claim_rate'])} | {_f(v['omission_rate'])} |"
                for op, v in summ.items()
            ]
        L.append("")
    L += [
        "### fleet (bm25 with the reference reader)",
        "",
        "| configuration | consistency | all agents correct | propagation lag | changes not reaching all | cost USD |",
        "|---|---|---|---|---|---|",
    ]
    for k in ("shared", "independent_same_cadence", "independent_staggered"):
        v = fleet[k]
        L.append(
            f"| {k} | {_f(v['cross_agent_consistency'])} | {_f(v['all_agents_correct'])} | {_f(v['propagation_lag_mean'])} | {v['changes_not_reaching_all_agents']} | {_f(v['fleet_cost_usd'])} |"
        )
    L += [
        "",
        f"Total time {round(time.time() - t0, 1)} s. Evaluation runs: {', '.join(os.path.basename(d) for d in run_dirs)}.",
    ]
    Path(out_dir, "REPORT.md").write_text("\n".join(L) + "\n")
    return out_dir


def _f(x) -> str:
    return "-" if x is None else f"{x:.2f}"
