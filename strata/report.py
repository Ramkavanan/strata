"""Markdown report from results.json. Per operator, family, budget and checkpoint. No composite score by design."""

from __future__ import annotations

import json
from typing import Optional

REQUIRED = [
    "adapter",
    "tokenizer",
    "harness_version",
    "git_sha",
    "scenario_run",
    "reader",
    "price_table",
    "repeats",
    "blocks",
]


def _f(x, nd=3) -> str:
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def check_reporting(results: dict) -> list[str]:
    """Appendix D checklist: fields every submitted result must carry."""
    miss = [k for k in REQUIRED if k not in results]
    a = results.get("adapter", {})
    for k in ("adapter", "provenance", "event_time_support", "config"):
        if k not in a:
            miss.append(f"adapter.{k}")
    pert = results.get("blocks", {}).get("perturbation")
    if pert and not all("confident_error_rate" in v for v in pert["operators"].values()):
        miss.append("confident error rate per operator")
    return miss


def render(results: dict, other: Optional[dict] = None) -> str:
    a = results["adapter"]
    L = [
        f"# STRATA results: {a.get('adapter')}",
        "",
        f"harness {results.get('harness_version')} ({results.get('git_sha') or 'no git sha'}), tokenizer {results.get('tokenizer')}, "
        f"reader {json.dumps(results.get('reader'))}, repeats {results.get('repeats')}, price table {results.get('price_table')}",
        f"event-time support: {a.get('event_time_support')}, provenance: {a.get('provenance')}",
        "",
    ]
    pert = results.get("blocks", {}).get("perturbation")
    if pert:
        L += [
            "## Perturbation block (budget 128k, final checkpoint)",
            "",
            "| operator | state acc (affected) | state acc (unaffected) | abstention BA | confident error | delta state acc, affected [95% CI] | class |",
            "|---|---|---|---|---|---|---|",
        ]
        for op, ag in pert["operators"].items():
            s = pert["sensitivity"].get(op)
            d = "-" if not s or s["mean"] is None else f"{s['mean']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
            L.append(
                f"| {op} | {_f(ag['state_accuracy_affected'])} | {_f(ag['state_accuracy_unaffected'])} | {_f(ag['abstention_balanced_accuracy'])} | "
                f"{_f(ag['confident_error_rate'])} | {d} | {(s['classification'] + ('*' if s.get('underpowered') else '')) if s else '-'} |"
            )
        L += [
            "",
            f"Staleness flag, balanced accuracy over paired clean and stale renders: {_f(pert.get('staleness_flag_balanced_accuracy'))}.",
        ]
        L += ["", "### Accuracy by task family (clean)", "", "| family | accuracy |", "|---|---|"]
        for fam, v in pert["operators"]["clean"]["accuracy_by_family"].items():
            L.append(f"| {fam} | {_f(v)} |")
        c = pert["operators"]["clean"]
        L += [
            "",
            f"Evidence precision {_f(c['evidence_precision'])}, recall {_f(c['evidence_recall'])}, recency {_f(c['evidence_recency'])}; "
            f"fact lineage recall {_f(c['fact_lineage_recall'])}, decision lineage recall {_f(c['decision_lineage_recall'])}; "
            f"drift F1 {_f(c['drift_f1'])}, question-resolution F1 {_f(c['question_resolution_f1'])}, open-inventory F1 {_f(c['open_inventory_f1'])}.",
            f"Attribution: {', '.join(c['attribution'])}.",
            "",
        ]
    att = results.get("blocks", {}).get("attribution")
    if att and "skipped" not in att:
        L += [
            "## Failure attribution (Section 6.1): the same consumer with no context, system context, oracle context",
            "",
            "| operator | Y none | Y system | Y oracle | context value V | context loss L | consumer gap G |",
            "|---|---|---|---|---|---|---|",
        ]
        for op, v in att.items():
            L.append(
                f"| {op} | {_f(v['y_none'])} | {_f(v['y_system'])} | {_f(v['y_oracle'])} | {_f(v['context_value'])} | "
                f"{_f(v['context_loss'])} | {_f(v['consumer_gap'])} |"
            )
        L += [
            "",
            "1 - Y none = V + L + G. A large L says the context layer is the bottleneck; a large G says even perfect "
            "context would not help this consumer.",
            "",
        ]
    cost = (pert or {}).get("cost")
    if cost and cost.get("clean"):
        L += [
            "## Cost (reference rates, not vendor prices)",
            "",
            "| operator | context tokens | system tokens | tool calls | query cost USD | ingest cost USD | CPCO at H=1000 |",
            "|---|---|---|---|---|---|---|",
        ]
        for op, c in cost.items():
            if c:
                L.append(
                    f"| {op} | {_f(c['mean_context_tokens'], 0)} | {_f(c['mean_system_input_tokens'], 0)} | {_f(c['mean_tool_calls'], 1)} | "
                    f"{_f(c['mean_query_cost_usd'], 6)} | {_f(c['mean_ingest_cost_usd'], 6)} | {_f(c['cpco_usd'], 6)} |"
                )
        L.append("")
    bud = results.get("blocks", {}).get("budget")
    if bud:
        keys = sorted(next(iter(bud.values()))["curve"], key=int)
        L += [
            "## Budget block",
            "",
            "| render | " + " | ".join(keys) + " | AUBC | B90 |",
            "|---|" + "---|" * (len(keys) + 2),
        ]
        for op, v in bud.items():
            L.append(f"| {op} | " + " | ".join(_f(v["curve"][k]) for k in keys) + f" | {_f(v['aubc'])} | {v['b90']} |")
        if all(x == 0 for v in bud.values() for x in v["truncated_share"].values()):
            L.append(
                "\nNote: no context was truncated at any budget, so the budget axis did not bind. Use smaller budgets (--budgets) or larger scenarios."
            )
        L.append("")
    gro = results.get("blocks", {}).get("growth_pooled")
    if gro:
        L += ["## Growth block", "", "| render | update latency (checkpoints) | stale-answer rate |", "|---|---|---|"]
        for op, v in gro.items():
            L.append(f"| {op} | {_f(v['update_latency_mean'])} | {_f(v['stale_answer_rate'])} |")
        L.append("")
    if pert and any(v.get("underpowered") for v in pert["sensitivity"].values()):
        L.append(
            "* fewer than 10 templates: the bootstrap over templates is too coarse to support an equivalence or degradation claim."
        )
    miss = check_reporting(results)
    L.append("Reporting checklist (Appendix D): " + ("complete" if not miss else "MISSING " + ", ".join(miss)))
    if other:
        L += [
            "",
            "## Comparison",
            "",
            "| operator | " + a.get("adapter", "A") + " delta | " + other["adapter"].get("adapter", "B") + " delta |",
            "|---|---|---|",
        ]
        sa = pert["sensitivity"] if pert else {}
        sb = other.get("blocks", {}).get("perturbation", {}).get("sensitivity", {})
        for op in sorted(set(sa) | set(sb)):
            L.append(f"| {op} | {_f((sa.get(op) or {}).get('mean'))} | {_f((sb.get(op) or {}).get('mean'))} |")
    return "\n".join(L) + "\n"
