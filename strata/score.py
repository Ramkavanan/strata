"""Scoring (Section 8). Everything is computed against ground truth derived from the world model; no judge involved."""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Iterable, Optional

from .scenario import Scenario
from .schema import CAPABILITY, STATE_FAMILIES, Item, Response

SESSION_TOLERANCE_DAYS = 7  # "within one session"


# ------------------------------------------------------------------ helpers
def _assertions(scn: Scenario, span_ids: Optional[Iterable[str]]) -> list[str]:
    return sorted({scn.span_map[s] for s in (span_ids or []) if scn.span_map.get(s)})


def _f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def _since_ok(resp_since: Optional[date], exp_since: Optional[str]) -> bool:
    if not exp_since:
        return True
    if resp_since is None:
        return False
    return abs((resp_since - date.fromisoformat(exp_since)).days) <= SESSION_TOLERANCE_DAYS


def _values(v) -> list[str]:
    return sorted(str(x) for x in (v if isinstance(v, (list, tuple)) else [v]))


def _step_match(step, exp, scn: Scenario) -> bool:
    exp_vals = _values(exp["value"])
    got = _values(step.value)
    if got != exp_vals:
        return False
    if not _since_ok(step.since, exp["since"]):
        return False
    if not exp["establishing"]:
        return True
    est = set(_assertions(scn, step.evidence))
    return bool(est & set(exp["establishing"]))


# ------------------------------------------------------------------ per-item scoring
def score_item(item: Item, resp: Response, scn: Scenario) -> dict:
    e, f = item.expected, item.family
    out: dict[str, Any] = {
        "item_id": item.id,
        "family": f,
        "capability": CAPABILITY[f],
        "affected": item.affected,
        "render_id": item.render_id,
        "scenario_id": item.scenario_id,
        "level": item.difficulty.get("level"),
        "explicitness": item.explicitness,
        "variable": item.variable,
    }
    cited = _assertions(scn, resp.evidence)
    out["abstained"] = bool(resp.abstain)
    expected_unanswerable = False  # truth is unanswerable or undetermined
    answerable = False
    correct = False
    partial: dict[str, float] = {}

    if f in ("T1", "T2", "T3", "T4", "T5"):
        status_exp = e["status"]
        if status_exp == "undetermined":
            expected_unanswerable = True
            correct = resp.abstain or (
                resp.state is not None and resp.state.status == "undetermined" and not resp.state.values
            )
        else:
            answerable = True
            st = resp.state
            ok_val = st is not None and _values(st.values) == _values(e["values"])
            ok_status = st is not None and st.status == status_exp
            ok_since = st is not None and (f == "T1" or _since_ok(st.since, e.get("since")))
            correct = (not resp.abstain) and ok_val and (ok_status or f == "T1") and ok_since
            partial = {"value": float(ok_val), "status": float(ok_status), "since": float(ok_since)}
        if f == "T4":
            gold = set(e["decider"]) | set(e["reasons"])
            lin = resp.lineage or {}
            pred = set(_assertions(scn, lin.get("decider", []))) | set(_assertions(scn, lin.get("reasons", [])))
            correct = bool(gold) and pred == gold
            partial["decider_reasons_f1"] = _f1(pred, gold)
    elif f == "T5b":
        answerable = True
        steps, exp = resp.chain or [], e["chain"]
        j = matched = 0
        for s in steps:
            if j < len(exp) and _step_match(s, exp[j], scn):
                matched += 1
                j += 1
        out["lineage_recall"] = matched / len(exp)
        out["lineage_exact"] = matched == len(exp) and len(steps) == len(exp)
        correct = out["lineage_exact"]
    elif f == "T6":
        answerable = True
        lin = resp.lineage or {}
        total = hit = 0
        exact = True
        for comp in ("trigger", "discussion", "reasons", "follow_ups"):
            gold = set(e[comp])
            pred = set(_assertions(scn, lin.get(comp, [])))
            total += len(gold)
            hit += len(gold & pred)
            exact &= pred == gold
        status_ok = lin.get("status") == e["status"]
        out["lineage_recall"] = hit / total if total else (1.0 if exact else 0.0)
        out["lineage_exact"] = exact and status_ok
        correct = out["lineage_exact"]
    elif f in ("T7", "T8", "T11"):
        answerable = True
        pred, gold = set(_assertions(scn, resp.ids)), set(e["ids"])
        out["set_f1"] = _f1(pred, gold)
        correct = out["set_f1"] == 1.0
    elif f == "T9":
        if e["drifted"] is None:
            expected_unanswerable = True
            correct = resp.abstain or resp.drifted is None
        else:
            answerable = True
            correct = (not resp.abstain) and resp.drifted == e["drifted"]
        out["drift_pos_gold"] = e["drifted"] is True
        out["drift_pos_pred"] = resp.drifted is True
    elif f == "T10":
        answerable = True
        got = scn.span_map.get(resp.resolver) if resp.resolver else None
        out["res_gold"] = bool(e["resolved"])
        out["res_pred"] = bool(resp.resolved)
        out["res_tp"] = bool(e["resolved"] and resp.resolved and got == e["resolver"])
        correct = (resp.resolved == e["resolved"]) and (not e["resolved"] or got == e["resolver"])
    elif f == "T12":
        expected_unanswerable = True
        correct = bool(resp.abstain)
    elif f == "T13":
        out["flag_gold"] = bool(e["staleness_flag"])
        out["flag_pred"] = bool(resp.staleness_flag)
        correct = out["flag_gold"] == out["flag_pred"]

    out["correct"] = bool(correct)
    out["expected_unanswerable"] = expected_unanswerable
    out["answerable"] = answerable
    out.update({f"part_{k}": v for k, v in partial.items()})

    # evidence attribution
    support, establishing = set(e.get("support", [])), set(e.get("establishing", []))
    if f not in ("T13",) and not (expected_unanswerable and resp.abstain):
        if resp.abstain and not cited:
            pass
        else:
            out["evidence_precision"] = (len(set(cited) & support) / len(cited)) if cited else 0.0
            if establishing:
                out["evidence_recall"] = len(set(cited) & establishing) / len(establishing)
    if f in STATE_FAMILIES and establishing and not resp.abstain:
        sup_ids = {
            a.id
            for a in scn.world.assertions
            if a.var == item.variable and scn.world.decision_status(a.id, item.query_time) == "superseded"
        }
        out["evidence_recency"] = bool(establishing <= set(cited) and not (set(cited) & sup_ids))
    # confident error: wrong, no abstention, no contested/undetermined status, no staleness flag
    st = resp.state
    hedged = resp.abstain or resp.staleness_flag or (st is not None and st.status in ("contested", "undetermined"))
    out["confident_error"] = (not correct) and not hedged and f != "T13"
    out["attribution"] = resp.attribution
    out["input_tokens"] = resp.usage.input_tokens
    out["output_tokens"] = resp.usage.output_tokens
    return out


# ------------------------------------------------------------------ aggregation
def _mean(xs: list[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def balanced_accuracy(tp: int, fn: int, tn: int, fp: int) -> Optional[float]:
    tpr = tp / (tp + fn) if (tp + fn) else None
    tnr = tn / (tn + fp) if (tn + fp) else None
    vals = [x for x in (tpr, tnr) if x is not None]
    return sum(vals) / len(vals) if vals else None


def aggregate(scores: list[dict]) -> dict:
    """All Section 8 metrics for one set of scored items (typically one operator and budget). No composite."""

    def sel(pred):
        return [s for s in scores if pred(s)]

    def acc(xs):
        return _mean([float(s["correct"]) for s in xs])

    out: dict[str, Any] = {"n": len(scores)}
    out["accuracy_by_family"] = {
        f: acc(sel(lambda s, f=f: s["family"] == f)) for f in sorted({s["family"] for s in scores})
    }
    out["accuracy_by_capability"] = {
        c: acc(sel(lambda s, c=c: s["capability"] == c)) for c in sorted({s["capability"] for s in scores})
    }
    out["accuracy_by_level"] = {
        int(l): acc(sel(lambda s, l=l: s["level"] == l)) for l in sorted({s["level"] for s in scores if s["level"]})
    }
    out["accuracy_by_explicitness"] = {
        x: acc(sel(lambda s, x=x: s["explicitness"] == x))
        for x in sorted({s["explicitness"] for s in scores if s["explicitness"]})
    }
    out["accuracy_affected"] = acc(sel(lambda s: s["affected"]))
    out["accuracy_unaffected"] = acc(sel(lambda s: not s["affected"]))
    state = sel(lambda s: s["family"] in STATE_FAMILIES)
    out["state_accuracy"] = acc(state)
    out["state_accuracy_affected"] = acc([s for s in state if s["affected"]])
    out["state_accuracy_unaffected"] = acc([s for s in state if not s["affected"]])
    out["fact_lineage_recall"] = _mean([s["lineage_recall"] for s in scores if s["family"] == "T5b"])
    out["fact_lineage_exact"] = acc(sel(lambda s: s["family"] == "T5b"))
    out["decision_lineage_recall"] = _mean([s["lineage_recall"] for s in scores if s["family"] == "T6"])
    out["decision_lineage_exact"] = acc(sel(lambda s: s["family"] == "T6"))
    t9 = sel(lambda s: s["family"] == "T9")
    tp = sum(1 for s in t9 if s["drift_pos_gold"] and s["drift_pos_pred"])
    fp = sum(1 for s in t9 if not s["drift_pos_gold"] and s["drift_pos_pred"])
    fn = sum(1 for s in t9 if s["drift_pos_gold"] and not s["drift_pos_pred"])
    out["drift_f1"] = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    t10 = sel(lambda s: s["family"] == "T10")
    tp = sum(1 for s in t10 if s["res_tp"])
    fp = sum(1 for s in t10 if s["res_pred"] and not s["res_tp"])
    fn = sum(1 for s in t10 if s["res_gold"] and not s["res_tp"])
    out["question_resolution_f1"] = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    out["open_inventory_f1"] = _mean([s["set_f1"] for s in scores if s["family"] == "T11"])
    out["reversal_f1"] = _mean([s["set_f1"] for s in scores if s["family"] == "T7"])
    out["evidence_precision"] = _mean([s["evidence_precision"] for s in scores if "evidence_precision" in s])
    out["evidence_recall"] = _mean([s["evidence_recall"] for s in scores if "evidence_recall" in s])
    out["evidence_recency"] = _mean([float(s["evidence_recency"]) for s in scores if "evidence_recency" in s])
    unans = sel(lambda s: s["expected_unanswerable"])
    ans = sel(lambda s: s["answerable"])
    out["abstention_balanced_accuracy"] = balanced_accuracy(
        tp=sum(1 for s in unans if s["correct"]),
        fn=sum(1 for s in unans if not s["correct"]),
        tn=sum(1 for s in ans if not s["abstained"]),
        fp=sum(1 for s in ans if s["abstained"]),
    )
    t13 = sel(lambda s: s["family"] == "T13")
    out["staleness_balanced_accuracy"] = balanced_accuracy(
        tp=sum(1 for s in t13 if s["flag_gold"] and s["flag_pred"]),
        fn=sum(1 for s in t13 if s["flag_gold"] and not s["flag_pred"]),
        tn=sum(1 for s in t13 if not s["flag_gold"] and not s["flag_pred"]),
        fp=sum(1 for s in t13 if not s["flag_gold"] and s["flag_pred"]),
    )
    out["confident_error_rate"] = _mean([float(s["confident_error"]) for s in scores])
    out["attribution"] = sorted({s["attribution"] for s in scores})
    return out


def sensitivity(
    op_scores: list[dict], clean_scores: list[dict], metric: str = "correct", only_affected: bool = True
) -> dict:
    """Paired difference M_pi - M_clean on the same items (matched by item key without render id)."""

    def key(s):
        return s["item_id"]

    clean = {key(s): s for s in clean_scores}
    diffs = []
    for s in op_scores:
        c = clean.get(key(s))
        if c is None or (only_affected and not s["affected"]):
            continue
        diffs.append(float(s[metric]) - float(c[metric]))
    return {"n_pairs": len(diffs), "delta": _mean(diffs), "diffs": diffs}


# ------------------------------------------------------------------ budget, growth, cost
def aubc(curve: dict[int, float]) -> Optional[float]:
    bs = sorted(curve)
    if len(bs) < 2 or any(curve[b] is None for b in bs):
        return None
    area = sum((curve[a] + curve[b]) / 2 * (math.log(b) - math.log(a)) for a, b in zip(bs, bs[1:]))
    return area / (math.log(bs[-1]) - math.log(bs[0]))


def b90(curve: dict[int, float]) -> Optional[int]:
    bs = sorted(curve)
    top = curve.get(bs[-1]) if bs else None
    if not top:
        return None
    for b in bs:
        if curve[b] is not None and curve[b] >= 0.9 * top:
            return b
    return None


def growth_metrics(expected: dict[str, list[tuple]], answered: dict[str, list[Optional[tuple]]]) -> dict:
    """expected[item][k] and answered[item][k] are value tuples per checkpoint k=0..K-1.

    Update latency: checkpoints between a change in valid state and the first checkpoint reporting the new state;
    changes not reflected by the last checkpoint are censored. Stale-answer rate: share of answers equal to a
    previously valid but superseded value. `events` lists one record per change for the hazard model."""
    latencies: list[int] = []
    events: list[dict] = []
    censored = stale_n = total_n = 0
    for item, exp in expected.items():
        ans = answered.get(item, [None] * len(exp))
        for k in range(1, len(exp)):
            if exp[k] != exp[k - 1] and exp[k]:
                found = None
                horizon = len(exp) - k
                for j in range(k, len(exp)):
                    if exp[j] != exp[k]:  # the state changed again before the system caught up
                        horizon = j - k
                        break
                    if ans[j] == exp[k]:
                        found = j - k
                        break
                if found is None:
                    censored += 1
                    events.append({"item": item, "at": k, "latency": None, "observed_for": horizon})
                else:
                    latencies.append(found)
                    events.append({"item": item, "at": k, "latency": found, "observed_for": found + 1})
        for k in range(len(exp)):
            total_n += 1
            prev_vals = {tuple(v) for v in exp[:k] if v}
            a_k = ans[k]
            if a_k and a_k != exp[k] and tuple(a_k) in prev_vals:
                stale_n += 1
    return {
        "update_latency_mean": _mean([float(x) for x in latencies]),
        "update_latency_median": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "changes_observed": len(latencies),
        "changes_censored": censored,
        "stale_answer_rate": stale_n / total_n if total_n else None,
        "events": events,
    }


def cost_usd(input_tokens: int, output_tokens: int, prices: dict, model: Optional[str] = "reader-default") -> float:
    p = prices.get("models", {}).get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def cpco(mean_query_cost: float, ingest_cost: float, accuracy: Optional[float], horizon: int = 1000) -> Optional[float]:
    """Cost per correct outcome: (mean query cost + ingest cost / H) / accuracy. Undefined when accuracy is 0."""
    if not accuracy:
        return None
    return (mean_query_cost + ingest_cost / horizon) / accuracy
