"""Question generation by deterministic replay of the (perturbed) world model. No model is involved."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from typing import Optional

from .core import WorldModel
from .render import staleness_gap_days
from .scenario import Scenario, sha
from .schema import STATE_FAMILIES, Item
from .templates import STALE_DAYS

# Difficulty thresholds (Section 3.7). Fixed, published, never tuned on system results.
DIFFICULTY_THRESHOLDS = {"sessions": 3, "transitions": 2, "hops": 2, "distractors": 2, "conflicts": 1}
# Item selection is a per-assertion hash draw, so whether an item exists never depends on other assertions.
# That keeps item identity stable across operators, which paired comparisons need.
P_T1 = 0.4
P_T6 = 1.0


def _draw(scn, family: str, key: str, p: float) -> bool:
    return int(sha([scn.seed, family, key]), 16) / 16**16 < p


def _iso(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d else None


def state_expected(w: WorldModel, var: str, t: date) -> dict:
    st = w.valid_state(var, t)
    vals = st["values"] if st["status"] == "contested" else ([st["values"]] if st["status"] == "valid" else [])
    support = sorted(
        {a.id for a in w.assertions if a.var == var and a.t <= t and a.kind in ("decide", "request", "revoke")}
    )
    return {
        "status": st["status"],
        "values": [str(v) for v in vals],
        "since": _iso(st["since"]),
        "establishing": st["establishing"],
        "support": support,
    }


def _depth(w: WorldModel, aid: str, seen: Optional[set] = None) -> int:
    seen = seen or set()
    a = w._by_id(aid)
    if a is None or aid in seen:
        return 0
    seen.add(aid)
    refs = [r for r in list(a.ref) + list(a.trigger) + list(a.discussion) if w._by_id(r)]
    return 1 + max([_depth(w, r, seen) for r in refs], default=0)


def difficulty(scn: Scenario, var: Optional[str], family: str, t: date) -> dict:
    w = scn.world
    if var is None:
        vec = {"sessions": 0, "transitions": 0, "hops": 0, "distractors": 0, "conflicts": 0, "compare": False}
        pts = 0
    else:
        asserts = [a for a in w.assertions if a.var == var and a.t <= t]
        und = {a.id for a in w.undominated(var, t)}
        sessions = len({a.t for a in asserts})
        transitions = max(0, len(w.fact_lineage(var, t)) - 1)
        hops = max([_depth(w, a.id) for a in asserts if a.id in und] + [0])
        dom = [a for a in asserts if a.kind in ("decide", "request") and a.id not in und]
        conflicts = len(dom) + (1 if w.valid_state(var, t)["status"] == "contested" else 0)
        compare = family == "T9"
        vec = {
            "sessions": sessions,
            "transitions": transitions,
            "hops": hops,
            "distractors": len(dom),
            "conflicts": conflicts,
            "compare": compare,
        }
        pts = sum(
            [
                sessions >= DIFFICULTY_THRESHOLDS["sessions"],
                transitions >= DIFFICULTY_THRESHOLDS["transitions"],
                hops >= DIFFICULTY_THRESHOLDS["hops"],
                len(dom) >= DIFFICULTY_THRESHOLDS["distractors"],
                conflicts >= DIFFICULTY_THRESHOLDS["conflicts"],
                compare,
            ]
        )
    base = 1 if family in ("T1",) else 1
    vec["level"] = int(min(5, base + pts))
    return vec


def _explicitness(scn: Scenario, ids: list[str]) -> Optional[str]:
    levels = [scn.explicitness.get(i, "explicit") for i in ids]
    for lv in ("diffuse", "hedged"):
        if lv in levels:
            return lv
    return "explicit" if levels else None


def _state_setting_ids(w: WorldModel, variables, t: date) -> list[str]:
    vs = set(variables)
    return sorted(a.id for a in w.assertions if a.var in vs and a.t <= t and a.kind in ("decide", "request", "revoke"))


def all_items(scn: Scenario) -> list[Item]:
    """Every item the world model supports on this render, before any panel selection."""
    w, t, L = scn.world, scn.query_time, scn.labels
    rid = scn.render_id
    aff = set(scn.affected_vars)
    items: list[Item] = []
    stale_op = scn.operator.get("name") == "stale"

    def add(
        family: str,
        key: str,
        question: str,
        var: Optional[str],
        expected: dict,
        params: Optional[dict] = None,
        affected: Optional[bool] = None,
        qt: Optional[date] = None,
        expl_ids: Optional[list[str]] = None,
    ):
        diff = difficulty(scn, var, family, qt or t)
        items.append(
            Item(
                id=f"{scn.id}/{family}/{key}",
                scenario_id=scn.id,
                render_id=rid,
                family=family,
                question=question,
                query_time=t,
                params=params or {},
                variable=var,
                expected=expected,
                affected=(var in aff) if affected is None else affected,
                difficulty=diff,
                explicitness=_explicitness(scn, expl_ids if expl_ids is not None else expected.get("establishing", [])),
            )
        )

    state_vars = [
        v
        for v in w.variables()
        if any(
            a.var == v and (a.kind in ("decide", "request", "revoke") or (a.kind == "report" and a.value is None))
            for a in w.assertions
        )
    ]
    # --- T1 recall (baseline): what a named speaker stated on a date ---
    decides = [a for a in w.assertions if a.kind == "decide" and w.auth(a) is not None]
    first = (
        min(decides, key=lambda x: (x.t, x.id)).id if decides else None
    )  # every scenario keeps at least one recall item
    for a in (x for x in decides if x.id == first or _draw(scn, "T1", x.id, P_T1)):
        name = scn.people.get(a.speaker, a.speaker)
        add(
            "T1",
            a.id,
            f'What value did {name} give for "{L[a.var]}" on {a.t.isoformat()}?',
            a.var,
            {"values": [str(a.value)], "status": "valid", "establishing": [a.id], "support": [a.id]},
            {"label": L[a.var], "date": a.t.isoformat(), "speaker_name": name},
            affected=a.var in aff,
        )
    # --- T2 state at time, T3 current state ---
    for var in state_vars:
        chain = w.fact_lineage(var, t)
        if len(chain) >= 2:
            tau = chain[-1]["since"] - timedelta(days=1)
            add(
                "T2",
                var,
                f'What was "{L[var]}" as of {tau.isoformat()}?',
                var,
                state_expected(w, var, tau),
                {"label": L[var], "as_of": tau.isoformat()},
            )
        add(
            "T3",
            var,
            f'What is the current state of "{L[var]}", and is it settled?',
            var,
            state_expected(w, var, t),
            {"label": L[var]},
        )
    # --- T4 transition reasoning ---
    for var in state_vars:
        chain = w.fact_lineage(var, t)
        if len(chain) >= 2 and chain[-1]["status"] == "valid" and chain[-2]["status"] == "valid":
            und = [a.id for a in w.undominated(var, t)]
            reasons = sorted({r for a in und for r in w._by_id(a).ref if w._by_id(r)})
            add(
                "T4",
                var,
                f'Why did "{L[var]}" change from {chain[-2]["value"]} to {chain[-1]["value"]}, and who decided?',
                var,
                {
                    "status": "valid",
                    "values": [str(chain[-1]["value"])],
                    "since": _iso(chain[-1]["since"]),
                    "establishing": chain[-1]["establishing"],
                    "support": chain[-1]["establishing"],
                    "decider": und,
                    "reasons": reasons,
                },
                {"label": L[var], "from": str(chain[-2]["value"]), "to": str(chain[-1]["value"])},
            )
    # --- T5 authority conflict ---
    for a in w.assertions:
        if a.kind != "decide" or w.auth(a) is None:
            continue
        und = w.undominated(a.var, t)
        valid = w.valid_state(a.var, t)
        if valid["status"] != "valid" or a in und:
            continue
        dom = [u for u in und if u.domain == a.domain and (w.auth(u) or 0) > (w.auth(a) or 0) and u.t <= a.t]
        if dom and str(a.value) != str(valid["values"]):
            name = scn.people.get(a.speaker, a.speaker)
            add(
                "T5",
                a.id,
                f'On {a.t.isoformat()}, {name} stated that "{L[a.var]}" is {a.value}. Is that now the state of "{L[a.var]}"?',
                a.var,
                state_expected(w, a.var, t),
                {"label": L[a.var], "claimed": str(a.value), "claimed_on": a.t.isoformat(), "speaker_name": name},
            )
    # --- T5b fact lineage ---
    for var in state_vars:
        chain = w.fact_lineage(var, t)
        if len(chain) >= 2:
            steps = [
                {
                    "value": c["value"] if c["status"] != "contested" else list(c["value"]),
                    "status": c["status"],
                    "since": _iso(c["since"]),
                    "establishing": c["establishing"],
                }
                for c in chain
            ]
            add(
                "T5b",
                var,
                f'Give the full history of "{L[var]}": every value, when it changed, who changed it and why.',
                var,
                {
                    "chain": steps,
                    "establishing": chain[-1]["establishing"],
                    "support": sorted(
                        {e for c in chain for e in c["establishing"]} | set(_state_setting_ids(w, [var], t))
                    ),
                },
                {"label": L[var]},
                expl_ids=sorted({e for c in chain for e in c["establishing"]}),
            )
    # --- T6 decision trace ---
    cands = []
    for a in w.assertions:
        if a.kind == "decide" and w.auth(a) is not None:
            lin = w.decision_lineage(a.id, t)
            ex_trig = [x for x in lin["trigger"] if w._by_id(x)]
            ex_disc = [x for x in lin["discussion"] if w._by_id(x)]
            ex_reas = [x for x in lin["reasons"] if w._by_id(x)]
            if ex_trig or ex_disc or ex_reas or lin["follow_ups"]:
                cands.append((a, lin, ex_trig, ex_disc, ex_reas))
    for a, lin, tr, di, re_ in (c for c in cands if _draw(scn, "T6", c[0].id, P_T6)):
        name = scn.people.get(a.speaker, a.speaker)
        add(
            "T6",
            a.id,
            f'What led to the {a.t.isoformat()} decision by {name} on "{L[a.var]}", and what followed?',
            a.var,
            {
                "decision": a.id,
                "trigger": tr,
                "discussion": di,
                "reasons": re_,
                "follow_ups": lin["follow_ups"],
                "status": lin["status"],
                "establishing": [a.id],
                "support": sorted(
                    set([a.id] + tr + di + re_ + lin["follow_ups"]) | set(_state_setting_ids(w, [a.var], t))
                ),
            },
            {"label": L[a.var], "date": a.t.isoformat(), "speaker_name": name},
        )
    # --- T7 reversal detection ---
    dates = sorted({a.t for a in w.assertions})
    if len(dates) >= 2:
        mid = dates[len(dates) // 2]
        for key, before in (("mid", mid), ("all", t + timedelta(days=1))):
            ids = w.not_in_force(t, before)
            aff7 = any(w._by_id(i).var in aff for i in ids) if ids else bool(aff)
            add(
                "T7",
                key,
                f"Which decisions made before {before.isoformat()} are no longer in force?",
                None,
                {
                    "ids": ids,
                    "establishing": ids,
                    # deciding that a decision is no longer in force needs the decisions that replaced or revoked it
                    "support": _state_setting_ids(w, {w._by_id(i).var for i in ids}, t),
                },
                {"before": before.isoformat()},
                affected=aff7,
            )
    # --- T8 requirement trace, T9 drift ---
    for var in w.variables():
        reqs = [a for a in w.assertions if a.var == var and a.kind == "request"]
        if reqs:
            st = state_expected(w, var, t)
            add(
                "T8",
                var,
                f'Which request produced the current "{L[var]}"?',
                var,
                {
                    "ids": [a.id for a in reqs],
                    "establishing": [a.id for a in reqs],
                    "support": [a.id for a in reqs] + st["establishing"],
                    "decision": st["establishing"],
                },
                {"label": L[var]},
            )
        if any(a.var == var and a.kind == "implement" for a in w.assertions):
            dr = w.drifted(var, t)
            impls = [a.id for a in w.assertions if a.var == var and a.kind == "implement"]
            st = state_expected(w, var, t)
            add(
                "T9",
                var,
                f'Does the shipped "{L[var]}" satisfy the specification now in force?',
                var,
                {"drifted": dr, "establishing": st["establishing"] + impls[-1:], "support": st["establishing"] + impls},
                {"label": L[var]},
            )
    # --- T10 question resolution, T11 open inventory ---
    asks = [a for a in w.assertions if a.kind == "ask"]
    for q in asks:
        ans = [a for a in w.assertions if a.kind == "answer" and q.id in a.ref and a.complete and a.t <= t]
        name = scn.people.get(q.speaker, q.speaker)
        add(
            "T10",
            q.id,
            f'Was the question {name} asked on {q.t.isoformat()} about "{L[q.var]}" answered, and where?',
            q.var,
            {
                "resolved": bool(ans),
                "resolver": ans[0].id if ans else None,
                "establishing": [a.id for a in ans] or [q.id],
                "support": [q.id] + [a.id for a in w.assertions if a.kind == "answer" and q.id in a.ref],
            },
            {"label": L[q.var], "date": q.t.isoformat(), "speaker_name": name},
        )
    if asks:
        open_ids = w.open_questions(t)
        add(
            "T11",
            "all",
            "Which questions remain unresolved?",
            None,
            {
                "ids": open_ids,
                "establishing": open_ids,
                "support": sorted(a.id for a in w.assertions if a.kind in ("ask", "answer")),
            },
            {},
            affected=any(w._by_id(i).var in aff for i in open_ids) if open_ids else bool(aff),
        )
    # --- T12 false premise ---
    for lab in scn.unanswerable:
        add(
            "T12",
            sha([lab])[:8],
            f'When was "{lab}" approved?',
            None,
            {"abstain": True, "establishing": [], "support": []},
            {"label": lab},
            affected=False,
        )
    # --- T13 staleness awareness ---
    gap = staleness_gap_days(scn.artifacts, t)
    add(
        "T13",
        "flag",
        "Is the information behind your answers recent enough to rely on?",
        None,
        {"staleness_flag": gap > STALE_DAYS, "establishing": [], "support": []},
        {},
        affected=stale_op,
    )
    return items


TARGETING_OPERATORS = ("stale", "contradict_low", "contradict_cross", "incomplete")


def targeted_variables(scn: Scenario) -> set[str]:
    """Variables some perturbation operator will target on this scenario. Operators are deterministic in the seed, so
    this is known when the panel is chosen, and the panel can make sure those variables have state questions."""
    from . import operators  # local import: operators imports this module's neighbours

    out: set[str] = set()
    for name in TARGETING_OPERATORS:
        out |= set(operators.apply(scn, name).affected_vars)
    return out


def select_panel(scn: Scenario, size: int) -> dict:
    """Choose about `size` items on the clean render (Appendix A.3: roughly 30 questions from all fourteen families).

    State questions about variables the operators target come first, because they are the affected items the
    primary hypothesis test is paired on; then one item per family in rounds until the panel is full. Order within
    a family is a hash of the item id, so the choice is deterministic and independent of any system evaluated."""
    items = all_items(scn)
    targeted = targeted_variables(scn)
    ordered = sorted(items, key=lambda i: sha([scn.seed, "panel", i.id]))
    is_target = lambda i: i.family in STATE_FAMILIES and i.variable in targeted
    selected: list[str] = []
    # 1. one item per family present, so every capability is covered (targeted ones first within a family)
    for fam in sorted({i.family for i in items}):
        pool = [i for i in ordered if i.family == fam]
        pick = next((i for i in pool if is_target(i)), pool[0])
        if len(selected) < size:
            selected.append(pick.id)
    # 2. state questions about the variables operators target: the affected items the primary test is paired on
    for it in ordered:
        if len(selected) < size and it.id not in selected and is_target(it):
            selected.append(it.id)
    # 3. the rest, one per family in rounds
    by_family: dict[str, list[Item]] = {}
    for it in ordered:
        if it.id not in selected:
            by_family.setdefault(it.family, []).append(it)
    rank = 0
    fams = sorted(by_family)
    while len(selected) < size and any(len(v) > rank for v in by_family.values()):
        for f in fams:
            if len(by_family[f]) > rank and len(selected) < size:
                selected.append(by_family[f][rank].id)
        rank += 1
    return {
        "selected": sorted(selected),
        "universe": sorted(i.id for i in items),
        "size": size,
        "targeted_variables": sorted(targeted),
    }


def generate_items(scn: Scenario) -> list[Item]:
    """The items to ask on this render. With a panel: the selected items, plus any item an operator created that has
    no counterpart on the clean render (those are the operator's own probes, such as a question about an injected
    statement)."""
    items = all_items(scn)
    if not scn.panel:
        return items
    keep, universe = set(scn.panel["selected"]), set(scn.panel["universe"])
    return [i for i in items if i.id in keep or i.id not in universe]


def ingested_world(scn: Scenario, cutoff: date) -> WorldModel:
    """World restricted to assertions whose source span was ingested by `cutoff` (growth protocol)."""
    ingested_spans = set()
    for a in scn.artifacts:
        if a.ingest_date <= cutoff:
            ingested_spans.update(s.id for s in a.spans)
    seen = {aid for sid, aid in scn.span_map.items() if sid in ingested_spans and aid}
    return replace(scn.world, assertions=[a for a in scn.world.assertions if a.id in seen])


def growth_items(scn: Scenario, k: int) -> list[Item]:
    """Fixed panel: current-state question for every state variable, answered as of checkpoint k."""
    cp = scn.checkpoints[k]
    w = ingested_world(scn, cp)
    out = []
    final = {i.variable: i for i in all_items(scn) if i.family == "T3"}
    for var, base in final.items():
        exp = state_expected(w, var, cp)
        out.append(base.model_copy(update={"query_time": cp, "expected": exp, "render_id": f"{scn.render_id}@cp{k}"}))
    return out
