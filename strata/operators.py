"""Perturbation operators (Section 4.1). Each maps a Scenario to a perturbed Scenario.

Ground truth for every item is computed from the perturbed world model, so a perfect system scores the
same under every operator. `late` changes ingestion order only and `noisy` adds decoys only, so their
ground truth equals clean.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta
from typing import Callable, Optional

from .core import Assertion, WorldModel
from .render import render_world
from .scenario import Scenario, sha
from .schema import Artifact, Span
from .templates import checkpoint_dates

OPERATORS = ["clean", "stale", "contradict_low", "contradict_cross", "incomplete", "late", "noisy"]


def _u(seed: int, *key) -> float:
    return int(sha([seed, *key]), 16) / 16**16


def _rank(items: list[str], seed: int, tag: str) -> list[str]:
    return sorted(items, key=lambda x: sha([seed, tag, x]))


def _last_day(w: WorldModel, var: str) -> date:
    return max(a.t for a in w.assertions if a.var == var)


def _rerender(
    scn: Scenario,
    world: WorldModel,
    op: dict,
    affected: list[str],
    extra_expl: Optional[dict] = None,
) -> Scenario:
    expl = dict(scn.explicitness)
    expl.update(extra_expl or {})
    start = min(a.t for a in scn.world.assertions)
    r = scn.renderer
    tl = tuple(date.fromisoformat(x) for x in r["timeline"]) if r.get("timeline") else None
    artifacts, span_map = render_world(
        world,
        labels=scn.labels,
        domain_labels=scn.domain_labels,
        people=scn.people,
        explicitness=expl,
        seed=scn.seed,
        start=start,
        chatter=int(r.get("chatter", 0)),
        filler_sessions=int(r.get("filler_sessions", 0)),
        timeline=tl,
    )
    cps = checkpoint_dates([a.ingest_date for a in artifacts if a.type != "org_chart"])
    return replace(
        scn,
        world=world,
        artifacts=artifacts,
        span_map=span_map,
        explicitness=expl,
        operator=op,
        affected_vars=sorted(affected),
        checkpoints=cps,
    )


TARGET_FRACTION = 0.25  # share of eligible variables a contradiction operator targets when k is not given


def _default_k(candidates: int) -> int:
    """k scales with the scenario: two variables at least, a quarter of the eligible ones otherwise."""
    return max(2, math.ceil(TARGET_FRACTION * candidates))


def clean(scn: Scenario, seed: int = 0) -> Scenario:
    return replace(scn, operator={"name": "clean"}, affected_vars=[])


def stale(scn: Scenario, seed: int = 0, tau_frac: float = 0.5, tau: Optional[date] = None) -> Scenario:
    w = scn.world
    first, last = min(a.t for a in w.assertions), max(a.t for a in w.assertions)
    cut = tau or first + timedelta(days=int((last - first).days * tau_frac))
    kept = [a for a in w.assertions if a.t <= cut]
    affected = sorted({a.var for a in w.assertions if a.t > cut})
    nw = replace(w, assertions=kept)
    return _rerender(scn, nw, {"name": "stale", "tau": cut.isoformat()}, affected)


def _lower_speaker(w: WorldModel, domain: str, level: int, taken: set) -> Optional[str]:
    cands = [
        p
        for p, r in sorted(w.roles.items())
        if w.authority.get(domain, {}).get(r) is not None and w.authority[domain][r] < level
    ]
    return cands[0] if cands else None


def _higher_none_speaker(w: WorldModel, domains: set, exclude: set) -> Optional[tuple[str, str]]:
    for dom in sorted(w.authority):
        if dom in domains:
            continue
        for p, r in sorted(w.roles.items()):
            if r in w.authority[dom]:
                return p, dom
    return None


def contradict_low(scn: Scenario, seed: int = 0, k: Optional[int] = None) -> Scenario:
    w = scn.world
    injected, affected, expl = [], [], {}
    cands = []
    for var in w.variables():
        st = w.valid_state(var, scn.query_time)
        und = w.undominated(var, scn.query_time)
        if st["status"] != "valid" or not und:
            continue
        top = und[0]
        lvl = w.auth(top)
        spk = _lower_speaker(w, top.domain, lvl, set())
        if spk:
            cands.append((var, top.domain, spk))
    k = k or _default_k(len(cands))
    chosen = _rank([c[0] for c in cands], seed, "low")[:k]
    for var, dom, spk in cands:
        if var in chosen:
            day = min(scn.query_time, _last_day(w, var) + timedelta(days=3))
            aid = f"inj-low-{var}"
            injected.append(Assertion(aid, day, spk, dom, "decide", var, "dropped"))
            expl[aid] = "hedged" if _u(seed, aid) < 0.5 else "explicit"
            affected.append(var)
    nw = replace(w, assertions=w.assertions + injected)
    return _rerender(scn, nw, {"name": "contradict_low", "k": k}, affected, expl)


def contradict_cross(scn: Scenario, seed: int = 0, k: Optional[int] = None) -> Scenario:
    w = scn.world
    gov = {v: set(d) for v, d in w.governance.items()}
    injected, affected, expl = [], [], {}
    cands = []
    for var in w.variables():
        st = w.valid_state(var, scn.query_time)
        if st["status"] != "valid":
            continue
        pick = _higher_none_speaker(w, gov.get(var, set()), set())
        if pick:
            cands.append((var, pick))
    k = k or _default_k(len(cands))
    chosen = _rank([c[0] for c in cands], seed, "cross")[:k]
    for var, (spk, dom) in cands:
        if var in chosen:
            gov[var] = gov[var] | {dom}
            day = min(scn.query_time, _last_day(w, var) + timedelta(days=3))
            aid = f"inj-cross-{var}"
            injected.append(Assertion(aid, day, spk, dom, "decide", var, "reprioritised"))
            expl[aid] = "explicit"
            affected.append(var)
    nw = replace(w, governance=gov, assertions=w.assertions + injected)
    return _rerender(scn, nw, {"name": "contradict_cross", "k": k}, affected, expl)


def _truncate(text: str) -> str:
    cut = max(24, len(text) // 3)
    return text[:cut].rstrip() + " ..."


def incomplete(scn: Scenario, seed: int = 0, rho: float = 0.25, mode: str = "omit") -> Scenario:
    assert mode in ("omit", "truncate")
    w = scn.world
    decided = sorted({a.var for a in w.assertions if a.kind in ("decide", "request")})
    n = max(1, round(rho * len(decided)))
    chosen = set(_rank(decided, seed, "inc")[:n])
    removed = [a for a in w.assertions if a.var in chosen and a.kind in ("decide", "request", "revoke")]
    kept = [a for a in w.assertions if a not in removed]
    indirect = []
    for var in sorted(chosen):
        rs = [a for a in removed if a.var == var]
        last = max(rs, key=lambda a: (a.t, a.id))
        day = min(scn.query_time, last.t + timedelta(days=7))
        indirect.append(
            Assertion(f"ind-{var}", day, last.speaker, last.domain, "report", var, None, tuple(a.id for a in rs))
        )
    nw = replace(w, assertions=kept + indirect)
    out = _rerender(scn, nw, {"name": "incomplete", "rho": rho, "mode": mode}, sorted(chosen))
    if mode == "truncate":
        # re-render the removed assertions, then keep only a truncated prefix in a partial artifact
        full, fmap = render_world(
            replace(w, assertions=removed),
            labels=scn.labels,
            domain_labels=scn.domain_labels,
            people=scn.people,
            explicitness=scn.explicitness,
            seed=scn.seed,
            start=min(a.t for a in w.assertions),
        )
        arts = list(out.artifacts)
        smap = dict(out.span_map)
        for art in full:
            if art.type == "org_chart":
                continue
            spans = [Span(id=s.id, text=_truncate(s.text)) for s in art.spans]
            for s in spans:
                smap[s.id] = None
            arts.append(
                Artifact(
                    id=art.id + "-p",
                    type=art.type,
                    title=art.title,
                    event_date=art.event_date,
                    ingest_date=art.ingest_date,
                    status="partial",
                    spans=spans,
                )
            )
        out = replace(out, artifacts=arts, span_map=smap)
    return out


def late(scn: Scenario, seed: int = 0, lam: float = 0.3) -> Scenario:
    arts = [a for a in scn.artifacts if a.type != "org_chart"]
    n = max(1, round(lam * len(arts)))
    chosen = set(a.id for a in sorted(arts, key=lambda a: sha([seed, "late", a.id]))[:n])
    last_event = max(a.event_date for a in arts)
    new = []
    affected: set[str] = set()
    for a in scn.artifacts:
        if a.id in chosen:
            delay = 1 + int(_u(seed, "latedelay", a.id) * 5)
            new.append(
                a.model_copy(
                    update={
                        "type": "document",
                        "title": f"Document revised {a.event_date.isoformat()}",
                        "ingest_date": last_event + timedelta(days=delay),
                    }
                )
            )
            for s in a.spans:
                aid = scn.span_map.get(s.id)
                if aid:
                    affected.add(next(x.var for x in scn.world.assertions if x.id == aid))
        else:
            new.append(a)
    qt = max([scn.query_time] + [a.ingest_date for a in new])
    cps = checkpoint_dates([a.ingest_date for a in new if a.type != "org_chart"])
    return replace(
        scn,
        artifacts=new,
        query_time=qt,
        checkpoints=cps,
        operator={"name": "late", "lam": lam},
        affected_vars=sorted(affected),
    )


DECOY_TOPICS = [
    "reporting dashboard",
    "analytics export",
    "sandbox environment",
    "partner webinar",
    "billing migration",
    "onboarding checklist",
    "legacy connector",
    "security questionnaire",
]
UNRELATED = [
    ("Team offsite venue", "lakeside lodge"),
    ("Quarterly newsletter topic", "customer stories"),
    ("Office plant schedule", "watering on Mondays"),
    ("Holiday rota", "two engineers on call"),
    ("Conference booth theme", "open source"),
    ("Parking policy", "reserved bays removed"),
]


def noisy(scn: Scenario, seed: int = 0, eta: float = 1.0) -> Scenario:
    from . import textformat as tf

    relevant = [a for a in scn.artifacts if a.type != "org_chart"]
    n = max(1, round(eta * len(relevant)))
    vars_ = sorted(scn.labels)
    first = min(a.event_date for a in relevant)
    last = max(a.event_date for a in relevant)
    span_days = max(1, (last - first).days)
    people = sorted(scn.world.roles.items())
    doms = sorted(scn.domain_labels.values())
    arts = list(scn.artifacts)
    smap = dict(scn.span_map)
    affected: set[str] = set()
    for i in range(n):
        near = i % 2 == 0
        day = first + timedelta(days=int(_u(seed, "noiseday", i) * span_days))
        spk, role = people[int(_u(seed, "noisespk", i) * len(people))]
        name = scn.people.get(spk, spk)
        dom = doms[int(_u(seed, "noisedom", i) * len(doms))]
        if near:
            var = vars_[(i // 2) % len(vars_)]
            base = " ".join(scn.labels[var].split()[:2])
            topic = DECOY_TOPICS[int(_u(seed, "topic", i) * len(DECOY_TOPICS))]
            label, value = (
                f"{base} {topic}",
                ["Q1", "Q2", "paused", "on track", "deferred"][int(_u(seed, "val", i) * 5)],
            )
            affected.add(var)
        else:
            label, value = UNRELATED[int(_u(seed, "unrel", i) * len(UNRELATED))]
        expl = ["explicit", "hedged", "diffuse"][int(_u(seed, "nexp", i) * 3)]
        text = tf.render_span(
            d=day, name=name, role=role, kind="decide", explicitness=expl, domain_label=dom, label=label, value=value
        )
        atype = ["meeting", "chat", "ticket"][int(_u(seed, "ntype", i) * 3)]
        sid = "sp-" + sha([seed, "noise", i])[:10]
        smap[sid] = None
        arts.append(
            Artifact(
                id="art-" + sha([seed, "noiseart", i])[:8],
                type=atype,
                title=f"{atype.title()} {day.isoformat()}",
                event_date=day,
                ingest_date=day,
                spans=[Span(id=sid, text=text)],
            )
        )
    return replace(
        scn, artifacts=arts, span_map=smap, operator={"name": "noisy", "eta": eta}, affected_vars=sorted(affected)
    )


_FUNCS: dict[str, Callable[..., Scenario]] = {
    "clean": clean,
    "stale": stale,
    "contradict_low": contradict_low,
    "contradict_cross": contradict_cross,
    "incomplete": incomplete,
    "late": late,
    "noisy": noisy,
}

DEFAULT_PARAMS: dict[str, dict] = {
    "clean": {},
    "stale": {"tau_frac": 0.5},
    "contradict_low": {"k": None},
    "contradict_cross": {"k": None},
    "incomplete": {"rho": 0.25, "mode": "omit"},
    "late": {"lam": 0.3},
    "noisy": {"eta": 1.0},
}


def apply(scn: Scenario, name: str, seed: Optional[int] = None, **params) -> Scenario:
    if name not in _FUNCS:
        raise KeyError(f"unknown operator {name}")
    p = {**DEFAULT_PARAMS[name], **params}
    return _FUNCS[name](scn, scn.seed if seed is None else seed, **p)


def standard_battery(scn: Scenario) -> list[Scenario]:
    return [apply(scn, name) for name in OPERATORS]
