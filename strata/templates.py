"""World-model templates (YAML): loading, invariant validation, instantiation into scenarios."""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, Optional

import yaml

from .core import Assertion, WorldModel
from .render import render_world, sample_explicitness
from .scenario import Scenario, sha

STALE_DAYS = 30  # staleness floor: flag when the newest artifact is older than this at query time
NUM_CHECKPOINTS = 5
KINDS = {"decide", "revoke", "report", "request", "implement", "ask", "answer"}
NAME_POOL = [
    "Alex Morgan",
    "Sam Rivera",
    "Jordan Lee",
    "Taylor Brooks",
    "Casey Nguyen",
    "Riley Patel",
    "Avery Kim",
    "Quinn Foster",
    "Rowan Silva",
    "Devon Clarke",
    "Harper Ito",
    "Emery Zhao",
]


def _as_date(x: Any) -> date:
    return x if isinstance(x, date) else date.fromisoformat(str(x))


def load_template(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def validate_template(tpl: dict) -> list[str]:
    """Return a list of invariant violations (empty when the template is sound)."""
    errs: list[str] = []
    ids = [a["id"] for a in tpl["assertions"]]
    if len(ids) != len(set(ids)):
        errs.append("duplicate assertion ids")
    roles = tpl["roles"]
    domains = tpl["domains"]
    variables = tpl["variables"]
    external = set(tpl.get("external_refs", []))
    qt = _as_date(tpl["query_time"])
    labels = [v["label"] for v in variables.values()]
    if len(labels) != len(set(labels)):
        errs.append("variable labels must be unique")
    for lab in labels + [d["label"] for d in domains.values()]:
        if ": " in lab or '"' in lab or "?" in lab:
            errs.append(f"label {lab!r} must not contain ': ', '\"' or '?'")
    for var, v in variables.items():
        for dom in v["governed_by"]:
            if dom not in domains:
                errs.append(f"{var} governed by unknown domain {dom}")
    idset = set(ids)
    for a in tpl["assertions"]:
        aid = a["id"]
        if a["kind"] not in KINDS:
            errs.append(f"{aid}: unknown kind {a['kind']}")
        if a["speaker"] not in roles:
            errs.append(f"{aid}: speaker {a['speaker']} has no role")
        if a["var"] not in variables:
            errs.append(f"{aid}: unknown variable {a['var']}")
        if a.get("domain") and a["domain"] not in domains:
            errs.append(f"{aid}: unknown domain {a['domain']}")
        if _as_date(a["t"]) > qt:
            errs.append(f"{aid}: dated after query time")
        for r in list(a.get("ref", [])) + list(a.get("trigger", [])) + list(a.get("discussion", [])):
            if r not in idset and r not in external:
                errs.append(f"{aid}: reference {r} is neither an assertion nor an external ref")
        val = a.get("value")
        if (
            a["kind"] in ("decide", "request", "implement", "report", "ask", "answer")
            and val is not None
            and '"' in str(val)
        ):
            errs.append(f"{aid}: value must not contain a double quote")
        if a["kind"] in ("decide", "request", "implement") and val is None:
            errs.append(f"{aid}: {a['kind']} needs a value")
        if a["kind"] in ("revoke", "answer") and not a.get("ref"):
            errs.append(f"{aid}: {a['kind']} needs a ref")
    if not errs:
        w = build_world(tpl)
        t = qt
        for var in tpl.get("planted_drift", []):
            if w.drifted(var, t) is not True:
                errs.append(f"planted drift on {var} is not drifted at query time")
        for var in tpl.get("planted_contested", []):
            if w.valid_state(var, t)["status"] != "contested":
                errs.append(f"planted conflict on {var} is not contested")
        if "planted_open" in tpl and sorted(w.open_questions(t)) != sorted(tpl["planted_open"]):
            errs.append(f"open questions {w.open_questions(t)} differ from planted {tpl['planted_open']}")
        last = max(_as_date(a["t"]) for a in tpl["assertions"])
        if (qt - last).days > STALE_DAYS:
            errs.append(
                f"clean corpus ends {(qt - last).days} days before the query time; the staleness floor needs <= {STALE_DAYS}"
            )
        # a revocation must target an assertion on the same variable
        by_id = {a.id: a for a in w.assertions}
        for a in w.assertions:
            if a.kind == "revoke":
                for r in a.ref:
                    if r in by_id and by_id[r].var != a.var:
                        errs.append(f"{a.id}: revocation targets a different variable")
    return errs


def shift_days(seed: int) -> int:
    return (seed % 6) * 7  # seed 0 reproduces the authored dates


def build_world(tpl: dict, shift: int = 0) -> WorldModel:
    delta = timedelta(days=shift)
    authority = {d: dict(spec["authority"]) for d, spec in tpl["domains"].items()}
    governance = {v: set(spec["governed_by"]) for v, spec in tpl["variables"].items()}
    assertions = []
    for a in tpl["assertions"]:
        assertions.append(
            Assertion(
                id=a["id"],
                t=_as_date(a["t"]) + delta,
                speaker=a["speaker"],
                domain=a.get("domain"),
                kind=a["kind"],
                var=a["var"],
                value=a.get("value"),
                ref=tuple(a.get("ref", ())),
                complete=a.get("complete", True),
                trigger=tuple(a.get("trigger", ())),
                discussion=tuple(a.get("discussion", ())),
            )
        )
    return WorldModel(roles=dict(tpl["roles"]), authority=authority, governance=governance, assertions=assertions)


def checkpoint_dates(ingest_dates: list[date], k: int = NUM_CHECKPOINTS) -> list[date]:
    ds = sorted(set(ingest_dates))
    if len(ds) <= k:
        return ds
    out = []
    for i in range(1, k + 1):
        idx = -(-i * len(ds) // k) - 1  # ceil(i*n/k) - 1
        out.append(ds[idx])
    return sorted(set(out))


def build_scenario(tpl: dict, seed: int = 0, explicitness_weights: Optional[dict] = None) -> Scenario:
    errs = validate_template(tpl)
    if errs:
        raise ValueError("template invalid: " + "; ".join(errs))
    shift = shift_days(seed)
    world = build_world(tpl, shift)
    labels = {v: spec["label"] for v, spec in tpl["variables"].items()}
    domain_labels = {d: spec["label"] for d, spec in tpl["domains"].items()}
    people = dict(tpl.get("people", {}))
    for i, p in enumerate(sorted(world.roles)):
        if p not in people:
            people[p] = NAME_POOL[(int(sha([seed, p]), 16) + i) % len(NAME_POOL)]
    forced = {a["id"]: a["explicit"] for a in tpl["assertions"] if a.get("explicit")}
    expl = sample_explicitness(world, seed, explicitness_weights, forced)
    start = min(a.t for a in world.assertions)
    end = max(a.t for a in world.assertions)
    rcfg = dict(tpl.get("render", {}))
    chatter, filler_sessions = int(rcfg.get("chatter", 0)), int(rcfg.get("filler_sessions", 0))
    artifacts, span_map = render_world(
        world,
        labels=labels,
        domain_labels=domain_labels,
        people=people,
        explicitness=expl,
        seed=seed,
        start=start,
        chatter=chatter,
        filler_sessions=filler_sessions,
        timeline=(start, end),
    )
    qt = _as_date(tpl["query_time"]) + timedelta(days=shift)
    sid = f"{tpl['id']}-s{seed}"
    renderer: dict = {"name": "template"}
    if chatter or filler_sessions:
        renderer.update(chatter=chatter, filler_sessions=filler_sessions, timeline=[start.isoformat(), end.isoformat()])
    scn = Scenario(
        id=sid,
        template_id=tpl["id"],
        archetype=tpl["archetype"],
        seed=seed,
        world=world,
        labels=labels,
        domain_labels=domain_labels,
        people=people,
        query_time=qt,
        artifacts=artifacts,
        span_map=span_map,
        explicitness=expl,
        unanswerable=list(tpl.get("unanswerable", [])),
        checkpoints=checkpoint_dates([a.ingest_date for a in artifacts if a.type != "org_chart"]),
        renderer=renderer,
    )
    # bring-your-own-data grids carry ingestion dates and processing status per source; apply them to the artifacts
    overrides = {k: _as_date(v) for k, v in (tpl.get("ingest_overrides") or {}).items()}
    partial = set(tpl.get("partial_assertions") or [])
    if overrides or partial:
        new = []
        for art in artifacts:
            aids = {span_map.get(sp.id) for sp in art.spans} - {None}
            late = [overrides[a] + timedelta(days=shift) for a in aids if a in overrides]
            upd: dict = {}
            if late:
                upd["ingest_date"] = max(late)
            if aids & partial:
                upd["status"] = "partial"
            new.append(art.model_copy(update=upd) if upd else art)
        scn.artifacts = new
        scn.checkpoints = checkpoint_dates([a.ingest_date for a in new if a.type != "org_chart"])
    if tpl.get("panel_size"):
        from .questions import select_panel  # local import: questions imports this module

        scn.panel = select_panel(scn, int(tpl["panel_size"]))
    return scn


def template_paths(directory: str) -> list[str]:
    return sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.endswith((".yaml", ".yml")))
