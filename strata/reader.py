"""Reference reader: turns a returned context into a structured Response with no model call.

It parses template spans, rebuilds a world model from what it was given and applies the engine's rules.
Switches mirror the modular reference system (Figure 4): temporal state, authority from the org chart,
lineage links, validation. With temporal and authority off it behaves like the recency heuristic that
many memory systems implement. It is a baseline and a diagnostic instrument, not a proposed product.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

from . import textformat as tf
from .core import Assertion, WorldModel
from .schema import ChainStep, ContextItem, Response, StateClaim, Task
from .templates import STALE_DAYS

_DATED = re.compile(r"^On (\d{4}-\d{2}-\d{2}),")

PRESETS = {
    "reference": dict(temporal=True, authority=True, lineage=True, validation=True, staleness=True, time_basis="event"),
    "recency": dict(
        temporal=False, authority=False, lineage=False, validation=False, staleness=False, time_basis="arrival"
    ),
}


@dataclass
class Parsed:
    sid: str
    kind: str
    d: date
    name: str
    role: str
    domain: Optional[str]
    label: str
    value: Optional[str]
    complete: bool
    target_date: Optional[date]
    refs: list
    ingest: Optional[date]


class ReferenceReader:
    def __init__(self, preset: str = "reference", **overrides):
        self.cfg = {**PRESETS[preset], **overrides}
        self.preset = preset

    def describe(self) -> dict:
        return {"reader": "reference", "preset": self.preset, **self.cfg}

    # ------------------------------------------------------------------ parsing
    def _parse_context(self, context: list[ContextItem]):
        spans: list[Parsed] = []
        authority: dict[str, dict[str, int]] = {}
        governance: dict[str, set[str]] = {}
        self._seen_dates: list[date] = []  # every dated line, assertion or not: what freshness is judged from
        for it in context:
            lines = [ln for ln in it.text.split("\n") if ln.strip()]
            ids = it.source_ids
            for i, ln in enumerate(lines):
                m = _DATED.match(ln.strip())
                if m:
                    self._seen_dates.append(date.fromisoformat(m.group(1)))
                sid = ids[i] if len(ids) == len(lines) else (ids[0] if len(ids) == 1 else f"unk-{hash(ln) & 0xFFFF:x}")
                org = tf.parse_org(ln.strip())
                if org:
                    kind, val = org
                    if kind == "authority":
                        authority[val[0]] = val[1]
                    elif kind == "governance":
                        governance[val[0]] = set(val[1])
                    continue
                p = tf.parse_span(ln.strip())
                if p:
                    spans.append(
                        Parsed(
                            sid,
                            p["kind"],
                            p["date"],
                            p["name"],
                            p["role"],
                            p["domain"],
                            p["label"],
                            p["value"],
                            p["complete"],
                            p["target_date"],
                            p["refs"],
                            it.ingest_date,
                        )
                    )
        return spans, authority, governance

    def _build(self, spans: list[Parsed], authority, governance) -> tuple[WorldModel, dict[str, Parsed]]:
        use_auth = self.cfg["authority"] and bool(authority)
        roles = {}
        for s in spans:
            roles[s.name] = s.role
        if use_auth:
            auth, gov = authority, {k: set(v) for k, v in governance.items()}
            dom_of = lambda s: s.domain
        else:
            allroles = {s.role for s in spans}
            auth = {"_": {r: 1 for r in allroles}}
            gov = {s.label: {"_"} for s in spans}
            dom_of = lambda s: "_"
        idx: dict[tuple, list[str]] = {}
        idx3: dict[tuple, list[str]] = {}
        for s in spans:
            idx.setdefault((s.d, s.label), []).append(s.sid)
            idx3.setdefault((s.d, s.label, s.name), []).append(s.sid)
        asserts = []
        by = {}
        for s in spans:
            ref: list[str] = []
            trig: list[str] = []
            disc: list[str] = []
            if s.kind in ("revoke", "answer") and s.target_date:
                ref += idx.get((s.target_date, s.label), [])
            for tag, rd, rl, rn in s.refs:
                tgt = [x for x in idx3.get((rd, rl, rn), []) if x != s.sid]
                {"see": ref, "triggered by": trig, "discussed in": disc}[tag].extend(tgt)
            t = s.d
            if self.cfg["time_basis"] == "arrival" and s.ingest is not None and not self.cfg["temporal"]:
                t = s.ingest
            a = Assertion(
                s.sid,
                t,
                s.name,
                dom_of(s) if s.kind not in ("implement",) else None,
                s.kind,
                s.label,
                s.value,
                tuple(ref),
                s.complete,
                tuple(trig),
                tuple(disc),
            )
            asserts.append(a)
            by[s.sid] = s
        # org-chart governance for implement/ask/answer variables is irrelevant to authority; ensure gov exists
        for a in asserts:
            gov.setdefault(a.var, {"_"} if not use_auth else set())
        return WorldModel(roles=roles, authority=auth, governance=gov, assertions=asserts), by

    # ------------------------------------------------------------------ answering
    def answer(self, task: Task, context: list[ContextItem]) -> Response:
        spans, authority, governance = self._parse_context(context)
        w, by = self._build(spans, authority, governance)
        fam = task.family
        label = task.params.get("label")
        qt = task.query_time
        r = Response(item_id=task.item_id)
        sorted({sid for it in context for sid in it.source_ids})

        def state_resp(as_of: date) -> Response:
            if self.cfg["temporal"]:
                st = w.valid_state(label, as_of)
                est = st["establishing"]
                vals = (
                    st["values"] if st["status"] == "contested" else ([st["values"]] if st["status"] == "valid" else [])
                )
                claim = StateClaim(values=[str(v) for v in vals], status=st["status"], since=st["since"])
            else:
                cands = [a for a in w.assertions if a.var == label and a.kind in ("decide", "request") and a.t <= as_of]
                if not cands:
                    claim, est = StateClaim(status="undetermined"), []
                else:
                    last = max(cands, key=lambda a: (a.t, a.id))
                    claim, est = StateClaim(values=[str(last.value)], status="valid", since=by[last.id].d), [last.id]
            if not self.cfg["validation"]:
                est = sorted(
                    {a.id for a in w.assertions if a.var == label and a.kind in ("decide", "request", "revoke")}
                )
            return r.model_copy(
                update={"state": claim, "evidence": est, "answer": f"{claim.status}: {', '.join(claim.values)}"}
            )

        if fam == "T1":
            hit = [
                s
                for s in spans
                if s.kind == "decide"
                and s.label == label
                and s.d.isoformat() == task.params.get("date")
                and s.name == task.params.get("speaker_name")
            ]
            if not hit:
                return r.model_copy(update={"abstain": True})
            h = hit[0]
            return r.model_copy(
                update={
                    "state": StateClaim(values=[str(h.value)], status="valid", since=h.d),
                    "evidence": [h.sid],
                    "answer": str(h.value),
                }
            )
        if fam == "T12":
            known = any(s.label == label and s.kind in ("decide", "request", "implement") for s in spans)
            return r.model_copy(update={"abstain": not known})
        if fam == "T13":
            flag = False
            if self.cfg["staleness"]:
                dates = self._seen_dates or [s.d for s in spans]
                flag = bool(dates) and (qt - max(dates)).days > STALE_DAYS
            return r.model_copy(update={"staleness_flag": flag})
        if fam in ("T2", "T3", "T5"):
            as_of = date.fromisoformat(task.params["as_of"]) if task.params.get("as_of") else qt
            return state_resp(as_of)
        if fam == "T4":
            out = state_resp(qt)
            if not self.cfg["lineage"]:
                return out
            und = [a for a in w.undominated(label, qt)]
            reasons = sorted({x for a in und for x in a.ref if w._by_id(x)})
            return out.model_copy(update={"lineage": {"decider": [a.id for a in und], "reasons": reasons}})
        if fam == "T5b":
            if not self.cfg["temporal"]:
                return r.model_copy(update={"abstain": True})
            chain = w.fact_lineage(label, qt)
            steps = [
                ChainStep(
                    value=c["value"] if c["status"] != "contested" else list(c["value"]),
                    since=c["since"],
                    status=c["status"],
                    evidence=c["establishing"],
                )
                for c in chain
            ]
            return r.model_copy(
                update={"chain": steps, "evidence": sorted({e for c in chain for e in c["establishing"]})}
            )
        if not self.cfg["lineage"]:
            return r.model_copy(update={"abstain": True})
        if fam == "T6":
            cand = [
                a
                for a in w.assertions
                if a.kind == "decide"
                and a.var == label
                and a.t.isoformat() == task.params.get("date")
                and by[a.id].name == task.params.get("speaker_name")
            ]
            if not cand:
                return r.model_copy(update={"abstain": True})
            lin = w.decision_lineage(cand[0].id, qt)
            lin = {k: lin[k] for k in ("trigger", "discussion", "reasons", "follow_ups", "status")}
            lin["trigger"] = [x for x in lin["trigger"] if w._by_id(x)]
            lin["discussion"] = [x for x in lin["discussion"] if w._by_id(x)]
            lin["reasons"] = [x for x in lin["reasons"] if w._by_id(x)]
            return r.model_copy(update={"lineage": lin, "evidence": [cand[0].id]})
        if fam == "T7":
            before = date.fromisoformat(task.params["before"])
            ids = w.not_in_force(qt, before)
            return r.model_copy(update={"ids": ids, "evidence": ids})
        if fam == "T8":
            ids = [a.id for a in w.assertions if a.var == label and a.kind == "request"]
            return r.model_copy(update={"ids": ids, "evidence": ids})
        if fam == "T9":
            dr = w.drifted(label, qt)
            if dr is None:
                return r.model_copy(update={"abstain": True})
            st = w.valid_state(label, qt)
            impl = [a.id for a in w.assertions if a.var == label and a.kind == "implement"]
            return r.model_copy(update={"drifted": dr, "evidence": st["establishing"] + impl[-1:]})
        if fam == "T10":
            qs = [
                a
                for a in w.assertions
                if a.kind == "ask"
                and a.var == label
                and a.t.isoformat() == task.params.get("date")
                and by[a.id].name == task.params.get("speaker_name")
            ]
            if not qs:
                return r.model_copy(update={"abstain": True})
            q = qs[0]
            ans = [a for a in w.assertions if a.kind == "answer" and q.id in a.ref and a.complete and a.t <= qt]
            return r.model_copy(
                update={
                    "resolved": bool(ans),
                    "resolver": ans[0].id if ans else None,
                    "evidence": [a.id for a in ans] or [q.id],
                }
            )
        if fam == "T11":
            ids = w.open_questions(qt)
            return r.model_copy(update={"ids": ids, "evidence": ids})
        return r.model_copy(update={"abstain": True})
