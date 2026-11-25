"""
STRATA engine: organizational world-model semantics (specification of record).

Implements Section 3 of the paper (valid state, supersession, establishing
sets, lineage status, drift, open questions) and the perturbation operators
of Section 5 (stale, contradict-low, contradict-cross, incomplete).

Pure Python, no dependencies. Moved from the Zenodo reference implementation
(10.5281/zenodo.23469830) without change to the semantics; the paper's examples
live in strata/examples.py and are asserted in tests/test_core_paper_examples.py.
Additions: trigger/discussion fields, fact and decision lineage, reversal listing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

BOTTOM = None  # authority ⊥


@dataclass(frozen=True)
class Assertion:
    id: str
    t: date
    speaker: str
    domain: Optional[str]
    kind: str  # decide | revoke | report | request | implement | ask | answer
    var: str
    value: object = None
    ref: Tuple[str, ...] = ()
    complete: bool = True  # for 'answer' assertions
    trigger: Tuple[str, ...] = ()  # event ids that triggered a decision (lineage, T6)
    discussion: Tuple[str, ...] = ()  # event ids of the discussion that preceded it


@dataclass
class WorldModel:
    roles: Dict[str, str]  # person -> role
    authority: Dict[str, Dict[str, int]]  # domain -> role -> level
    governance: Dict[str, Set[str]]  # variable -> governing domains
    assertions: List[Assertion] = field(default_factory=list)

    # ---------- authority and supersession (Section 3.2) ----------
    def auth(self, s: Assertion) -> Optional[int]:
        """Authority of a decide/revoke/request assertion on its own variable."""
        if s.domain is None or s.domain not in self.governance.get(s.var, set()):
            return BOTTOM
        role = self.roles.get(s.speaker)
        return self.authority.get(s.domain, {}).get(role, BOTTOM)

    def comparable(self, a: Assertion, b: Assertion) -> bool:
        return a.domain == b.domain

    def supersedes(self, s2: Assertion, s1: Assertion) -> bool:
        """s2 ▷ s1 : same variable, same domain, higher authority, or equal authority and later."""
        if s1.var != s2.var or not self.comparable(s1, s2):
            return False
        a1, a2 = self.auth(s1), self.auth(s2)
        if a1 is BOTTOM or a2 is BOTTOM:
            return False
        return a2 > a1 or (a2 == a1 and s2.t > s1.t)

    def _state_setting(self, var: str, t: date) -> List[Assertion]:
        """decide (and request, for requirement variables) assertions on var at or before t,
        with authority, not removed by a valid revocation at or before t."""
        cands = [
            s
            for s in self.assertions
            if s.var == var and s.kind in ("decide", "request") and s.t <= t and self.auth(s) is not BOTTOM
        ]
        revokes = [r for r in self.assertions if r.kind == "revoke" and r.t <= t]
        alive = []
        for s in cands:
            removed = any(
                s.id in r.ref
                and self.comparable(r, s)
                and self.auth(replace(r, var=s.var)) is not BOTTOM
                and self.auth(replace(r, var=s.var)) >= self.auth(s)
                for r in revokes
            )
            if not removed:
                alive.append(s)
        return alive

    def undominated(self, var: str, t: date) -> List[Assertion]:
        S = self._state_setting(var, t)
        return [s for s in S if not any(self.supersedes(s2, s) for s2 in S if s2 is not s)]

    def valid_state(self, var: str, t: date) -> dict:
        """Returns status (valid|contested|undetermined), values, since, establishing set."""
        U = self.undominated(var, t)
        if not U:
            ever = any(s.var == var and s.kind in ("decide", "request") and s.t <= t for s in self.assertions)
            return {
                "status": "undetermined",
                "values": [],
                "since": None,
                "establishing": [],
                "reason": "revoked" if ever else "never_decided",
            }
        values = sorted({str(s.value) for s in U})
        reasons = [r for s in U for r in s.ref if self._by_id(r) is not None]
        est = sorted({s.id for s in U} | set(reasons))
        status = "valid" if len(values) == 1 else "contested"
        return {
            "status": status,
            "values": values if status == "contested" else values[0],
            "since": max(s.t for s in U),
            "establishing": est,
        }

    def _by_id(self, aid: str) -> Optional[Assertion]:
        return next((s for s in self.assertions if s.id == aid), None)

    # ---------- lineage, drift, open questions (Section 3.3) ----------
    def decision_status(self, decision_id: str, t: date, actions: Set[str] = frozenset()) -> str:
        d = self._by_id(decision_id)
        if d is None or d.t > t:
            return "absent"
        if d not in self._state_setting(d.var, t):
            return "revoked"
        if any(self.supersedes(s2, d) for s2 in self._state_setting(d.var, t)):
            return "superseded"
        followed = any(
            a.kind != "revoke" and decision_id in a.ref and a.t <= t and a.id in actions for a in self.assertions
        )
        return "in_force" if followed or not actions else "unactioned"

    def drifted(self, spec_var: str, t: date) -> Optional[bool]:
        spec = self.valid_state(spec_var, t)
        impls = [s for s in self.assertions if s.kind == "implement" and s.var == spec_var and s.t <= t]
        if spec["status"] != "valid" or not impls:
            return None
        latest = max(impls, key=lambda s: s.t)
        return str(latest.value) != str(spec["values"])

    def open_questions(self, t: date) -> List[str]:
        asks = [s for s in self.assertions if s.kind == "ask" and s.t <= t]
        out = []
        for q in asks:
            resolved = any(a.kind == "answer" and q.id in a.ref and a.complete and a.t <= t for a in self.assertions)
            if not resolved:
                out.append(q.id)
        return out

    # ---------- lineage (Section 3.3) ----------
    def fact_lineage(self, var: str, t: date) -> List[dict]:
        """F(x,t): ordered chain of valid states with the time each took effect."""
        times = sorted({s.t for s in self.assertions if s.var == var and s.t <= t})
        chain: List[dict] = []
        last = None
        for ti in times:
            st = self.valid_state(var, ti)
            key = (st["status"], str(st["values"]))
            if not chain and st["status"] == "undetermined":
                continue  # lineage starts at the first state that is not "nothing decided yet"
            if key != last:
                chain.append(
                    {"since": ti, "status": st["status"], "value": st["values"], "establishing": st["establishing"]}
                )
                last = key
        return chain

    def follow_ups(self, decision_id: str, t: date) -> List[str]:
        return sorted(
            a.id
            for a in self.assertions
            if decision_id in a.ref
            and a.t <= t
            and a.kind in ("implement", "request", "report", "decide")
            and a.id != decision_id
        )

    def decision_lineage(self, decision_id: str, t: date) -> Optional[dict]:
        """Lambda(d): trigger, discussion, decider, domain, reasons, follow-ups, status."""
        d = self._by_id(decision_id)
        if d is None or d.t > t:
            return None
        fu = [a for a in self.follow_ups(decision_id, t) if (self._by_id(a) or d).kind != "decide"]
        status = self.decision_status(decision_id, t)
        if status == "in_force" and not fu:
            status = "unactioned"
        return {
            "decision": decision_id,
            "trigger": sorted(d.trigger),
            "discussion": sorted(d.discussion),
            "decider": d.speaker,
            "domain": d.domain,
            "reasons": sorted(d.ref),
            "follow_ups": fu,
            "status": status,
        }

    def decisions(self, t: date) -> List[Assertion]:
        return [
            s for s in self.assertions if s.kind in ("decide", "request") and s.t <= t and self.auth(s) is not BOTTOM
        ]

    def not_in_force(self, t: date, before: Optional[date] = None) -> List[str]:
        """T7: decisions (made before `before`) that took force when made and are superseded or revoked at t.

        A decision that was already dominated when it was made (a later statement by a lower authority)
        never took force, so it is not listed as "no longer in force"."""
        out = []
        for s in self.decisions(t):
            if before is not None and s.t >= before:
                continue
            if s not in self.undominated(s.var, s.t):
                continue
            if self.decision_status(s.id, t) in ("superseded", "revoked"):
                out.append(s.id)
        return sorted(out)

    def variables(self) -> List[str]:
        seen, out = set(), []
        for s in self.assertions:
            if s.var not in seen:
                seen.add(s.var)
                out.append(s.var)
        return out


# ---------- perturbation operators (Section 5.1) ----------
def stale(w: WorldModel, tau: date) -> WorldModel:
    return replace(w, assertions=[s for s in w.assertions if s.t <= tau])


def contradict_low(w: WorldModel, injected: List[Assertion]) -> WorldModel:
    """Inject later decide assertions in the same domain by lower-authority roles."""
    for s in injected:
        assert s.kind == "decide"
    return replace(w, assertions=w.assertions + injected)


def contradict_cross(w: WorldModel, var: str, new_domain: str, injected: List[Assertion]) -> WorldModel:
    gov = {k: set(v) for k, v in w.governance.items()}
    gov[var] = gov.get(var, set()) | {new_domain}
    return replace(w, governance=gov, assertions=w.assertions + injected)


def incomplete(w: WorldModel, remove_ids: Set[str], keep_indirect: List[Assertion] = ()) -> WorldModel:
    kept = [s for s in w.assertions if s.id not in remove_ids]
    return replace(w, assertions=kept + list(keep_indirect))
