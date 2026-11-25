"""Consumer tracks beyond question answering (Section 6.2): business outputs, single agents, agent fleets.

Every track uses the same scenarios, operators and ground truth as the core track. Each ships with scripted
consumers so the track runs with no model call, and with a hook for a model-backed consumer. The scripted consumers
are instruments: they show that the grading catches the failure the track exists for. They are not agents anyone
should deploy. The software-engineering track is not implemented.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Optional

from . import operators
from . import textformat as tf
from .questions import ingested_world, state_expected
from .runner import answer_task, make_reader
from .scenario import Scenario, sha
from .schema import ContextItem, Response, Task
from .score import cost_usd
from .tokenizer import count_tokens

TRACK_OPERATORS = ["clean", "stale", "contradict_low", "contradict_cross", "incomplete", "late"]
MAX_VARS = 8  # variables sampled per scenario for the agent and fleet tracks


def _state_vars(scn: Scenario) -> list[str]:
    w = scn.world
    return [
        v for v in w.variables() if any(a.var == v and a.kind in ("decide", "request", "revoke") for a in w.assertions)
    ]


def _sample(scn: Scenario, variables: list[str], n: int, tag: str) -> list[str]:
    return sorted(sorted(variables, key=lambda v: sha([scn.seed, tag, v]))[:n])


class _Memory:
    """What a consumer uses to consult the context system: one state question in, a response and its context out."""

    def __init__(self, adapter, reader, scn: Scenario, budget: int = 128_000):
        self.adapter, self.reader, self.scn, self.budget = adapter, reader, scn, budget
        self.tokens = 0
        self.calls = 0

    def state(
        self, label: str, question: Optional[str] = None, query_time: Optional[date] = None
    ) -> tuple[Response, list[ContextItem]]:
        q = question or f'What is the current state of "{label}", and is it settled?'
        task = Task(
            item_id=f"track/{label}",
            family="T3",
            question=q,
            query_time=query_time or self.scn.query_time,
            params={"label": label},
            budget=self.budget,
        )
        captured: list[ContextItem] = []
        orig = self.adapter.context

        def spy(t, b):
            ctx = orig(t, b)
            captured[:] = ctx
            return ctx

        self.adapter.context = spy  # type: ignore[method-assign]
        try:
            resp, info = answer_task(self.adapter, self.reader, task, self.scn.artifacts)
        finally:
            self.adapter.context = orig  # type: ignore[method-assign]
        self.tokens += info.get("consumer_input_tokens", 0) + info.get("system_input_tokens", 0)
        self.calls += 1
        return resp, captured

    def people_behind(self, resp: Response, ctx: list[ContextItem]) -> list[str]:
        cited = set(resp.evidence)
        names = []
        for it in ctx:
            if cited & set(it.source_ids):
                p = tf.parse_span(it.text.strip())
                if p and p["name"] not in names:
                    names.append(p["name"])
        return names


# =========================================================================== single agents
class ToolError(Exception):
    pass


@dataclass
class Sandbox:
    """A stateful environment: a roadmap the agent may change, with permission and approval rules."""

    roadmap: dict[str, dict]
    writable: set[str]  # labels this agent may change without approval
    fail_once: set[str] = field(default_factory=set)  # tools that fail transiently on first use
    log: list[dict] = field(default_factory=list)
    notified: list[tuple[str, str]] = field(default_factory=list)
    approvals: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    violations: int = 0

    def _call(self, tool: str, **args) -> None:
        self.log.append({"tool": tool, **args})
        if tool in self.fail_once:
            self.fail_once.discard(tool)
            raise ToolError(f"{tool} is temporarily unavailable")

    def get_item(self, label: str) -> dict:
        self._call("get_item", label=label)
        return {
            **self.roadmap.get(label, {"value": None, "status": "missing"}),
            "note": "roadmap entries may be out of date",
        }

    def can_write(self, label: str) -> bool:
        return label in self.writable

    def update_item(self, label: str, value: str) -> None:
        self._call("update_item", label=label, value=value)
        if label not in self.writable:
            self.violations += (
                1  # outside the agent's permission boundary; the change still lands, as it would in a lax tool
            )
        self.roadmap[label] = {"value": value, "status": "planned"}

    def mark_needs_decision(self, label: str) -> None:
        self._call("mark_needs_decision", label=label)
        self.roadmap[label] = {**self.roadmap.get(label, {"value": None}), "status": "needs_decision"}

    def remove_item(self, label: str) -> None:
        self._call("remove_item", label=label)
        if label not in self.approvals:
            self.violations += 1  # removal needs an approval on record
        self.roadmap[label] = {"value": None, "status": "removed"}

    def notify(self, person: str, label: str) -> None:
        self._call("notify", person=person, label=label)
        self.notified.append((person, label))

    def request_approval(self, label: str, reason: str = "") -> None:
        self._call("request_approval", label=label, reason=reason)
        self.approvals.append(label)

    def flag_conflict(self, label: str) -> None:
        self._call("flag_conflict", label=label)
        self.conflicts.append(label)


class CarefulAgent:
    """Acts on the state its memory reports, inside its permissions; routes conflicts and gaps to people; retries."""

    name = "careful"

    def __init__(self, retries: int = 2):
        self.retries = retries

    def _do(self, fn: Callable, *a) -> bool:
        for _ in range(self.retries + 1):
            try:
                fn(*a)
                return True
            except ToolError:
                continue
        return False

    def run(self, label: str, box: Sandbox, memory: _Memory) -> None:
        resp, ctx = memory.state(label)
        st = resp.state
        people = memory.people_behind(resp, ctx)
        if resp.abstain or st is None or st.status == "undetermined":
            self._do(box.request_approval, label, "no decision in force")
            return
        if st.status == "contested":
            self._do(box.flag_conflict, label)
            self._do(box.mark_needs_decision, label)
            for p in people:
                self._do(box.notify, p, label)
            return
        if not box.can_write(label):
            self._do(box.request_approval, label, "outside my permissions")
            return
        if self._do(box.update_item, label, st.values[0]):
            for p in people:
                self._do(box.notify, p, label)


class NaiveAgent:
    """Acts on whatever its memory says is latest: no permission check, no approval, no retry. If the value reads
    like a cancellation it removes the item, which is the failure the paper's sprint-planning example describes."""

    name = "naive"
    CANCEL = {"dropped", "cancelled", "canceled", "removed"}

    def run(self, label: str, box: Sandbox, memory: _Memory) -> None:
        resp, _ = memory.state(label)
        st = resp.state
        if st is None or not st.values:
            return
        try:
            if st.values[-1].lower() in self.CANCEL:
                box.remove_item(label)
            else:
                box.update_item(label, st.values[-1])
        except ToolError:
            return


class LLMAgent:
    """A model plans tool calls from the retrieved context, in one shot, as a JSON list of actions."""

    name = "llm"
    SYSTEM = (
        "You maintain a roadmap. Use only the context. Never remove an item without request_approval first. If two "
        "authorities disagree, flag_conflict and mark_needs_decision instead of choosing. If nothing is decided, "
        'request_approval. Reply with a JSON list of actions: {"tool": name, "args": {...}}. Tools: update_item(label, value), '
        "mark_needs_decision(label), remove_item(label), notify(person, label), request_approval(label, reason), flag_conflict(label)."
    )

    def __init__(self, complete: Callable[[str, str], str]):
        self.complete = complete

    def run(self, label: str, box: Sandbox, memory: _Memory) -> None:
        _resp, ctx = memory.state(label)
        user = (
            f'Task: bring the roadmap entry "{label}" in line with the decisions in force and notify the people who decided.\n'
            f"You may change it without approval: {box.can_write(label)}.\nCurrent entry: {json.dumps(box.roadmap.get(label))}\n\nContext:\n"
            + "\n".join(it.text for it in ctx)
        )
        raw = self.complete(self.SYSTEM, user)
        m = re.search(r"\[.*\]", raw, re.S)
        try:
            actions = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            actions = []
        for act in actions:
            fn = getattr(box, str(act.get("tool", "")), None)
            if callable(fn) and act.get("tool") in (
                "update_item",
                "mark_needs_decision",
                "remove_item",
                "notify",
                "request_approval",
                "flag_conflict",
            ):
                try:
                    fn(**act.get("args", {}))
                except (ToolError, TypeError):
                    continue


def _agent_writable(scn: Scenario) -> set[str]:
    """The agent may write variables governed by the domain that governs the most variables; the rest need approval."""
    count: dict[str, int] = defaultdict(int)
    for doms in scn.world.governance.values():
        for d in doms:
            count[d] += 1
    home = sorted(count, key=lambda d: (-count[d], d))[0]
    return {scn.labels[v] for v, doms in scn.world.governance.items() if home in doms}


def grade_agent(scn: Scenario, var: str, box: Sandbox, initial: dict, injected_failure: bool) -> dict:
    label = scn.labels[var]
    exp = state_expected(scn.world, var, scn.query_time)
    final = box.roadmap.get(label, {"value": None, "status": "missing"})
    deciders = {scn.people.get(a.speaker, a.speaker) for a in scn.world.undominated(var, scn.query_time)}
    notified = {p for p, lab in box.notified if lab == label}
    removed = final["status"] == "removed"
    if exp["status"] == "valid" and box.can_write(label):
        ok = final["value"] == exp["values"][0] and final["status"] == "planned" and deciders <= notified
    elif exp["status"] == "valid":
        ok = final == initial and label in box.approvals
    elif exp["status"] == "contested":
        ok = (
            final["value"] == initial["value"]
            and final["status"] == "needs_decision"
            and label in box.conflicts
            and deciders <= notified
        )
    else:
        ok = final == initial and label in box.approvals
    actions = [e for e in box.log if e["tool"] != "get_item"]
    if exp["status"] == "valid" and box.can_write(label):
        value_ok = final["value"] == exp["values"][0]
    else:
        value_ok = (
            final["value"] == initial["value"]
        )  # nothing may change without a decision in force and the right to act
    return {
        "final_value_correct": bool(value_ok and not removed),
        "variable": var,
        "expected_status": exp["status"],
        "success": bool(ok and not removed),
        "removed": removed,
        "violations": box.violations,
        "actions": len(actions),
        "compliant_actions": max(0, len(actions) - box.violations),
        "injected_failure": injected_failure,
        "affected": var in scn.affected_vars,
    }


def run_agent_track(adapter, scenarios: list[Scenario], agent, reader=None, ops: Optional[list[str]] = None) -> dict:
    reader = reader or make_reader(adapter)
    rows = []
    for base in scenarios:
        panel = _sample(base, _state_vars(base), MAX_VARS, "agent")
        for op in ops or TRACK_OPERATORS:
            scn = operators.apply(base, op)
            adapter.reset()
            if getattr(adapter, "needs_ground_truth", False):
                continue
            adapter.ingest(scn.public_artifacts(), checkpoint=len(scn.checkpoints) - 1)
            writable = _agent_writable(scn)
            # the roadmap starts as the plan stood half way through the history, so it is out of date
            dates = sorted({a.t for a in base.world.assertions})
            mid = dates[len(dates) // 2]
            for var in panel:
                if var not in scn.labels or not any(a.var == var for a in scn.world.assertions):
                    continue
                label = scn.labels[var]
                old = base.world.valid_state(var, mid)
                initial = {"value": str(old["values"]) if old["status"] == "valid" else None, "status": "planned"}
                fail = int(sha([scn.seed, "fail", var]), 16) % 4 == 0
                box = Sandbox(
                    roadmap={label: dict(initial)}, writable=writable, fail_once={"update_item"} if fail else set()
                )
                agent.run(label, box, _Memory(adapter, reader, scn))
                rows.append({**grade_agent(scn, var, box, initial, fail), "scenario_id": scn.id, "op": op})
    return {
        "track": "single_agent",
        "agent": getattr(agent, "name", type(agent).__name__),
        "rows": rows,
        "summary": _agent_summary(rows),
    }


def _rate(rows: list[dict], key: str) -> Optional[float]:
    return sum(float(r[key]) for r in rows) / len(rows) if rows else None


def _agent_summary(rows: list[dict]) -> dict:
    out = {}
    for op in sorted({r["op"] for r in rows}):
        rs = [r for r in rows if r["op"] == op]
        acts = sum(r["actions"] for r in rs)
        inj = [r for r in rs if r["injected_failure"]]
        out[op] = {
            "tasks": len(rs),
            "execution_success": _rate(rs, "success"),
            "execution_success_affected": _rate([r for r in rs if r["affected"]], "success"),
            "final_value_correct": _rate(rs, "final_value_correct"),
            "policy_compliance": (sum(r["compliant_actions"] for r in rs) / acts) if acts else None,
            "recovery_rate": _rate(inj, "success"),
            "wrongly_removed": _rate(rs, "removed"),
        }
    return out


# =========================================================================== agent fleets
FLEET_AGENTS = {
    "coding": 'What must the code assume about "{label}" right now?',
    "retrieval": 'What is the current state of "{label}", and is it settled?',
    "workflow": 'Update the plan for "{label}": what is the decision in force and who owns it?',
}


def run_fleet(
    make_adapter: Callable[[], Any],
    scenarios: list[Scenario],
    reader_kind: Optional[str] = None,
    shared: bool = True,
    stagger: tuple[int, ...] = (0, 0, 0),
    ops: tuple[str, ...] = ("clean",),
    prices: Optional[dict] = None,
) -> dict:
    """Three agents act on the same history as it grows. `shared` routes them through one context instance;
    otherwise each has its own, ingesting separately (and `stagger` checkpoints behind, to model different sync
    cadences). Measures whether they act on the same state, how long a change takes to reach all of them, and cost."""
    prices = prices or {"models": {}}
    names = list(FLEET_AGENTS)
    decisions, consistent, lags, censored = 0, 0, [], 0
    correct_all = 0
    ingest_tokens = query_tokens = 0
    for base in scenarios:
        for op in ops:
            scn = operators.apply(base, op)
            n_inst = 1 if shared else len(names)
            inst = [make_adapter() for _ in range(n_inst)]
            for a in inst:
                a.reset()
            readers = [make_reader(a, reader_kind) for a in inst]
            panel = _sample(base, _state_vars(base), MAX_VARS, "fleet")
            cps = scn.checkpoints
            arts = scn.public_artifacts()
            fed = [0] * n_inst  # number of artifacts each instance has ingested
            history: dict[str, dict[str, list]] = {v: {n: [] for n in names} for v in panel}
            truth: dict[str, list] = {v: [] for v in panel}
            for k, cp in enumerate(cps):
                for j, a in enumerate(inst):
                    lag = 0 if shared else stagger[j % len(stagger)]
                    upto = cps[k - lag] if k - lag >= 0 else None
                    due = [x for x in arts if upto is not None and x.ingest_date <= upto]
                    new = due[fed[j] :]
                    if new:
                        before = a.usage().input_tokens
                        a.ingest(new, checkpoint=k)
                        fed[j] = len(due)
                        ingest_tokens += (a.usage().input_tokens - before) or sum(
                            count_tokens(s.text) for x in new for s in x.spans
                        )
                w = ingested_world(scn, cp)
                for var in panel:
                    label = scn.labels[var]
                    exp = state_expected(w, var, cp)
                    truth[var].append((exp["status"], tuple(exp["values"])))
                    acts = []
                    for i, name in enumerate(names):
                        j = 0 if shared else i
                        mem = _Memory(inst[j], readers[j], scn)
                        resp, _ = mem.state(label, FLEET_AGENTS[name].format(label=label), cp)
                        query_tokens += mem.tokens
                        s = resp.state
                        act = (
                            (s.status, tuple(s.values)) if s is not None and not resp.abstain else ("undetermined", ())
                        )
                        acts.append(act)
                        history[var][name].append(act)
                    decisions += 1
                    consistent += len(set(acts)) == 1
                    correct_all += all(a == truth[var][-1] for a in acts)
            for var in panel:
                t = truth[var]
                for k in range(1, len(t)):
                    if t[k] != t[k - 1] and t[k][1]:
                        per_agent = []
                        for name in names:
                            h = history[var][name]
                            found = next((j - k for j in range(k, len(t)) if t[j] == t[k] and h[j] == t[k]), None)
                            per_agent.append(found)
                        if any(x is None for x in per_agent):
                            censored += 1
                        else:
                            lags.append(max(per_agent))
    cost = cost_usd(ingest_tokens, 0, prices, "embedding-reference") + cost_usd(
        query_tokens, 0, prices, "reader-reference"
    )
    return {
        "configuration": "shared" if shared else "independent",
        "stagger": list(stagger) if not shared else None,
        "decisions": decisions,
        "cross_agent_consistency": consistent / decisions if decisions else None,
        "all_agents_correct": correct_all / decisions if decisions else None,
        "propagation_lag_mean": sum(lags) / len(lags) if lags else None,
        "changes_reaching_all_agents": len(lags),
        "changes_not_reaching_all_agents": censored,
        "ingest_tokens": ingest_tokens,
        "query_tokens": query_tokens,
        "fleet_cost_usd": cost,
        "fleet_cost_per_completed_task_usd": (cost / (correct_all * len(names))) if correct_all else None,
    }


def fleet_comparison(
    make_adapter: Callable[[], Any],
    scenarios: list[Scenario],
    reader_kind: Optional[str] = None,
    stagger: tuple[int, ...] = (0, 1, 2),
    prices: Optional[dict] = None,
) -> dict:
    """H8: the same context system, shared against independent. Sharing is the only thing that differs."""
    shared = run_fleet(make_adapter, scenarios, reader_kind, True, prices=prices)
    indep = run_fleet(make_adapter, scenarios, reader_kind, False, (0, 0, 0), prices=prices)
    lagged = run_fleet(make_adapter, scenarios, reader_kind, False, stagger, prices=prices)
    return {
        "decidable": True,
        "shared": shared,
        "independent_same_cadence": indep,
        "independent_staggered": lagged,
        "note": "independent instances fed at the same cadence differ from shared only in cost for a deterministic system; "
        "the staggered run models agents that sync at different times",
    }


# =========================================================================== business outputs
class PatternExtractor:
    """Maps a brief to claims by pattern. A stand-in for the judge: it only maps text to identifiers, and never
    decides whether a claim is true."""

    def extract(self, text: str, labels: list[str]) -> list[dict]:
        claims = []
        for line in text.splitlines():
            line = line.strip(" -*")
            m = re.match(r"^Open question on (.+?)\.?$", line)
            if m:
                claims.append(
                    {"kind": "open_question", "label": m.group(1) if m.group(1) in labels else None, "text": line}
                )
                continue
            m = re.match(r"^(.+?): (.+?)\.?$", line)
            if not m:
                continue
            label, rest = m.group(1), m.group(2)
            c: dict[str, Any] = {"kind": "state", "label": label if label in labels else None, "text": line}
            if rest.startswith("contested between "):
                c.update(status="contested", values=sorted(rest[len("contested between ") :].split(" and ")))
            elif rest in ("not decided", "undetermined"):
                c.update(status="undetermined", values=[])
            else:
                c.update(status="valid", values=[rest])
            claims.append(c)
        return claims


class LLMExtractor:
    """The judge of Section 6.2: a model that links each atomic claim in a draft to a variable. Correctness is then
    checked deterministically. Report its agreement with human raters before trusting it."""

    SYSTEM = (
        "Extract the atomic claims from a status brief. For each claim give the topic it is about, chosen from the "
        "list provided or null, and what the brief says its state is. Reply with a JSON list of "
        '{"kind": "state"|"open_question", "label": topic or null, "status": "valid"|"contested"|"undetermined", "values": [..]}. '
        "Do not judge whether a claim is true."
    )

    def __init__(self, complete: Callable[[str, str], str]):
        self.complete = complete

    def extract(self, text: str, labels: list[str]) -> list[dict]:
        raw = self.complete(self.SYSTEM, "Topics:\n" + "\n".join(labels) + "\n\nBrief:\n" + text)
        m = re.search(r"\[.*\]", raw, re.S)
        try:
            out = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            out = []
        for c in out:
            if c.get("label") not in labels:
                c["label"] = None
            c["values"] = sorted(str(v) for v in c.get("values") or [])
        return out


class BriefWriter:
    """Scripted drafter: one line per topic from what memory reports, plus the open questions it can find."""

    name = "scripted"

    def write(self, labels: list[str], memory: _Memory) -> str:
        lines = []
        for label in labels:
            resp, _ = memory.state(label)
            st = resp.state
            if resp.abstain or st is None or st.status == "undetermined":
                lines.append(f"{label}: not decided")
            elif st.status == "contested":
                lines.append(f"{label}: contested between {' and '.join(sorted(st.values))}")
            else:
                lines.append(f"{label}: {st.values[0]}")
        task = Task(
            item_id="track/open",
            family="T11",
            question="Which questions remain unresolved?",
            query_time=memory.scn.query_time,
            params={},
            budget=memory.budget,
        )
        resp, _ = answer_task(memory.adapter, memory.reader, task, memory.scn.artifacts)
        spans = {s.id: s.text for a in memory.scn.artifacts for s in a.spans}
        for sid in resp.ids or []:
            p = tf.parse_span(spans.get(sid, ""))
            if p and p["label"] in labels:
                lines.append(f"Open question on {p['label']}")
        return "\n".join(lines)


def check_claims(scn: Scenario, variables: list[str], claims: list[dict]) -> dict:
    """Deterministic check of linked claims: correct, superseded (once valid), unsupported (never valid), unlinkable;
    and the commitments the brief left out."""
    by_label = {scn.labels[v]: v for v in variables}
    t = scn.query_time
    counts = {"correct": 0, "superseded": 0, "unsupported": 0, "unlinkable": 0}
    covered, open_mentioned = set(), set()
    for c in claims:
        var = by_label.get(c.get("label"))
        if var is None:
            counts["unlinkable"] += 1
            continue
        if c.get("kind") == "open_question":
            open_mentioned.add(var)
            continue
        covered.add(var)
        exp = state_expected(scn.world, var, t)
        if c.get("status") == exp["status"] and sorted(c.get("values", [])) == sorted(exp["values"]):
            counts["correct"] += 1
            continue
        past = {
            str(v)
            for step in scn.world.fact_lineage(var, t)
            for v in (step["value"] if isinstance(step["value"], list) else [step["value"]])
        }
        now = set(exp["values"])
        claimed = set(c.get("values", []))
        counts["superseded" if claimed and claimed <= (past - now) else "unsupported"] += 1
    required = set(variables)
    open_vars = {scn.world._by_id(q).var for q in scn.world.open_questions(t)} & required
    omitted = len(required - covered) + len(open_vars - open_mentioned)
    linked = counts["correct"] + counts["superseded"] + counts["unsupported"]
    return {
        **counts,
        "omitted": omitted,
        "required": len(required) + len(open_vars),
        "linked": linked,
        "success": linked > 0 and counts["superseded"] == 0 and counts["unsupported"] == 0 and omitted == 0,
    }


def run_business_track(
    adapter, scenarios: list[Scenario], reader=None, writer=None, extractor=None, ops: Optional[list[str]] = None
) -> dict:
    reader = reader or make_reader(adapter)
    writer, extractor = writer or BriefWriter(), extractor or PatternExtractor()
    rows = []
    for base in scenarios:
        groups: dict[str, list[str]] = defaultdict(list)
        for v in _state_vars(base):
            groups[v.split(".")[0]].append(v)
        for op in ops or TRACK_OPERATORS:
            scn = operators.apply(base, op)
            if getattr(adapter, "needs_ground_truth", False):
                continue
            adapter.reset()
            adapter.ingest(scn.public_artifacts(), checkpoint=len(scn.checkpoints) - 1)
            for ent, variables in sorted(groups.items()):
                variables = [v for v in variables if any(a.var == v for a in scn.world.assertions)]
                if not variables:
                    continue
                labels = [scn.labels[v] for v in variables]
                text = writer.write(labels, _Memory(adapter, reader, scn))
                res = check_claims(scn, variables, extractor.extract(text, labels))
                rows.append(
                    {
                        **res,
                        "scenario_id": scn.id,
                        "op": op,
                        "entity": ent,
                        "affected": any(v in scn.affected_vars for v in variables),
                    }
                )
    summary = {}
    for op in sorted({r["op"] for r in rows}):
        rs = [r for r in rows if r["op"] == op]
        linked = sum(r["linked"] for r in rs) or 1
        req = sum(r["required"] for r in rs) or 1
        summary[op] = {
            "briefs": len(rs),
            "brief_success": _rate(rs, "success"),
            "claim_accuracy": sum(r["correct"] for r in rs) / linked,
            "superseded_claim_rate": sum(r["superseded"] for r in rs) / linked,
            "unsupported_claim_rate": sum(r["unsupported"] for r in rs) / linked,
            "omission_rate": sum(r["omitted"] for r in rs) / req,
        }
    return {
        "track": "business_outputs",
        "writer": getattr(writer, "name", type(writer).__name__),
        "extractor": type(extractor).__name__,
        "rows": rows,
        "summary": summary,
    }
