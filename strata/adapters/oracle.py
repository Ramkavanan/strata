"""Oracle: the establishing set passed directly. An upper bound that separates context failures from consumer
failures. It reads the hidden item, so it is a diagnostic, not a system."""

from __future__ import annotations

from datetime import date

from ..scenario import Scenario
from ..schema import ChainStep, ContextItem, Item, Response, StateClaim, Task
from .base import BaseAdapter


class OracleAdapter(BaseAdapter):
    name = "oracle"
    own_reader = True
    needs_ground_truth = True

    def __init__(self, **config):
        super().__init__(**config)
        self.scn: Scenario | None = None
        self.items: dict[str, Item] = {}

    def attach(self, scn: Scenario, items: list[Item]) -> None:
        self.scn = scn
        self.items = {i.id: i for i in items}

    def context(self, task: Task, budget: int) -> list[ContextItem]:
        return []

    def _spans(self, ids) -> list[str]:
        m = self.scn.assertion_spans()
        return [m[i][0] for i in ids if i in m]

    def answer(self, task: Task) -> Response:
        it = self.items[task.item_id]
        e, f = it.expected, it.family
        sp = self._spans
        r = Response(item_id=it.id, evidence=sp(e.get("establishing", [])))
        if f in ("T1", "T2", "T3", "T5", "T4"):
            r.state = StateClaim(
                values=e["values"], status=e["status"], since=date.fromisoformat(e["since"]) if e.get("since") else None
            )
            if f == "T4":
                r.lineage = {"decider": sp(e["decider"]), "reasons": sp(e["reasons"])}
        elif f == "T5b":
            r.chain = [
                ChainStep(
                    value=c["value"],
                    since=date.fromisoformat(c["since"]),
                    status=c["status"],
                    evidence=sp(c["establishing"]),
                )
                for c in e["chain"]
            ]
        elif f == "T6":
            r.lineage = {
                "trigger": sp(e["trigger"]),
                "discussion": sp(e["discussion"]),
                "reasons": sp(e["reasons"]),
                "follow_ups": sp(e["follow_ups"]),
                "status": e["status"],
            }
        elif f in ("T7", "T8", "T11"):
            r.ids = sp(e["ids"])
        elif f == "T9":
            r.drifted = e["drifted"]
            r.abstain = e["drifted"] is None
        elif f == "T10":
            r.resolved = e["resolved"]
            r.resolver = sp([e["resolver"]])[0] if e["resolver"] else None
        elif f == "T12":
            r.abstain = True
        elif f == "T13":
            r.staleness_flag = e["staleness_flag"]
        return r


STATE_LIKE = {"T1", "T2", "T3", "T4", "T5"}


class OracleContext(BaseAdapter):
    """Oracle context: the establishing set, passed as context to the consumer under test (Section 6.1).

    Unlike `oracle`, this does not answer. It hands the reader perfect context, so whatever the reader still gets
    wrong is the consumer gap. For state questions the context is the establishing set; for lineage, requirement
    and question tasks it is the supporting history those tasks are about. The organisation chart is always
    included, because every deployment has one."""

    name = "oracle_context"
    needs_ground_truth = True
    reader = "reference"

    def __init__(self, **config):
        super().__init__(**config)
        self.scn: Scenario | None = None
        self.items: dict[str, Item] = {}

    def attach(self, scn: Scenario, items: list[Item]) -> None:
        self.scn = scn
        self.items = {i.id: i for i in items}

    def context(self, task: Task, budget: int) -> list[ContextItem]:
        it = self.items[task.item_id]
        e = it.expected
        ids = set(e.get("establishing", []))
        if it.family not in STATE_LIKE:
            ids |= set(e.get("support", []))
        by_span = {s["id"]: s for s in self.spans}
        amap = self.scn.assertion_spans()
        org = [s for s in self.spans if s["type"] == "org_chart"]
        if it.family == "T13":
            body = sorted((s for s in self.spans if s["type"] != "org_chart"), key=lambda s: (s["event"], s["id"]))[-3:]
        else:
            sids = [sid for aid in sorted(ids) for sid in amap.get(aid, []) if sid in by_span]
            body = sorted((by_span[s] for s in sids), key=lambda s: (s["event"], s["id"]))
        return [self._item(s) for s in org + body]
