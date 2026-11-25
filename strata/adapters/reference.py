"""Modular reference context system (Figure 4): evidence store, temporal state, authority, lineage, validation.

It receives exactly what every other system receives (the rendered corpus and the task) and never sees the world
model. The evidence store keeps the spans that assert something, with the topic and kind extracted from each, so
small talk never reaches the consumer. Mechanisms are switches on its reader; retrieval always leads with the
parts of the organisation chart the question needs.
"""

from __future__ import annotations

from .. import textformat as tf
from ..schema import ContextItem, Task
from ..tokenizer import count_tokens
from .base import BM25, BaseAdapter

STATE_KINDS = {"decide", "request", "revoke"}
QUESTION_KINDS = {"ask", "answer"}


class ReferenceSystem(BaseAdapter):
    name = "reference"
    reader = "reference"
    ingest_price_key = "reader-reference"  # extracting assertions from prose is a model pass over the corpus

    def __init__(
        self,
        temporal: bool = True,
        authority: bool = True,
        lineage: bool = True,
        validation: bool = True,
        staleness: bool = True,
        **config,
    ):
        super().__init__(
            temporal=temporal,
            authority=authority,
            lineage=lineage,
            validation=validation,
            staleness=staleness,
            **config,
        )
        self.reader_overrides = dict(
            temporal=temporal, authority=authority, lineage=lineage, validation=validation, staleness=staleness
        )
        self.evidence: list[dict] = []
        self._index = None
        self._seen = 0

    def reset(self) -> None:
        super().reset()
        self.evidence, self._seen = [], 0

    def ingest(self, artifacts, checkpoint: int = 0) -> None:
        super().ingest(artifacts, checkpoint)
        for s in self.spans[self._seen :]:
            self._usage.input_tokens += count_tokens(s["text"])
            if s["type"] == "org_chart":
                continue
            p = tf.parse_span(s["text"])
            if p:
                self.evidence.append({**s, "label": p["label"], "kind": p["kind"], "refs": {r[2] for r in p["refs"]}})
        self._seen = len(self.spans)
        self._index = BM25([s["text"] for s in self.evidence])

    def _org(self, labels: set[str] | None) -> list[dict]:
        out = []
        for s in self.spans:
            if s["type"] != "org_chart":
                continue
            o = tf.parse_org(s["text"])
            if o and (o[0] == "authority" or (o[0] == "governance" and (labels is None or o[1][0] in labels))):
                out.append(s)
        return out

    def context(self, task: Task, budget: int) -> list[ContextItem]:
        newest_first = lambda xs: sorted(xs, key=lambda s: (s["event"], s["id"]), reverse=True)
        label = task.params.get("label", "")
        labels: set[str] | None
        if task.family == "T13":  # the newest material of any kind, so the reader can judge freshness
            body = [s for s in self.spans if s["type"] != "org_chart"]
            return [self._item(s) for s in newest_first(body)[:3]]
        if task.family == "T7":
            pick, labels = newest_first(e for e in self.evidence if e["kind"] in STATE_KINDS), None
        elif task.family == "T11":
            pick, labels = newest_first(e for e in self.evidence if e["kind"] in QUESTION_KINDS), None
        else:
            own = [e for e in self.evidence if e["label"] == label]
            cited = {r for e in own for r in e["refs"]} - {label}
            pick = newest_first(own) + newest_first(e for e in self.evidence if e["label"] in cited)
            labels = {label} | cited
            if not own and self.evidence:  # nothing on that topic: fall back to lexical retrieval
                sc = self._index.scores(task.question)
                pick = [self.evidence[i] for i in sorted(range(len(sc)), key=lambda i: -sc[i])[:8] if sc[i] > 0]
                labels = {e["label"] for e in pick}
        # the organisation chart first, then evidence in priority order: this is the order the budget cuts from the end
        return [self._item(s) for s in self._org(labels) + pick]
