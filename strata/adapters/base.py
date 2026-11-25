"""The system boundary. A context system implements ContextSystem; everything downstream is adapter-agnostic.

Minimum for a vendor: `ingest` and `context`. Provide source span ids in each ContextItem (native provenance);
if you cannot, the harness aligns your text to spans and flags the evidence scores as aligned.
"""

from __future__ import annotations

import importlib
import math
import re
from collections import Counter
from typing import Optional, Protocol, runtime_checkable

from ..schema import Artifact, ContextItem, Task, Usage


@runtime_checkable
class ContextSystem(Protocol):
    name: str

    def reset(self) -> None: ...
    def ingest(self, artifacts: list[Artifact], checkpoint: int = 0) -> None: ...
    def context(self, task: Task, budget: int) -> list[ContextItem]: ...
    def describe(self) -> dict: ...
    def usage(self) -> Usage: ...


class BaseAdapter:
    """Convenience base: span store, usage counters, describe(). Subclasses override ingest/context."""

    name = "base"
    event_time_support = True  # does the system accept the event time of an artifact?
    provenance = "native"  # native | aligned
    reader = "recency"  # reference reader preset used to turn context into a Response
    reader_overrides: dict = {}
    own_reader = False  # True when the system answers directly via answer()
    # Price-table key for tokens this system runs through a model at ingest (embedding or extraction); None is free.
    ingest_price_key: Optional[str] = None

    def __init__(self, **config):
        self.config = config
        self._usage = Usage()
        self.spans: list[dict] = []
        if "reader" in config:
            self.reader = config["reader"]

    def reset(self) -> None:
        self.spans = []
        self._usage = Usage()

    def ingest(self, artifacts: list[Artifact], checkpoint: int = 0) -> None:
        for a in artifacts:
            for s in a.spans:
                self.spans.append(
                    {
                        "id": s.id,
                        "text": s.text,
                        "event": a.event_date,
                        "ingest": a.ingest_date,
                        "type": a.type,
                        "art": a.id,
                        "status": a.status,
                    }
                )

    def context(self, task: Task, budget: int) -> list[ContextItem]:
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "adapter": self.name,
            "ingest_price_key": self.ingest_price_key,
            "event_time_support": self.event_time_support,
            "provenance": self.provenance,
            "reader": self.reader,
            "config": self.config,
        }

    def usage(self) -> Usage:
        return self._usage.model_copy()

    def _item(self, span: dict, score: Optional[float] = None) -> ContextItem:
        return ContextItem(text=span["text"], source_ids=[span["id"]], score=score, ingest_date=span["ingest"])


# ------------------------------------------------------------------ BM25 (no dependencies)
_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [words(d) for d in docs]
        self.avg = (sum(len(t) for t in self.toks) / len(self.toks)) if self.toks else 1.0
        df: Counter = Counter()
        for t in self.toks:
            df.update(set(t))
        n = len(self.toks)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def scores(self, query: str) -> list[float]:
        q = words(query)
        out = []
        for i, tf in enumerate(self.tf):
            dl = len(self.toks[i]) or 1
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += (
                        self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
                    )
            out.append(s)
        return out


# ------------------------------------------------------------------ registry
BUILTIN = {
    "no_context": "strata.adapters.baselines:NoContext",
    "long_context": "strata.adapters.baselines:LongContext",
    "bm25": "strata.adapters.baselines:BM25Retrieval",
    "oracle": "strata.adapters.oracle:OracleAdapter",
    "oracle_context": "strata.adapters.oracle:OracleContext",
    "reference": "strata.adapters.reference:ReferenceSystem",
    "vector": "strata.adapters.baselines:VectorRetrieval",
    "vector_rerank": "strata.adapters.baselines:VectorRerank",
    "graph": "strata.adapters.baselines:GraphRetrieval",
    "agentic": "strata.adapters.baselines:AgenticRetrieval",
    "http": "strata.adapters.http:HTTPAdapter",
}


def get_adapter(name: str, **config) -> ContextSystem:
    """Resolve a built-in name or a 'package.module:ClassName' path to an adapter instance."""
    path = BUILTIN.get(name, name)
    if ":" not in path:
        raise KeyError(f"unknown adapter {name!r}; built-ins: {sorted(BUILTIN)}; or pass 'module:Class'")
    mod, cls = path.split(":")
    return getattr(importlib.import_module(mod), cls)(**config)
