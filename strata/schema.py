"""Appendix C records as pydantic models (scenario artifacts, tasks, responses, results).

Hidden fields (expected answers, span-to-assertion map) live in Item.expected and
Scenario.span_map and are never passed to a system under test; see Task.
"""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

FAMILIES = ["T1", "T2", "T3", "T4", "T5", "T5b", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13"]
CAPABILITY = {
    "T1": "baseline",
    "T2": "state",
    "T3": "state",
    "T4": "state",
    "T5": "state",
    "T5b": "state",
    "T6": "decision",
    "T7": "decision",
    "T8": "requirement",
    "T9": "requirement",
    "T10": "open_question",
    "T11": "open_question",
    "T12": "abstention",
    "T13": "staleness",
}
STATE_FAMILIES = {"T2", "T3", "T5"}

Status = Literal["valid", "contested", "undetermined"]


class Span(BaseModel):
    id: str
    text: str


class Artifact(BaseModel):
    """What a system under test ingests. Contains no hidden ground truth."""

    id: str
    type: str
    title: str
    event_date: date  # when it happened (for a document: last revised)
    ingest_date: date  # when it became available to the system
    status: Literal["success", "partial"] = "success"
    spans: list[Span]

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.spans)

    # native file form of each artifact type: what a real ingestion layer would receive
    FORMATS: ClassVar[dict[str, tuple[str, str]]] = {
        "meeting": ("txt", "text/plain"),  # transcript with speaker turns
        "customer_call": ("txt", "text/plain"),
        "chat": ("jsonl", "application/x-ndjson"),  # one message per line, as chat exports are
        "ticket": ("csv", "text/csv"),  # one comment per row
        "document": ("md", "text/markdown"),
        "release_note": ("md", "text/markdown"),
        "org_chart": ("csv", "text/csv"),  # people and roles; authority and governance as rows too
        "source_code": ("py", "text/x-python"),
    }

    @property
    def filename(self) -> str:
        ext = self.FORMATS.get(self.type, ("txt", "text/plain"))[0]
        if self.type == "source_code":
            return self.title
        return f"{self.event_date.isoformat()}_{self.type}_{self.id}.{ext}"

    @property
    def mime(self) -> str:
        return self.FORMATS.get(self.type, ("txt", "text/plain"))[1]

    def to_file_text(self) -> str:
        """The artifact in its native format. Spans keep their order; every span is one unit (line, row or message)."""
        import csv as _csv
        import io
        import json as _json

        head = f"{self.title} | {self.event_date.isoformat()} | received {self.ingest_date.isoformat()}"
        if self.status == "partial":
            head += " | PARTIAL: processing stopped early"
        if self.type == "chat":
            return (
                "\n".join(
                    _json.dumps({"id": s.id, "ts": self.event_date.isoformat(), "text": s.text}) for s in self.spans
                )
                + "\n"
            )
        if self.type == "ticket":
            out = io.StringIO()
            w = _csv.writer(out)
            w.writerow(["ticket", "comment_id", "date", "comment"])
            for s in self.spans:
                w.writerow([self.id, s.id, self.event_date.isoformat(), s.text])
            return out.getvalue()
        if self.type == "org_chart":
            out = io.StringIO()
            w = _csv.writer(out)
            w.writerow(["record_id", "kind", "text"])
            for s in self.spans:
                kind = (
                    "person"
                    if s.text.startswith("Team member")
                    else ("authority" if s.text.startswith("Authority") else "governance")
                )
                w.writerow([s.id, kind, s.text])
            return out.getvalue()
        if self.type == "source_code":
            return self.text
        if self.type in ("document", "release_note"):
            return f"# {head}\n\n" + "\n\n".join(s.text for s in self.spans) + "\n"
        return f"{head}\n\n" + "\n".join(s.text for s in self.spans) + "\n"


class Task(BaseModel):
    """The only item view a system under test sees."""

    item_id: str
    family: str
    question: str
    query_time: date
    params: dict[str, Any] = Field(default_factory=dict)
    budget: int


class Item(BaseModel):
    id: str
    scenario_id: str
    render_id: str
    family: str
    question: str
    query_time: date
    params: dict[str, Any] = Field(default_factory=dict)
    variable: Optional[str] = None
    expected: dict[str, Any]
    affected: bool = False
    difficulty: dict[str, Any] = Field(default_factory=dict)
    explicitness: Optional[str] = None

    def task(self, budget: int) -> Task:
        return Task(
            item_id=self.id,
            family=self.family,
            question=self.question,
            query_time=self.query_time,
            params=self.params,
            budget=budget,
        )


class ContextItem(BaseModel):
    text: str
    source_ids: list[str] = Field(default_factory=list)  # span ids this item derives from
    score: Optional[float] = None
    ingest_date: Optional[date] = None  # arrival time of the source, if the system tracks it


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    retries: int = 0
    latency_ms: float = 0.0


class StateClaim(BaseModel):
    values: list[str] = Field(default_factory=list)
    status: Status = "undetermined"
    since: Optional[date] = None


class ChainStep(BaseModel):
    value: Any = None
    since: Optional[date] = None
    status: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)


class Response(BaseModel):
    item_id: str
    answer: str = ""
    state: Optional[StateClaim] = None
    evidence: list[str] = Field(default_factory=list)
    abstain: bool = False
    staleness_flag: bool = False
    chain: Optional[list[ChainStep]] = None  # T5b fact lineage
    lineage: Optional[dict[str, Any]] = None  # T4/T6: decider, reasons, trigger, discussion, follow_ups, status
    ids: Optional[list[str]] = None  # T7 decisions not in force, T8 request, T11 open questions (span ids)
    drifted: Optional[bool] = None  # T9
    resolved: Optional[bool] = None  # T10
    resolver: Optional[str] = None  # T10 resolving answer span id
    attribution: Literal["native", "aligned"] = "native"
    usage: Usage = Field(default_factory=Usage)


class RunManifest(BaseModel):
    run_type: str  # scenarios | ingests | evaluations | submissions
    run_number: int
    timestamp: str
    schema_version: str = SCHEMA_VERSION
    harness_version: str = ""
    git_sha: str = ""
    parent: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    hashes: dict[str, Any] = Field(default_factory=dict)


def export_json_schemas(directory: str) -> list[str]:
    import json
    import os

    os.makedirs(directory, exist_ok=True)
    written = []
    for model in (Artifact, Task, Item, ContextItem, Response, RunManifest):
        path = os.path.join(directory, f"{model.__name__.lower()}.schema.json")
        with open(path, "w") as f:
            json.dump(model.model_json_schema(), f, indent=2, sort_keys=True)
            f.write("\n")
        written.append(path)
    return written
