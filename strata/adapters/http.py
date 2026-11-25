"""Generic HTTP adapter: evaluate any context engine that exposes an ingest endpoint and a query endpoint, from a
JSON config and no Python. Secrets come from the environment: a header value "${MY_KEY}" is replaced at start.

Config (every key but `ingest` and `query` is optional):
{
  "name": "my-engine",
  "headers": {"Authorization": "Bearer ${MY_ENGINE_API_KEY}"},
  "reset":  {"url": "https://api.example.com/v1/workspaces/{session}", "method": "DELETE"},
  "ingest": {"url": "https://api.example.com/v1/documents", "method": "POST", "per": "artifact",
             "body": {"workspace": "{session}", "id": "{id}", "title": "{title}", "text": "{text}", "mime": "{mime}",
                      "occurred_at": "{event_date}", "received_at": "{ingest_date}", "units": "{unit_ids}"}},
  "query":  {"url": "https://api.example.com/v1/query", "method": "POST",
             "body": {"workspace": "{session}", "question": "{question}", "as_of": "{query_time}", "max_tokens": "{budget}"}},
  "results": {"items": "results", "text": "text", "ids": "metadata.unit_ids", "score": "score", "when": "metadata.occurred_at"},
  "event_time_support": true, "settle_seconds": 0, "reader": "llm"
}
`per` is "artifact" (one call per artifact, text in native file form) or "unit" (one call per span, `{text}` is the
span and `{unit_ids}` its id). `results.ids` may point at a string or a list; ids that are STRATA unit ids make
provenance native, otherwise the harness aligns text to spans and marks the evidence scores as aligned.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Optional

from ..schema import ContextItem, Task
from .base import BaseAdapter

_ENV = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _env(v: Any) -> Any:
    if isinstance(v, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), ""), v)
    if isinstance(v, dict):
        return {k: _env(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_env(x) for x in v]
    return v


def _fill(template: Any, values: dict[str, Any]) -> Any:
    """Substitute {placeholders}; a value that is exactly one placeholder keeps the placeholder's type."""
    if isinstance(template, str):
        m = re.fullmatch(r"\{([a-z_]+)\}", template)
        if m and m.group(1) in values:
            return values[m.group(1)]
        return re.sub(r"\{([a-z_]+)\}", lambda mm: str(values.get(mm.group(1), mm.group(0))), template)
    if isinstance(template, dict):
        return {k: _fill(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [_fill(v, values) for v in template]
    return template


def _path(obj: Any, path: Optional[str]) -> Any:
    if not path:
        return obj
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        elif isinstance(obj, list) and part.isdigit():
            obj = obj[int(part)]
        else:
            return None
    return obj


class HTTPAdapter(BaseAdapter):
    name = "http"
    reader = "llm"

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[dict] = None,
        retries: int = 3,
        timeout: float = 120,
        **kw,
    ):
        cfg = config or json.load(open(config_path))  # type: ignore[arg-type]
        self.cfg = _env(cfg)
        super().__init__(config_path=config_path, **kw)
        self.name = self.cfg.get("name", "http")
        self.reader = self.cfg.get("reader", "llm")
        self.event_time_support = bool(self.cfg.get("event_time_support", True))
        self.provenance = "native" if self.cfg.get("results", {}).get("ids") else "aligned"
        self.retries, self.timeout = retries, timeout
        self.session = uuid.uuid4().hex[:12]

    # ---- transport
    def _call(self, spec: dict, values: dict[str, Any]) -> Any:
        url = _fill(spec["url"], values)
        method = spec.get("method", "POST").upper()
        body = _fill(spec.get("body"), values) if spec.get("body") is not None else None
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json", **self.cfg.get("headers", {}), **spec.get("headers", {})}
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    self._usage.tool_calls += 1
                    raw = r.read()
                    return json.loads(raw) if raw.strip() else None
            except urllib.error.HTTPError as e:
                last = e
                if e.code < 500 and e.code != 429:
                    raise
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
            self._usage.retries += 1
            time.sleep(min(30, 2**attempt))
        raise RuntimeError(f"{method} {url} failed after {self.retries + 1} attempts: {last}")

    def _values(self, **extra) -> dict[str, Any]:
        return {"session": self.session, **extra}

    # ---- protocol
    def reset(self) -> None:
        super().reset()
        if self.cfg.get("reset"):
            try:
                self._call(self.cfg["reset"], self._values())
            except Exception:
                pass
        self.session = uuid.uuid4().hex[:12]

    def ingest(self, artifacts, checkpoint: int = 0) -> None:
        super().ingest(artifacts, checkpoint)
        spec = self.cfg["ingest"]
        for a in artifacts:
            base = {
                "id": a.id,
                "title": a.title,
                "type": a.type,
                "mime": a.mime,
                "filename": a.filename,
                "event_date": a.event_date.isoformat(),
                "ingest_date": a.ingest_date.isoformat(),
                "status": a.status,
                "checkpoint": checkpoint,
            }
            if spec.get("per", "artifact") == "unit":
                for s in a.spans:
                    self._call(spec, self._values(**base, text=s.text, unit_ids=[s.id]))
            else:
                self._call(spec, self._values(**base, text=a.to_file_text(), unit_ids=[s.id for s in a.spans]))
        time.sleep(float(self.cfg.get("settle_seconds", 0)))

    def context(self, task: Task, budget: int) -> list[ContextItem]:
        res = self.cfg.get("results", {})
        out = self._call(
            self.cfg["query"],
            self._values(
                question=task.question,
                query_time=task.query_time.isoformat(),
                budget=budget,
                family=task.family,
                label=task.params.get("label", ""),
            ),
        )
        items = _path(out, res.get("items")) or []
        if isinstance(items, dict):
            items = [items]
        known = {s["id"] for s in self.spans}
        ctx = []
        for it in items:
            text = _path(it, res.get("text", "text"))
            if text is None:
                text = json.dumps(it)
            ids = _path(it, res.get("ids"))
            ids = [ids] if isinstance(ids, str) else list(ids or [])
            ids = [i for i in ids if i in known]
            when = _path(it, res.get("when"))
            prefix = f"{when} " if when and res.get("prefix_when", True) else ""
            ctx.append(ContextItem(text=f"{prefix}{text}", source_ids=ids, score=_path(it, res.get("score"))))
        return ctx

    def describe(self) -> dict:
        d = super().describe()
        d.update(
            {
                "adapter": self.name,
                "endpoint": self.cfg["query"]["url"],
                "config": {k: v for k, v in self.cfg.items() if k != "headers"},
            }
        )
        return d
