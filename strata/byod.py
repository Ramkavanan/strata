"""Bring your own data: a team annotates its own history in the grid format (paper Appendix B) and gets a full
scenario with computed ground truth, questions and renders, which it can then run against its own system.

The grid is the annotation schema: one column per source in the order it was processed, one row per fact, each cell
the value that source asserted. Header rows carry the dates, the processing status and, optionally, the speaker's
role. Human effort goes on observation (what did this source say); truth, lineage and every operator's expected
behaviour are derived. Nothing a model wrote enters the ground truth.

Real text is not needed for the evaluation to be valid, but the renders a system ingests will be the template
renderer's sentences, not the team's own documents. To evaluate a system on the team's real documents, the grid must
also say which document each cell came from (`Source file` row); the export then copies those files into the corpus
and the hashes pin them.
"""

from __future__ import annotations

import csv
import re
from datetime import date
from pathlib import Path
from typing import Any, Optional

import yaml

HEADER_ROWS = {
    "sources",
    "date",
    "ingested",
    "processing status",
    "source type",
    "speaker",
    "role",
    "source file",
    "domain",
}
TYPE_MAP = {
    "meeting": "meeting",
    "kick-off meeting": "meeting",
    "sprint planning": "meeting",
    "architecture review": "meeting",
    "steering review": "meeting",
    "engineering sync": "meeting",
    "design doc": "document",
    "doc": "document",
    "document": "document",
    "standards": "document",
    "chat": "chat",
    "ticket": "ticket",
    "call": "customer_call",
    "release": "release_note",
    "code": "source_code",
}


def _slug(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", x.lower()).strip("_")


def grid_to_template(
    path: str,
    *,
    template_id: Optional[str] = None,
    domain: str = "team",
    domain_label: str = "the team",
    authority: Optional[dict[str, int]] = None,
    archetype: str = "bring_your_own",
    query_time: Optional[str] = None,
    render: Optional[dict] = None,
    panel_size: Optional[int] = None,
) -> dict:
    """A STRATA template from a grid CSV. Without a Role row and an authority map every speaker has equal authority and
    the grid tests time ordering, late arrival and partial ingestion only."""
    rows = list(csv.reader(open(path, newline="")))
    head = {r[0].strip().lower(): r[1:] for r in rows if r and r[0].strip().lower() in HEADER_ROWS}
    sources = [c.strip() for c in head["sources"]]
    n = len(sources)
    col = lambda key, default="": [(head.get(key, [default] * n) + [default] * n)[i].strip() for i in range(n)]
    dates = [date.fromisoformat(d) for d in col("date")]
    ingested = [date.fromisoformat(d) if d else dates[i] for i, d in enumerate(col("ingested"))]
    status = [(s or "success").lower() for s in col("processing status")]
    stype = col("source type")
    roles = col("role") if "role" in head else col("speaker")
    roles = [r or "member" for r in roles]
    files = col("source file")
    fact_rows = [r for r in rows if r and r[0].strip() and r[0].strip().lower() not in HEADER_ROWS]
    people = {f"spk_{i}": f"Speaker {i + 1}" for i in range(n)}
    role_of = {f"spk_{i}": roles[i] for i in range(n)}
    auth = authority or {r: 1 for r in sorted(set(roles))}
    for r in set(roles):
        auth.setdefault(r, 1)
    variables, assertions, overrides, partial = {}, [], {}, []
    for r in fact_rows:
        label = r[0].strip()
        var = _slug(label)
        variables[var] = {"label": label, "governed_by": [domain]}
        for i, cell in enumerate(r[1 : n + 1]):
            v = cell.strip()
            if not v:
                continue
            aid = f"{_slug(sources[i])}__{var}"
            assertions.append(
                {
                    "id": aid,
                    "t": dates[i],
                    "speaker": f"spk_{i}",
                    "domain": domain,
                    "kind": "decide",
                    "var": var,
                    "value": v,
                    "explicit": "explicit",
                    "source_type": TYPE_MAP.get(stype[i].lower(), "document") if stype[i] else None,
                }
            )
            if ingested[i] != dates[i]:
                overrides[aid] = ingested[i]
            if status[i] != "success":
                partial.append(aid)
    qt = date.fromisoformat(query_time) if query_time else max(ingested)
    tpl: dict[str, Any] = {
        "id": template_id or f"byod-{_slug(Path(path).stem)}",
        "archetype": archetype,
        "query_time": qt,
        "people": people,
        "roles": role_of,
        "domains": {domain: {"label": domain_label, "authority": auth}},
        "variables": variables,
        "assertions": assertions,
        "sources": [
            {
                "id": s,
                "date": dates[i],
                "ingested": ingested[i],
                "status": status[i],
                "type": stype[i],
                "file": files[i],
            }
            for i, s in enumerate(sources)
        ],
        "ingest_overrides": {k: v for k, v in overrides.items()},
        "partial_assertions": partial,
    }
    if render:
        tpl["render"] = dict(render)
    if panel_size:
        tpl["panel_size"] = panel_size
    return tpl


def write_template(tpl: dict, out: str) -> str:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(yaml.safe_dump(tpl, sort_keys=False, allow_unicode=True, width=120))
    return out
