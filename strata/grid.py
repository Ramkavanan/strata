"""
STRATA grid authoring format.

A scenario can be authored as a grid: one column per source in processing
(ingestion) order, one row per fact, each cell the value that source asserted.
Header rows carry source metadata: event date, ingestion date, processing
status, source type and (optionally) speaker role. The engine converts the
grid to world-model assertions and COMPUTES the final truth column; it is
never hand-entered.

Run `strata grid data/golden/table11_grid.csv` to print fact lineage and final truth.
"""

from __future__ import annotations

import csv
from datetime import date

from .core import Assertion, WorldModel

HEADER_ROWS = {"sources", "date", "ingested", "processing status", "source type", "speaker", "role"}


def parse_date(s: str) -> date:
    return date.fromisoformat(s.strip())


def load_grid(path: str, domain: str = "team", default_role: str = "member", authority: dict | None = None):
    rows = list(csv.reader(open(path, newline="")))
    head = {r[0].strip().lower(): r[1:] for r in rows if r and r[0].strip().lower() in HEADER_ROWS}
    sources = [c.strip() for c in head["sources"]]
    ncol = len(sources)
    dates = [parse_date(c) for c in head["date"][:ncol]]
    ingested = [
        parse_date(c) if c.strip() else dates[i] for i, c in enumerate(head.get("ingested", [""] * ncol)[:ncol])
    ]
    status = [c.strip().lower() or "success" for c in head.get("processing status", [""] * ncol)[:ncol]]
    stype = [c.strip() for c in head.get("source type", [""] * ncol)[:ncol]]
    roles = [c.strip() or default_role for c in head.get("role", head.get("speaker", [""] * ncol))[:ncol]]

    fact_rows = [r for r in rows if r and r[0].strip() and r[0].strip().lower() not in HEADER_ROWS]
    facts = [r[0].strip() for r in fact_rows]
    # one speaker per source column; role per column
    speakers = {f"spk_{i}": roles[i] for i in range(ncol)}
    auth = authority or {domain: {r: 1 for r in set(roles)}}  # equal authority unless given
    gov = {f: {domain} for f in facts}
    assertions = []
    for r in fact_rows:
        fact = r[0].strip()
        for i, cell in enumerate(r[1 : ncol + 1]):
            v = cell.strip()
            if not v:
                continue
            assertions.append(
                Assertion(
                    id=f"{sources[i]}:{fact}",
                    t=dates[i],
                    speaker=f"spk_{i}",
                    domain=domain,
                    kind="decide",
                    var=fact,
                    value=v,
                    ref=(),
                    complete=True,
                )
            )
    meta = [
        {
            "source": sources[i],
            "date": dates[i],
            "ingested": ingested[i],
            "status": status[i],
            "type": stype[i],
            "role": roles[i],
        }
        for i in range(ncol)
    ]
    return WorldModel(roles=speakers, authority=auth, governance=gov, assertions=assertions), facts, meta


def fact_lineage(w: WorldModel, var: str, t: date):
    """Ordered chain of (since, value, establishing assertion ids) for var up to t."""
    return w.fact_lineage(var, t)


def as_of_ingestion(w: WorldModel, meta, cutoff: date) -> WorldModel:
    """Growth protocol by INGESTION order: keep assertions whose source was ingested by cutoff."""
    seen = {m["source"] for m in meta if m["ingested"] <= cutoff}
    from dataclasses import replace

    return replace(w, assertions=[s for s in w.assertions if s.id.split(":")[0] in seen])


def render_report(path: str) -> str:
    w, facts, meta = load_grid(path)
    now = max(m["ingested"] for m in meta)
    lines = [f"{'Fact':<28}{'Final truth':<20}Lineage (since: value <- establishing)"]
    for f in facts:
        st = w.valid_state(f, now)
        chain = " -> ".join(
            f"{c['since'].strftime('%d %b')}: {c['value']} <- {','.join(c['establishing'])}"
            for c in fact_lineage(w, f, now)
        )
        final = st["values"] if st["status"] == "valid" else f"{st['status']} {st['values']}"
        lines.append(f"{f:<28}{str(final):<20}{chain}")
    lines.append("")
    lines.append("Source order checks:")
    for m in meta:
        if m["ingested"] > m["date"]:
            lines.append(
                f"  {m['source']}: event date {m['date']} but ingested {m['ingested']} (late-arriving evidence)"
            )
        if m["status"] != "success":
            lines.append(f"  {m['source']}: processing status '{m['status']}' (partially ingested source)")
    return "\n".join(lines)
