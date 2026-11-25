"""Span text format shared by the template renderer and the reference reader.

The template renderer is a deliberately simple, parsable realisation of the world model. It exists so the
harness runs with no model calls and so the reference reader can be validated by round trip. LLM renderers
(strata.llm_render) produce freer prose; they are what realism claims rest on.
"""

from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from typing import Any, Optional

REF_RE = re.compile(r' \[(see|triggered by|discussed in) (\d{4}-\d{2}-\d{2}) "([^"]+)" by ([^\]]+)\]')
PREFIX = r"On (?P<d>\d{4}-\d{2}-\d{2}), (?P<name>.+?) \((?P<role>[^()]+)\) "
PATTERNS: list[tuple[str, str, str]] = [
    # (kind, explicitness, regex)
    ("decide", "explicit", PREFIX + r'decided in (?P<dom>.+?): "(?P<label>.+?): (?P<value>.+)\."$'),
    (
        "decide",
        "hedged",
        PREFIX
        + r'said in (?P<dom>.+?): "I think we should probably go with (?P<label>.+?): (?P<value>.+), at this point\."$',
    ),
    ("decide", "diffuse", PREFIX + r'noted in (?P<dom>.+?): "Plan updated, (?P<label>.+?) now reads (?P<value>.+)\."$'),
    ("request", "explicit", PREFIX + r'requested in (?P<dom>.+?): "(?P<label>.+?): (?P<value>.+)\."$'),
    (
        "revoke",
        "explicit",
        PREFIX + r'withdrew in (?P<dom>.+?) the (?P<tdate>\d{4}-\d{2}-\d{2}) decision on "(?P<label>.+?)"\.$',
    ),
    ("implement", "explicit", PREFIX + r'shipped in release: "(?P<label>.+?): (?P<value>.+)\."$'),
    ("report", "explicit", PREFIX + r'reported in (?P<dom>.+?): "(?P<label>.+?): (?P<value>.+)\."$'),
    (
        "report",
        "indirect",
        PREFIX + r'mentioned in (?P<dom>.+?): "after last week\'s decision, work on (?P<label>.+?) continues\."$',
    ),
    ("ask", "explicit", PREFIX + r'asked in (?P<dom>.+?): "(?P<label>.+?)\? (?P<value>.+)"$'),
    (
        "answer",
        "explicit",
        PREFIX
        + r'answered in (?P<dom>.+?) \((?P<cmp>complete|partial)\) the (?P<tdate>\d{4}-\d{2}-\d{2}) question on "(?P<label>.+?)": "(?P<value>.+)"$',
    ),
]
_COMPILED = [(k, e, re.compile(r)) for k, e, r in PATTERNS]


def render_span(
    *,
    d: date,
    name: str,
    role: str,
    kind: str,
    explicitness: str,
    domain_label: Optional[str],
    label: str,
    value: Any,
    complete: bool = True,
    target_date: Optional[date] = None,
    refs: list[tuple[str, date, str, str]] = (),
) -> str:
    pre = f"On {d.isoformat()}, {name} ({role}) "
    dom = domain_label or "release"
    if kind == "decide":
        if explicitness == "hedged":
            body = f'said in {dom}: "I think we should probably go with {label}: {value}, at this point."'
        elif explicitness == "diffuse":
            body = f'noted in {dom}: "Plan updated, {label} now reads {value}."'
        else:
            body = f'decided in {dom}: "{label}: {value}."'
    elif kind == "request":
        body = f'requested in {dom}: "{label}: {value}."'
    elif kind == "revoke":
        body = (
            f'withdrew in {dom} the {target_date.isoformat() if target_date else d.isoformat()} decision on "{label}".'
        )
    elif kind == "implement":
        body = f'shipped in release: "{label}: {value}."'
    elif kind == "report" and value is None:
        body = f'mentioned in {dom}: "after last week\'s decision, work on {label} continues."'
    elif kind == "report":
        body = f'reported in {dom}: "{label}: {value}."'
    elif kind == "ask":
        body = f'asked in {dom}: "{label}? {value}"'
    elif kind == "answer":
        body = (
            f"answered in {dom} ({'complete' if complete else 'partial'}) the "
            f'{target_date.isoformat() if target_date else d.isoformat()} question on "{label}": "{value}"'
        )
    else:
        raise ValueError(f"cannot render kind {kind}")
    suffix = "".join(f' [{t} {rd.isoformat()} "{rl}" by {rn}]' for t, rd, rl, rn in refs)
    return pre + body + suffix


@lru_cache(maxsize=400_000)
def parse_span(text: str) -> Optional[dict]:
    # cached: callers must treat the returned dict as read-only
    refs = [(t, date.fromisoformat(dt), lab, nm) for t, dt, lab, nm in REF_RE.findall(text)]
    core = REF_RE.sub("", text).strip()
    for kind, expl, rx in _COMPILED:
        m = rx.match(core)
        if m:
            g = m.groupdict()
            return {
                "kind": kind,
                "explicitness": expl,
                "date": date.fromisoformat(g["d"]),
                "name": g["name"],
                "role": g["role"],
                "domain": g.get("dom"),
                "label": g["label"],
                "value": g.get("value"),
                "complete": g.get("cmp") != "partial",
                "target_date": date.fromisoformat(g["tdate"]) if g.get("tdate") else None,
                "refs": refs,
            }
    return None


# ---- organisation chart spans ----
def org_team(name: str, role: str) -> str:
    return f"Team member {name}: role {role}."


def org_authority(domain_label: str, levels: dict[str, int]) -> str:
    return f"Authority in {domain_label}: " + "; ".join(f"{r}={l}" for r, l in sorted(levels.items())) + "."


def org_governance(label: str, domain_labels: list[str]) -> str:
    return f'Governance: "{label}" is governed by ' + ", ".join(sorted(domain_labels)) + "."


ORG_TEAM_RE = re.compile(r"^Team member (.+): role (.+)\.$")
ORG_AUTH_RE = re.compile(r"^Authority in (.+?): (.+)\.$")
ORG_GOV_RE = re.compile(r'^Governance: "(.+)" is governed by (.+)\.$')


@lru_cache(maxsize=50_000)
def parse_org(text: str) -> Optional[tuple[str, tuple]]:
    m = ORG_TEAM_RE.match(text)
    if m:
        return "team", (m.group(1), m.group(2))
    m = ORG_AUTH_RE.match(text)
    if m:
        levels = {}
        for part in m.group(2).split("; "):
            r, _, l = part.rpartition("=")
            levels[r] = int(l)
        return "authority", (m.group(1), levels)
    m = ORG_GOV_RE.match(text)
    if m:
        return "governance", (m.group(1), [x.strip() for x in m.group(2).split(", ")])
    return None


# ---- in-artifact chatter (never parses as an assertion) ----
def render_filler(d: date, name: str, role: str, line: str) -> str:
    return f'On {d.isoformat()}, {name} ({role}) said: "{line}"'
