"""Template renderer: world model -> artifacts (meeting, chat, ticket, document, release note, call, org chart)."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional

from . import textformat as tf
from .core import Assertion, WorldModel
from .scenario import sha
from .schema import Artifact, Span

KIND_TYPES = {
    "decide": ["meeting", "chat", "ticket"],
    "request": ["customer_call"],
    "implement": ["release_note"],
    "report": ["chat", "meeting"],
    "ask": ["chat"],
    "answer": ["chat"],
    "revoke": ["meeting", "chat"],
}
DEFAULT_EXPLICITNESS_WEIGHTS = {"explicit": 0.6, "hedged": 0.25, "diffuse": 0.15}


def sample_explicitness(
    world: WorldModel, seed: int, weights: dict[str, float] = None, forced: dict[str, str] = None
) -> dict[str, str]:
    weights = weights or DEFAULT_EXPLICITNESS_WEIGHTS
    forced = forced or {}
    out = {}
    order = ["explicit", "hedged", "diffuse"]
    for a in world.assertions:
        if a.id in forced:
            out[a.id] = forced[a.id]
            continue
        if a.kind != "decide":
            out[a.id] = "explicit"
            continue
        u = int(sha([seed, a.id, "expl"]), 16) / 16**16
        acc = 0.0
        pick = "explicit"
        for lv in order:
            acc += weights.get(lv, 0)
            if u < acc:
                pick = lv
                break
        out[a.id] = pick
    return out


def _pick(options: list[str], seed: int, key: str) -> str:
    return options[int(sha([seed, key, "type"]), 16) % len(options)]


def render_org_chart(
    world: WorldModel,
    domain_labels: dict[str, str],
    labels: dict[str, str],
    people: dict[str, str],
    seed: int,
    day: date,
) -> Artifact:
    spans: list[Span] = []

    def add(text: str, key: str):
        spans.append(Span(id="sp-" + sha([seed, "org", key])[:10], text=text))

    for person, role in sorted(world.roles.items()):
        add(tf.org_team(people.get(person, person), role), f"team:{person}")
    for dom, levels in sorted(world.authority.items()):
        add(tf.org_authority(domain_labels.get(dom, dom), levels), f"auth:{dom}")
    for var, doms in sorted(world.governance.items()):
        add(tf.org_governance(labels.get(var, var), [domain_labels.get(d, d) for d in doms]), f"gov:{var}")
    return Artifact(
        id="art-" + sha([seed, "org", day.isoformat()])[:8],
        type="org_chart",
        title="Organization chart and governance",
        event_date=day,
        ingest_date=day,
        spans=spans,
    )


FILLER_LINES = [
    "Quick update, the staging environment is back up.",
    "Can someone share the notes from yesterday?",
    "I will follow up with the vendor after lunch.",
    "Reminder that the retro is moved by half an hour.",
    "The dashboard numbers look about the same as last week.",
    "Thanks everyone, that covers what I had.",
    "I am still waiting on access to the shared drive.",
    "Let us park that and come back to it next week.",
    "The build was flaky again this morning, rerunning it now.",
    "Nothing new from my side, carrying on with the same work.",
    "Could we keep this to fifteen minutes today?",
    "I added a few comments to the draft, nothing blocking.",
    "The customer call got pushed, I will send a new invite.",
    "Heads up, I am out on Friday.",
    "We should clean up the old tickets at some point.",
    "I will pair with whoever picks that up.",
    "Docs are slightly out of date, I can tidy them.",
    "Sorry, I dropped off for a minute, what did I miss?",
    "That matches what I heard in the hallway.",
    "Let me check and get back to you on that.",
    "The intern demo went well, by the way.",
    "We are fine on capacity for this sprint.",
    "I have not had time to look at it yet.",
    "Can we get a second reviewer on that change?",
    "Lunch and learn is on Thursday this week.",
    "Good point, I will write that down.",
    "The test data needs refreshing before we try again.",
    "No objections from me.",
    "I can take notes today.",
    "The link in the invite is broken, use the one in chat.",
    "Still seeing the odd timeout in the logs, not urgent.",
    "Let us circle back once the numbers are in.",
]


def _u(seed: int, *key) -> float:
    return int(sha([seed, *key]), 16) / 16**16


def render_world(
    world: WorldModel,
    *,
    labels: dict[str, str],
    domain_labels: dict[str, str],
    people: dict[str, str],
    explicitness: dict[str, str],
    seed: int,
    start: date,
    chatter: int = 0,
    filler_sessions: int = 0,
    timeline: Optional[tuple[date, date]] = None,
) -> tuple[list[Artifact], dict[str, Optional[str]]]:
    """Render every assertion as one span; group spans by (day, type, speaker) into artifacts.

    `chatter` adds on average that many filler lines to each artifact, and `filler_sessions` adds sessions that
    carry no assertion at all, spread over `timeline`. Filler never parses as an assertion and maps to no ground
    truth; it gives artifacts the length of real transcripts so the context budget has something to cut.
    """
    span_map: dict[str, Optional[str]] = {}
    groups: dict[tuple, list[tuple[Assertion, str, str]]] = {}
    by_id = {a.id: a for a in world.assertions}
    roster = sorted(world.roles.items())

    def filler(key: str, day: date, n: int) -> list[Span]:
        out = []
        for i in range(n):
            person, role = roster[int(_u(seed, "fillspk", key, i) * len(roster))]
            line = FILLER_LINES[int(_u(seed, "fillline", key, i) * len(FILLER_LINES))]
            sid = "sp-" + sha([seed, "fill", key, i])[:10]
            span_map[sid] = None
            out.append(Span(id=sid, text=tf.render_filler(day, people.get(person, person), role, line)))
        return out

    for a in sorted(world.assertions, key=lambda x: (x.t, x.id)):
        role = world.roles.get(a.speaker, "member")
        name = people.get(a.speaker, a.speaker)
        expl = explicitness.get(a.id, "explicit")
        refs: list[tuple[str, date, str, str]] = []
        tdate = None
        if a.kind not in ("answer", "revoke"):
            for tag, ids in (("see", a.ref), ("triggered by", a.trigger), ("discussed in", a.discussion)):
                for rid in ids:
                    tgt = by_id.get(rid)
                    if tgt is not None:
                        refs.append((tag, tgt.t, labels.get(tgt.var, tgt.var), people.get(tgt.speaker, tgt.speaker)))
        if a.kind in ("answer", "revoke") and a.ref:
            tgt = by_id.get(a.ref[0])
            tdate = tgt.t if tgt else None
        text = tf.render_span(
            d=a.t,
            name=name,
            role=role,
            kind=a.kind,
            explicitness=expl,
            domain_label=domain_labels.get(a.domain, a.domain) if a.domain else None,
            label=labels.get(a.var, a.var),
            value=a.value,
            complete=a.complete,
            target_date=tdate,
            refs=refs,
        )
        atype = _pick(KIND_TYPES[a.kind], seed, a.id)
        groups.setdefault((a.t, atype, a.speaker), []).append((a, text, name))

    artifacts: list[Artifact] = []
    for (day, atype, speaker), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        aid = "art-" + sha([seed, day.isoformat(), atype, speaker])[:8]
        spans = []
        for i, (a, text, name) in enumerate(items):
            sid = "sp-" + sha([seed, a.id, "span", i])[:10]
            spans.append(Span(id=sid, text=text))
            span_map[sid] = a.id
        if chatter:
            n = int(_u(seed, "filln", aid) * (2 * chatter + 1))
            extra = filler(aid, day, n)
            spans = extra[: (n + 1) // 2] + spans + extra[(n + 1) // 2 :]
        artifacts.append(
            Artifact(
                id=aid,
                type=atype,
                title=f"{atype.replace('_', ' ').title()} {day.isoformat()}",
                event_date=day,
                ingest_date=day,
                spans=spans,
            )
        )
    if filler_sessions and world.assertions:
        t0, t1 = timeline or (min(a.t for a in world.assertions), max(a.t for a in world.assertions))
        last = max(a.t for a in world.assertions)  # a stale cut removes the sessions after it as well
        for j in range(filler_sessions):
            day = t0 + timedelta(days=int(_u(seed, "fillday", j) * max(1, (t1 - t0).days)))
            if day > last:
                continue
            aid = "art-" + sha([seed, "fillsess", j])[:8]
            artifacts.append(
                Artifact(
                    id=aid,
                    type="meeting",
                    title=f"Weekly Sync {day.isoformat()}",
                    event_date=day,
                    ingest_date=day,
                    spans=filler(aid, day, 2 * chatter + 2),
                )
            )
        artifacts.sort(key=lambda a: (a.event_date, a.id))
    # what shipped also exists as code: a config constant in a small source file, dated by the release
    for a in world.assertions:
        if a.kind != "implement" or a.value is None:
            continue
        const = re.sub(r"[^A-Z0-9]+", "_", labels.get(a.var, a.var).upper()).strip("_")
        sid = "sp-" + sha([seed, a.id, "code"])[:10]
        span_map[sid] = a.id
        code = (
            f"# {labels.get(a.var, a.var)}\n"
            f"# shipped {a.t.isoformat()} by {people.get(a.speaker, a.speaker)}\n"
            f'{const} = "{a.value}"\n'
        )
        artifacts.append(
            Artifact(
                id="art-" + sha([seed, a.id, "codefile"])[:8],
                type="source_code",
                title=f"src/config/{const.lower()}.py",
                event_date=a.t,
                ingest_date=a.t,
                spans=[Span(id=sid, text=code)],
            )
        )
    artifacts.sort(key=lambda x: (x.event_date, x.id))
    org = render_org_chart(world, domain_labels, labels, people, seed, start)
    for s_ in org.spans:
        span_map[s_.id] = None
    return [org] + artifacts, span_map


def staleness_gap_days(artifacts: list[Artifact], query_time: date) -> int:
    """Days between the query time and the latest non-org-chart artifact event date."""
    dates = [a.event_date for a in artifacts if a.type not in ("org_chart", "noise")]
    return (query_time - max(dates)).days if dates else 10**6
