"""Procedural world-model generator: seeded templates at the scale of a full scenario (paper Appendix A.3).

A template is assembled from motifs, each a small story the benchmark needs to contain: a reversal, a later
statement by someone with less authority, a conflict between two authority domains, a revocation, a value that
returns, a requirement that drifted, a question left open. The generator plants them, records what it planted, and
the template validator then re-derives every planted property from the world model, so a template that does not
contain what it claims is rejected.

No model is involved. Generated templates are a starting point, not a substitute for templates written by domain
experts: they share the generator's regularities. The dataset card says so.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any, Optional

from .scenario import sha

GENERATOR_VERSION = "1"

# Appendix A.3 targets. `shape_ok` enforces them; realised values are reported with every dataset.
TARGETS = {
    "history_days": (180, 275),
    "sessions": (40, 120),
    "people": (8, 15),
    "roles": (5, 8),
    "entities": (3, 5),
    "variables_per_entity": (2, 6),
    "decisions": (15, 40),
    "share_superseded_or_revoked": (0.20, 0.30),
    "requirements": (5, 10),
    "drifted_requirements": (1, 3),
    "questions": (5, 10),
    "open_share": (0.40, 0.60),
    "authority_conflicts": (2, 4),
    "cross_domain_conflicts": (1, 4),
}

NAMES = [
    "Priya Nair",
    "Arjun Mehta",
    "Dev Rao",
    "Maya Iyer",
    "Sofia Lindqvist",
    "Lena Fischer",
    "Ravi Menon",
    "Ana Costa",
    "Omar Haddad",
    "Tom Becker",
    "Kofi Mensah",
    "Hannah Ward",
    "Elena Petrova",
    "Noor Ali",
    "Ben Carter",
    "Mei Tanaka",
    "Pedro Alves",
    "Sam Okoye",
    "Iris Novak",
    "Lucas Martin",
    "Aisha Khan",
    "Jonas Weber",
    "Chloe Dubois",
    "Diego Ramos",
    "Yuki Sato",
    "Nadia Rahman",
    "Marco Bianchi",
    "Zara Ahmed",
    "Felix Braun",
    "Ingrid Olsen",
    "Tariq Aziz",
    "Grace Kim",
    "Mateo Silva",
    "Anya Sokolova",
    "Victor Hugo Lima",
    "Leila Farah",
    "Oscar Nilsson",
    "Rina Kapoor",
    "Hugo Laurent",
    "Amara Obi",
    "Stefan Kovac",
    "Mina Park",
    "Caleb Stone",
    "Farah Siddiqui",
    "Jorge Castillo",
    "Emma Lund",
]
REASONS = [
    "capacity constraints",
    "a customer escalation",
    "a vendor delay",
    "a security review finding",
    "the budget review",
    "a slipped dependency",
    "incident follow-up",
    "legal feedback",
    "new usage data",
    "a partner request",
]
QUESTIONS = [
    "who signs off on this",
    "does this apply to enterprise plans",
    "is the date firm",
    "what is the fallback if this slips",
    "has finance approved this",
    "which customers are affected",
    "is legal aware of this",
    "what does this depend on",
    "who is the escalation contact",
    "does this change the contract",
]
ANSWERS = [
    "yes it was confirmed in the review",
    "finance approved it last week",
    "only the enterprise plans are in scope",
    "the platform migration is the only dependency",
    "the contract is unchanged",
    "the regional lead is the contact",
]
PARTIAL_ANSWERS = ["still checking with the team", "will confirm next week", "it depends on the vendor review"]

ARCHETYPES: dict[str, dict[str, Any]] = {
    "b2b_product_team": {
        "short": "b2b",
        "domains": {
            "product_scope": ("product scope", {"engineer": 1, "product_manager": 2, "cto": 3}),
            "commercial": ("commercial terms", {"support_lead": 1, "sales_head": 2}),
            "architecture": ("architecture", {"engineer": 1, "tech_lead": 2, "principal_engineer": 3}),
            "requirements": ("requirements", {"customer": 1, "support_lead": 2, "product_manager": 3}),
        },
        "role_counts": {
            "engineer": (2, 3),
            "product_manager": (1, 2),
            "cto": (1, 1),
            "sales_head": (1, 1),
            "tech_lead": (1, 2),
            "principal_engineer": (1, 1),
            "support_lead": (1, 1),
            "customer": (1, 3),
        },
        "implementer": "engineer",
        "entities": [
            "Salesforce integration",
            "HubSpot integration",
            "bulk export",
            "single sign-on",
            "audit log",
            "usage dashboard",
            "mobile app",
            "webhooks",
            "billing portal",
            "data residency option",
            "API gateway",
            "notification centre",
        ],
        "attributes": [
            ("target", "release target", "product_scope", None, ["Q1", "Q2", "Q3", "Q4", "next half"]),
            ("priority", "priority", "product_scope", "commercial", ["P0", "P1", "P2", "deferred"]),
            (
                "owner",
                "owning team",
                "product_scope",
                None,
                ["platform team", "growth team", "integrations team", "core team"],
            ),
            (
                "tier",
                "pricing tier",
                "commercial",
                "product_scope",
                ["included in all plans", "enterprise only", "paid add-on", "free during beta"],
            ),
            (
                "rollout",
                "rollout plan",
                "product_scope",
                "commercial",
                ["beta customers first", "general availability", "internal only", "staged by region"],
            ),
            (
                "arch",
                "architecture rule",
                "architecture",
                None,
                [
                    "async job service only",
                    "sync allowed under 1k rows",
                    "event bus only",
                    "direct database access forbidden",
                ],
            ),
        ],
        "requirements": [
            (
                "spec",
                "specification",
                [
                    "all records as CSV",
                    "CSV with a 50k row cap",
                    "CSV with a 100k row cap",
                    "CSV and JSON with a 50k row cap",
                ],
            ),
            (
                "sla",
                "service level requirement",
                [
                    "99.9 percent uptime",
                    "99.5 percent uptime",
                    "99.9 percent uptime in business hours",
                    "99 percent uptime",
                ],
            ),
            (
                "retention",
                "data retention requirement",
                ["keep 24 months", "keep 12 months", "keep 6 months", "keep 36 months"],
            ),
        ],
        "unanswerable": [
            "general availability approval",
            "security audit sign-off",
            "pricing committee approval",
            "legal clearance",
        ],
    },
    "engineering_incident": {
        "short": "inc",
        "domains": {
            "operations": ("operations", {"sre": 1, "incident_lead": 2, "vp_engineering": 3}),
            "architecture": ("architecture", {"engineer": 1, "tech_lead": 2, "principal_engineer": 3}),
            "customer_comms": ("customer communications", {"incident_lead": 1, "support_lead": 2, "vp_engineering": 3}),
            "requirements": ("requirements", {"customer": 1, "support_lead": 2, "tech_lead": 3}),
        },
        "role_counts": {
            "sre": (2, 3),
            "incident_lead": (1, 2),
            "vp_engineering": (1, 1),
            "engineer": (1, 2),
            "tech_lead": (1, 2),
            "principal_engineer": (1, 1),
            "support_lead": (1, 1),
            "customer": (1, 3),
        },
        "implementer": "engineer",
        "entities": [
            "payments service",
            "checkout API",
            "search cluster",
            "notification pipeline",
            "auth gateway",
            "data warehouse sync",
            "image CDN",
            "billing ledger",
            "session store",
            "feature flag service",
        ],
        "attributes": [
            (
                "rollback",
                "rollback plan",
                "operations",
                None,
                [
                    "roll back the release",
                    "roll forward with a hotfix",
                    "disable the feature flag",
                    "fail over to the standby",
                ],
            ),
            (
                "freeze",
                "deploy freeze",
                "operations",
                "customer_comms",
                ["freeze for one week", "freeze for two weeks", "no freeze", "freeze lifted for hotfixes only"],
            ),
            (
                "oncall",
                "on-call owner",
                "operations",
                None,
                ["payments squad", "platform squad", "reliability squad", "data squad"],
            ),
            (
                "status",
                "status page wording",
                "customer_comms",
                "operations",
                ["investigating", "identified", "monitoring", "resolved"],
            ),
            (
                "retry",
                "retry policy",
                "architecture",
                None,
                [
                    "exponential backoff with 3 retries",
                    "linear retry with 5 retries",
                    "no automatic retry",
                    "circuit breaker after 2 failures",
                ],
            ),
            (
                "postmortem",
                "postmortem deadline",
                "operations",
                None,
                ["within 5 working days", "within 10 working days", "end of the month", "before the next release"],
            ),
        ],
        "requirements": [
            (
                "remediation",
                "remediation requirement",
                [
                    "alert within 5 minutes",
                    "alert within 2 minutes",
                    "alert within 1 minute",
                    "alert within 10 minutes",
                ],
            ),
            (
                "credit",
                "customer credit requirement",
                [
                    "credit for all affected customers",
                    "credit for enterprise customers only",
                    "credit on request",
                    "no credit",
                ],
            ),
            (
                "rto",
                "recovery time requirement",
                [
                    "recover within 4 hours",
                    "recover within 1 hour",
                    "recover within 30 minutes",
                    "recover within 8 hours",
                ],
            ),
        ],
        "unanswerable": [
            "root cause sign-off",
            "regulator notification approval",
            "insurance claim approval",
            "vendor penalty approval",
        ],
    },
    "customer_success_team": {
        "short": "cs",
        "domains": {
            "account_terms": ("account terms", {"account_manager": 1, "cs_head": 2, "cro": 3}),
            "renewal": ("renewal strategy", {"csm": 1, "account_manager": 2, "cro": 3}),
            "legal_review": ("legal review", {"account_manager": 1, "legal_counsel": 2}),
            "requirements": ("requirements", {"customer": 1, "csm": 2, "cs_head": 3}),
        },
        "role_counts": {
            "csm": (2, 3),
            "account_manager": (1, 2),
            "cs_head": (1, 1),
            "cro": (1, 1),
            "legal_counsel": (1, 1),
            "support_lead": (1, 1),
            "solutions_engineer": (1, 2),
            "customer": (1, 3),
        },
        "implementer": "solutions_engineer",
        "entities": [
            "Atlas account",
            "Borealis account",
            "Cobalt account",
            "Delphi account",
            "Everest account",
            "Fjord account",
            "Garnet account",
            "Helix account",
            "Indigo account",
            "Juniper account",
        ],
        "attributes": [
            (
                "discount",
                "renewal discount",
                "account_terms",
                "renewal",
                ["10 percent", "15 percent", "20 percent", "no discount"],
            ),
            (
                "sla",
                "support level",
                "account_terms",
                None,
                ["4 hour response", "2 hour response", "next business day response", "1 hour response"],
            ),
            (
                "sponsor",
                "executive sponsor",
                "renewal",
                None,
                [
                    "the chief revenue officer",
                    "the head of customer success",
                    "the account manager",
                    "the chief executive",
                ],
            ),
            (
                "dpa",
                "data processing addendum",
                "legal_review",
                "account_terms",
                [
                    "standard addendum version 3",
                    "customer paper with edits",
                    "standard addendum version 4",
                    "no addendum required",
                ],
            ),
            (
                "term",
                "renewal term",
                "renewal",
                "account_terms",
                ["12 months", "24 months", "36 months", "month to month"],
            ),
            (
                "plan",
                "success plan",
                "account_terms",
                None,
                ["quarterly business reviews", "monthly check-ins", "self-serve only", "dedicated workshops"],
            ),
        ],
        "requirements": [
            (
                "onboarding",
                "onboarding requirement",
                ["train 50 users", "train 20 users", "train 100 users", "train 10 users"],
            ),
            (
                "integration",
                "integration requirement",
                [
                    "connect the CRM and the helpdesk",
                    "connect the CRM only",
                    "connect the helpdesk only",
                    "connect the CRM the helpdesk and billing",
                ],
            ),
            (
                "reporting",
                "reporting requirement",
                ["weekly usage report", "monthly usage report", "quarterly usage report", "daily usage report"],
            ),
        ],
        "unanswerable": [
            "multi-year renewal signature",
            "security questionnaire approval",
            "reference call approval",
            "expansion order approval",
        ],
    },
}

DECISION_MOTIFS = [
    ("stable", 0.46),
    ("update", 0.12),
    ("escalation", 0.12),
    ("peer", 0.06),
    ("returns", 0.04),
    ("revoked", 0.05),
    ("revoked_restore", 0.03),
    ("no_authority", 0.12),
]
REQUIREMENT_MOTIFS = [("req_accepted", 0.35), ("req_narrowed", 0.45), ("req_unimplemented", 0.20)]
DRIFT_MOTIFS = ["req_drift_impl", "req_changed_after_ship"]


def _weighted(rng: random.Random, options: list[tuple[str, float]]) -> str:
    x = rng.random() * sum(w for _, w in options)
    for name, w in options:
        x -= w
        if x <= 0:
            return name
    return options[-1][0]


class _Builder:
    def __init__(self, cfg: dict, rng: random.Random, start: date, end: date):
        self.cfg, self.rng, self.start, self.end = cfg, rng, start, end
        self.assertions: list[dict] = []
        self.used: dict[str, set[date]] = {}
        self.roles: dict[str, str] = {}
        self.people: dict[str, str] = {}
        self.n = 0

    # ---- people
    def staff(self) -> None:
        names = self.rng.sample(NAMES, len(NAMES))
        for role, (lo, hi) in self.cfg["role_counts"].items():
            for i in range(self.rng.randint(lo, hi)):
                pid = f"{role}_{i + 1}"
                self.roles[pid] = role
                self.people[pid] = names.pop()

    def who(self, role: str) -> str:
        return self.rng.choice(sorted(p for p, r in self.roles.items() if r == role))

    def levels(self, domain: str) -> list[tuple[int, str]]:
        return sorted((lvl, role) for role, lvl in self.cfg["domains"][domain][1].items())

    def outsider(self, domain: str) -> str:
        roles = sorted(set(self.roles.values()) - set(self.cfg["domains"][domain][1]))
        return self.who(self.rng.choice(roles))

    # ---- time
    def dates(self, var: str, k: int, lo: int = 6, hi: int = 32) -> list[date]:
        span = (self.end - self.start).days
        gaps = [self.rng.randint(lo, hi) for _ in range(k - 1)]
        first = self.start + timedelta(days=self.rng.randint(0, max(1, span - sum(gaps) - 8)))
        out, d = [], first
        for i in range(k):
            if i:
                d = d + timedelta(days=gaps[i - 1])
            out.append(self.free(var, d))
            d = out[-1]
        return out

    def free(self, var: str, d: date) -> date:
        """No two assertions about one variable share a day, so every reference by date is unambiguous."""
        used = self.used.setdefault(var, set())
        while d in used:
            d += timedelta(days=1)
        used.add(d)
        return min(d, self.end)

    def add(self, t: date, speaker: str, domain: Optional[str], kind: str, var: str, value=None, **kw) -> str:
        self.n += 1
        aid = f"a{self.n:03d}"
        rec = {"id": aid, "t": t, "speaker": speaker, "domain": domain, "kind": kind, "var": var}
        if value is not None:
            rec["value"] = value
        rec.update({k: v for k, v in kw.items() if v not in (None, [], ())})
        self.assertions.append(rec)
        return aid

    # ---- decision motifs
    def decision_motif(self, motif: str, var: str, domain: str, values: list[str], cross: Optional[str] = None) -> None:
        rng = self.rng
        lv = self.levels(domain)
        lo_role, hi_role = lv[0][1], lv[-1][1]
        mid_role = lv[len(lv) // 2][1]
        vals = rng.sample(values, len(values))
        if motif == "stable":
            (d1,) = self.dates(var, 1)
            self.add(d1, self.who(rng.choice([mid_role, hi_role])), domain, "decide", var, vals[0])
        elif motif == "update":
            d1, d2, d3 = self.dates(var, 3)
            who = self.who(rng.choice([mid_role, hi_role]))
            self.add(d1, who, domain, "decide", var, vals[0])
            r = self.add(d2, self.who(lo_role), domain, "report", var, rng.choice(REASONS))
            self.add(d3, who, domain, "decide", var, vals[1], ref=[r], trigger=[r])
        elif motif == "escalation":
            d1, d2 = self.dates(var, 2)
            first = self.add(
                d1,
                self.who(lo_role if len(lv) < 3 else rng.choice([lo_role, mid_role])),
                domain,
                "decide",
                var,
                vals[0],
            )
            self.add(d2, self.who(hi_role), domain, "decide", var, vals[1], discussion=[first])
        elif motif == "peer":
            d1, d2 = self.dates(var, 2)
            role = rng.choice([mid_role, hi_role])
            first = self.add(d1, self.who(role), domain, "decide", var, vals[0])
            self.add(d2, self.who(role), domain, "decide", var, vals[1], ref=[first])
        elif motif == "returns":
            d1, d2, d3 = self.dates(var, 3)
            who = self.who(mid_role)
            self.add(d1, who, domain, "decide", var, vals[0])
            self.add(d2, who, domain, "decide", var, vals[1])
            self.add(d3, who, domain, "decide", var, vals[0])
        elif motif == "revoked":
            d1, d2 = self.dates(var, 2)
            first = self.add(d1, self.who(mid_role), domain, "decide", var, vals[0])
            self.add(d2, self.who(hi_role), domain, "revoke", var, ref=[first])
        elif motif == "revoked_restore":
            d1, d2, d3 = self.dates(var, 3)
            self.add(d1, self.who(lo_role), domain, "decide", var, vals[0])
            second = self.add(d2, self.who(hi_role), domain, "decide", var, vals[1])
            self.add(d3, self.who(hi_role), domain, "revoke", var, ref=[second])
        elif motif == "no_authority":
            d1, d2 = self.dates(var, 2)
            self.add(d1, self.who(rng.choice([mid_role, hi_role])), domain, "decide", var, vals[0])
            self.add(d2, self.outsider(domain), domain, "decide", var, vals[1])
        elif motif == "low_override":  # planted: a later statement by a lower authority in the same domain
            d1, d2 = self.dates(var, 2)
            self.add(d1, self.who(hi_role), domain, "decide", var, vals[0])
            self.add(d2, self.who(rng.choice([r for _, r in lv[:-1]])), domain, "decide", var, vals[1])
        elif motif == "cross_conflict":  # planted: two authority domains disagree
            assert cross
            d1, d2 = self.dates(var, 2)
            self.add(d1, self.who(rng.choice([mid_role, hi_role])), domain, "decide", var, vals[0])
            self.add(d2, self.who(self.levels(cross)[-1][1]), cross, "decide", var, vals[1])
        else:
            raise ValueError(motif)

    # ---- requirement motifs
    def requirement_motif(self, motif: str, var: str, variants: list[str]) -> None:
        lv = self.levels("requirements")
        customer, top = lv[0][1], lv[-1][1]
        impl = self.who(self.cfg["implementer"])
        requested, narrowed, changed, wrong = variants
        k = {
            "req_accepted": 3,
            "req_narrowed": 3,
            "req_unimplemented": 2,
            "req_drift_impl": 3,
            "req_changed_after_ship": 4,
        }[motif]
        ds = self.dates(var, k, lo=5, hi=24)
        req = self.add(ds[0], self.who(customer), "requirements", "request", var, requested)
        spec = requested if motif == "req_accepted" else narrowed
        dec = self.add(ds[1], self.who(top), "requirements", "decide", var, spec, ref=[req])
        if motif == "req_unimplemented":
            return
        shipped = wrong if motif == "req_drift_impl" else spec
        self.add(ds[2], impl, None, "implement", var, shipped, ref=[dec])
        if motif == "req_changed_after_ship":
            self.add(ds[3], self.who(top), "requirements", "decide", var, changed, ref=[dec])


def _try(
    archetype: str, index: int, seed: int, attempt: int, render: Optional[dict], panel_size: Optional[int]
) -> dict:
    cfg = ARCHETYPES[archetype]
    rng = random.Random(int(sha([GENERATOR_VERSION, seed, archetype, index, attempt]), 16))
    start = date(2026, 1, 5) + timedelta(days=rng.randrange(0, 60))
    end = start + timedelta(days=rng.randint(*TARGETS["history_days"]) - 5)
    b = _Builder(cfg, rng, start, end)
    b.staff()
    entities = rng.sample(cfg["entities"], rng.randint(*TARGETS["entities"]))
    variables: dict[str, dict] = {}
    dec_vars: list[tuple[str, str, Optional[str], list[str]]] = []
    req_vars: list[tuple[str, list[str]]] = []
    for ei, ent in enumerate(entities):
        for key, suffix, dom, cross, values in rng.sample(cfg["attributes"], rng.randint(2, 3)):
            var = f"e{ei + 1}.{key}"
            variables[var] = {"label": f"{ent} {suffix}", "governed_by": [dom]}
            dec_vars.append((var, dom, cross, values))
        for key, suffix, variants in rng.sample(cfg["requirements"], rng.randint(1, 2)):
            var = f"e{ei + 1}.{key}"
            variables[var] = {"label": f"{ent} {suffix}", "governed_by": ["requirements"]}
            req_vars.append((var, variants))
    # planted authority conflicts: 2 to 4, at least one across domains
    n_conf = rng.randint(*TARGETS["authority_conflicts"])
    crossable = [v for v in dec_vars if v[2]]
    n_cross = min(len(crossable), rng.randint(1, 2), n_conf)
    cross_pick = rng.sample(crossable, n_cross)
    rest = [v for v in dec_vars if v not in cross_pick]
    low_pick = rng.sample(rest, min(len(rest), n_conf - n_cross))
    planted_contested, planted_low = [], []
    for var, dom, cross, values in dec_vars:
        if (var, dom, cross, values) in cross_pick:
            variables[var]["governed_by"] = [dom, cross]
            b.decision_motif("cross_conflict", var, dom, values, cross)
            planted_contested.append(var)
        elif (var, dom, cross, values) in low_pick:
            b.decision_motif("low_override", var, dom, values)
            planted_low.append(var)
        else:
            b.decision_motif(_weighted(rng, DECISION_MOTIFS), var, dom, values)
    n_drift = min(len(req_vars), rng.randint(*TARGETS["drifted_requirements"]))
    drift_pick = rng.sample(req_vars, n_drift)
    planted_drift = []
    for var, variants in req_vars:
        if (var, variants) in drift_pick:
            b.requirement_motif(rng.choice(DRIFT_MOTIFS), var, variants)
            planted_drift.append(var)
        else:
            b.requirement_motif(_weighted(rng, REQUIREMENT_MOTIFS), var, variants)
    # questions: about half stay open, some of those with a partial answer
    nq = rng.randint(*TARGETS["questions"])
    n_open = rng.choice([nq // 2, (nq + 1) // 2])
    planted_open = []
    everyone = sorted(b.roles)
    for qi in range(nq):
        var = rng.choice(sorted(variables))
        dom = variables[var]["governed_by"][0]
        d1, d2 = b.dates(var, 2, lo=3, hi=14)
        q = b.add(d1, rng.choice(everyone), dom, "ask", var, rng.choice(QUESTIONS))
        if qi < n_open:
            planted_open.append(q)
            if rng.random() < 0.3:
                b.add(
                    d2, rng.choice(everyone), dom, "answer", var, rng.choice(PARTIAL_ANSWERS), ref=[q], complete=False
                )
        else:
            b.add(d2, rng.choice(everyone), dom, "answer", var, rng.choice(ANSWERS), ref=[q], complete=True)
    # a last observation near the end, so the clean corpus reaches the query time (the staleness floor test)
    var = rng.choice(sorted(variables))
    b.add(
        b.free(var, end - timedelta(days=rng.randint(0, 5))),
        rng.choice(everyone),
        variables[var]["governed_by"][0],
        "report",
        var,
        rng.choice(REASONS),
    )
    tpl: dict[str, Any] = {
        "id": f"{cfg['short']}-{index:02d}",
        "archetype": archetype,
        "generator": {"version": GENERATOR_VERSION, "attempt": attempt},
        "query_time": end + timedelta(days=rng.randint(2, 9)),
        "people": b.people,
        "roles": b.roles,
        "domains": {d: {"label": lab, "authority": dict(auth)} for d, (lab, auth) in cfg["domains"].items()},
        "variables": variables,
        "entities": entities,
        "unanswerable": [f"{rng.choice(entities)} {u}" for u in rng.sample(cfg["unanswerable"], rng.randint(2, 3))],
        "planted_drift": sorted(planted_drift),
        "planted_contested": sorted(planted_contested),
        "planted_low_authority": sorted(planted_low),
        "planted_open": sorted(planted_open),
        "assertions": sorted(b.assertions, key=lambda a: (a["t"], a["id"])),
    }
    if render:
        tpl["render"] = dict(render)
    if panel_size:
        tpl["panel_size"] = panel_size
    return tpl


def shape(tpl: dict) -> dict:
    """Realised Appendix A.3 statistics of a template (the numbers a dataset card reports)."""
    from .templates import _as_date, build_scenario, build_world

    w = build_world(tpl)
    qt = _as_date(tpl["query_time"])
    scn = build_scenario(tpl, 0)
    decides = [a for a in w.assertions if a.kind == "decide" and w.auth(a) is not None]
    gone = [
        a
        for a in decides
        if a in w.undominated(a.var, a.t) and w.decision_status(a.id, qt) in ("superseded", "revoked")
    ]
    asks = [a for a in w.assertions if a.kind == "ask"]
    reqs = sorted({a.var for a in w.assertions if a.kind == "request"})
    per_entity: dict[str, int] = {}
    for v in tpl["variables"]:
        per_entity[v.split(".")[0]] = per_entity.get(v.split(".")[0], 0) + 1
    low = [v for v in tpl.get("planted_low_authority", [])]
    return {
        "history_days": (max(a.t for a in w.assertions) - min(a.t for a in w.assertions)).days,
        "sessions": sum(1 for a in scn.artifacts if a.type != "org_chart"),
        "people": len(tpl["roles"]),
        "roles": len(set(tpl["roles"].values())),
        "entities": len(per_entity),
        "variables_per_entity": (min(per_entity.values()), max(per_entity.values())),
        "variables": len(tpl["variables"]),
        "assertions": len(w.assertions),
        "decisions": len(decides),
        "share_superseded_or_revoked": len(gone) / len(decides) if decides else 0.0,
        "requirements": len(reqs),
        "drifted_requirements": sum(1 for v in reqs if w.drifted(v, qt) is True),
        "questions": len(asks),
        "open_share": len(w.open_questions(qt)) / len(asks) if asks else 0.0,
        "authority_conflicts": len(low)
        + sum(1 for v in tpl["variables"] if w.valid_state(v, qt)["status"] == "contested"),
        "cross_domain_conflicts": sum(1 for v in tpl["variables"] if w.valid_state(v, qt)["status"] == "contested"),
    }


def shape_ok(stats: dict) -> list[str]:
    bad = []
    for key, (lo, hi) in TARGETS.items():
        val = stats[key]
        if isinstance(val, tuple):
            if val[0] < lo or val[1] > hi:
                bad.append(f"{key}={val}")
        elif not (lo <= val <= hi):
            bad.append(f"{key}={val:.3g}")
    return bad


def generate_template(
    archetype: str,
    index: int,
    seed: int = 0,
    render: Optional[dict] = None,
    panel_size: Optional[int] = None,
    max_attempts: int = 400,
) -> dict:
    """A template of the given archetype whose realised shape lies inside every Appendix A.3 range."""
    from .templates import validate_template

    last: list[str] = []
    for attempt in range(max_attempts):
        tpl = _try(archetype, index, seed, attempt, render, panel_size)
        errs = validate_template(tpl)
        if errs:
            last = errs
            continue
        last = shape_ok(shape(tpl))
        if not last:
            return tpl
    raise RuntimeError(f"no {archetype} template within the target shape after {max_attempts} attempts: {last}")
