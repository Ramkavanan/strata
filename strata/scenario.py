"""Scenario container: hidden world model plus rendered artifacts, with stable serialisation and hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Optional

from .core import Assertion, WorldModel
from .schema import Artifact


def _d(x: Optional[date]) -> Optional[str]:
    return x.isoformat() if x else None


def assertion_to_dict(a: Assertion) -> dict:
    return {
        "id": a.id,
        "t": _d(a.t),
        "speaker": a.speaker,
        "domain": a.domain,
        "kind": a.kind,
        "var": a.var,
        "value": a.value,
        "ref": list(a.ref),
        "complete": a.complete,
        "trigger": list(a.trigger),
        "discussion": list(a.discussion),
    }


def assertion_from_dict(d: dict) -> Assertion:
    return Assertion(
        id=d["id"],
        t=date.fromisoformat(d["t"]),
        speaker=d["speaker"],
        domain=d.get("domain"),
        kind=d["kind"],
        var=d["var"],
        value=d.get("value"),
        ref=tuple(d.get("ref", ())),
        complete=d.get("complete", True),
        trigger=tuple(d.get("trigger", ())),
        discussion=tuple(d.get("discussion", ())),
    )


def world_to_dict(w: WorldModel) -> dict:
    return {
        "roles": dict(sorted(w.roles.items())),
        "authority": {k: dict(sorted(v.items())) for k, v in sorted(w.authority.items())},
        "governance": {k: sorted(v) for k, v in sorted(w.governance.items())},
        "assertions": [assertion_to_dict(a) for a in w.assertions],
    }


def world_from_dict(d: dict) -> WorldModel:
    return WorldModel(
        roles=d["roles"],
        authority=d["authority"],
        governance={k: set(v) for k, v in d["governance"].items()},
        assertions=[assertion_from_dict(a) for a in d["assertions"]],
    )


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()[:16]


@dataclass
class Scenario:
    id: str
    template_id: str
    archetype: str
    seed: int
    world: WorldModel
    labels: dict[str, str]  # variable -> natural-language label
    domain_labels: dict[str, str]  # domain id -> label
    people: dict[str, str]  # speaker id -> display name
    query_time: date
    artifacts: list[Artifact]
    span_map: dict[str, Optional[str]]  # span id -> assertion id (None for decoys/org chart); HIDDEN
    explicitness: dict[str, str]  # assertion id -> explicit|hedged|diffuse
    unanswerable: list[str] = field(default_factory=list)  # labels for false-premise questions
    checkpoints: list[date] = field(default_factory=list)
    operator: dict = field(default_factory=lambda: {"name": "clean"})
    affected_vars: list[str] = field(default_factory=list)  # variables this render's operator targets
    renderer: dict = field(default_factory=lambda: {"name": "template"})
    # Optional question panel: {"selected": [...item ids chosen on the clean render], "universe": [...all clean item ids]}.
    # Under an operator, an item is kept if it is selected, or if it does not exist on the clean render at all.
    panel: Optional[dict] = None

    @property
    def render_id(self) -> str:
        op = self.operator
        params = ",".join(f"{k}={v}" for k, v in sorted(op.items()) if k != "name")
        return f"{op['name']}({params})" if params else op["name"]

    def with_(self, **kw) -> "Scenario":
        return replace(self, **kw)

    # ---- serialisation ----
    def to_dict(self) -> dict:
        return {
            "scenario_id": self.id,
            "template_id": self.template_id,
            "archetype": self.archetype,
            "seed": self.seed,
            "world": world_to_dict(self.world),
            "labels": self.labels,
            "domain_labels": self.domain_labels,
            "people": self.people,
            "query_time": _d(self.query_time),
            "artifacts": [json.loads(a.model_dump_json()) for a in self.artifacts],
            "span_map": self.span_map,
            "explicitness": self.explicitness,
            "unanswerable": self.unanswerable,
            "checkpoints": [_d(c) for c in self.checkpoints],
            "operator": self.operator,
            "affected_vars": self.affected_vars,
            "renderer": self.renderer,
            "panel": self.panel,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(
            id=d["scenario_id"],
            template_id=d["template_id"],
            archetype=d["archetype"],
            seed=d["seed"],
            world=world_from_dict(d["world"]),
            labels=d["labels"],
            domain_labels=d["domain_labels"],
            people=d["people"],
            query_time=date.fromisoformat(d["query_time"]),
            artifacts=[Artifact.model_validate(a) for a in d["artifacts"]],
            span_map=d["span_map"],
            explicitness=d["explicitness"],
            unanswerable=d.get("unanswerable", []),
            checkpoints=[date.fromisoformat(c) for c in d.get("checkpoints", [])],
            operator=d.get("operator", {"name": "clean"}),
            affected_vars=d.get("affected_vars", []),
            renderer=d.get("renderer", {"name": "template"}),
            panel=d.get("panel"),
        )

    # ---- hashes ----
    def world_hash(self) -> str:
        return sha(world_to_dict(self.world))

    def render_hash(self) -> str:
        """Hash of exactly what a system sees: artifacts only."""
        return sha([json.loads(a.model_dump_json()) for a in self.artifacts])

    def public_artifacts(self) -> list[Artifact]:
        return sorted(self.artifacts, key=lambda a: (a.ingest_date, a.event_date, a.id))

    def span_assertion(self, span_id: str) -> Optional[str]:
        return self.span_map.get(span_id)

    def assertion_spans(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for sid, aid in self.span_map.items():
            if aid:
                out.setdefault(aid, []).append(sid)
        return out
