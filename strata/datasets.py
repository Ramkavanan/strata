"""Golden datasets: build, load, verify, export.

A dataset is a set of world-model templates plus everything derived from them: the corpus a system ingests under
each operator, and the goldens (question, expected output, ideal context) it is scored against. The derivation
is deterministic, so the public development split ships templates and goldens and regenerates corpora on demand,
checked against pinned hashes. The validation split ships corpora and questions only; its world models and
answers stay with the maintainers, and the public manifest carries a commitment to the seed that produced them.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import secrets
import statistics
from datetime import date
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from . import __version__, operators, worldgen
from .questions import generate_items
from .scenario import Scenario, sha
from .schema import CAPABILITY, STATE_FAMILIES, Artifact, Item
from .templates import build_scenario
from .tokenizer import count_tokens

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "data" / "datasets"
DEFAULT_DATASET = "strata-core-v0"

SPEC: dict[str, Any] = {
    "name": DEFAULT_DATASET,
    "version": "0.1.0",
    "public_seed": 20261019,
    "templates_per_archetype": 10,
    "dev_indices": [0, 1, 2, 3],  # 4 per archetype -> 12 development templates with answers
    "validation_indices": [4, 5, 6, 7, 8, 9],  # 6 per archetype -> 18 validation templates, answers withheld
    "render": {"chatter": 10, "filler_sessions": 25},
    "panel_size": 30,
    "instantiation_seed": 0,
}


class Golden(BaseModel):
    """One evaluation case, in the shape DeepEval calls a golden: an input, the expected output and the ideal
    context, plus the structured ground truth STRATA scores against."""

    id: str
    input: str
    expected_output: Optional[str] = None
    context: Optional[list[str]] = None
    scenario_id: str
    operator: str
    family: str
    capability: str
    query_time: date
    params: dict[str, Any] = Field(default_factory=dict)
    expected: Optional[dict[str, Any]] = None  # structured ground truth; None on hidden splits
    evidence_ids: Optional[list[str]] = None  # span ids of the ideal context
    affected: Optional[bool] = None
    difficulty: Optional[int] = None
    explicitness: Optional[str] = None


# --------------------------------------------------------------------------- expected output as text
def _describe(scn: Scenario, aid: str) -> str:
    a = scn.world._by_id(aid)
    if a is None:
        return aid
    who = scn.people.get(a.speaker, a.speaker)
    val = "" if a.value is None else f" = {a.value}"
    return f"{a.t.isoformat()} {who} ({scn.world.roles.get(a.speaker, '')}): {scn.labels.get(a.var, a.var)}{val}"


def expected_text(item: Item, scn: Scenario) -> str:
    e, f = item.expected, item.family
    d = lambda ids: "; ".join(_describe(scn, i) for i in ids) or "none"
    if f in ("T1", "T2", "T3", "T5"):
        if e["status"] == "valid":
            base = f"{e['values'][0]}" + (f" (in force since {e['since']})" if e.get("since") and f != "T1" else "")
            if f == "T5":
                return f"No. The state is {base}. The later statement did not have the authority to change it."
            return base
        if e["status"] == "contested":
            return f"Contested between {' and '.join(e['values'])} since {e['since']}: two authorities disagree and neither can overrule the other."
        return "Undetermined: no decision is in force."
    if f == "T4":
        return f"It is now {e['values'][0]} since {e['since']}. Decided by: {d(e['decider'])}. Reasons cited: {d(e['reasons'])}."
    if f == "T5b":
        steps = []
        for c in e["chain"]:
            v = " versus ".join(c["value"]) if isinstance(c["value"], list) else (c["value"] or "nothing in force")
            steps.append(f"{c['since']}: {v} [{c['status']}]")
        return " -> ".join(steps)
    if f == "T6":
        return (
            f"Triggered by: {d(e['trigger'])}. Discussed in: {d(e['discussion'])}. Reasons: {d(e['reasons'])}. "
            f"Followed by: {d(e['follow_ups'])}. Status: {e['status']}."
        )
    if f in ("T7", "T8", "T11"):
        return d(e["ids"])
    if f == "T9":
        if e["drifted"] is None:
            return "Cannot be determined: there is no valid specification to compare the shipped behaviour with."
        return (
            "No. What shipped has drifted from the specification now in force."
            if e["drifted"]
            else "Yes. What shipped matches the specification now in force."
        )
    if f == "T10":
        return f"Yes, answered: {_describe(scn, e['resolver'])}" if e["resolved"] else "No. The question is still open."
    if f == "T12":
        return "This did not happen. The question rests on a false premise."
    if f == "T13":
        return (
            "No. The newest material is well older than the question, so the answer may be out of date."
            if e["staleness_flag"]
            else "Yes. The material extends to the time of the question."
        )
    return ""


def golden_from_item(item: Item, scn: Scenario, hidden: bool = False) -> Golden:
    op = scn.operator["name"]
    if hidden:
        # the family is needed to answer (it fixes the response shape); the capability and the operator are not
        return Golden(
            id=item.id,
            input=item.question,
            scenario_id=scn.id,
            operator=op,
            family=item.family,
            capability="hidden",
            query_time=item.query_time,
            params=item.params,
        )
    ids = (
        item.expected.get("establishing", [])
        if item.family in STATE_FAMILIES | {"T1", "T4"}
        else sorted(set(item.expected.get("establishing", [])) | set(item.expected.get("support", [])))
    )
    amap = scn.assertion_spans()
    text = {s.id: s.text for a in scn.artifacts for s in a.spans}
    sids = [sid for aid in ids for sid in amap.get(aid, [])]
    return Golden(
        id=item.id,
        input=item.question,
        expected_output=expected_text(item, scn),
        context=[text[s] for s in sids],
        scenario_id=scn.id,
        operator=op,
        family=item.family,
        capability=CAPABILITY[item.family],
        query_time=item.query_time,
        params=item.params,
        expected=item.expected,
        evidence_ids=sids,
        affected=item.affected,
        difficulty=item.difficulty.get("level"),
        explicitness=item.explicitness,
    )


# --------------------------------------------------------------------------- build
def _dump_template(tpl: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(tpl, sort_keys=False, allow_unicode=True, width=120))


def _corpus_lines(scn: Scenario) -> list[str]:
    return [a.model_dump_json() for a in scn.public_artifacts()]


def _stats(templates: list[dict], scns: list[Scenario]) -> dict:
    rows = [worldgen.shape(t) for t in templates]
    out: dict[str, Any] = {}
    for k in rows[0]:
        vals = [r[k][1] if isinstance(r[k], tuple) else r[k] for r in rows]
        out[k] = {"min": round(min(vals), 3), "median": round(statistics.median(vals), 3), "max": round(max(vals), 3)}
    toks = [sum(count_tokens(s.text) for a in scn.artifacts for s in a.spans) for scn in scns]
    out["corpus_tokens"] = {"min": min(toks), "median": int(statistics.median(toks)), "max": max(toks)}
    return out


def build_dataset(
    out_dir: Optional[str] = None, private_dir: Optional[str] = None, spec: Optional[dict] = None
) -> dict:
    """Build the public dataset under `out_dir` and, when `private_dir` is given, the hidden validation answers
    there. The validation templates come from a secret seed kept in the private directory."""
    spec = {**SPEC, **(spec or {})}
    out = Path(out_dir) if out_dir else DATASETS_DIR / spec["name"]
    (out / "dev" / "templates").mkdir(parents=True, exist_ok=True)
    (out / "dev" / "goldens").mkdir(parents=True, exist_ok=True)
    man: dict[str, Any] = {
        "name": spec["name"],
        "version": spec["version"],
        "harness_version": __version__,
        "generator_version": worldgen.GENERATOR_VERSION,
        "spec": spec,
        "splits": {},
    }
    seed0 = spec["instantiation_seed"]

    def make(indices, seed):
        tpls = [
            worldgen.generate_template(arch, i, seed, spec["render"], spec["panel_size"])
            for arch in worldgen.ARCHETYPES
            for i in indices
        ]
        return tpls, [build_scenario(t, seed0) for t in tpls]

    # ---- development split: templates and goldens, corpora regenerated on demand
    tpls, scns = make(spec["dev_indices"], spec["public_seed"])
    dev: dict[str, Any] = {"hidden": False, "templates": {}, "counts": {}, "stats": _stats(tpls, scns)}
    by_op: dict[str, list[Golden]] = {op: [] for op in operators.OPERATORS}
    for tpl, base in zip(tpls, scns):
        _dump_template(tpl, out / "dev" / "templates" / f"{tpl['id']}.yaml")
        hashes = {}
        for op in operators.OPERATORS:
            r = operators.apply(base, op)
            items = generate_items(r)
            by_op[op].extend(golden_from_item(i, r) for i in items)
            hashes[op] = {
                "world": r.world_hash(),
                "render": r.render_hash(),
                "items": sha([i.model_dump(mode="json") for i in items]),
            }
        dev["templates"][tpl["id"]] = {
            "scenario_id": base.id,
            "archetype": tpl["archetype"],
            "template": sha(_canon(tpl)),
            "operators": hashes,
        }
    for op, gs in by_op.items():
        with open(out / "dev" / "goldens" / f"{op}.jsonl", "w") as f:
            for g in gs:
                f.write(g.model_dump_json() + "\n")
        fam: dict[str, int] = {}
        for g in gs:
            fam[g.family] = fam.get(g.family, 0) + 1
        dev["counts"][op] = {"goldens": len(gs), "by_family": dict(sorted(fam.items()))}
    man["splits"]["dev"] = dev

    # ---- validation split: corpora and questions only; world models and answers go to the private directory
    if private_dir:
        priv = Path(private_dir) / spec["name"] / "validation"
        (priv / "templates").mkdir(parents=True, exist_ok=True)
        (priv / "goldens").mkdir(parents=True, exist_ok=True)
        seed_file = priv / "seed.txt"
        if not seed_file.exists():
            seed_file.write_text(str(secrets.randbits(63)))
        secret = int(seed_file.read_text().strip())
        tpls, scns = make(spec["validation_indices"], secret)
        (out / "validation").mkdir(parents=True, exist_ok=True)
        val: dict[str, Any] = {
            "hidden": True,
            "seed_commitment": hashlib.sha256(str(secret).encode()).hexdigest(),
            "scenarios": {},
            "counts": {},
            "stats": _stats(tpls, scns),
        }
        mapping: dict[str, dict[str, str]] = {}
        task_ids: dict[str, str] = {}
        tasks_by_code: dict[str, list[Golden]] = {}
        answers_by_op: dict[str, list[Golden]] = {op: [] for op in operators.OPERATORS}
        for n, (tpl, base) in enumerate(zip(tpls, scns)):
            # public identifiers carry no template id, operator name or seed
            public_sid = f"val-{n + 1:02d}"
            _dump_template(tpl, priv / "templates" / f"{tpl['id']}.yaml")
            mapping[public_sid] = {"template": tpl["id"], "scenario_id": base.id}
            cdir = out / "validation" / "corpus" / public_sid
            cdir.mkdir(parents=True, exist_ok=True)
            val["scenarios"][public_sid] = {"archetype": tpl["archetype"], "conditions": {}}
            for op in operators.OPERATORS:
                r = operators.apply(base, op)
                code = "c-" + hashlib.sha256(f"{secret}:{public_sid}:{op}".encode()).hexdigest()[:6]
                mapping[public_sid][code] = op
                with gzip.open(cdir / f"{code}.jsonl.gz", "wt") as f:
                    f.write("\n".join(_corpus_lines(r)) + "\n")
                items = generate_items(r)
                for i in items:
                    g = golden_from_item(i, r, hidden=True)
                    # opaque id: the real one names assertions and injected statements
                    g.id = f"{public_sid}/{code}/q-" + hashlib.sha256(f"{secret}:{op}:{i.id}".encode()).hexdigest()[:10]
                    task_ids[g.id] = i.id
                    g.scenario_id, g.operator = public_sid, code
                    tasks_by_code.setdefault(code, []).append(g)
                    answers_by_op[op].append(golden_from_item(i, r))
                val["scenarios"][public_sid]["conditions"][code] = {
                    "render": sha([json.loads(line) for line in _corpus_lines(r)]),
                    "tasks": len(items),
                }
        with open(out / "validation" / "tasks.jsonl", "w") as f:
            for code in sorted(tasks_by_code):
                for g in tasks_by_code[code]:
                    f.write(g.model_dump_json(exclude_none=True) + "\n")
        for op, gs in answers_by_op.items():
            with open(priv / "goldens" / f"{op}.jsonl", "w") as f:
                for g in gs:
                    f.write(g.model_dump_json() + "\n")
        (priv / "mapping.json").write_text(json.dumps(mapping, indent=1, sort_keys=True))
        (priv / "task_ids.json").write_text(json.dumps(task_ids, indent=0, sort_keys=True))
        val["counts"] = {"scenarios": len(scns), "tasks": sum(len(v) for v in tasks_by_code.values())}
        man["splits"]["validation"] = val
    (out / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True, default=str) + "\n")
    (out / "DATASET_CARD.md").write_text(dataset_card(man))
    return man


def _canon(tpl: dict) -> Any:
    return json.loads(json.dumps(tpl, sort_keys=True, default=str))


# --------------------------------------------------------------------------- load
class Dataset:
    def __init__(self, path: Path, split: str):
        self.path, self.split = path, split
        self.manifest = json.loads((path / "manifest.json").read_text())
        if split not in self.manifest["splits"]:
            raise KeyError(f"split {split!r} not in dataset; available: {sorted(self.manifest['splits'])}")
        self.info = self.manifest["splits"][split]
        self.name, self.version, self.hidden = self.manifest["name"], self.manifest["version"], self.info["hidden"]
        self._scn: dict[str, Scenario] = {}

    # ---- development split
    def templates(self) -> list[dict]:
        self._need_answers()
        return [yaml.safe_load(p.read_text()) for p in sorted((self.path / self.split / "templates").glob("*.yaml"))]

    def scenarios(self) -> list[Scenario]:
        """Clean scenarios, rebuilt from templates."""
        self._need_answers()
        if not self._scn:
            seed = self.manifest["spec"]["instantiation_seed"]
            for t in self.templates():
                s = build_scenario(t, seed)
                self._scn[s.id] = s
        return list(self._scn.values())

    def render(self, scenario_id: str, operator: str = "clean") -> Scenario:
        self.scenarios()
        return operators.apply(self._scn[scenario_id], operator)

    def goldens(self, operator: str = "clean") -> list[Golden]:
        if self.hidden:
            return [g for g in self.tasks() if operator in (None, g.operator)]
        path = self.path / self.split / "goldens" / f"{operator}.jsonl"
        return [Golden.model_validate_json(line) for line in path.read_text().splitlines() if line]

    def corpus(self, scenario_id: str, operator: str = "clean") -> list[Artifact]:
        """The artifacts a system ingests for one scenario under one operator (or condition code on a hidden split)."""
        if self.hidden:
            with gzip.open(self.path / self.split / "corpus" / scenario_id / f"{operator}.jsonl.gz", "rt") as f:
                return [Artifact.model_validate_json(line) for line in f if line.strip()]
        return self.render(scenario_id, operator).public_artifacts()

    # ---- hidden split
    def tasks(self) -> list[Golden]:
        path = self.path / self.split / "tasks.jsonl"
        return [Golden.model_validate_json(line) for line in path.read_text().splitlines() if line]

    def conditions(self) -> list[tuple[str, str]]:
        return [(sid, code) for sid, v in sorted(self.info["scenarios"].items()) for code in sorted(v["conditions"])]

    def _need_answers(self) -> None:
        if self.hidden:
            raise PermissionError(f"the {self.split} split ships no world models or answers; use tasks() and corpus()")

    # ---- integrity
    def verify(self) -> list[str]:
        """Re-derive everything the development split pins and compare with the manifest."""
        problems: list[str] = []
        if self.hidden:
            for sid, code in self.conditions():
                arts = self.corpus(sid, code)
                h = sha([json.loads(a.model_dump_json()) for a in arts])
                if h != self.info["scenarios"][sid]["conditions"][code]["render"]:
                    problems.append(f"{sid}/{code}: corpus hash differs from the manifest")
            return problems
        spec = self.manifest["spec"]
        for tpl in self.templates():
            pin = self.info["templates"].get(tpl["id"])
            if pin is None:
                problems.append(f"{tpl['id']}: not in manifest")
                continue
            if sha(_canon(tpl)) != pin["template"]:
                problems.append(f"{tpl['id']}: template file differs from the manifest")
            arch, idx = tpl["archetype"], int(tpl["id"].rsplit("-", 1)[1])
            regen = worldgen.generate_template(arch, idx, spec["public_seed"], spec["render"], spec["panel_size"])
            if sha(_canon(regen)) != pin["template"]:
                problems.append(f"{tpl['id']}: the generator no longer reproduces this template")
            base = build_scenario(tpl, spec["instantiation_seed"])
            for op, h in pin["operators"].items():
                r = operators.apply(base, op)
                items = generate_items(r)
                got = {
                    "world": r.world_hash(),
                    "render": r.render_hash(),
                    "items": sha([i.model_dump(mode="json") for i in items]),
                }
                for k in got:
                    if got[k] != h[k]:
                        problems.append(f"{tpl['id']}/{op}: {k} hash changed")
        for op in operators.OPERATORS:
            want = self.info["counts"][op]["goldens"]
            if len(self.goldens(op)) != want:
                problems.append(f"goldens/{op}.jsonl: {len(self.goldens(op))} lines, manifest says {want}")
        return problems


def load_dataset(name: str = DEFAULT_DATASET, split: str = "dev", root: Optional[str] = None) -> Dataset:
    path = Path(root) / name if root else DATASETS_DIR / name
    if not (path / "manifest.json").exists():
        raise FileNotFoundError(f"dataset {name!r} not found under {path.parent}; build it with `strata dataset build`")
    return Dataset(path, split)


# --------------------------------------------------------------------------- export
def export_deepeval(ds: Dataset, out_dir: str, operators_: Optional[list[str]] = None, corpus: bool = True) -> dict:
    """Write goldens as a JSON list with the keys DeepEval's dataset loader reads (input, expected_output, context),
    one file per operator, and optionally the corpus as plain text files, one per artifact."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {"goldens": {}, "corpus_files": 0}
    ops = operators_ or operators.OPERATORS
    for op in ops:
        recs = []
        for g in ds.goldens(op):
            recs.append(
                {
                    "name": g.id,
                    "comments": f"{g.family} ({g.capability}), operator {op}",
                    "input": g.input,
                    "expected_output": g.expected_output,
                    "context": g.context,
                    "source_file": f"corpus/{g.scenario_id}/{op}",
                    "additional_metadata": {
                        "id": g.id,
                        "family": g.family,
                        "capability": g.capability,
                        "operator": op,
                        "scenario_id": g.scenario_id,
                        "query_time": g.query_time.isoformat(),
                        "affected": g.affected,
                        "difficulty": g.difficulty,
                        "expected": g.expected,
                        "evidence_ids": g.evidence_ids,
                    },
                }
            )
        (out / f"goldens.{op}.json").write_text(json.dumps(recs, indent=1, default=str))
        written["goldens"][op] = len(recs)
    if corpus and not ds.hidden:
        for scn in ds.scenarios():
            for op in ops:
                d = out / "corpus" / scn.id / op
                d.mkdir(parents=True, exist_ok=True)
                for a in ds.corpus(scn.id, op):
                    path = d / a.filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(a.to_file_text())
                    written["corpus_files"] += 1
                (d / "MANIFEST.jsonl").write_text(
                    "".join(
                        json.dumps(
                            {
                                "file": a.filename,
                                "id": a.id,
                                "type": a.type,
                                "mime": a.mime,
                                "event_date": a.event_date.isoformat(),
                                "ingest_date": a.ingest_date.isoformat(),
                                "status": a.status,
                                "units": [sp.id for sp in a.spans],
                            }
                        )
                        + "\n"
                        for a in ds.corpus(scn.id, op)
                    )
                )
    return written


# --------------------------------------------------------------------------- dataset card
def dataset_card(man: dict) -> str:
    spec = man["spec"]
    dev = man["splits"]["dev"]
    val = man["splits"].get("validation")
    L = [
        f"# {man['name']} (version {man['version']})",
        "",
        "Golden dataset for the STRATA benchmark (DOI 10.5281/zenodo.23469830): organisational histories with a hidden "
        "world model, the corpus a context system ingests, and the questions it is scored on.",
        "",
        "## Composition",
        "",
        f"- {spec['templates_per_archetype']} templates for each of {len(worldgen.ARCHETYPES)} archetypes "
        f"({', '.join(worldgen.ARCHETYPES)}), {spec['templates_per_archetype'] * len(worldgen.ARCHETYPES)} in total.",
        f"- Development split: {len(dev['templates'])} templates, with world models and answers.",
    ]
    if val:
        L.append(
            f"- Validation split: {val['counts']['scenarios']} scenarios, corpus and questions only. "
            "World models, answers and the operator behind each condition code are held by the maintainers."
        )
    L += [
        "- Each scenario is rendered under 7 operators (clean, stale, contradict_low, contradict_cross, incomplete, late, noisy).",
        f"- A panel of about {spec['panel_size']} questions per scenario covers the 14 task families (T1 to T13 and T5b).",
        "",
        "| operator | development goldens | "
        + " | ".join(sorted({f for c in dev["counts"].values() for f in c["by_family"]}))
        + " |",
        "|---|---|" + "---|" * len({f for c in dev["counts"].values() for f in c["by_family"]}),
    ]
    fams = sorted({f for c in dev["counts"].values() for f in c["by_family"]})
    for op, c in dev["counts"].items():
        L.append(f"| {op} | {c['goldens']} | " + " | ".join(str(c["by_family"].get(f, 0)) for f in fams) + " |")
    L += [
        "",
        "## Realised scenario shape against the paper's targets (Appendix A.3)",
        "",
        "| statistic | target | development min / median / max |" + (" validation min / median / max |" if val else ""),
        "|---|---|---|" + ("---|" if val else ""),
    ]
    for k, tgt in worldgen.TARGETS.items():
        d = dev["stats"][k]
        row = f"| {k.replace('_', ' ')} | {tgt[0]} to {tgt[1]} | {d['min']} / {d['median']} / {d['max']} |"
        if val:
            v = val["stats"][k]
            row += f" {v['min']} / {v['median']} / {v['max']} |"
        L.append(row)
    d = dev["stats"]["corpus_tokens"]
    L.append(
        f"| corpus tokens per scenario | 150k to 600k in the paper | {d['min']} / {d['median']} / {d['max']} |"
        + (
            f" {val['stats']['corpus_tokens']['min']} / {val['stats']['corpus_tokens']['median']} / {val['stats']['corpus_tokens']['max']} |"
            if val
            else ""
        )
    )
    L += [
        "",
        "## Files",
        "",
        "```",
        "manifest.json                       hashes of every template, render and item set",
        "dev/templates/<id>.yaml             world-model templates (the hidden truth, public for this split)",
        "dev/goldens/<operator>.jsonl        goldens: input, expected_output, context, structured expected answer",
        "validation/corpus/<scenario>/<condition>.jsonl.gz   artifacts to ingest",
        "validation/tasks.jsonl              questions without answers",
        "```",
        "",
        "Corpora for the development split are regenerated from the templates and checked against the manifest "
        "(`strata dataset verify`). `strata dataset export --format deepeval` writes them as text files with goldens.",
        "",
        "## How it was made",
        "",
        "World models are produced by a seeded procedural generator (`strata/worldgen.py`) that plants reversals, "
        "later statements by lower authorities, cross-domain conflicts, revocations, values that return, drifted "
        "requirements and open questions, then rejects any template outside the target shape above. A validator "
        "re-derives every planted property from the world model. Artifacts are rendered from templates of text "
        f"with {spec['render']['chatter']} filler lines per artifact on average and {spec['render']['filler_sessions']} "
        "sessions that carry no decision. Questions and answers are computed by deterministic replay. No model is "
        "involved at any step.",
        "",
        "## Limitations you must keep in mind",
        "",
        "- The templates are generated, not written by independent domain experts as the paper requires for version 1. "
        "They share the generator's regularities, which a system could learn.",
        "- Artifacts come from a template renderer whose sentences are regular enough to parse. A system that parses "
        "them is not thereby shown to handle real organisational text. Model-rendered corpora with a human audit are "
        "the missing step.",
        "- Corpora are about 30k tokens, well below the paper's estimate of 150k to 600k. The 2k and 8k budgets bind; "
        "128k does not.",
        "- The validation split hides answers and operator names, but anyone holding the corpora can compare "
        "conditions of one scenario. A private test split of fresh instantiations is what protects a leaderboard.",
        "- Authority is modelled as levels per domain with one role per person. Real authority is messier.",
        "",
        "## Intended use",
        "",
        "Evaluating context and memory systems on authority, time and attribution; regression testing a system "
        "between releases; teaching. Not intended as evidence about any vendor until that vendor has reviewed its "
        "adapter and configuration.",
        "",
        "## Maintenance",
        "",
        f"Built with harness {man['harness_version']}, generator version {man['generator_version']}, public seed "
        f"{spec['public_seed']}."
        + (f" Validation seed commitment (SHA-256): `{val['seed_commitment']}`." if val else ""),
        "All scenarios are synthetic and contain no personal data; names are drawn from a fixed list.",
        "Licence: CC BY 4.0.",
        "",
    ]
    return "\n".join(L)
