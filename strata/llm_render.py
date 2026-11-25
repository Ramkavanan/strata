"""LLM renderers, the entailment and leakage checker, and the human-audit calibration tools (Section 5).

Rule: models write prose, never ground truth. A renderer rewrites each template span as natural text for its
assertion, keeping span ids and the span-to-assertion map. A checker from a different model family verifies that
each rewritten span entails its assertion and leaks nothing it should not. Failing spans are regenerated.
Nothing here is needed to run the offline harness.
"""

from __future__ import annotations

import csv
import io
import random
from collections import defaultdict
from dataclasses import replace
from typing import Callable, Optional

from .scenario import Scenario, sha
from .schema import Span

Complete = Callable[[str, str], str]  # (system, user) -> text

EXPLICITNESS_GUIDE = {
    "explicit": "State the decision outright, as people do in a recap (for example 'we are moving the target to Q4').",
    "hedged": "State it tentatively, as people do mid-meeting (for example 'I think we should probably plan on Q4 at this point').",
    "diffuse": "Never state it as a decision. Mention it only through a changed plan or an offhand reference so that a reader must infer it.",
}
TYPE_GUIDE = {
    "meeting": "a line of meeting transcript",
    "chat": "a chat message",
    "ticket": "a ticket comment",
    "document": "a sentence from a document revision",
    "release_note": "a release note line",
    "customer_call": "a line from a customer call",
}


def render_prompt(
    *,
    name: str,
    role: str,
    date_iso: str,
    artifact_type: str,
    kind: str,
    domain: str,
    label: str,
    value: Optional[str],
    explicitness: str,
) -> tuple[str, str]:
    system = (
        "You write one realistic line of workplace communication. Never mention that it is synthetic. "
        "Keep every fact exact; do not add facts. Output only the line."
    )
    user = (
        f"Write {TYPE_GUIDE.get(artifact_type, 'a message')} by {name} ({role}) dated {date_iso}, area: {domain}.\n"
        f"Kind of statement: {kind}. Topic: {label}. Value: {value}.\n{EXPLICITNESS_GUIDE.get(explicitness, '')}\n"
        f"Include the date {date_iso} and the speaker's name in the line."
    )
    return system, user


def entails_deterministic(
    text: str, *, kind: str, value: Optional[str], other_values: list[str], date_iso: str
) -> list[str]:
    """Cheap, model-free checks. Returns a list of defects (empty means pass)."""
    defects = []
    low = text.lower()
    if (
        kind in ("decide", "request", "implement", "report", "answer", "ask")
        and value is not None
        and str(value).lower() not in low
    ):
        defects.append("value missing")
    if value is None:
        pass
    for ov in other_values:
        if ov and str(ov).lower() != str(value).lower() and str(ov).lower() in low:
            defects.append(f"leaks value of another assertion: {ov!r}")
    if date_iso not in text:
        defects.append("date missing")
    return defects


def entails_llm(checker: Complete, text: str, *, label: str, value: Optional[str], kind: str) -> bool:
    system = "You verify claims against text. Answer with exactly YES or NO."
    claim = (
        f"the speaker {kind}s that '{label}' is '{value}'"
        if value is not None
        else f"the speaker mentions '{label}' without stating its value"
    )
    ans = checker(system, f"Text: {text}\n\nDoes the text support this claim: {claim}?")
    return ans.strip().upper().startswith("YES")


class LLMRenderer:
    def __init__(
        self, family_a: Complete, family_b: Complete, checker: Complete, names=("a", "b"), max_retries: int = 2
    ):
        self.families = {names[0]: family_a, names[1]: family_b}
        self.checker = checker
        self.max_retries = max_retries
        self.log: list[dict] = []

    def family_for(self, scn: Scenario) -> str:
        names = sorted(self.families)
        return names[int(sha([scn.seed, scn.id, "family"]), 16) % len(names)]

    def render(self, scn: Scenario) -> Scenario:
        fam = self.family_for(scn)
        gen = self.families[fam]
        by_id = {a.id: a for a in scn.world.assertions}
        by_var: dict[str, list] = defaultdict(list)
        for a in scn.world.assertions:
            by_var[a.var].append(a)
        arts = []
        for art in scn.artifacts:
            spans = []
            for sp in art.spans:
                aid = scn.span_map.get(sp.id)
                if not aid or art.type == "org_chart":
                    spans.append(sp)
                    continue
                a = by_id[aid]
                others = [str(x.value) for x in by_var[a.var] if x.id != a.id and x.value is not None]
                new_text, ok, tries = sp.text, False, 0
                while tries <= self.max_retries and not ok:
                    sys_p, usr_p = render_prompt(
                        name=scn.people.get(a.speaker, a.speaker),
                        role=scn.world.roles.get(a.speaker, ""),
                        date_iso=a.t.isoformat(),
                        artifact_type=art.type,
                        kind=a.kind,
                        domain=scn.domain_labels.get(a.domain or "", ""),
                        label=scn.labels[a.var],
                        value=None if a.value is None else str(a.value),
                        explicitness=scn.explicitness.get(a.id, "explicit"),
                    )
                    cand = gen(sys_p, usr_p).strip()
                    defects = entails_deterministic(
                        cand,
                        kind=a.kind,
                        value=None if a.value is None else str(a.value),
                        other_values=others,
                        date_iso=a.t.isoformat(),
                    )
                    if not defects and a.value is not None:
                        defects = (
                            []
                            if entails_llm(self.checker, cand, label=scn.labels[a.var], value=str(a.value), kind=a.kind)
                            else ["checker: not entailed"]
                        )
                    self.log.append({"span": sp.id, "assertion": a.id, "try": tries, "family": fam, "defects": defects})
                    if not defects:
                        new_text, ok = cand, True
                    tries += 1
                spans.append(Span(id=sp.id, text=new_text if ok else sp.text))
            arts.append(art.model_copy(update={"spans": spans}))
        return replace(
            scn,
            artifacts=arts,
            renderer={
                "name": "llm",
                "family": fam,
                "failed_spans": sum(1 for r in self.log if r["defects"] and r["try"] == self.max_retries),
            },
        )


# ------------------------------------------------------------------ human audit
def audit_sample(scn: Scenario, frac: float = 0.2, seed: int = 0) -> str:
    """CSV of a stratified sample (by artifact type and explicitness) for two annotators to label."""
    by_id = {a.id: a for a in scn.world.assertions}
    strata: dict[tuple, list] = defaultdict(list)
    for art in scn.artifacts:
        for sp in art.spans:
            aid = scn.span_map.get(sp.id)
            if aid:
                strata[(art.type, scn.explicitness.get(aid, "explicit"))].append((art, sp, by_id[aid]))
    rng = random.Random(seed)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(
        [
            "span_id",
            "artifact_type",
            "explicitness",
            "text",
            "claim_label",
            "claim_value",
            "entails_annotator_1",
            "entails_annotator_2",
            "realistic_annotator_1",
            "realistic_annotator_2",
            "is_synthetic_guess_1",
            "is_synthetic_guess_2",
        ]
    )
    for key, rows in sorted(strata.items()):
        k = max(1, round(frac * len(rows)))
        for art, sp, a in rng.sample(rows, min(k, len(rows))):
            w.writerow([sp.id, key[0], key[1], sp.text, scn.labels[a.var], a.value, "", "", "", "", "", ""])
    return out.getvalue()


def cohens_kappa(a: list[bool], b: list[bool]) -> Optional[float]:
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def checker_calibration(rows: list[dict]) -> dict:
    """Precision and recall of the checker as a defect detector against the human audit, per artifact type and level.

    Each row: artifact_type, explicitness, checker_flags_defect (bool), human_flags_defect (bool)."""
    out: dict = {}
    keys = {("all", "all")} | {(r["artifact_type"], "all") for r in rows} | {("all", r["explicitness"]) for r in rows}
    for t, e in sorted(keys):
        sel = [r for r in rows if t in ("all", r["artifact_type"]) and e in ("all", r["explicitness"])]
        tp = sum(r["checker_flags_defect"] and r["human_flags_defect"] for r in sel)
        fp = sum(r["checker_flags_defect"] and not r["human_flags_defect"] for r in sel)
        fn = sum(not r["checker_flags_defect"] and r["human_flags_defect"] for r in sel)
        out[f"{t}/{e}"] = {
            "n": len(sel),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    return out


def realism_discrimination(guesses: list[bool], truth_synthetic: list[bool]) -> Optional[float]:
    """Share of artifacts annotators classify correctly as synthetic or real. Near 0.5 is the realism target."""
    if not guesses or len(guesses) != len(truth_synthetic):
        return None
    return sum(g == t for g, t in zip(guesses, truth_synthetic)) / len(guesses)
