import json
from datetime import date

from strata.llm import LLMReader
from strata.llm_render import (
    LLMRenderer,
    audit_sample,
    checker_calibration,
    cohens_kappa,
    entails_deterministic,
    realism_discrimination,
)
from strata.schema import ContextItem, Task


def good_writer(system, user):
    import re

    name = re.search(r"by (.+?) \(", user).group(1)
    date_ = re.search(r"dated (\d{4}-\d{2}-\d{2})", user).group(1)
    topic = re.search(r"Topic: (.+?)\. Value: (.+?)\.\n", user)
    return f"{date_}: {name} here. For {topic.group(1)}, we are going with {topic.group(2)}."


def yes_checker(system, user):
    return "YES"


def test_llm_renderer_keeps_ids_map_and_world(northwind):
    r = LLMRenderer(good_writer, good_writer, yes_checker, names=("alpha", "beta"))
    out = r.render(northwind)
    assert out.span_map == northwind.span_map and out.world_hash() == northwind.world_hash()
    assert out.renderer["name"] == "llm" and out.renderer["family"] in ("alpha", "beta")
    changed = [s.text for a in out.artifacts if a.type != "org_chart" for s in a.spans if s.text.startswith("2026")]
    assert changed


def test_leakage_and_missing_value_are_rejected():
    assert (
        entails_deterministic(
            "on 2026-03-18 we chose Q4", kind="decide", value="Q4", other_values=["Q3"], date_iso="2026-03-18"
        )
        == []
    )
    assert entails_deterministic(
        "on 2026-03-18 we chose Q4 not Q3", kind="decide", value="Q4", other_values=["Q3"], date_iso="2026-03-18"
    )
    assert entails_deterministic("we chose it", kind="decide", value="Q4", other_values=[], date_iso="2026-03-18")


def test_failing_spans_fall_back_to_the_template_text(northwind):
    bad = lambda s, u: "no facts here"
    out = LLMRenderer(bad, bad, yes_checker, max_retries=1).render(northwind)
    assert out.renderer["failed_spans"] > 0
    assert out.render_hash() == northwind.render_hash()


def test_audit_sample_is_stratified_and_calibration_math():
    # 20% of each (type, explicitness) stratum, at least one
    from strata.templates import build_scenario, load_template

    scn = build_scenario(load_template("templates/b2b_northwind.yaml"), 0)
    csv_text = audit_sample(scn, 0.2, 0)
    assert csv_text.count("\n") >= 5 and "entails_annotator_1" in csv_text
    cal = checker_calibration(
        [
            {
                "artifact_type": "chat",
                "explicitness": "diffuse",
                "checker_flags_defect": True,
                "human_flags_defect": True,
            },
            {
                "artifact_type": "chat",
                "explicitness": "diffuse",
                "checker_flags_defect": True,
                "human_flags_defect": False,
            },
            {
                "artifact_type": "chat",
                "explicitness": "diffuse",
                "checker_flags_defect": False,
                "human_flags_defect": True,
            },
        ]
    )
    assert cal["chat/all"]["precision"] == 0.5 and cal["chat/all"]["recall"] == 0.5


def test_kappa_and_realism_discrimination():
    assert cohens_kappa([True, True, False, False], [True, True, False, False]) == 1.0
    assert cohens_kappa([True, False, True, False], [False, True, False, True]) == -1.0
    assert realism_discrimination([True, False, True, False], [True, True, False, False]) == 0.5


def test_llm_reader_filters_unknown_evidence_and_flags_alignment():
    reply = json.dumps(
        {
            "answer": "Q4",
            "state": {"values": ["Q4"], "status": "valid", "since": "2026-03-18"},
            "evidence": ["sp-1", "sp-ghost"],
        }
    )
    reader = LLMReader(lambda s, u: "here you go: " + reply)
    task = Task(item_id="i", family="T3", question="q", query_time=date(2026, 6, 1), budget=1000)
    r = reader.answer(task, [ContextItem(text="x", source_ids=["sp-1"])])
    assert (
        r.evidence == ["sp-1"] and r.state.values == ["Q4"] and r.attribution == "native" and r.usage.model_calls == 1
    )
    assert reader.answer(task, [ContextItem(text="x")]).attribution == "aligned"
    assert LLMReader(lambda s, u: "garbage").answer(task, []).abstain is True
