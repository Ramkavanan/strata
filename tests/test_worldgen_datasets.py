import json

import pytest

from strata import evaluate, load_dataset, operators, worldgen
from strata import textformat as tf
from strata.adapters.base import get_adapter
from strata.datasets import Golden, build_dataset, export_deepeval
from strata.datasets import load_dataset as load_from
from strata.questions import all_items, generate_items
from strata.reader import ReferenceReader
from strata.runner import evaluate_render
from strata.scenario import sha
from strata.schema import FAMILIES
from strata.submission import answer_split, score_submission
from strata.templates import build_scenario, validate_template

RENDER = {"chatter": 10, "filler_sessions": 25}


@pytest.fixture(scope="module")
def generated():
    return {a: worldgen.generate_template(a, 0, 7, RENDER, 30) for a in worldgen.ARCHETYPES}


@pytest.fixture(scope="module")
def gen_scn(generated):
    return build_scenario(generated["b2b_product_team"], 0)


@pytest.mark.parametrize("arch", list(worldgen.ARCHETYPES))
def test_generated_templates_are_valid_and_inside_the_target_shape(generated, arch):
    tpl = generated[arch]
    assert validate_template(tpl) == []
    assert worldgen.shape_ok(worldgen.shape(tpl)) == []


def test_generation_is_deterministic_and_varies_with_its_inputs():
    a = worldgen.generate_template("b2b_product_team", 3, 11)
    assert sha(json.loads(json.dumps(a, default=str))) == sha(
        json.loads(json.dumps(worldgen.generate_template("b2b_product_team", 3, 11), default=str))
    )
    assert a["assertions"] != worldgen.generate_template("b2b_product_team", 4, 11)["assertions"]
    assert a["assertions"] != worldgen.generate_template("b2b_product_team", 3, 12)["assertions"]


@pytest.mark.parametrize("arch", list(worldgen.ARCHETYPES))
def test_everything_the_generator_plants_is_really_there(generated, arch):
    tpl = generated[arch]
    scn = build_scenario(tpl, 0)
    w, t = scn.world, scn.query_time
    for var in tpl["planted_contested"]:
        assert w.valid_state(var, t)["status"] == "contested" and len(w.governance[var]) == 2
    for var in tpl["planted_low_authority"]:
        decides = sorted((a for a in w.assertions if a.var == var and a.kind == "decide"), key=lambda a: a.t)
        assert w.auth(decides[-1]) < w.auth(decides[0])  # a later, lower authority
        assert w.valid_state(var, t)["values"] == str(decides[0].value)  # did not change the state
    for var in tpl["planted_drift"]:
        assert w.drifted(var, t) is True
    assert sorted(w.open_questions(t)) == tpl["planted_open"]
    # no two assertions about one variable share a day, so references by date are unambiguous
    seen = set()
    for a in w.assertions:
        assert (a.var, a.t) not in seen
        seen.add((a.var, a.t))


def test_filler_never_parses_as_an_assertion_and_carries_no_ground_truth(gen_scn):
    filler = [s for a in gen_scn.artifacts if a.type != "org_chart" for s in a.spans if gen_scn.span_map[s.id] is None]
    assert len(filler) > 300
    assert all(tf.parse_span(s.text) is None for s in filler)
    assert sum(1 for a in gen_scn.artifacts if a.title.startswith("Weekly Sync")) > 5


def test_a_stale_cut_also_removes_the_sessions_after_it(gen_scn):
    s = operators.apply(gen_scn, "stale")
    cut = max(a.t for a in s.world.assertions)
    assert all(a.event_date <= cut for a in s.artifacts if a.type != "org_chart")


def test_chatter_does_not_change_the_world_or_the_answers(generated):
    tpl = dict(generated["engineering_incident"])
    plain = {k: v for k, v in tpl.items() if k not in ("render", "panel_size")}
    a, b = build_scenario(tpl, 0), build_scenario(plain, 0)
    assert a.world_hash() == b.world_hash() and a.render_hash() != b.render_hash()
    key = lambda i: (i.id, json.dumps({k: v for k, v in i.expected.items()}, sort_keys=True, default=str))
    assert {key(i) for i in all_items(a)} == {key(i) for i in all_items(b)}


def test_panel_is_about_thirty_questions_from_every_family_present(gen_scn):
    items = generate_items(gen_scn)
    assert len(items) == 30 and len(all_items(gen_scn)) > 60
    present = {i.family for i in all_items(gen_scn)}
    assert {i.family for i in items} == present and len(present) >= 12
    assert (
        gen_scn.panel["selected"]
        == build_scenario(worldgen.generate_template("b2b_product_team", 0, 7, RENDER, 30), 0).panel["selected"]
    )


def test_panel_items_stay_paired_and_operator_probes_are_kept(gen_scn):
    clean_ids = {i.id for i in generate_items(gen_scn)}
    low = operators.apply(gen_scn, "contradict_low")
    ids = {i.id for i in generate_items(low)}
    assert {i.id for i in generate_items(operators.apply(gen_scn, "late"))} == clean_ids
    probes = ids - set(gen_scn.panel["universe"])
    assert probes and all("inj-low" in p for p in probes if "/T5/" in p)
    assert not (ids & (set(gen_scn.panel["universe"]) - clean_ids))  # nothing unselected sneaks in


@pytest.mark.parametrize("op", operators.OPERATORS)
def test_reference_system_is_perfect_on_a_generated_scenario(gen_scn, op):
    ev = evaluate_render(get_adapter("reference"), operators.apply(gen_scn, op))
    assert all(r["correct"] for r in ev["rows"]), [r["item_id"] for r in ev["rows"] if not r["correct"]]


def test_contradiction_operators_scale_with_scenario_size(gen_scn, scenarios):
    assert len(operators.apply(gen_scn, "contradict_low").affected_vars) > 2
    assert len(operators.apply(scenarios["paper-a1"], "contradict_low").affected_vars) <= 2


# --------------------------------------------------------------------------- the shipped dataset
@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def test_shipped_dataset_matches_the_paper_composition(ds):
    spec = ds.manifest["spec"]
    assert spec["templates_per_archetype"] * len(worldgen.ARCHETYPES) == 30
    assert len(ds.scenarios()) == 12 and ds.manifest["splits"]["validation"]["counts"]["scenarios"] == 18
    clean = ds.goldens("clean")
    assert len(clean) == 12 * spec["panel_size"] == ds.info["counts"]["clean"]["goldens"]
    assert set(ds.info["counts"]["clean"]["by_family"]) <= set(FAMILIES)


def test_goldens_carry_input_expected_output_and_ideal_context(ds):
    for g in ds.goldens("contradict_low")[:60]:
        assert isinstance(g, Golden) and g.input and g.expected_output and g.expected is not None
        if g.family in ("T2", "T3") and g.expected["status"] != "undetermined":
            assert g.context and g.expected["values"][0] in " ".join(g.context)


def test_shipped_dataset_verifies_against_its_manifest(ds):
    assert ds.verify() == []
    assert load_dataset(split="validation").verify() == []


def test_hidden_split_ships_no_answers_and_no_tell_tale_ids(ds):
    val = load_dataset(split="validation")
    tasks = val.tasks()
    assert tasks and all(t.expected is None and t.expected_output is None and t.context is None for t in tasks)
    text = (val.path / "validation" / "tasks.jsonl").read_text()
    assert "inj-" not in text and "b2b-" not in text and "contradict" not in text and "stale" not in text
    with pytest.raises(PermissionError):
        val.scenarios()
    assert len(val.conditions()) == 18 * 7
    assert len(val.corpus(*val.conditions()[0])) > 40
    assert len(ds.manifest["splits"]["validation"]["seed_commitment"]) == 64


def test_build_answer_score_round_trip(tmp_path):
    spec = {"name": "tiny", "dev_indices": [0], "validation_indices": [4]}
    build_dataset(str(tmp_path / "tiny"), str(tmp_path / "private"), spec)
    dev = load_from("tiny", "dev", str(tmp_path))
    assert dev.verify() == []
    val = load_from("tiny", "validation", str(tmp_path))
    adapter = get_adapter("reference")
    recs = answer_split(adapter, val, ReferenceReader("reference"), limit=1)
    res = score_submission(
        recs, str(tmp_path / "private"), "tiny", adapter.describe(), {"system": "reference+reference"}
    )
    assert res["coverage"] == 1.0 and res["unknown_records"] == 0
    assert all(r["correct"] for r in res["rows"]) and len(res["rows"]) > 150
    with pytest.raises(ValueError):
        answer_split(get_adapter("oracle"), val)
    # a response with an id the maintainers never issued is counted, not scored
    recs[0]["response"]["item_id"] = "val-01/c-000000/q-forged"
    assert score_submission(recs, str(tmp_path / "private"), "tiny", adapter.describe(), {})["unknown_records"] == 1
    # tampering with a template is caught
    p = next((tmp_path / "tiny" / "dev" / "templates").glob("*.yaml"))
    p.write_text(p.read_text().replace("decide", "report", 1))
    assert load_from("tiny", "dev", str(tmp_path)).verify()


def test_deepeval_export_layout(ds, tmp_path):
    w = export_deepeval(ds, str(tmp_path), ["clean"], corpus=False)
    recs = json.loads((tmp_path / "goldens.clean.json").read_text())
    assert w["goldens"]["clean"] == len(recs) == 360
    assert {"input", "expected_output", "context", "additional_metadata", "name", "source_file"} <= set(recs[0])


def test_python_api(ds):
    r = evaluate("reference", limit=2, operators=["clean", "contradict_low"])
    assert r["system"] == "reference+reference" and all(x["correct"] for x in r["rows"])
    assert set(r["blocks"]["perturbation"]["operators"]) == {"clean", "contradict_low"}
    weak = evaluate("bm25", reader="recency", limit=2, operators=["clean"])
    strong = evaluate("bm25", reader="reference", limit=2, operators=["clean"])
    acc = lambda res: sum(x["correct"] for x in res["rows"]) / len(res["rows"])
    assert acc(strong) > acc(weak)
