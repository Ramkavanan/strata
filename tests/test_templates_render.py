import copy

import pytest
from conftest import TEMPLATES

from strata import textformat as tf
from strata.templates import build_scenario, load_template, validate_template


@pytest.mark.parametrize("path", TEMPLATES)
def test_every_template_is_valid(path):
    assert validate_template(load_template(path)) == []


def test_validator_catches_each_invariant(root):
    base = load_template(f"{root}/templates/b2b_northwind.yaml")

    def broken(mut):
        t = copy.deepcopy(base)
        mut(t)
        return validate_template(t)

    assert broken(lambda t: t["assertions"].append(dict(t["assertions"][0])))  # duplicate id
    assert broken(lambda t: t["assertions"][0].update(speaker="nobody"))  # unknown speaker
    assert broken(lambda t: t["assertions"][0].update(var="nope"))  # unknown variable
    assert broken(lambda t: t.update(planted_drift=["sf.target"]))  # planted drift that is not drift
    assert broken(lambda t: t.update(planted_contested=["sf.target"]))  # planted conflict that is not contested
    assert broken(lambda t: t.update(planted_open=["q1"]))  # wrong open set
    assert broken(lambda t: t["assertions"][0].update(value='say "hi"'))  # quote in value
    assert broken(lambda t: t.update(query_time="2026-12-31"))  # corpus too old for the staleness floor
    assert broken(lambda t: t["assertions"][0].update(ref=["ghost"]))  # dangling reference


@pytest.mark.parametrize("seed", [0, 1, 4])
@pytest.mark.parametrize("path", TEMPLATES)
def test_every_rendered_span_parses_back_to_its_assertion(path, seed):
    scn = build_scenario(load_template(path), seed)
    by_id = {a.id: a for a in scn.world.assertions}
    for art in scn.artifacts:
        for sp in art.spans:
            if art.type == "org_chart":
                assert tf.parse_org(sp.text) is not None, sp.text
                continue
            if art.type == "source_code":  # code is not a statement; it maps to the implementation it realises
                assert by_id[scn.span_map[sp.id]].kind == "implement"
                continue
            p = tf.parse_span(sp.text)
            assert p is not None, sp.text
            a = by_id[scn.span_map[sp.id]]
            assert p["kind"] == a.kind and p["label"] == scn.labels[a.var] and p["date"] == a.t
            if a.value is not None:
                assert p["value"] == str(a.value)


def test_same_seed_same_bytes(root):
    p = f"{root}/templates/b2b_northwind.yaml"
    a, b = build_scenario(load_template(p), 3), build_scenario(load_template(p), 3)
    assert a.render_hash() == b.render_hash() and a.world_hash() == b.world_hash()
    assert build_scenario(load_template(p), 4).render_hash() != a.render_hash()


def test_seed_zero_reproduces_the_paper_dates(scenarios):
    assert min(a.t for a in scenarios["paper-a1"].world.assertions).isoformat() == "2026-01-12"


def test_span_ids_are_opaque_and_unique(northwind):
    ids = [s.id for a in northwind.artifacts for s in a.spans]
    assert len(ids) == len(set(ids))
    import re

    assert all(re.fullmatch(r"sp-[0-9a-f]{10}", i) for i in ids)


def test_public_artifacts_carry_no_ground_truth(northwind):
    dumped = " ".join(a.model_dump_json() for a in northwind.public_artifacts())
    assert "span_map" not in dumped and "explicit" not in dumped.lower().replace("explicitly", "")
