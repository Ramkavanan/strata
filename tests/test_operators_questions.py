import pytest

from strata.adapters.base import get_adapter
from strata.operators import OPERATORS, apply
from strata.questions import difficulty, generate_items, growth_items, ingested_world
from strata.runner import evaluate_render
from strata.schema import FAMILIES


def test_clean_has_no_affected_items(northwind):
    assert not any(i.affected for i in generate_items(apply(northwind, "clean")))


def test_stale_cuts_the_corpus_and_raises_the_flag(northwind):
    s = apply(northwind, "stale")
    assert max(a.t for a in s.world.assertions) < max(a.t for a in northwind.world.assertions)
    flag = [i for i in generate_items(s) if i.family == "T13"][0]
    assert flag.expected["staleness_flag"] is True and flag.affected
    assert [i for i in generate_items(northwind) if i.family == "T13"][0].expected["staleness_flag"] is False


def test_contradict_low_does_not_change_valid_state(northwind):
    s = apply(northwind, "contradict_low", k=2)
    assert s.affected_vars
    for var in s.affected_vars:
        assert s.world.valid_state(var, s.query_time) == northwind.world.valid_state(var, s.query_time)
    assert any(a.id.startswith("inj-low-") for a in s.world.assertions)


def test_contradict_cross_makes_variables_contested(northwind):
    s = apply(northwind, "contradict_cross", k=2)
    assert s.affected_vars
    for var in s.affected_vars:
        assert s.world.valid_state(var, s.query_time)["status"] == "contested"
        assert len(s.world.governance[var]) == len(northwind.world.governance[var]) + 1


@pytest.mark.parametrize("mode", ["omit", "truncate"])
def test_incomplete_removes_every_establishing_decision(northwind, mode):
    s = apply(northwind, "incomplete", rho=0.25, mode=mode)
    for var in s.affected_vars:
        assert s.world.valid_state(var, s.query_time)["status"] == "undetermined"
        assert not [a for a in s.world.assertions if a.var == var and a.kind in ("decide", "request")]
        assert any(a.kind == "report" and a.value is None for a in s.world.assertions if a.var == var)
    if mode == "truncate":
        assert any(a.status == "partial" for a in s.artifacts)


def test_late_changes_ingestion_order_only(northwind):
    s = apply(northwind, "late", lam=0.3)
    assert s.world_hash() == northwind.world_hash()
    assert [i.expected for i in generate_items(s)] == [i.expected for i in generate_items(northwind)]
    late_docs = [a for a in s.artifacts if a.ingest_date > a.event_date]
    assert late_docs and all(a.type == "document" for a in late_docs)


def test_noisy_adds_decoys_without_touching_the_world(northwind):
    s = apply(northwind, "noisy", eta=1.0)
    assert s.world_hash() == northwind.world_hash()
    assert len(s.artifacts) > len(northwind.artifacts)
    assert [i.expected for i in generate_items(s)] == [i.expected for i in generate_items(northwind)]
    decoys = [sid for sid, a in s.span_map.items() if a is None and sid not in northwind.span_map]
    assert decoys


@pytest.mark.parametrize("op", OPERATORS)
def test_operators_are_deterministic(northwind, op):
    assert apply(northwind, op).render_hash() == apply(northwind, op).render_hash()


def test_every_family_is_generated_on_the_b2b_scenario(northwind):
    assert {i.family for i in generate_items(northwind)} == set(FAMILIES)


def test_items_have_unique_ids_and_levels(scenarios):
    for scn in scenarios.values():
        its = generate_items(scn)
        assert len({i.id for i in its}) == len(its)
        assert all(1 <= i.difficulty["level"] <= 5 for i in its)


def test_items_are_paired_across_operators(northwind):
    clean_ids = {i.id for i in generate_items(northwind) if i.family in ("T1", "T3", "T5b")}
    for op in ("late", "noisy", "contradict_low"):
        assert clean_ids <= {i.id for i in generate_items(apply(northwind, op))}


def test_difficulty_rises_with_conflict(northwind):
    hard = difficulty(northwind, "hub.target", "T3", northwind.query_time)
    easy = difficulty(northwind, "pricing.tiers", "T3", northwind.query_time)
    assert hard["level"] >= easy["level"] and hard["distractors"] >= 1


@pytest.mark.parametrize("op", OPERATORS)
def test_oracle_is_perfect_under_every_operator(scenarios, op):
    ad = get_adapter("oracle")
    for scn in scenarios.values():
        ev = evaluate_render(ad, apply(scn, op), 128_000)
        assert all(r["correct"] for r in ev["rows"]), [r["item_id"] for r in ev["rows"] if not r["correct"]]


def test_growth_panel_changes_between_checkpoints(northwind):
    first, last = growth_items(northwind, 0), growth_items(northwind, len(northwind.checkpoints) - 1)
    assert {i.id for i in first} == {i.id for i in last}
    assert [i.expected for i in first] != [i.expected for i in last]
    assert len(ingested_world(northwind, northwind.checkpoints[0]).assertions) < len(northwind.world.assertions)
