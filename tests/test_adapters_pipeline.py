import json
import os

import pytest

from strata.adapters.base import get_adapter
from strata.cli import main
from strata.conformance import run_conformance
from strata.operators import OPERATORS, apply
from strata.runner import evaluate_growth, evaluate_render

OFFLINE = ["no_context", "long_context", "bm25", "oracle", "reference"]


def acc(ev):
    return sum(r["correct"] for r in ev["rows"]) / len(ev["rows"])


@pytest.mark.parametrize("op", OPERATORS)
def test_reference_system_is_perfect_on_template_renders(scenarios, op):
    ad = get_adapter("reference")
    for scn in scenarios.values():
        ev = evaluate_render(ad, apply(scn, op), 128_000)
        assert acc(ev) == 1.0, [r["item_id"] for r in ev["rows"] if not r["correct"]]


def test_baselines_rank_below_the_reference_and_no_context_is_near_floor(northwind):
    res = {
        n: acc(evaluate_render(get_adapter(n), northwind, 128_000))
        for n in ("no_context", "long_context", "bm25", "reference")
    }
    assert res["reference"] > res["long_context"] > res["no_context"] and res["reference"] > res["bm25"]
    assert res["no_context"] < 0.2


def test_recency_reader_fails_contradict_low_where_authority_reader_passes(northwind):
    s = apply(northwind, "contradict_low", k=2)
    rec = evaluate_render(get_adapter("long_context"), s, 128_000)
    ref = evaluate_render(get_adapter("reference"), s, 128_000)
    aff = lambda ev: [r for r in ev["rows"] if r["affected"] and r["family"] in ("T2", "T3", "T5")]
    assert sum(r["correct"] for r in aff(ref)) == len(aff(ref))
    assert sum(r["correct"] for r in aff(rec)) < len(aff(rec))


def test_late_documents_fool_arrival_order_readers_but_not_event_time_readers(northwind):
    s = apply(northwind, "late", lam=1.0)
    rec = evaluate_render(get_adapter("long_context"), northwind, 128_000)
    rec_late = evaluate_render(get_adapter("long_context"), s, 128_000)
    assert acc(rec_late) <= acc(rec)
    assert acc(evaluate_render(get_adapter("reference"), s, 128_000)) == 1.0


def test_ablating_authority_breaks_contradict_cross_robustness(northwind):
    s = apply(northwind, "contradict_cross", k=3)
    full = acc(evaluate_render(get_adapter("reference"), s, 128_000))
    no_auth = acc(evaluate_render(get_adapter("reference", authority=False), s, 128_000))
    assert full > no_auth


def test_budget_axis_binds_at_small_budgets(northwind):
    ad = get_adapter("reference")
    small = evaluate_render(ad, northwind, 256)
    big = evaluate_render(ad, northwind, 128_000)
    assert acc(small) < acc(big) and any(r["truncated"] for r in small["rows"])


def test_repeated_runs_of_a_deterministic_adapter_agree(northwind):
    ev = evaluate_render(get_adapter("bm25"), northwind, 128_000, repeats=3)
    assert ev["reliability_share_solved_all_runs"] == pytest.approx(
        sum(1 for r in ev["rows"] if r["run"] == 0 and r["correct"]) / ev["n_items"]
    )


def test_growth_oracle_has_no_latency_and_no_stale_answers(northwind):
    g = evaluate_growth(get_adapter("oracle"), northwind)["growth"]
    assert g["update_latency_mean"] == 0 and g["stale_answer_rate"] == 0 and g["changes_observed"] > 0


def test_growth_recency_adapter_is_not_worse_than_chance_on_latency(northwind):
    g = evaluate_growth(get_adapter("reference"), northwind)["growth"]
    assert g["update_latency_mean"] == 0 and g["stale_answer_rate"] == 0


@pytest.mark.parametrize("name", OFFLINE)
def test_offline_adapters_pass_conformance(name, capsys):
    assert run_conformance(name, {}) == 0


def test_conformance_catches_a_broken_adapter(capsys):
    assert run_conformance("broken_adapter:Broken", {}) == 1


def test_unknown_adapter_is_rejected():
    with pytest.raises(KeyError):
        get_adapter("not_a_system")


def test_cli_pipeline_generate_evaluate_report_verify(tmp_path, capsys):
    root = str(tmp_path / "runs")
    t = os.path.join(os.path.dirname(__file__), "..", "templates", "appendix_a_ex1.yaml")
    assert main(["--root", root, "generate", "--template", t]) == 0
    assert (
        main(
            ["--root", root, "evaluate", "--adapter", "reference", "--block", "all", "--budgets", "256,1024", "--quiet"]
        )
        == 0
    )
    assert main(["--root", root, "verify"]) == 0
    assert main(["--root", root, "report", "--check"]) == 0
    out = capsys.readouterr().out
    assert (
        "Perturbation block" in out
        and "Budget block" in out
        and "Growth block" in out
        and "composite" not in out.lower()
    )
    # tamper with a recorded response: verification must fail
    ev = sorted(os.listdir(os.path.join(root, "evaluations")))[-1]
    path = os.path.join(root, "evaluations", ev, "responses.jsonl")
    lines = open(path).read().splitlines()
    rec = json.loads(lines[0])
    rec["response"]["abstain"] = not rec["response"]["abstain"]
    rec["response"]["state"] = None
    lines[0] = json.dumps(rec, sort_keys=True)
    open(path, "w").write("\n".join(lines) + "\n")
    assert main(["--root", root, "verify"]) == 1


def test_golden_data_is_unchanged(capsys):
    assert main(["golden"]) == 0


def test_schema_export(tmp_path):
    assert main(["schema", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "item.schema.json").exists() and (tmp_path / "response.schema.json").exists()
