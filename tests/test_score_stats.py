import pytest

from strata.schema import ContextItem
from strata.score import _f1, aubc, b90, balanced_accuracy, cost_usd, cpco, growth_metrics
from strata.stats import classify_effect, cluster_bootstrap, holm, kendall_tau_b
from strata.tokenizer import count_tokens, enforce_budget, truncate_to_tokens


def test_f1():
    assert _f1({"a", "b"}, {"a", "b"}) == 1.0 and _f1(set(), set()) == 1.0
    assert _f1({"a"}, {"a", "b"}) == pytest.approx(2 / 3) and _f1({"x"}, {"a"}) == 0.0


def test_balanced_accuracy_punishes_refusing_everything():
    assert balanced_accuracy(tp=10, fn=0, tn=0, fp=10) == 0.5
    assert balanced_accuracy(tp=5, fn=5, tn=8, fp=2) == pytest.approx((0.5 + 0.8) / 2)


def test_aubc_and_b90_hand_computed():
    flat = {2000: 0.5, 8000: 0.5, 32000: 0.5, 128000: 0.5}
    assert aubc(flat) == pytest.approx(0.5)
    ramp = {2000: 0.0, 8000: 1.0, 32000: 1.0, 128000: 1.0}
    # trapezoids on a log axis: first segment contributes 0.5 * ln4, others 1.0 * ln4 each; total span 3 ln4
    assert aubc(ramp) == pytest.approx((0.5 + 1 + 1) / 3)
    assert b90(ramp) == 8000 and b90({2000: 0.0, 128000: 0.0}) is None


def test_growth_metrics_reproduce_figure_3a():
    exp = {"x": [("Q3",), ("Q3",), ("Q4",), ("Q4",), ("Q4",)]}
    a = growth_metrics(exp, {"x": [("Q3",), ("Q3",), ("Q4",), ("Q4",), ("Q4",)]})
    b = growth_metrics(exp, {"x": [("Q3",), ("Q3",), ("Q3",), ("Q3",), ("Q4",)]})
    assert a["update_latency_mean"] == 0 and a["stale_answer_rate"] == 0
    assert b["update_latency_mean"] == 2 and b["stale_answer_rate"] == pytest.approx(2 / 5)
    never = growth_metrics(exp, {"x": [("Q3",)] * 5})
    assert never["changes_censored"] == 1 and never["update_latency_mean"] is None


def test_cpco_formula_and_undefined_at_zero_accuracy():
    assert cpco(0.002, 1.0, 0.8, horizon=1000) == pytest.approx((0.002 + 0.001) / 0.8)
    assert cpco(0.002, 1.0, 0.0) is None
    assert cost_usd(1_000_000, 0, {"models": {"m": {"input": 3.0, "output": 15.0}}}, "m") == 3.0


def test_equivalence_classification():
    mk = lambda vals: cluster_bootstrap({f"t{i}": [v] * 20 for i, v in enumerate(vals)}, replicates=500)
    assert classify_effect(mk([-0.3, -0.35, -0.28, -0.32])) == "degraded"
    assert classify_effect(mk([0.3, 0.35, 0.28, 0.32])) == "improved"
    assert classify_effect(mk([0.0, 0.01, -0.01, 0.0])) == "equivalent"
    assert classify_effect(mk([-0.3, 0.3, -0.2, 0.25])) == "inconclusive"


def test_holm_is_monotone_and_conservative():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)


def test_kendall_tau_b():
    assert kendall_tau_b([1, 2, 3], [1, 2, 3]) == 1.0 and kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0


def test_tokenizer_and_budget():
    assert count_tokens("Hello, world 42") == 4
    assert truncate_to_tokens("one two three four", 2) == "one two"
    items = [ContextItem(text="a b c d e"), ContextItem(text="f g h i j"), ContextItem(text="k l m")]
    kept, info = enforce_budget(items, 7)
    assert info["tokens"] == 7 and info["truncated"] and len(kept) == 2 and count_tokens(kept[1].text) == 2
    assert info["dropped_items"] == 1
