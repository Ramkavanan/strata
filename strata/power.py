"""Sample size by simulation (Section 9.2). Needs numpy (the `stats` extra).

Paired correctness is simulated for affected items nested in templates: a template effect on the logit scale sized
from the intra-template correlation, and a pairing correlation between an item's clean and perturbed outcome. Each
replicate is classified with the same cluster bootstrap and the same rule the analysis uses, so the power reported
is the power of the procedure actually run.
"""

from __future__ import annotations

import math
from typing import Optional

from .stats import SESOI


def _np():
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover
        raise ImportError("the power simulation needs numpy: pip install 'strata-bench[stats]'") from e
    return np


def simulate(
    n_templates: int,
    items_per_template: int,
    p_clean: float,
    true_effect: float,
    icc: float,
    pairing: float = 0.5,
    replicates: int = 300,
    boot: int = 400,
    sesoi: float = SESOI,
    seed: int = 0,
) -> dict:
    """Share of replicates classified as degraded under two rules.

    rule "beyond_sesoi": the 95% interval lies wholly below -SESOI (the classification rule of Section 9.1).
    rule "excludes_zero": the 95% interval lies wholly below zero (ordinary detection of a degradation)."""
    np = _np()
    rng = np.random.default_rng(seed)
    sigma = math.sqrt(icc / (1 - icc) * (math.pi**2 / 3)) if 0 < icc < 1 else 0.0
    base = math.log(p_clean / (1 - p_clean))
    beyond = zero = equiv = 0
    for _ in range(replicates):
        u = rng.normal(0, sigma, n_templates)
        pc = 1 / (1 + np.exp(-(base + u)))  # clean accuracy per template
        pp = np.clip(pc - true_effect, 0.0, 1.0)  # perturbed accuracy per template
        # correlated Bernoulli pair per item through a shared uniform (pairing = share of items that use it)
        shape = (n_templates, items_per_template)
        shared = rng.random(shape)
        own = rng.random(shape)
        use = rng.random(shape) < pairing
        clean = shared < pc[:, None]
        pert = np.where(use, shared, own) < pp[:, None]
        d = pert.astype(float) - clean.astype(float)
        sums = d.sum(axis=1)
        idx = rng.integers(0, n_templates, (boot, n_templates))
        means = sums[idx].sum(axis=1) / (n_templates * items_per_template)
        lo95, hi95 = np.quantile(means, [0.025, 0.975])
        lo90, hi90 = np.quantile(means, [0.05, 0.95])
        beyond += hi95 < -sesoi
        zero += hi95 < 0
        equiv += (lo90 > -sesoi) and (hi90 < sesoi)
    return {
        "power_beyond_sesoi": beyond / replicates,
        "power_excludes_zero": zero / replicates,
        "share_equivalent": equiv / replicates,
    }


def power_table(
    items_per_template: int,
    p_clean: float,
    pairing: float = 0.5,
    templates=(10, 20, 30, 45, 60, 90, 120),
    effects=(0.05, 0.075, 0.10, 0.15),
    iccs=(0.05, 0.15, 0.30),
    replicates: int = 300,
    seed: int = 0,
) -> list[dict]:
    out = []
    for icc in iccs:
        for eff in effects:
            for n in templates:
                r = simulate(n, items_per_template, p_clean, eff, icc, pairing, replicates, seed=seed)
                out.append({"icc": icc, "true_effect": eff, "templates": n, **r})
    return out


def required_templates(
    table: list[dict], icc: float, true_effect: float, rule: str = "power_beyond_sesoi", target: float = 0.8
) -> Optional[int]:
    rows = sorted(
        (r for r in table if r["icc"] == icc and r["true_effect"] == true_effect), key=lambda r: r["templates"]
    )
    for r in rows:
        if r[rule] >= target:
            return r["templates"]
    return None
