"""Statistics from Section 9.2: cluster bootstrap over templates, Holm-Bonferroni, equivalence classification.

Not implemented in v0.1: the binomial GLMM with random template and item intercepts and the complementary
log-log hazard model. The cluster bootstrap and the equivalence tests are what the pre-registered classification
(degraded, improved, equivalent, inconclusive) is computed from here; see docs/statistics.md.
"""

from __future__ import annotations

import math
import random
from typing import Optional

SESOI = 0.05
BOOTSTRAP_REPLICATES = 2000


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def cluster_bootstrap(
    values_by_cluster: dict[str, list[float]], replicates: int = BOOTSTRAP_REPLICATES, seed: int = 0
) -> dict:
    """Mean over all items with clusters (templates) resampled with replacement."""
    clusters = [v for v in values_by_cluster.values() if v]
    if not clusters:
        return {"mean": None, "ci95": None, "ci90": None, "p": None, "n_clusters": 0}
    flat = [x for c in clusters for x in c]
    mean = sum(flat) / len(flat)
    rng = random.Random(seed)
    boots = []
    for _ in range(replicates):
        pick = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        xs = [x for c in pick for x in c]
        boots.append(sum(xs) / len(xs))
    boots.sort()
    le0 = sum(1 for b in boots if b <= 0) / replicates
    ge0 = sum(1 for b in boots if b >= 0) / replicates
    p = min(1.0, 2 * min(le0, ge0))
    return {
        "mean": mean,
        "ci95": [_pct(boots, 0.025), _pct(boots, 0.975)],
        "ci90": [_pct(boots, 0.05), _pct(boots, 0.95)],
        "p": p,
        "n_clusters": len(clusters),
    }


def classify_effect(boot: dict, sesoi: float = SESOI) -> str:
    """Degraded or improved if the 95% interval lies wholly beyond +-SESOI; equivalent if the 90% interval lies
    wholly within +-SESOI (two one-sided tests); otherwise inconclusive."""
    if boot.get("mean") is None:
        return "no_data"
    lo95, hi95 = boot["ci95"]
    lo90, hi90 = boot["ci90"]
    if hi95 < -sesoi:
        return "degraded"
    if lo95 > sesoi:
        return "improved"
    if lo90 > -sesoi and hi90 < sesoi:
        return "equivalent"
    return "inconclusive"


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def kendall_tau_b(x: list[float], y: list[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / denom if denom else None


def permutation_interaction(
    deltas: dict[str, dict[str, dict[str, float]]], replicates: int = 2000, seed: int = 0
) -> dict:
    """Operator-by-system interaction, tested without a model.

    `deltas[system][operator][item]` is the paired difference (perturbed minus clean) for one affected item. Because
    clean is the reference level, a system whose difference under some operator departs from the other systems' is
    exactly an operator-by-system interaction. Under the null the system labels of one item's differences are
    exchangeable, so they are permuted within each (operator, item) cell, which leaves the template structure
    intact. The statistic is the between-system variance of the mean difference, summed over operators."""
    systems = sorted(deltas)
    ops = sorted({op for s in systems for op in deltas[s]})
    cells: dict[str, list[list[float]]] = {}
    for op in ops:
        shared = set.intersection(*[set(deltas[s].get(op, {})) for s in systems]) if systems else set()
        cells[op] = [[deltas[s][op][i] for s in systems] for i in sorted(shared)]

    def stat(c: dict[str, list[list[float]]]) -> float:
        total = 0.0
        for rows in c.values():
            if not rows:
                continue
            means = [sum(r[j] for r in rows) / len(rows) for j in range(len(systems))]
            g = sum(means) / len(means)
            total += sum((m - g) ** 2 for m in means) / len(means)
        return total

    observed = stat(cells)
    n_items = sum(len(v) for v in cells.values())
    if len(systems) < 2 or n_items == 0:
        return {"statistic": observed, "p": None, "systems": len(systems), "items": n_items}
    rng = random.Random(seed)
    hits = 0
    for _ in range(replicates):
        perm = {op: [rng.sample(r, len(r)) for r in rows] for op, rows in cells.items()}
        if stat(perm) >= observed - 1e-15:
            hits += 1
    return {
        "statistic": observed,
        "p": (hits + 1) / (replicates + 1),
        "systems": len(systems),
        "items": n_items,
        "replicates": replicates,
    }


def icc_anova(values_by_cluster: dict[str, list[float]]) -> Optional[float]:
    """Intra-cluster correlation by the one-way ANOVA estimator (clusters are templates)."""
    groups = [v for v in values_by_cluster.values() if len(v) > 1]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return None
    grand = sum(sum(g) for g in groups) / n
    msb = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups) / (n - k)
    n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return max(0.0, (msb - msw) / denom) if denom > 0 else None
