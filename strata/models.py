"""The pre-registered models of Section 9.2. Needs the `stats` extra (statsmodels, pandas).

- `fit_glmm`: binomial mixed model of item correctness with random intercepts for template and item.
- `fit_hazard`: discrete-time hazard of reflecting an update, complementary log-log link, censored at the last
  checkpoint.

The mixed model is fitted by variational Bayes (statsmodels has no frequentist binomial GLMM). Its joint test of
the interaction terms is a Wald statistic on posterior means and standard deviations, which treats coefficients as
independent; read it next to the permutation test in strata.stats, which makes no such assumption.
"""

from __future__ import annotations

from typing import Optional


def _need():
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import statsmodels  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("the mixed and hazard models need: pip install 'strata-bench[stats]'") from e


def fit_glmm(rows: list[dict], reference_op: str = "clean") -> dict:
    """logit Pr(correct) = system + operator + system x operator + affected x operator + family + u_template + u_item."""
    _need()
    import numpy as np
    import pandas as pd
    import patsy
    from scipy import sparse
    from scipy.stats import chi2
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    df = pd.DataFrame(
        [
            {
                "correct": int(r["correct"]),
                "system": r["system"],
                "op": r["op"],
                "affected": int(bool(r["affected"])),
                "family": r["family"],
                "template": r["scenario_id"],
                "item": r["item_id"],
            }
            for r in rows
        ]
    )
    systems, ops = sorted(df["system"].unique()), sorted(df["op"].unique())
    if len(systems) < 2 or len(ops) < 2:
        return {"fitted": False, "reason": "needs at least two systems and two operators"}
    formula = (
        f"C(system) * C(op, Treatment('{reference_op}')) + C(op, Treatment('{reference_op}')):affected + C(family)"
    )
    X = patsy.dmatrix(formula, df, return_type="dataframe")
    keep = [c for c in X.columns if c == "Intercept" or X[c].std() > 0]
    X = X[keep].loc[:, ~X[keep].T.duplicated()]
    t_codes, t_levels = pd.factorize(df["template"])
    i_codes, i_levels = pd.factorize(df["item"])
    n = len(df)
    vc = sparse.hstack(
        [
            sparse.csr_matrix((np.ones(n), (np.arange(n), t_codes)), shape=(n, len(t_levels))),
            sparse.csr_matrix((np.ones(n), (np.arange(n), i_codes)), shape=(n, len(i_levels))),
        ]
    ).tocsr()
    ident = np.r_[np.zeros(len(t_levels), dtype=int), np.ones(len(i_levels), dtype=int)]
    model = BinomialBayesMixedGLM(
        df["correct"].to_numpy(),
        X.to_numpy(),
        vc,
        ident,
        vcp_p=2,
        fe_p=3,
        fep_names=list(X.columns),
        vcp_names=["template", "item"],
    )
    res = model.fit_vb(verbose=False)
    fe = {name: {"mean": float(m), "sd": float(s)} for name, m, s in zip(X.columns, res.fe_mean, res.fe_sd)}
    inter = [c for c in X.columns if "C(system)" in c and "C(op" in c]
    wald = float(sum((fe[c]["mean"] / fe[c]["sd"]) ** 2 for c in inter)) if inter else None
    sd_t, sd_i = float(np.exp(res.vcp_mean[0])), float(np.exp(res.vcp_mean[1]))
    return {
        "fitted": True,
        "n_rows": n,
        "systems": systems,
        "operators": ops,
        "fixed_effects": fe,
        "interaction_terms": inter,
        "interaction_wald_chi2": wald,
        "interaction_df": len(inter),
        "interaction_p": float(chi2.sf(wald, len(inter))) if inter else None,
        "sd_template": sd_t,
        "sd_item": sd_i,
        "icc_template": sd_t**2 / (sd_t**2 + sd_i**2 + 3.289868),  # pi^2 / 3 is the logistic residual variance
        "method": "variational Bayes; Wald test on posterior means and sds (approximate)",
    }


def fit_hazard(events: list[dict]) -> dict:
    """Discrete-time hazard of a system reflecting a changed decision.

    Each event: {system, template, latency (checkpoints, or None when censored), observed_for (checkpoints at risk)}.
    Expanded to one row per checkpoint at risk; cloglog Pr(update at period) = system + period."""
    _need()
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    pp = []
    for e in events:
        for period in range(int(e["observed_for"])):
            pp.append(
                {
                    "system": e["system"],
                    "template": e["template"],
                    "period": period,
                    "y": int(e["latency"] is not None and period == e["latency"]),
                }
            )
    df = pd.DataFrame(pp)
    if df.empty or df["system"].nunique() < 2:
        return {"fitted": False, "reason": "needs change events from at least two systems"}
    by_sys = df.groupby("system")["y"].mean()
    if df["y"].nunique() < 2 or by_sys.nunique() < 2:
        return {
            "fitted": False,
            "reason": "no variation in update timing across systems (every system updates at the same checkpoint)",
            "hazard_by_system": {k: float(v) for k, v in by_sys.items()},
        }
    formula = "y ~ C(system)" + (" + C(period)" if df["period"].nunique() > 1 else "")
    try:
        fit = smf.glm(formula, df, family=sm.families.Binomial(sm.families.links.CLogLog())).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(df["template"])[0]}
        )
    except Exception as exc:  # perfect separation and similar
        return {
            "fitted": False,
            "reason": f"model did not converge: {type(exc).__name__}",
            "hazard_by_system": {k: float(v) for k, v in by_sys.items()},
        }
    terms = {k: {"coef": float(fit.params[k]), "p": float(fit.pvalues[k])} for k in fit.params.index if "system" in k}
    if any(abs(t["coef"]) > 15 for t in terms.values()):
        return {
            "fitted": False,
            "reason": "separation: at least one system always or never updates at a checkpoint, so its hazard has no finite estimate",
            "hazard_by_system": {k: float(v) for k, v in by_sys.items()},
        }
    return {
        "fitted": True,
        "n_person_periods": len(df),
        "system_terms": terms,
        "hazard_by_system": {k: float(v) for k, v in by_sys.items()},
        "method": "GLM binomial, complementary log-log link, standard errors clustered by template",
    }


def median_latency(events: list[dict], horizon: int) -> Optional[float]:
    """Median update latency in checkpoints; a censored change counts as later than the horizon."""
    if not events:
        return None
    vals = sorted(e["latency"] if e["latency"] is not None else horizon + 1 for e in events)
    m = len(vals) // 2
    return float(vals[m]) if len(vals) % 2 else (vals[m - 1] + vals[m]) / 2
