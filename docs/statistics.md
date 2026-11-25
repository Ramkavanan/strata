# Statistics

Implemented:

- cluster bootstrap over templates (2,000 replicates) for every paired difference;
- classification against a 5-point smallest effect of interest: degraded or improved when the 95% interval lies
  wholly beyond the bound, equivalent when the 90% interval lies wholly inside it (two one-sided tests),
  inconclusive otherwise;
- Holm-Bonferroni across the operators of one hypothesis family;
- an operator-by-system interaction test by permutation of system labels within each (operator, item) cell, which
  keeps the template structure intact and needs no model;
- the binomial mixed model of Section 9.2, fitted by variational Bayes (statsmodels has no frequentist binomial
  GLMM), with a Wald test of the interaction terms on posterior means and standard deviations;
- the discrete-time hazard of reflecting an update with a complementary log-log link, clustered by template;
- intra-template correlation by one-way ANOVA;
- Kendall tau-b with a bootstrap interval for the rank comparisons;
- a power simulation of the full procedure (see docs/hypotheses.md).

Read every classification next to the number of templates. Below 10 the report marks cells underpowered. The
mixed model's interaction test treats coefficients as independent; the permutation test does not, and when they
disagree the permutation test is the one to trust.
