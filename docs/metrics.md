# Metrics (paper Section 8)

State accuracy needs value, status and time-from-which (within 7 days) to match. Fact lineage is ordered recall over (value, time, establishing id) plus exact chain. Decision lineage is identifier recall over trigger, discussion, reasons and follow-ups plus exact match. Drift and question resolution are F1; the open inventory and reversal list are set F1.

Evidence precision is |cited and supporting| / |cited| (0 for a non-abstaining answer with no citations, not computed for a correct abstention). Evidence recall is over the establishing set. Evidence recency holds when every establishing id is cited and no superseded assertion is cited.

Abstention and staleness are balanced accuracies, so refusing everything does not score well. A confident error is wrong, with no abstention, no contested or undetermined status and no staleness flag.

Sensitivity is the paired difference perturbed minus clean on the same items. Budget curves use AUBC on a log axis (trapezoid over the budgets) and B90. Growth reports update latency in checkpoints (changes not reflected by the last checkpoint are censored) and the stale-answer rate. CPCO is (mean query cost + ingest cost / 1000) / accuracy. There is no composite score.

Item selection. T1 and T6 items are selected by a per-assertion hash, so an item exists or not independently of other assertions. This keeps item identity stable across operators, which paired comparison needs.
