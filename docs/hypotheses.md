# Hypotheses and how the harness tests them

`strata study` runs every block for every (system, reader) pair, the mechanism ablation, the other consumer tracks
and the fleet comparison, then classifies the pre-registered hypotheses (paper Section 9.1). `strata hypotheses
--evaluations 1,2,3` classifies from existing evaluation runs. Verdicts are supported, disconfirmed, inconclusive or
not decidable, and every section says what it rested on.

| hypothesis | implemented test | needs |
|---|---|---|
| H1 perturbations cause measurable, architecture-dependent changes | at least one operator classified degraded for at least half the systems (state accuracy on affected items), and an operator-by-system interaction by a permutation test within item | 2 or more systems with memory (`no_context` is excluded: it has nothing to degrade) |
| H2 mechanisms help against some failures, not all | difference in sensitivity between the full reference system and each ablation, per operator; supported if some mechanism is improved under one operator and equivalent under another | `strata ablate` or the study |
| H3 systems differ in detecting and recovering from change | median update latency of the most and least responsive systems at least one checkpoint apart (growth block), and best-against-worst accuracy under `contradict_low` beyond the smallest effect | growth and perturbation blocks |
| H4 quality ranking differs from cost ranking | 95% interval for Kendall tau-b between the accuracy ranking and the cost-per-correct-outcome ranking excludes 1, plus at least one pair reversing with both differences outside their intervals | 3 or more systems; costs at the reference rates in the price table |
| H5 context architectures pay for themselves | outcome improved over `no_context` and CPCO below both `long_context` and `vector`, with the same reader | those three baselines run with the same reader |
| H6 (exploratory) AUBC ordering differs from the ordering at the largest budget | tau-b between the two | budget block for 3 or more systems |
| H7 (exploratory) staleness-flag accuracy weakly related to clean accuracy | tau-b | perturbation block for 3 or more systems |
| H8 (exploratory) shared and independent fleets differ | consistency, propagation lag and cost of shared against independent instances of one system | fleet track |

Also reported: the binomial mixed model with template and item random intercepts (variational Bayes, `stats`
extra) and its joint test of the system-by-operator terms, next to the permutation test. The hazard model for update
latency is fitted when the data allow it; with deterministic systems that always update at the same checkpoint it
reports separation instead of a coefficient.

## Power

`strata power` runs the five pilot templates (excluded from every split), measures the quantities the simulation
needs (affected state items per template, clean accuracy, intra-template correlation) and simulates the whole
procedure: bootstrap over templates, then the classification rule. It reports power for two rules:

- the rule as written in Section 9.1 (95% interval wholly beyond the 5-point bound), and
- ordinary detection (95% interval wholly below zero).

The first rule cannot classify a true effect equal to the bound with useful power at any sample size: the interval is
centred on the bound, so about 2.5% of it lies beyond. The paper's power target should name a larger effect to
classify, or the detection rule. This is a finding about the pre-registration, not about any system.
