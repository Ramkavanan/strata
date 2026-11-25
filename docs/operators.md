# Operators

Ground truth for every item is computed from the perturbed world model, so a perfect system scores the same under every operator. Items about the targeted variables form the affected set.

| operator | transformation | correct behaviour |
|---|---|---|
| clean | none | correct answer |
| stale(tau) | events after tau removed, queries posed at the original time | state as of tau and raise the staleness flag |
| contradict_low(k) | k later decisions in the same domain by lower-authority roles; k defaults to a quarter of the eligible variables, at least 2 | answer unchanged |
| contradict_cross(k) | a second governing domain plus a conflicting decision in it; k as above | report contested and cite both sides |
| incomplete(rho, mode) | every establishing decision removed for a fraction of variables, indirect mentions kept (omit, or truncate and mark partial) | undetermined or abstain |
| late(lambda) | a fraction of artifacts become documents ingested after later events | order by event time; ground truth equals clean |
| noisy(eta) | decoy artifacts at ratio eta, half near-duplicates about similarly named entities | correct answer despite volume |

Reading `incomplete`: the correct answer is undetermined, so a system that retrieves nothing is right. Compare with clean accuracy before drawing conclusions.
