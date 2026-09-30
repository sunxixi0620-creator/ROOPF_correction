# Fixed continuous solver: three-seed within-family replication

This protocol is written after seeing the acquisition-factorial development
results. It tests a follow-up hypothesis and does not replace that study's two
failed primary interaction tests. It is not a never-seen-function-family test.

Freeze the exact existing acquisition_search.py solver, all GP parameters, the
10-observation initialization, prior withdrawal at 40, and the 300-call budget.
No new training, checkpoint selection or parameter changes are permitted.

36 configurations × 4 new instances × (W seeds 0/1/2 + one O + one WA) gives
720 trajectories, 216,000 paid calls. Independent task/RNG role is
continuous_prior_replication_v1. Shared O/WA controls are evaluated once per
task, never counted as three independent observations. All methods share the
same initial paid points and objective values. All twelve recipes remain.

Primary claims: continuous W saves >=5 evaluations versus each of O and WA,
with a positive adjusted interval lower bound and positive effect for every
frozen training seed. Use the same 0.5-initial-sample-SD quality target,
failure-censored T=301, all failures retained. Average seed effects within each
task, then three scales/four instances within each of twelve recipe clusters.
10,000 cluster bootstrap replicates, 97.5% intervals for two primary comparisons.

Separately disclose the 300-call terminal bounded-improvement difference and
attainment difference, with two-comparison 97.5% intervals per endpoint. Keep
the old protective thresholds visible: terminal lower bound > -0.005 and mean
attainment difference >=0. Joint scientific acceptance requires both primary
efficiency comparisons and both protective conditions. No claim of a single
97.5% simultaneous interval over all endpoints or of 600-call noninferiority.
All learning-seed effects and all recipe effects are reported.

All models were frozen in the earlier fewshot study. New W uses the corresponding
fold/seed weights; O/WA have no training seed. Replication tests learning's
incremental efficiency under the new solver, not necessity of old residual,
fixed withdrawal, or all possible offline knowledge.

Before main execution, compare the seed-0 loop to the frozen factorial loop for
O/WA/W at 45 calls (270 total contract calls), and run two 45-call seed-1/2 smoke
contracts (90); total contracts 360, recorded separately. Do not use their
objective performance to tune anything. Resume verified cases only. CPU worker
choice reuses the completed resource test; GPU verifies frozen-prior parity.

If this single bounded replication fails, retain the result and stop extending
this same-family test for significance. Do not search seeds, tune windows, add
epochs or change the effect threshold. If it passes, freeze this as a candidate
for strong baseline/external task evaluation; it still does not establish broad
generality or paper-level novelty.
