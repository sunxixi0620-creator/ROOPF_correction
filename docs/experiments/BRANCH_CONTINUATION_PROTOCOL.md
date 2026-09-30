# One-intervention continuation audit v1

Freeze current uncalibrated New (three selected contextual proposal models).
No fitting, no new gate or residual. Dimension20, budget600, all36 procedural
configurations, one fresh branch_v1 task instance per configuration, four initial
populations, three fixed model seeds. Run each population separately (batch1)
to avoid cross-population RNG effects. 432 base trajectories, no outcome-based
state filtering. Fixed decision states after170,310,520 paid evaluations.

At each state save population, fitness, archive, operator memory, candidates,
scores, prior and RNG. After completing base rollout query a separate teacher
for two new proposals and the two actual chosen points (4 calls/state). Choose
the best new proposal only when it is strictly better than incumbent and both
actual choices. If beneficial, keep the actual first point and replace the
second with that new point. Otherwise leave selection unchanged. No duplicate
index is allowed. This is an oracle-assisted single intervention, not deployable
performance; its value does not bound all possible interventions or policies.

For each state restart from identical population and random seed, verify the
complete captured state and RNG fingerprint at intervention, and verify every
prefix point/trail. Force only the single chosen pair; subsequent decisions use
unchanged New. Branches all finish at600 with matched remaining budgets430,290,80.
For no-op interventions require full trajectory exact equality. Per-population
runs isolate all random streams. Store complete base/branch trails and points.
Actual replay cost:432*600 +1296*600 =1,036,800 function calls; includes432,000
redundant prefix calls in branches. Teacher cost432*3*4=5,184. Do not report these
oracle-assisted branches as new deployable-method trajectories.

Report immediate best-value and terminal best-value advantage using each case's
initial std and a COMMON base initial best, with u=g/(1+g). Also lead after20/60
additional evaluations (capped at600), persistence among true opportunities,
harm/no-op frequencies. Conditional opportunities and all scheduled states both
reported; samples correlated. Bootstrap12 recipe clusters5000 resamples,
95% descriptive intervals, three training seeds fixed. Aggregate all three stages
equally as primary; stage tables descriptive, no post-hoc best-stage selection.

Direction rule: only prioritize further selector development if aggregate final
advantage>=.001,95%lower>0,all three seed means>0. If upper<.001, evidence is below
this practical threshold for THIS one-intervention policy; do not infer all
fusion methods impossible. Otherwise inconclusive: no automatic new training.
No parameter sweep or second intervention variant in this round. Conditional
opportunity results explain mechanism, not an alternative acceptance endpoint.

Contracts on a separate function/seed test captured base versus original exact
parity, prefix state restoration, no-op full parity, selected-slot replacement,
true-call counts and frozen weights. Costs separate. Freeze protocol/sources
before opening branch_v1 outcomes. Old calibration failure remains unchanged.
