# Frozen external verification of the simplified unified candidate

Stage3 selected `no_residual` by the predeclared rule. Full-vs-anchor paired
bounded-improvement effect was .0373435; full-vs-no-residual was -.0007357 with
an interval crossing zero. Protection did not meet its risk criterion. These
development results justify testing a simpler candidate, not claiming residual
or protection benefit. No new loss, residual model, or threshold is selected.

## Fixed candidate and matched controls

Primary method: unified `no_residual`, using the three selected10-D anchors from
`artifacts/unified_revision_v1/anchor_{0,1,2}.pt`. Keep all scoring constants,
error margin and update rules unchanged. Add `no_residual_score_only` as a
*matched ablation*, because the earlier score-only retained residual and cannot
isolate protection in the selected simpler method. This control is not a new
candidate selection round: external results will not switch the primary method.

Native20-D anchors use the same architecture family,80-epoch maximum, batch16,
three seeds, quality loss,300-NFE training rollout, validation selection and
early-stop rule. Only dimension-dependent tensor shapes and generated tasks
change. Train from scratch on the same predeclared generator, never on external
tasks. Do not transfer10-D weights or describe this as a high-dimensional
residual validation. No20-D residual training is needed for the selected method.
The common experimental API may load the legacy27-feature template, but
no-residual/anchor-only variants never execute its prediction network or use its
output. Report actual component execution, not just loaded modules.

## New test source and scope

Use all12 CEC2022 classes from the installed, version-recorded opfunu package.
Repository search found no previous CEC2022 experiment in this project. This is
a new suite for this candidate, not a claim that all mathematical primitives are
unseen or that historical BBOB/CEC development contact disappeared. Record exact
implementation and support-data hashes. Diagnostic evaluation at the declared
optimum passed for all12 functions at10-D and20-D (24 implementation-check
queries; no candidate comparison or tuning). These diagnostics are separate
from optimization budgets and never provide points or values to the optimizer.

Three fixed dimension/budget conditions: (10,300), (20,300), (20,600).
All12 functions, no selection by performance. Use supplied shifts/rotations;
30 optimizer runs per function/condition/method. Five methods: selected
no_residual, same-anchor-only, matched no_residual_score_only, CMA-ES, GP-EI.
Total12*3*30*5=5400 trajectories and2,160,000 objective evaluations.

The30 run indices are partitioned into three groups of10; learned methods use
anchor seed floor(run_index/10), and classical methods use the same30 run groups
with distinct optimizer seeds. Within each run group all methods see the same
objective and budget; learned comparisons share initial points and policy RNG.
Training seed and run-group variability are not fully separated in this external
design; the earlier development experiment crossed shared populations with all
three training seeds. Do not pool duplicate classical runs or manufacture an
extra baseline training-replicate sample size. Native GP and CMA initialization
are retained and fully counted; do not force GP to spend100 initial evaluations.

## Input and evaluation contract

Use the same affine coordinate interface for every bounded objective: internal
box[-5,5]^d maps to the declared physical bounds. Training/development boxes
already equal[-5,5], so this map is exactly identity there; it is fixed before
external comparisons and does not use the optimum. Baselines use the same
objective interface (GP additionally parameterizes it in[0,1]). No objective
normalization using hidden test statistics is introduced. The neural controller
retains float32 inference; objective values are evaluated/tracked in float64 by
the external implementation, with float32 copies passed to the neural archive.
Report final performance from the actual objective trace, not rounded neural
trail values. This precision difference remains an implementation limitation.

True evaluation counts include initialization. Store all evaluated normalized
points, raw objective values, cumulative best trace, seeds, weights and code
hashes. Teacher calls are forbidden. The known optimum is used only in reporting
and the disclosed implementation checks, never acquisition, early stopping or
candidate generation. Every job uses a verified locked artifact; all workers
finish before completion markers/archiving.

## Baseline settings

CMA-ES: pycma recorded version, internal starting mean0, sigma2, population10,
box[-5,5], exactly the specified budget, no optimum-based stopping. GP-EI:
20-point Latin hypercube initialization; constant times ARD Matern5/2 kernel,
noise1e-6, normalize_y=True; optimize kernel every20 observations using L-BFGS-B,
max60 iterations, no restart;512 global Sobol plus512 local normal acquisition
candidates (sigma.1 in unit coordinates), EI xi0, avoid exact duplicate queries.
These match the prior baseline recipe; no external-performance-driven tuning.

## Reporting and decisions

Primary external metric: log10(max(final_objective-f_global,1e-8)), lower better.
Report raw errors/medians and per-function means alongside it. Pair all methods
by function, run group and run index. Report each dimension/budget separately;
the three conditions are not36 independent function families. Use5000 paired
bootstrap draws over functions, run groups and within-group run indices. Treat
the12 suite functions as a finite suite; intervals are not universal guarantees.

Report selected-vs-anchor, selected-vs-matched-no-protection, selected-vs-CMA,
selected-vs-GP. Positive reported advantage is log-error(comparator)-log-error
(selected). Practical tie for descriptive counts is absolute paired mean
log-error difference<=.01. For matched protection, also report fraction of runs
worse than same-anchor by>.02 in initial-standardized bounded utility (using the
learned methods' common initial population). Do not infer protection from mere
existence of a fallback. No parameter changes, benchmark exclusions or
post-hoc widening of the test set based on these outcomes.

If strong baselines remain better, report that limitation and narrow the paper;
do not start another residual/threshold search. Keep original final checkpoints
unchanged and report this research candidate separately. Serial timing uses a
preselected small panel after the parallel run, with the same settings and
separate counters; those repeats are timing observations, not independent
performance samples. Offline training and all extra diagnostics are listed
separately. Shared-host timings are not exclusive-device benchmarks.
