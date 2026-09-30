# Acquisition optimizer × prior: development factorial

Frozen before new main-task evaluation. This is a development mechanism study,
not independent-family confirmation, not new training and not original ROOPF.

## Design

36 existing procedural configurations × 2 new instances × 3 priors (O, W, WA)
× 2 acquisition solvers (pool, continuous) = 432 trajectories. Every trajectory
has 300 paid evaluations, including 10 identical initial points. Main cost is
129,600 calls. Model seed 0 is frozen. All 12 recipes and all 3 scales are kept.
Task role: acquisition_factorial_v1. W/WA drop the prior after 40 paid calls and
retain all observations. The original 600-step local bandwidth schedule remains
unchanged; 300 is a truncated prefix, not a rescaled schedule.

Use the parent seven-group study's exact frozen checkpoints, normalization,
fixed GP hyperparameters, duplicate tolerance and original candidate RNG.
No function ID or hidden task parameters enter acquisition scores.

Pool uses the original implementation and raw-EI selection unchanged.
Continuous uses the original 128 candidates as starts/fallbacks. Select the
top 8 by stable LogEI, jointly optimize their summed LogEI with bounded L-BFGS-B
in 160 variables: 40 iterations, maxfun 65 (SciPy may finish an iteration beyond
this limit), maxls 10, maxcor 10, ftol 1e-9, gtol 1e-6. This is batched multistart
search with shared line search, not eight separately converged optimizations.
Retain 8 returned candidates and the best visited candidate; cast them to
float32 and rescore all 137 candidates with the original GP/prior. Stable LogEI
breaks underflow degeneracy if encountered. Exclude paid duplicates at 1e-6.
Original candidates remain present, so returned LogEI cannot be worse than the
original-pool maximum at the SAME state. This is not a true-objective guarantee.
Record raw-EI versus LogEI original-pool ranking discrepancies; otherwise any
observed difference includes this numerical selection change.

Optimize a double-precision smooth extension of the old model with analytic GP
derivatives and neural autograd. Fixed float32 context statistics and normalized
context targets are retained. Small dtype differences during the continuous
search are checked; final scores/GP updates always use the original predictor.
Do not fit new GP hyperparameters or change the mean/variance model.

## Contracts, hardware and cost

Before main runs: check stable LogEI against high-precision reference, analytic
gradients against finite differences and an independent torch autograd GP,
original-pool trajectory equivalence, zero-prior equivalence, paid count,
duplicate exclusion and the 40-point switch. Numerical contracts may use saved
data. Any newly evaluated contract trajectories use a separate role and their
calls are recorded separately. Contracts with a short budget do not choose the
scientific configuration based on performance.

Benchmark CPU workers and CPU/GPU numerical primitives using synthetic data or
saved states, no objective labels newly queried. Choose resources by throughput,
never objective performance. Limit BLAS/Torch CPU threads per worker to one.
Use GPU for independent derivative/parity checks and large batches if faster;
do not force small CPU-bound solves onto GPU just to increase utilization.
Preserve immutable inputs, atomic per-trajectory records and resumability.

The six trajectories at fid 0 / instance 0 are executed serially first to measure
real per-case compute cost; these remain part of the 432 main trajectories and
are not extra scientific runs. Report that their timing is a single-task example,
not a general runtime estimate. Remaining trajectories run in parallel. GPU/CPU
microbenchmark time and contract objective calls are reported separately.

## Analysis fixed before main outcomes

Primary response T: earliest paid count reaching initial_best minus 0.5 times
the initial sample standard deviation. If not reached by 300, T=301. Include
all failures. Secondary: attainment by 300, bounded improvement at 300,
the full improvement curve and average bounded improvement over counts 10–300.
Do not claim 600-evaluation noninferiority from a 300-evaluation experiment.

Primary interactions, positive means learning benefits more from the new solver:

  I_O  = (T_O - T_W)_continuous - (T_O - T_W)_pool
  I_WA = (T_WA - T_W)_continuous - (T_WA - T_W)_pool

Cluster by 12 recipes, averaging 3 scales and 2 instances per cluster.
Use 10,000 cluster bootstrap replicates and 97.5% intervals for the two primary
interactions (Bonferroni within this family). An interaction is practically
positive only if its mean is at least 5 calls and its lower bound exceeds 0.

Six secondary time contrasts (three within-prior solver improvements; W−O and
W−WA under continuous; W−WA under pool) receive 99.1667% intervals within their
own family. Learning-vs-WA efficiency requires mean time saved >=5 and lower
bound >0. Also disclose attainment and terminal intervals (95%, descriptive),
including negative outcomes. Do not combine this into a new reliability guarantee
or alter the old seven-group joint gate. One model seed precludes the old
three-seed confirmation claim.

If O/WA improve similarly, report a common engine benefit. If W still does not
beat WA, stop window/epoch search on this current mean-prior model. In either
case retain this result and proceed, if warranted, to separately specified
strong baselines and established task-transfer benchmarks. Do not replace tasks,
budgets, seeds, effects or models based on this study's outcomes.
