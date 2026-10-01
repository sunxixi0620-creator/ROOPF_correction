# Frozen fitted-online reference audit

This is a development comparison on previously exposed procedural families, not
external confirmation or a new attempt to establish W > WA. No prior retraining.

## Tasks and accounting

- Reuse `continuous_prior_replication_v1` task role, all 36 configurations (12
  recipes × 3 scales), instances 0 and 1, dimension 20. The first two instances
  are selected by index before running the new methods, never by outcomes.
- Budget 300 including exactly the same first 10 points as existing O/WA/W.
- Two new methods, 144 trajectories, 43,200 paid evaluations; separate two-step
  reproducibility contracts, 48 evaluations. Do not rerun W or copy shared O/WA
  controls into independent replicates. W averages its three frozen model seeds.
- No true gradients, teacher calls, target extrema, or source data in optimizers.
  Repeated/invalid proposals are rejected without objective access. Restarts are
  charged; no termination based on unknown optimum.

## Baselines fixed before outcomes

1. **GP_LogEI**: BoTorch 0.16.1 SingleTaskGP defaults: learned constant mean,
   dimension-scaled RBF lengthscale prior and learned Gaussian likelihood.
   Input box mapped to [0,1]^20; Standardize uses only paid responses. Minimize
   the objective by fitting its negative. Float64 GP, original float32 objective.
2. **TuRBO_LogEI**: one trust region, sequential analytic LogEI variant based on
   the official BoTorch TuRBO tutorial. ARD Matern-5/2 with ScaleKernel,
   lengthscale bounds [0.005,4], noise [1e-8,1e-3]. Trust-region initial/min/max
   lengths 0.8 / 0.5^7 / 1.6; success tolerance 10 (tutorial), failure tolerance
   20 at q=1, relative success 1e-3. ARD weights have geometric mean one.
   Local data reset on restart; ten new Sobol points are charged and capped by
   remaining budget. Global incumbent remains in the reporting ledger.

Both refit marginal likelihood after every observation, max 100 L-BFGS-B
iterations, ftol 1e-9, BoTorch retry handling. Warm-start only learned parameters;
recompute outcome statistics from currently available data. Optimize LogEI using
10 starts, 512 raw samples, max 200 iterations, batch limit 5. Rescore all ten
returned starts and 512 deterministic Sobol fallbacks at the exact float32 points
sent to the evaluator. Exclude global duplicates within 1e-6. No silent fallback
to random search on fit errors: persist error and stop for engineering review.

This TuRBO variant uses q=1, initial 10, tutorial success=10, and analytic LogEI;
it is **not** the original author's TS implementation (which uses success=3,
Adam 50 steps, and sampled candidates), and must always be labeled accordingly.
The comparison tests stronger online references; it is not a matched component
ablation because kernels, fitting, and acquisition engines differ.

## Fixed analysis

Reuse the previous target: improvement of 0.5 initial sample standard deviations
over the best of the first ten. Nonattainment is 301. Report restricted mean time,
attainment at 300, bounded improvement g/(1+g), full curves and costs. Primary
descriptive contrasts: W vs each new baseline and WA vs each new baseline, with
12-recipe clustered bootstrap (10,000, seed 2026100101), 98.75% intervals for four
comparisons within each endpoint. Report O also. No claim of global family-wise
coverage across endpoints. Same practical scale of five evaluations; no further
same-family sample addition if the learned increment remains insufficient.

Audit every case's budget, unique points, finite values, exact common initial
data, trace/paid-step correspondence and trust-region bounds. Replay predictive
scores at paid counts 10,39,40,149,299 where not a restart, including an independent
GPU replay. Distinguish concurrent wall-clock from per-case CPU elapsed cost.

References: [BoTorch getting started](https://botorch.org/docs/v0.16.1/getting_started),
[official TuRBO tutorial](https://botorch.org/docs/v0.16.1/tutorials/turbo_1),
[author implementation](https://github.com/uber-research/TuRBO/tree/de0db39f481d9505bb3610b7b7aa0ebf7702e4a5).
The newer BoTorch releases require Python >=3.11; 0.16.1 is pinned to avoid
changing this project's Python 3.10 / torch environment.
