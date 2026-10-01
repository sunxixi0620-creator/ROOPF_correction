# HPO-B-derived development transfer reference, V1

This protocol is fixed before any target optimization on the raw-grouped cohort.
It follows HPOB_RAW_GROUPED_PROTOCOL.md: 344 eligible source tasks, all 66
development tasks, five paired seeds 0..4, 105 calls including five common
input-only random initial configurations. Confirmation responses are not
exported or queried. This is not the official HPO-B-v3 test protocol.

Source acquisition: retain at most 256 distinct rows per source task, selected
without y by a seeded permutation (`hpob_transfer_source_v1`, space, task ID).
The bank is fixed across target seeds; uncertainty over acquisition of source
data is not covered by these five seeds. Only selected source labels may be
consumed by training. Report their count separately from target calls. All
eligible source groups in the same search space are used. No task is selected
by observed performance. Every candidate X is public; source mean predictions
may be cached on these X, without target labels.

Four methods, all maximizing the original table response:

* O: an online GP with constant mean and ARD Matern-5/2 kernel, BoTorch 0.16.1
  dimension-scaled lengthscale prior, unit kernel amplitude, default learned
  noise prior. Standardize only the current paid y (sample SD; SD below 1e-8
  becomes 1). Fit marginal likelihood every step with L-BFGS-B, maxiter 100,
  ftol 1e-9. Warm-start parameters only. Failed warm fitting retries one fresh
  model; an unrecovered failure is retained, not replaced by another optimizer.
  LogEI uses maximum posterior mean at paid points as incumbent.
* P: offline-only equal average of standardized source GP posterior means.
  After the common five points, choose its highest unqueried mean. No online
  fit or adaptation of the prior; target responses only update reported best.
* A: simple fixed source mean plus online GP on residuals. Let m(x) be exactly
  P's average. At each step normalize paid target y, fit the same GP to
  z-m(X_paid), then add m(x) back in prediction. LogEI and fitting settings
  match O. No learned source weighting, manual withdrawal window or neural net.
* R: RGPE-TAF/Matern, an independent implementation of Feurer et al. (2022),
  https://arxiv.org/html/1802.02219v4, Algorithm 1 and Sections 3.2, 4.1,
  4.2.4, 4.3. Use 1,000 paired bootstrap index samples of the paid observations,
  source posterior-mean ranking losses (Eq. 3), and target leave-one-out mean
  ranking against observed target values (Eq. 4). LOO holds full-data fitted
  hyperparameters/normalization fixed, using exact Gaussian conditioning.
  Split ties equally (Eq. 5). Independently drop each source with probability
  1-(1-n/105)*Pr_bootstrap(loss_source < loss_target), then recompute weights.
  Target is never dropped. TAF is w_target*EI_target plus weighted positive
  source-mean improvements over each source's best prediction at paid X.
  Compute the sum in log space. This uses the 2022 bootstrap formulation,
  not the older BoTorch tutorial's posterior-sampling/qNEI formulation.

All source GPs use the same GP construction and fit limit as O; source labels
are standardized per selected source task. R is a mechanism reimplementation,
not a bit-for-bit reproduction of the authors' SMAC implementation: library,
GP priors, source cap, discrete candidate pool, and initial design differ.
No claim of the paper's theoretical bound for this implementation is made.

Score the complete remaining finite candidate pool in batches; do not subsample
the acquisition search. Ties choose smallest canonical index. If TAF is zero
everywhere (zero target weight and exhausted source improvements), use that
same tie rule and record a raw zero score instead of a nonfinite log score;
do not silently change to another acquisition function. Optimizers see
only public X, the source bank, and purchased labels. A separate oracle process
journals each call durably before returning it. Duplicate/over-budget calls
are errors. Recovery reads prior paid responses, never purchases them again.
Save model parameters, selected score, source weights, warnings and timings
before each query. All methods receive identical first five indices per seed.
Contract tests use fabricated responses, not extra development calls.

Primary endpoint: average normalized simple regret at calls 5..40 inclusive.
Normalization uses full target max/min only in the evaluator after 105 calls;
these quantities never reach the optimizer. For a constant table regret is 0.
Secondary: AUC 5..105, terminal regret, first attainment of regret <=0.05
(unattained=106), and source-weight trajectory. Average seeds within task,
tasks within raw dataset group, then equally weight the 11 represented groups.
Use 10,000 group bootstrap samples, seed 2026100102. Three primary contrasts
R-O, R-A, R-P use 98.333333% percentile intervals (Bonferroni family of three).
Positive regret differences favor R. Show every task, space and group.

Entry condition for a new neural prior: R improves primary normalized regret
over O by >=0.01 with corrected lower bound >0, and R's terminal-regret excess
over O has corrected upper bound <=0.01. This is a development go/no-go rule,
not external confirmation. R-P and R-A distinguish fusion from static prior
and sophisticated weighting from a simple prior. Failure does not establish
transfer is impossible; investigate diagnosed implementation/family mismatch
before training. Success does not establish neural necessity or ROOPF novelty.

CPU fitting runs in spawned workers with one BLAS/PyTorch thread each. GPU may
accelerate source prediction caches after CPU/GPU numerical contracts. Choose
resource placement from throughput, not target outcome. Record wall time,
per-stage model cost, CPU/GPU environment and all 138,600 target calls planned
(66*5*4*105); no neural training epochs are involved in this reference stage.

Preflight amendment before target calls: an initial source bank/cache (344/66,
33.9/7.2 seconds concurrent wall time) was retained under
results/hpob_transfer_preflight_v1 with its exact code and identity. The
all-zero-TAF boundary case was identified in code review and clarified above.
Re-freeze/recompute source fits and caches, using exactly the same source rows;
the extra fitting cost is reported separately, and no additional unique source
labels or target responses are acquired. No performance results informed this.
