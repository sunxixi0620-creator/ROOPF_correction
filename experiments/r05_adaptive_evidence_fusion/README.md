# R05: Adaptive evidence fusion

This round addresses the three jointly requested changes. It is a combined intervention, not a factorial ablation and not a blind confirmatory benchmark.

## Information and responsibilities

The optimizer still generates the original36 portfolio candidates. The controller receives the competing Anchor plus all36 candidates, rather than two already shortlisted portfolio candidates. Extra scoring predicts objectives; it never calls the true objective and preserves the solver RNG. The separate first Anchor slot remains protected. The exact original shortlist choice is mapped into the expanded pool, so `release` preserves its actual point.

Only previously evaluated protected/competing pairs update smoothed pairwise rank-agreement statistics, separately for proxy mean and frozen Residual support. These are selected-pair descriptive estimates, not unbiased reliability or calibrated probabilities. Their ratio updates the Residual coefficient, and their uncertainty updates acceptance margins. Each configuration also records the running mean of its observed normalized advantage relative to the simultaneously evaluated protected Anchor. That is not a counterfactual estimate of a different configuration's outcome.

The LLM selects among precomputed protected options: `release`, `adaptive_fusion`, and `online_only`. The last option removes Residual from both scoring and acceptance/veto logic. Its thresholds depend only on proxy rank reliability and its own observed returns, not Residual reliability. Anchor generation and unrelated structural protections remain; “online_only” does not mean an optimizer without any offline components.

R04 cases provide context only. Their old three-candidate action labels are explicitly not transferred as counterfactual values for new full-pool actions. All feedback snapshots are deep-copied before subsequent evaluation.

## Frozen experiment

-35 functions,100 paired trajectories per function, D10,300 evaluations including initial100.
- Arms: original-equivalent Agent-off; deterministic adaptive controller (legacy output name `retrieval`); real DeepSeek API controller (`llm_live`).
- All original100-trajectory baseline observations and RNG endpoints must match bitwise.
- Original late-stage activation and CEC structured-system locks remain. BBOB20 is reported separately because its original objective is batch-coupled and stochastic.
- API epochs0 and23 of45 active rounds, at most two per batch, with one-round configuration duration. This differs from R04's five-round cadence and must not be hidden when comparing rounds.
- Total request cap600 includes preparation calls. Failed or unknown paid requests are not automatically repeated. Objective caching requires complete request identity and stochastic RNG; it does not expose unevaluated candidate outcomes to the controller.
- User explicitly authorized the full-pool evidence and observed feedback transmission to WorldCodes/DeepSeek. Credentials are loaded outside the repository and never logged.

## Preparation issue

The first version of `online_only` reused margins derived jointly from proxy and Residual reliability. It was stopped because that was an indirect offline dependency inconsistent with the intended boundary. Its partial outputs, request journals, costs, and old code binding are retained under `setup_attempt_v1`; they are not included in formal efficacy. The formal version adds an explicit test that changing Residual reliability leaves the online-only margin unchanged. This correction was based on contract inspection, not on performance tuning.

## Reproduction and acceptance

Run `scripts/verify_r05.py`, then `scripts/experiment_r05_full.py run`, then `scripts/finalize_r05_full.py` with the existing environment. The runner resumes only matching frozen bindings. A health batch is part of the full experiment, not a replacement for it.

Only audited `COMPLETE.json` establishes full completion. The report includes per-function means and population standard deviations, paired wins/ties/losses, changed points, failures/fallbacks, actual API usage, provider credit units without inferred currency, and native/cached objective evaluations. Successful API use is not evidence of improvement.
