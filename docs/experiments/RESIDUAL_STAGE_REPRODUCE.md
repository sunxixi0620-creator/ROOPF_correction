# Reproduce shared-state stage diagnostics

Restore the six trained checkpoints from `artifacts/residual_top_study` into `results/residual_top_study`; retain original checkpoints. No training dataset or neural retraining is required.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/residual_stage_audit.py
```

Eight CPU workers execute36 cases. Each case runs the frozen no-residual behavior twice, once without diagnostics and once with all eight scoring configurations observed at identical late states. Point, trajectory and RNG equality are mandatory for every case. Diagnostic truth values are never passed to model features or behavior decisions. No-residual recomputed decisions must match actual decisions.

Case CSVs retain state keys, candidate indices, anchor/pool/shortlist/pre-gate/post-gate truth values and regret components. Case NPZs retain unchanged factual trajectories and points. The finalizer verifies nonnegative shortlist/rerank/final regret, the exact additive decomposition within tolerance, unique keys and checkpoint hashes, then archives all files with member hashes.

For a clean rerun preserve/move `results/residual_stage_audit`. Completed cases can resume only with the same pinned script. To review delivered output, restore `artifacts/residual_stage_audit/data.zip` into that results folder and verify manifest SHA256. Running the script on complete restored cases regenerates summaries without new objective evaluations.

The shared state distribution and success-memory counters come from the no-residual policy. This isolates conditional selection differences but does not establish each model's full-policy performance or attribute final optimization gains to individual stages. Normalized local regret and blocked-proposal counts should not be reported as final performance percentages.

Before shared-state execution, the first case is also run under each of the eight native configurations, checking reconstructed shortlist/choice and diagnostic-on/off full trajectory parity. An initial implementation inherited the behavior policy pool-router switch; its completed preflight (86400 main points,239760 teacher points) is isolated under results/residual_stage_preflight_disabled_router and excluded from formal evidence. The corrected run sets the switch per model explicitly.
