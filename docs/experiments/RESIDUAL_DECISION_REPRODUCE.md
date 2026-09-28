# Reproduce residual decision diagnostics

Prerequisites: repository environment and original checkpoints, plus all four trained residual checkpoints from `artifacts/residual_target_study` restored into `results/residual_target_study`. Run from repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/residual_decision_audit.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/finish_residual_decision_audit.py
```

The first command runs eight CPU workers; no neural training or GPU-heavy work is necessary. The second audits fixed source/protocol/checkpoint hashes and summarizes results. For a clean rerun, preserve/move existing `results/residual_decision_audit` first. Complete cases can resume under the same pinned script. Never mix modified script outputs with existing cases.

For read-only verification, restore ZIP members from `artifacts/residual_decision_audit` into `results/residual_decision_audit`, checking archive and member SHA256 against `manifest.json`. Per-case NPZ files retain factual points/trails and, where a changed decision exists, replay points/trails. Per-case CSV files retain diagnostic teacher values. JSON files record which first round was disabled and final paired outcomes.

Normal factual diagnostic execution never mutates the optimizer state with teacher outcomes. The replay intentionally removes residual for one specified round, recomputes ranking and gate, then restores it. Every replay checks exact prefix points/trails and exact equality of the substituted second candidate to the factual state's no-residual counterfactual. If a case has no selected-point changes, it has no replay. Within replayed batches, some trajectories may have unchanged choices; the report separates them from the actually changed subset.

Interpretation is in `RESIDUAL_DECISION_DECISIONS.zh-CN.md`. This is not a decomposition of the full residual contribution, nor an external benchmark performance claim.
