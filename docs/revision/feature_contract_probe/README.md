# Audit evidence and reproduction

These are bounded implementation/provenance checks, not new optimizer performance experiments. No weights or archived training cases were overwritten.

Prerequisite: restore artifacts/residual_target_study into results/residual_target_study.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_training_feature_parity.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_residual_dataset_replay.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_legacy_case_recovery.py
```

The feature probe selects indices0/24/48 before inspecting results and compares logged38-candidate scores with native36/3-candidate scores. It separately records whether current replay matches historical data; it does not silently treat a historical mismatch as equality.

The dataset audit replays all144 archived behavior trajectories;140 exactly match the declared settings. Four mismatches are training indices24–27. The legacy recovery uses the saved four legacy tasks in recovered_legacy_tasks.pt and the old policy seed offset, recovering all features, labels, teacher fitness and full trails exactly. Function parameters equal declared parameters; populations/policy seeds differ. The effective manifest records actual recovered metadata without rewriting the original protocol or data.

These checks do not prove the feature mismatch caused all negative results, and they do not establish that every historical benchmark is invalid. Read the parent EXPERIMENT_CHAIN_REVIEW.zh-CN.md for evidence boundaries and decisions. Audit scripts can repeat objective calls; their per-run counters are in the manifests. Initial failed diagnostic attempts and repeated recovery checks are not additional independent samples.
