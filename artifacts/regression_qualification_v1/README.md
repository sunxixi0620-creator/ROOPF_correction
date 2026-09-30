# Cross-dataset regression task qualification v1

Joint task-screen gate failed: only one of three development datasets meets the fixed difficulty criterion. No fusion optimizers run, no original checkpoint changed, no neural epochs, no confirmation data downloaded. Six dataset identities, roles, 128 configurations, tolerance and screening rules were frozen before fitting in commit `fdd82e0`.

The ZIP contains official UCI raw downloads, processed arrays, exact group splits, all18 tables,54 repeated evaluations, elapsed times, source identity, reports and figures. Raw data and processed versions are CC BY4.0; attribution, DOIs and modifications are in `docs/experiments/REGRESSION_TASK_QUALIFICATION_PROTOCOL.md` and `results/regression_qualification_20260930/DATA_PROVENANCE.json`. Wine raw CSV contains both colors; only red rows enter this experiment. No confirmation datasets are included.

Extract at repository root and verify `MANIFEST.json`. Recompute audit/figures without any additional SVR fits:

```bash
.venv/bin/python scripts/regression_task_qualification.py audit
.venv/bin/python scripts/finalize_regression_qualification.py
```

For a fresh repeat, use a separate checkout, move aside the extracted run directory, and run phases `prepare`, `run --workers 24`, `audit`, then the finalizer. Total2358 SVR fits, including54 deterministic repeats. Python/numpy/scipy/pandas/sklearn versions recorded in identity.json; requests fetches official files, matplotlib generates figures.

This is a finite-grid development qualification, not optimizer benchmarking, a claim of physical-system optimization, or an independent confirmation result. Exact duplicate observations removed; identical predictor rows grouped to prevent train/validation overlap. Each standardizer fitted on training rows only. Outcomes from all six datasets are retained.
