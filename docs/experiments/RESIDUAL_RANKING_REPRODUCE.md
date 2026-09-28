# Reproduce ranking/magnitude residual experiments

Restore `artifacts/residual_target_study/part*.zip` into `results/residual_target_study` first. Its train/validation cases and two second-anchor checkpoints are prerequisites. Original checkpoints remain under `checkpoints`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/residual_ranking_study.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/finish_residual_ranking_study.py
```

Training uses GPU if available, and confirmation/diagnostic evaluation uses eight CPU workers. Both objectives use the same two training seeds. Complete confirmation cases can resume; unfinished training restarts deterministically until all four selections are frozen. For a clean rerun preserve/move `results/residual_ranking_study`; never combine changed source versions with existing outputs.

The finalizer checks pinned train/validation file hashes, original checkpoints, training-only feature statistics, validation-based checkpoint selection, exact normal/diagnostic final values, and archive members. Each teacher confirmation run checks point/trajectory/RNG equality with its ordinary no-residual control. Main objective budget and teacher points are separate in the report. No previous confirmation or decision-audit case is training data.

To review delivered evidence, restore all archive members from `artifacts/residual_ranking_study` into `results/residual_ranking_study` and verify the manifest. Raw confirmation results, early-stop histories, trained checkpoints, teacher features/truth and decision CSVs are retained. All eight configurations remain in the report; none is selected by confirmation performance. Keep scientific interpretation in `RESIDUAL_RANKING_DECISIONS.zh-CN.md` separate from regenerated tables.
