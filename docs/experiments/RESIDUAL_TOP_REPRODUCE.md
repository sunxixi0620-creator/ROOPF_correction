# Reproduce top-candidate residual study

Restore the training/validation cases in `artifacts/residual_target_study` to `results/residual_target_study`. Original frozen checkpoints remain under `checkpoints`. No prior confirmation cases are inputs to training.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/residual_top_study.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/finish_residual_top_study.py
```

Training uses CUDA when available. Confirmation and decision diagnostics use eight CPU workers. All three objectives use the same two random seeds and validation top1-regret selection, with40epoch cap/patience8. Only after all six selections freeze does the script prepare fresh confirmation instances. Repeated teacher and diagnostic executions are separately counted; they are not additional independent samples.

Preserve/move `results/residual_top_study` before a clean rerun. Cached complete confirmation/diagnostic cases can resume under the same source version; unfinished training before the selection marker restarts deterministically. Do not mix cached outputs from modified sources. Checkpoint/normalization/data/source hashes are verified by the finalizer; it also verifies selection from recorded validation regret and diagnostic final-value equality. Every teacher case checks exact normal/teacher trajectory, search-point and RNG equality.

To inspect delivered evidence, restore archive members into `results/residual_top_study` after checking manifest hashes. Training histories, all six checkpoints, confirmation labels/features, raw final outcomes and all decision observations are retained. Both seeds of every objective remain in the report. `RESIDUAL_TOP_DECISIONS.zh-CN.md` contains interpretation separate from generated results.

Current confirmation metrics describe known generated families with fresh parameters. The validation split has been reused across development rounds; results do not constitute a new unseen-family benchmark test or a guarantee of generalization. Ranking scores use the existing runtime sigmoid/gate interface without a calibrated-probability claim.
