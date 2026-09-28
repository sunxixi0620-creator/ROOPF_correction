# Reproduce residual target development study

Use the repository `.venv` with PyTorch, NumPy and pandas. Run from repository root. The script uses eight CPU workers for collection and confirmation; training uses CUDA when available. Original frozen checkpoints and `artifacts/training_provenance/generated36_curated.py` are required.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u scripts/residual_target_study.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u scripts/finish_residual_target_study.py
```

For a clean rerun, preserve/move any existing `results/residual_target_study` first. Do not combine outputs from modified script versions: the protocol pins the script hash. Existing complete collection cases and confirmation cases can be resumed; training before `SELECTION_FROZEN.json` restarts deterministically. Confirmation labels are collected only after all four model selections are frozen.

To audit the delivered outputs without rerunning, extract all `artifacts/residual_target_study/part*.zip` files into `results/residual_target_study`, verify ZIP/member SHA256 against the manifest, and run the finalization script. It checks feature finiteness, both labels against saved truth, disjoint seed namespaces, training-only normalization, validation-based selected epoch, original checkpoint hashes and paired outcome counts. The finalization script regenerates the results document and archive; interpretation is separately maintained in `RESIDUAL_TARGET_DECISIONS.zh-CN.md`.

Each case contains 400 pools (100 rounds ×4 trajectories), each with two anchor candidates and36 portfolio candidates. Rows preserve pool order, allowing reconstruction of the second-anchor target from `fit[:,1:2]`. The initial-best feature reconstructs the incumbent target. No candidate truth or anchor truth is a predictor feature. Case identity/split/parameters/populations are in `tasks.pt`.

This experiment does not train the anchor, gate or online ridge model. Full-policy confirmation changes only the residual checkpoint, leaving the final unlocked policy's remaining settings fixed. Confirmation label metrics use the no-residual behavior distribution; standalone predictive accuracy is not a claim of long-term optimization gain.
