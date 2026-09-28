# Reproduce the controlled revision

This experiment is separate from the original `final_unlocked` entry. The latter remains available through `run_roopf.py` and `roopf.factory.build_final`. The research candidate is built only through `roopf.unified.build_unified`; its selected external variant is `no_residual`. Do not rename an original-checkpoint run as a new-candidate run.

## Inspect completed evidence without training again

- Development: `docs/revision/unified_execution/development.csv`, `DECISION.json`, `EARLY_EFFECTS.json`, and `decision_timing.csv`.
- External: `docs/revision/unified_external/results.csv`, `function_summary.csv`, `COMPARISONS.json`, and `VERIFICATION.json` after finalization.
- Training: selected epoch/total epoch in the corresponding `TRAINING.json` and `NATIVE20_TRAINING.json`, with full validation histories.
- Models and raw case artifacts: `artifacts/unified_revision_v1` and `artifacts/unified_external_v1`. ZIP files retain the per-case content/identity manifests. The external archives are split by dimension, budget, and method; each complete archive contains 360 trajectories.
- Costs: both stages' `COSTS.json`; serial repeats in `serial_timing.csv` are excluded from the performance sample size. Do not use heavily concurrent or paused workers' wall times as serial algorithm timings.

The old run's source hashes intentionally differ from the current dimension-generalized implementation. Reusing an old run directory with changed source/configuration is rejected. Inspect its archived data, or use its source snapshot in a separate directory.

## Rebuild stage three

Extract `artifacts/unified_revision_v1/source_snapshot.zip` into a separate checkout/directory and copy the two original files under `checkpoints/` there. The snapshot includes the exact computation source and frozen protocol used for the 5,184 development trajectories. Use the recorded environment in `docs/revision/unified_execution/environment.json` and `identity.json`.

From that separate directory, with the matching Python environment:

```bash
python scripts/unified_revision.py prepare --run results/new_development_run
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/unified_revision.py all --run results/new_development_run --gpu-workers 3 --workers 8
```

This retrains three anchors, collects teacher labels under each selected anchor, trains three residuals with validation stopping, freezes all selected models, and evaluates the six fixed configurations. It is a full reproduction, not a quick smoke test. Separate CPU/GPU scheduling can change elapsed time; do not change objective definitions, splits, model selection, or optimizer settings. The recorded CUDA environment is shared, and exact floating-point agreement across different devices/library versions is not promised.

## Rebuild stage four

Use the current frozen external sources, also archived in `sources_and_timing.zip` after completion. Required 10D assets are the three anchor and three residual payloads under `artifacts/unified_revision_v1`; the residual prediction network is not executed by any external learned method. Train native20 anchors from scratch under the fixed generator and validation protocol, or restore the selected native20 artifacts and their recorded metadata.

The external driver expects the native20 training directory at `results/unified_native20_20260928`. In a fresh checkout, recreate its `identity.json` from the external identity's `native20_training` field. Restore each `anchor_s/selected.pt` from `artifacts/unified_external_v1/anchor20_s.pt`, its `COMPLETE.json` from the corresponding entry of `NATIVE20_TRAINING.json`, and its history from `anchor20_s_history.json`. For archive-only verification, also restore `NATIVE20_COMPLETE.json` as the native20 directory's `COMPLETE.json`. These are historical completion records, not evidence that a new training run has executed.

For an actual fresh native20 training run at that expected path, first ensure the directory does not already exist, then run:

```bash
python scripts/unified_revision.py prepare --run results/unified_native20_20260928 --dimension 20
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/unified_revision.py anchor --run results/unified_native20_20260928 --gpu-workers 3
```

The `anchor` mode joins its workers before returning. Record its successful process exit and root completion metadata before full finalization. It does not train a 20D residual model.

With verified native20 selections and matching versions of opfunu, SciPy, scikit-learn, NumPy, PyTorch and pycma, run in a new external directory:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/unified_external.py prepare --run results/new_external_run
CUDA_VISIBLE_DEVICES='' python scripts/unified_external.py baselines --run results/new_external_run --workers 12
CUDA_VISIBLE_DEVICES='' python scripts/unified_external.py learned10 --run results/new_external_run --workers 8
CUDA_VISIBLE_DEVICES='' python scripts/unified_external.py learned20 --run results/new_external_run --workers 8
```

These modes can run concurrently subject to memory/CPU limits. `prefetch_external20.py` and `overlap_external_baselines.py` are optional scheduling helpers, use the same locked case function, and must be joined before final timing. This recorded run used both. For reproducing its strict final audit, preserve the corresponding scheduler completion records; do not fabricate records for helpers that were never run.

Use `finalize_unified_external.py timing` for the predetermined serial panel and `finalize_unified_external.py finalize` for verification, paired analysis, plots, and archives. The supplied finalizer intentionally expects this recorded run's scheduler metadata and native-training completion record. A fresh execution that uses different scheduling should document that scheduling and adapt only the completion/provenance checks, retaining the numerical analysis and scientific protocol. The performance runner is usable independently of this archival convenience script.

To verify existing raw external artifacts, extract the 15 case ZIPs under one results directory and copy the archived identity and completion metadata there; restore native20 model metadata as above. `CaseStore.load` checks code/configuration/seed identity and actual data hashes, not just file existence. Repeated scheduling of a completed case does not create a new independent trajectory.

## Interpretation constraints

All three training seeds are reported. Development and external conditions use different primary scales, so their effect sizes are not directly interchangeable. External intervals are nominal pointwise intervals over the fixed suite hierarchy. Known optima are accessed only for the disclosed implementation checks and post-optimization reporting. Original BBOB/CEC development exposure and the 12 shared procedural recipes remain disclosed. No external result selects a new model or authorizes another local search in this protocol.
