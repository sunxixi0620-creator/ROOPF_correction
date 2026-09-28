# Reproducing the remaining experiments

Use repository root and the package versions in `requirements-supplement-final-20260928.lock.txt`. The original final checkpoints remain under `checkpoints/`; reconstructed checkpoints go only to their result directories. Recover archived evidence by extracting every ZIP part under `artifacts/remaining_20260928/` at repository root. SHA256 and ZIP member paths are in its manifest.

For a fresh execution, choose an empty working copy or move existing result directories aside. Each experiment checks its frozen protocol before resuming; do not edit existing result records to make a changed protocol pass. Training resumption restores optimizer, RNG, parameters and history. CPU commands should set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''`; GPU training should omit the CUDA mask and use two CPU numerical-library threads.

## Main remaining evaluations

```bash
.venv/bin/python scripts/remaining_mechanisms.py --stage operators --output results/remaining_operators_20260928 --workers 8
.venv/bin/python scripts/remaining_mechanisms.py --stage replay --output results/remaining_replay_20260928 --workers 6
.venv/bin/python scripts/remaining_uav.py --output results/remaining_uav_20260928 --workers 6
.venv/bin/python scripts/remaining_gp_ei.py --output results/remaining_gp_ei_20260928 --workers 8
.venv/bin/python scripts/remaining_surr_rlde.py --output results/remaining_surr_rlde_20260928 --workers 8
.venv/bin/python scripts/random_shortlist_control.py --output results/random_shortlist_20260928 --workers 8
```

## Controlled training and dependent evaluation

Launch the following two training jobs with GPU access (concurrently only if memory permits). Both use64trajectories per function,80epochs,36functions and population100; there is no test-based checkpoint selection.

```bash
.venv/bin/python scripts/reconstruct_anchor_training.py --dim 20 --order canonical --output results/retrain_d20_canonical_20260928
.venv/bin/python scripts/reconstruct_anchor_training.py --dim 10 --order original --output results/retrain_d10_original_20260928
```

The orchestrator below waits for the20-D completion marker, launches the paired10-D curated-order training, generates native20-D residual labels, trains the native residual using the copied historical trainer, and runs all prescribed20-D and training-order comparisons. It sets CPU/GPU environments itself. Start it after the two training jobs have been launched, or after they finish; it does not launch those first two jobs.

```bash
.venv/bin/python scripts/finish_remaining_pipeline.py
```

Observe individual `results/<stage>.log` files and completion markers. A marker is written only after successful budget/finite-value checks. The orchestration manifest records commands and durations. If a predecessor fails, inspect and fix that failure rather than waiting for its marker indefinitely; the orchestrator is designed for this monitored session, not as a cluster scheduler.

## Final serial timing, tables, report and archive

After the other jobs finish, run the timing command with a single CPU numerical-library thread and no CUDA device. Then generate results and archives:

```bash
.venv/bin/python scripts/serial_external_timing.py
.venv/bin/python scripts/summarize_remaining.py
.venv/bin/python scripts/plot_remaining.py
.venv/bin/python scripts/write_remaining_report.py
.venv/bin/python scripts/archive_remaining.py --require-all
```

`write_remaining_report.py` and `archive_remaining.py --require-all` require every remaining stage to be complete. The report separates online experiments, separately charged diagnostics and offline training/teacher evaluations. Search-seed bootstrap intervals do not describe training-seed uncertainty or correct for multiple hypotheses.

All original algorithm checks and the previous supplementary stages remain documented in their own protocols. No claim is made that the newly reconstructed training reproduces the exact historical80-epoch command or random seed.
