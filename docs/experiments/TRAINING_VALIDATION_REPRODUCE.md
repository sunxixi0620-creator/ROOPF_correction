# Reproduce the training-validation development study

Use the existing supplementary environment and repository root. Restore the
previous `retrain_d10_curated_20260928` archive first: both new arms require its
epoch80 `resume.pt`, not the selected epoch79 `anchor.pt`. The source checkpoint
is never overwritten. Use empty result directories for a new execution; do not
combine changed protocols with existing records.

1. Prepare CPU-generated, serialized task instances and initial populations:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/training_validation.py prepare
```

2. Run the two continuations. They may run concurrently if GPU memory permits
(about4–5GB allocated per training process here, plus other applications):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/training_validation.py train --arm legacy
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/training_validation.py train --arm shared_scale
```

3. Run the CPU gate controls and separately counted teacher diagnostics:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/development_auxiliary.py --workers 6
```

4. Restore `historical_label_availability.json` from the development archive if
the historical logs are unavailable. That audit counted the `is_baseline`,
`pool_size`, and missing `baseline_candidate_fit` columns of the two historical
CSV files named in the JSON. It is provenance evidence, not input to training
or model selection. All3,240,000 historical rows lack baseline truth; new paired
labels are recomputed in step3 from both anchor proposals rather than inferred
from that missing data.

5. Start the monitored finalizer (it waits for the two training arms and CPU
study). If any predecessor fails, resolve it instead of waiting indefinitely:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/finish_training_validation.py
.venv/bin/python scripts/verify_training_validation.py
```

The finalizer writes checkpoint selections before opening confirmation results,
evaluates only the frozen choices and declared references, then writes tables,
plots, and independent ZIP parts. The verifier checks selection criteria,
identical starting validation records, unique trajectory IDs, budgets, paired
labels, original final checkpoint hashes, and archive coverage/hashes.

This is an80-to160 continuation with one paired random stream and an unchanged
post80 schedule. It cannot establish which from-scratch epoch budget is best,
that all neural modules should be trained, or that a newly selected model beats
external optimizers. Generated function families are known; only the sampled
validation/confirmation instances and initial populations are separate.
