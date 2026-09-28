# Checkpoint before supplementary experiments

`results-before-supplement-20260928.zip` preserves all 319 files currently under
`results/`, including historical experiment outputs, logs, and protocols. The
adjacent JSON records each uncompressed file's SHA-256 and the archive SHA-256.
Every archived file was read back and checked against its original hash.

The selected final ROOPF configuration is `build('unlocked')` in
`scripts/coordinate_experiment.py`. The original `run_roopf.py` default remains
unchanged. The checkpoint freezes the current source, the two model artifacts,
revision plans, and historical results before new experiment instrumentation.
Historical protocols retain their original hashes; some driver scripts have
changed since those historical experiments, as recorded in the revision audit.

Virtual environments, caches, editor settings and ignored paper-audit extracts
are not part of this Git snapshot. The original manuscript PDFs are outside the
repository and are referenced by path and hash in the final-method support plan.

To restore the archived results without touching an existing results directory,
unzip the archive into a new directory. Keep subsequent experiment outputs in
new directories and do not overwrite this archive.
