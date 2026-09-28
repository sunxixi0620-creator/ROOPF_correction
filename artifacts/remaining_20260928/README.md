# Remaining supplementary evidence

Extract **every** ZIP part of a stage at repository root; each part is an independent valid ZIP containing original `results/` paths. No binary concatenation is needed. `manifest.json` records every file and ZIP SHA256; CRC validation was performed after archive creation. Files are grouped by size, never selected by performance.

`summaries/` contains browsable protocols, tables and training curves. Paired tables and final interpretation are under `docs/experiments/`. Reconstructed training checkpoints are supplementary controls, not replacements for the frozen main checkpoints.

Stages appear only after their completion markers exist. The final archive command uses `--require-all` so a missing stage is an error.
