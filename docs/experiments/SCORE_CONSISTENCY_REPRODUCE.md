# Reproduce two-stage score diagnostics

Original repository checkpoints and generated36 source are sufficient; no training or prior experimental models are needed.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/score_consistency_audit.py
```

The script uses8CPU workers. Both full and no_residual run on36 fresh instances. Each is executed with and without observation, requiring exact points/trail/RNG equality. Additional acquisition recomputations restore saved output fields and RNG. Teacher queries are separate and never influence factual search.

The decomposition captures native36-candidate and3-candidate calls. Residual correction is the difference between native scoring and recomputation with the learned router disabled on the identical candidate set. The other term is the remaining structural/guard/operator correction, so the sum reconstructs actual scores. Candidate order changes are evaluated using teacher truth. Single-component rollback counts may overlap and are not additive causal contributions.

The prior-preserving comparison recomputes second-stage scoring and gate on the same candidates using the original two portfolio priors and anchor prior1. It is a conditional one-state comparison, not a complete-policy performance result.

To rerun cleanly, preserve/move results/score_consistency_audit. Resume only under the same pinned script; complete case JSONs skip execution. For review, restore archive data.zip into that directory and verify ZIP/member hashes in manifest.json. Case CSVs contain all five score components for both shortlisted candidates at both stages; NPZs preserve factual trajectories and query points. Final summaries can be regenerated from restored complete cases without new objective calls.
