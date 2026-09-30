# Reproduce the contextual proposal pilot

Protocol: [frozen design](../../experiments/COMPLEMENTARY_PROPOSAL_PROTOCOL.md).
Run from repository root using the existing environment:

```bash
.venv/bin/python scripts/complementary_proposal_study.py --workers 12
```

Default output: `results/complementary_proposal_20260930`. `identity.json`
binds all execution sources and the three previously selected20D anchors.
Do not edit frozen sources and resume the same directory. Committed source
snapshot is the reproduction reference. Data/search case files have atomic
commit manifests and SHA256 validation. Completed training jobs are hash-checked;
an interrupted, incomplete training job restarts from its original anchor,
not from its `last.pt`. Retain interrupted-attempt costs separately if this occurs.

`STATUS.json` records the stage. `model_{0,1,2}/PROGRESS.json` and `history.json`
record training. `MODELS_FROZEN.json` binds selected weights before checking
proposal potential. `PROPOSAL_GATE.json` controls whether search is opened.
`SEARCH_GATE.json` exists only when that gate passes. `COMPLETE.json` means all
workers joined and all protocol stages required by the observed gates finished.

Contract costs in the run's `CONTRACTS.json` are separate from preliminary
developer fixture costs in [PRECHECK_COSTS.json](PRECHECK_COSTS.json).
Behavior collection, teacher labels, training/validation teacher calls,
proposal diagnostics and actual paid search calls are reported separately.
Teacher labels and objective derivatives are allowed only for the synthetic
offline mechanism study. The deployed proposal module has no teacher evaluator.

Interpretation limits: shared procedural families, three fixed training seeds,
recipe-cluster intervals rather than new-seed uncertainty, domain bounds[-5,5],
and joint modification of conditioning/training objective. New-A is contextual
fusion versus the old standalone anchor; it is not a matched standalone version
of the new module. A positive result needs subsequent isolation/independent
validation before a paper can claim general complementarity.
