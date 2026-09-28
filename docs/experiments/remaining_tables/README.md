# Tables for manuscript revision

`*_means.csv` contains every recorded condition, with the number of search initializations, arithmetic mean, sample standard deviation, median and endpoint range. These are raw objective values; do not average them across functions with different scales. The native 20-D table uses separately reconstructed checkpoints, not the frozen 10-D model.

`*_paired.csv` reports paired endpoint differences and 5,000-resample percentile bootstrap intervals over search initializations within each fixed condition. A negative difference favors the first method (usually full ROOPF; the training-order table uses curated minus original). No multiplicity-adjusted significance claim is made. Strict numerical equality defines ties.

The replay table instead uses forced branch minus original full trajectory. Its changed-only counts exclude interventions that leave the chosen point unchanged. Operator-origin and incumbent-decrease concentration tables are descriptive, not additive causal attributions.

The main report defines benchmark provenance, task routing, timing boundaries, training controls and all limitations. `final_integrity_checks.json` is written only after all 15 stages and their archives pass verification.
