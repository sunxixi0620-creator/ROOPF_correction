# Zero-query search-engine audit, 2026-09-30

This is an exploratory engineering audit of existing development trajectories,
not a new optimization benchmark or a change to any frozen experiment.

Before running the audit, fix the following selection: all 36 configurations,
instance 0, model seed 0, methods O/W/WA, at paid counts 20, 39, 100, 300.
This gives 108 trajectories and 432 states. Load them from the verified
cold_start_components_v1 store. Reconstruct GP targets from saved decision
records; replay the original selection and verify its predicted values and point.

Diagnostics:

1. Finite/zero/negative EI scores, winning EI magnitude, standardized improvement
   z, posterior standard-deviation spread, and number of numerically tied maxima.
2. Fraction of local candidate coordinates exactly on the domain boundary,
   fraction of local candidates touching a boundary, and selection of global vs
   local proposals. These are geometric diagnostics, not objective performance.
3. Maximize the same acquisition over an expanded pool: original 128 candidates
   plus 2,048 new uniform and 2,048 new Gaussian/clamped proposals generated with
   the original bandwidth and an independent deterministic audit seed. No
   candidate objective values may be computed. An acquisition gain is evidence
   of a finite-pool approximation gap, not proof of improved real optimization.

The original engine uses double precision GP algebra and its original query
dtypes. Do not change GP hyperparameters, fit new models, call an objective, or
rerun rollouts. Disable ProceduralTask.calfitness during the audit so accidental
objective evaluation fails. Record sources and selected input artifact hashes.

Report all selected states, method/count summaries and descriptive fractions.
No significance claim, independent-family claim, new training, or hidden truth
is licensed by this audit. Keep existing statistical gates unchanged. A clean
numerical result rules out the checked symptom only on these selected states.
