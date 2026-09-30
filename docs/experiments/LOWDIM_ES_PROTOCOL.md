# Low-dimensional ES reliability audit v1

Freeze before outcomes. Start from terminal_training_v1 selected checkpoints
(seed 0/1/2 at update 0/20/10), not original unlocked production weights.
One candidate only: freeze all parameters except an additive rank-4 update to
context.2.weight (40x200) and context.2.bias (40). A fixed 200x4 orthonormal
basis Q, generated with role-hashed seed, maps 40x4 coefficients C to C Q^T.
The 200 optimization coordinates comprise 160 coefficients plus 40 biases.
Use existing per-tensor RMS scales from terminal origin; no extra norm matching.
Zero coordinates exactly reproduce the selected model. This restricts function
class and total perturbation norm as well as dimension; a difference does not
isolate dimension alone. No basis/rank/noise/lr search.

Matched Full comparator updates all 58,504 coordinates. For Low and Full,
3 model seeds x A/B independent sets x8 Gaussian antithetic directions x2 signs
x6 fixed configurations (RNG 20261003) x2 populations. sigma=.02, new role split
lowdim_es_v1_estimation. Objective/population/policy RNG shared across methods,
signs and direction sets. 1,152 batches, 2,304 trajectories, 1,382,400 calls.
Directions are independent across methods; allocation and seeds fixed ex ante.
Estimate sum((Uplus-Uminus)*noise)/(2*.02*8). Each A/B update uses fresh Adam
lr=.01, gradient norm clip1, at the frozen selected center, not sequentially.
O cancels from the antithetic difference. No teacher truth outside budget.

Fresh transfer split lowdim_es_v1_transfer: Base/Low_A/Low_B/Full_A/Full_B,
3 model seeds x36 configurations x2 instances x4 populations x600 budget.
1,080 batches, 4,320 trajectories, 2,592,000 calls.
Primary: Low_A-Base and Low_B-Base, 12-recipe cluster bootstrap5000, Bonferroni
97.5% intervals. BOTH require mean>=.0001, lower>0, all3 seed means>0.
Full_A/B-Base and Low_A/B-Full_A/B: descriptive95% intervals, no confirmatory
superiority claims. Report all outcomes, no selection by transfer. Small positive
means or low gradient cosine alone cannot establish success or failure.
Local gate is not the paper's .005 practical fusion threshold.
If gate fails, stop this restricted-parameter route without rank/lr sweep or
full retraining. If gate passes, specify a bounded retraining protocol before
training; this script performs audit only. Shared-family development evidence,
not external generalization or F-vs-O evidence. Fresh Adam is not continuation
of previous optimizer momentum. Report limitation of three selected centers.

Reuse prior CPU throughput decision:32 workers,1 thread each, CPU only. Check
>=26GiB available before launch (about18GiB workers plus8GiB reserve); fallback
24/16 with documented availability if needed. No redundant CUDA benchmark.
Contracts: zero-offset equals parent output, only final context weights/bias
change for Low, basis orthonormal, same task/population initial identity, exact
600 NFE and zero teacher calls, frozen parameter fingerprint. Two repeated
zero-offset full rollouts (batch2) add2,400 calls. Total3,976,800 calls including
contracts. Save source/model hashes, gradients, coordinates, paired utilities,
best-value trails and points fingerprints; not full evaluated coordinates.
