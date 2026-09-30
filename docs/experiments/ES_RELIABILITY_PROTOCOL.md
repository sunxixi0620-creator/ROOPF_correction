# ES direction reliability audit v1

No retraining or new method adoption. Freeze three terminal_training_v1 selected
proposal checkpoints. Test the current full58,504-parameter estimator with two
independent sets A/B of8 Gaussian antithetic directions, sigma.02, original frozen
per-layer scales. Same6 configurations chosen by fixed20261001 permutation, new
reliability_v1_estimation instance0, two populations,600 budget. Common objective,
populations/policy RNG across +/- and A/B. 576 batches,691200 calls.

Estimate g_phi=mean((Uplus-Uminus)*epsilon)/(2sigma). O cancels algebraically, so
no redundant O evaluations. Report cosine and norm, but do NOT interpret near-zero
cosine alone as failure: random8-direction estimates in58k dimensions can be nearly
orthogonal even when useful. Test functional transfer instead.

From each fixed center separately apply ONE fresh Adam lr.01 step (clip gradient
norm1) using estimate A or B; no momentum history or sequential updates. This is
a fresh-optimizer local audit, not exact continuation of previous Adam moments.
Transfer set: all36 configurations x2 fresh instances x4 populations per model
seed, reliability_v1_transfer, Base/A/B. 648 batches,1555200 calls. No selection
using transfer outcomes. Compare A-Base and B-Base, recipe-cluster bootstrap5000,
Bonferroni97.5% CIs. Local reliability requires each mean>=.0001,lower>0,all3 seed
means>0. This diagnostic threshold is not the paper's .005 method-gain threshold.
If failed/mixed, prioritize testing a lower-dimensional estimator before claiming
architectural lack of potential. If passed, training is locally useful but not
proved converged; consider the proposal-role hypothesis separately. No automatic
network training, width/noise/step sweeps or new exploration mechanism this round.

Resource benchmark BEFORE direction outcomes:16/24/32 workers, each runs the same
64 full600-budget batches2. Record worker RSS and choose lowest wall time; allow32
only with8GiB memory headroom. Single CPU and CUDA batch2 comparisons on same task
are timing-only; CUDA results NEVER mixed into scientific comparisons because RNG
and numerical paths differ. All scientific comparisons remain CPU to preserve
matching. Limit per-worker BLAS/PyTorch threads to1. Choose worker count on speed,
not scores. Timing cost3*64*1200 +2*1200=232800 when all settings complete.
Archive hardware/throughput, exact repeated-CPU fingerprints, source/model hashes,
all estimates and paired final utilities. Shared-family development evidence only.
