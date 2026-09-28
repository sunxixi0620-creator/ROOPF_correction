# Supplementary development experiments, frozen before the main run

The selected final ROOPF remains the unlocked configuration with the original
anchor/residual weights and 70% warm-up. No optimization rule is repaired here.
The original code/checkpoints remain available at
`roopf-final-before-supplement-20260928`.

## Stage 1

- 23 distributed fixed BBOB cases (all except f20), six zero-shift CEC-style
  cases and six shifted CEC-style cases, with shift seed 20260925.
- These are development/regression instances, not new independent test families.
- 30 initializations per case, seeds 20263000–20263029, in three batches of ten.
- Five variants: full, identical frozen anchor only, no residual, score-only,
  and no proxy with structural scores. Total: 5,250 trajectories at 300 NFE,
  including 100 initialization points (1,575,000 point evaluations).
- CPU, one PyTorch/OMP/MKL thread; all variants use the same platform. Component
  timers are synchronized and nested. Batch times are not 10 independent runtime
  observations. Initialization/model construction is outside the optimizer timer.
- Structural-score jitter in no-proxy runs has an isolated RNG, preventing it
  from advancing the candidate-generation random stream. The no-proxy ablation
  retains the residual/gate with zero proxy features and is interpreted as this
  specific operational ablation, not a calibrated alternative predictor.
- Anchor-only skips unused pool/proxy computation. Exact candidate, population
  and trajectory equality against the frozen baseline_only path is tested.
- Every run stores population hash, model/source hashes, exact objective count,
  complete trajectory, selected-candidate records and gate decisions. Runtime
  includes ordinary instrumentation; diagnostic observer time is separately
  reported. Existing timing smoke measurements are not headline results.
- No full-pool teacher logging; no counterfactual objective queries in Stage 1.
- Full-method default decisions and RNG state are tested against the frozen Git
  implementation, both with and without diagnostic observers.

### Benchmark exception discovered before aggregation

The supplied f20 calls `f1_1` on each evaluation, randomly resampling signs, and
its first objective term sums across the entire input batch. It is therefore
stochastic and batch-coupled, rather than a fixed noiseless objective. Exclude it
from the primary paired aggregate, preserve its source and report the issue.
Other small batched/singleton differences flagged in f11/f16/f19 are numerical
checks, not grounds for post-hoc exclusion. This study is a comparison on the
repository's definitions and is not a certification of official COCO conformity.

## Stage 2

After Stage 1 completion: full ROOPF on BBOB f9/f11/f15 and shifted CEC-style
f1/f3/f6, 30 paired initializations each; warm-ups 1/3, .50, .70, .90.
Task selection is fixed here before Stage 1 results are inspected. Default stays
.70; no re-tuning on final test data. Counterfactual diagnostics are enabled at
every tenth outer iteration: re-evaluate the competing anchor and pre-gate
proposal on a separate diagnostic counter, preserving optimizer RNG and state.
Also replay the no-residual ranking and gate on the same state, including
shortlist reconstruction. These calls do not influence online updates or count
as part of the main 300-NFE run. Immediate diagnostic effects are not long-term
causal effects. CEC skips residual routing in the selected final method; report
that structural fact rather than interpreting ties as universal ineffectiveness.

## Interpretation and later stages

Report per-case paired endpoint differences, seed-pair bootstrap intervals
conditional on the tested instance, exact w/t/l, convergence and decision rates.
Do not average raw objective values across different functions. Report results
against anchor-only and component ablations, not against a legacy release.

Independent suites, external baselines, higher dimensions and matched training
suite experiments follow only after their specific protocols/assets are ready.
Do not silently invent checkpoints, infer full-pipeline holdouts from residual
holdouts, or mark these stages complete from the development results.
