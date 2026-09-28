# Unified revision v1: frozen development protocol

Authorized by the user after `EXECUTION_DIRECTION.zh-CN.md`. Original final
weights remain frozen. This is one new candidate, not a claim that fixes already
establish algorithmic benefit. No results from this protocol were examined when
the formulas, thresholds, and stop rules below were chosen.

## Scope and source isolation

The first decision is a 10-D, 300-NFE development gate. The procedural generator
in `roopf/revision_tasks.py` retains all 12 predeclared coefficient recipes x 3
scales. No benchmark, ELA statistic, performance ranking, or original 36-function
curation is imported. These generic ingredients do overlap familiar mathematical
primitives; do not claim all primitives are novel or that historical development
contact disappeared. Training, validation and development have distinct role-
hashed parameter/population/policy seeds, persisted per case. The same 36 recipes
occur in the three splits: this first gate measures instance generalization,
not unseen-family generalization.

Final external source selection/manifest must be completed without evaluating
candidate performance before any final runs. Previously examined COCO/CEC
results stay development/history evidence. Final expanded testing is conditional
on this development gate; it must include a separately documented unused task
source and native middle dimension, not just renamed old instances.

## Training

Three model seeds: 0, 1, 2; no best-seed selection. All full/ablated runs within a
seed use exactly the same anchor. Every anchor has the existing architecture,
100 initial points and 300-NFE rollouts, trained from scratch with Adam lr .001,
gradient clipping 10, batch 16, 36 tasks/epoch and accumulation over 4 tasks.
Maximum 80 epochs, validation every 5, earliest stopping after epoch 20 with 3
non-improving validation checks (improvement tolerance 1e-5). Include epoch 0 in
validation and checkpoint selection, and report if training never beats it.

The quality loss is mean((best candidate - best parent)/parent std), with detached
shared parent scale (floor 1e-8). This uses the already tested shared-scale recipe
from training-validation evidence; there is no new loss comparison. The plan's
instruction to retain the objective is refined here to retain its *quality
improvement aim*, while avoiding the audited separately standardized parent/
child values that erase whole-set shifts. Record this as a changed controlled
training recipe, not historical reproduction or proof of a novel training loss.
The original architecture, original weights and historical results are retained.
Validation is 36 fixed tasks x 2 populations, scored by bounded normalized
improvement below; it is reused only for model selection and explicitly marked.

Residual: original 27-input, 128-hidden MLP, incumbent-improvement label, Adam
.001, batch4096, unweighted BCE, max60 epochs, patience8 and tolerance1e-5 on
validation BCE. Two training instances/recipe x2 populations; one validation
instance/recipe x2. Collect all 38 candidates at every decision on the unified
no-residual behavior policy, using the matching newly trained anchor. Labels
are offline teacher data; exact deployed features are saved before normalization.
Compute normalization from training only. The three residual seeds match the
three anchor seeds. No target/width/epoch grid search and no retraining of the
14 historical models. These choices define a new controlled recipe, not the
original weighted-BCE training history.

## The single candidate

Keep the six operators (6 candidates each), frozen auxiliary initialization,
ridge ensemble, success memory and one protected anchor slot. Auxiliary networks
are not described as trained. Each step scores 2 anchors +36 candidates together;
retain these scores through selection rather than rescoring a smaller shortlist.

Let s and c be archive std/mean, d_best normalized distance to the *actual*
archive best, d_archive distance to nearest observation, r remaining budget and
h normalized stagnation. All comparisons minimize:

    a = (mu-c)/s - .3125*sigma/s
        + (1-r)^2*(1-.7*h)*d_best
        - (.15+.35*r+.35*h)*d_archive - .03*log(prior)
    a_res = a - .008*clip((p-mean(p))/std(p), -2.5, 2.5)

The 38-point context defines both residual input and normalization of its score.
Both anchors have prior1/6. Residual p is a learned incumbent-improvement score;
do not claim a calibrated replacement probability. Surrogate distance_scale=.03.
Archive std floor1e-8, residual-score std floor1e-6. No task name or known optimum
enters the optimizer. Keep best100 plus recent128 when archive exceeds256.

First anchor is always evaluated. Minimize a_res (a for no-residual) among the
second anchor and36 pool points for slot2. Predict both paid points *before*
observing truth. Error evidence is mean absolute prediction error divided by
the pre-observation archive std, clipped at10, over up to the last8 rounds.
After at least4 rounds (8 paid prediction outcomes), permit a portfolio point
only if its score advantage over anchor2 exceeds .10 + .25*error. Before that,
fall back to anchor. Thus the earliest allowed replacement is NFE108, not zero
observations. Score-only removes this protection and readiness requirement.
Late arms additionally require used_budget>=.70. Update success memory and
portfolio counters on all actually paid outcomes, never teacher outcomes.

This combines common scoring and one evidence margin; stage comparisons support
claims about this whole candidate. Do not attribute all gains to one change or
claim formal safe/non-degrading optimization. Error estimates concern adaptively
selected paid points, not unseen candidates with guaranteed coverage.

## Development comparison and stopping

Fixed six configurations: anchor_only, full, no_residual, score_only, late_full,
late_no_residual. Two new development instances per recipe, four population
seeds per instance, all three training seeds. 36*2*4*3*6=5184 trajectories.
Pair initial populations and role-based policy seeds; trajectories may diverge.
No opportunistic extension or alternative loss search in this v1 run.

For each trajectory g=max(0,(initial_best-final)/initial_population_std),
u=g/(1+g). Higher is better. This scale-free bounded measure is not a success
probability; report original final objective per condition alongside it.
Primary effect is mean paired u(full)-u(anchor), with recipes equally weighted.
Main gate: effect>=.005 and95% lower bootstrap endpoint>0, with all three training
seed aggregate effects positive. The .005 is a prechosen practical threshold,
not a power guarantee. Resample the12 coefficient families, instance and training
seed levels in a paired manner (5000 draws); retain the3 fixed scales and4
populations within each sampled family/instance. Candidate rows are not replicates.

Report residual effect full-no_residual using same threshold/interval rule.
If unsupported, do not automatically train another objective. A simpler method
may be considered only via the predeclared full/no_residual comparison and the
same overall gate; this is simplification, not residual evidence.

Protection value: quality difference full-score_only lower95% endpoint>=-.005,
and risk reduction lower95% endpoint>0. Risk is fraction of paired trajectories
with u(method)-u(anchor)<-.02, averaged by recipe/instance/training seed. Also
report mean quality so a gate is not claimed useful from adoption rate alone.
If full improves quality but risk criterion fails, describe quality results,
not empirically established protection. Early benefit: paired quality at NFE210
and final full-late_full, plus actual adoption timing; do not infer early benefit
solely from allowing earlier intervention.

If neither full nor predeclared simplified no_residual meets the overall gate,
stop this candidate before final testing. Strong-baseline comparison and native
20-D/unused-source expansion are then unnecessary for promoting this candidate;
keep earlier baseline evidence and report the failure. If a candidate passes,
run matched strong-baseline checks (CMA-ES, GP-EI), freeze the surviving candidate
and complete the final-source protocol before expansion. This ordering limits
cost and does not suppress completed unfavorable baseline results.

## Integrity, costs and execution

Unique run directory; complete source/config/checkpoint/task/population/RNG
fingerprints; locked atomic writes; verify completed file hashes on every reuse.
All spawned workers must be joined before COMPLETE. Record parent orchestration
and worker status; resume rejects changed source/config. Ordinary logging and
teacher-mode parity must pass, including no extra online objective queries.
Stage1 uses frozen-code trajectory comparisons; stage2 checks label-blindness,
feature roundtrip, finite gradients, resume rejection and budget boundaries.
Teacher objective evaluations and training/validation cost are reported
separately from each300-NFE main trajectory. GPU is shared with other work;
timing cannot be labeled exclusive hardware benchmarking.
