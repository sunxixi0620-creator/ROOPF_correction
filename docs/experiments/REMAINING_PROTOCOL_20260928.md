# Remaining supplementary protocol (fixed before new outcome inspection)

Final 10-D checkpoint and default configuration remain frozen. All new training
below is explicitly a reconstructed controlled experiment, never the recovered
historical run or a replacement for the final checkpoint.

1. Fixed-width operators: the same six diagnostic cases as Stage 2; 30 seeds
20269000–20269029; full, six leave-one-family-out, six single-family portfolios.
All have 36 candidate slots and 300 NFE. Surviving original candidates are kept;
removed slots are replaced by fresh draws from surviving operators, evenly in
cyclic operator order. Extra draws use an isolated random stream. Deterministic
operators may produce duplicate points (especially the neural-only condition);
36 slots does not imply 36 distinct coordinates. These are diagnostic tasks,
not an independent generalization test.

2. Single-intervention replay: same cases/seeds, one forced competing anchor or
one forced pre-gate proposal at NFE210/240/270, then resume the unchanged full
policy. Verify all evaluated points before intervention exactly match the full
run. Report null interventions as null, not successful changes. At NFE220/260
of the unmodified run evaluate all 36 pool points on a separate diagnostic
counter; report proxy ranking and prediction error. No diagnostic truths feed
back into selection. This estimates individual conditional trajectory effects,
not an additive decomposition of all interventions.

3. UAV: recover the historical five 10-D path-planning proxy scenarios; 30
seeds20270000–20270029 per method, 300 NFE. Full, anchor-only, no-residual,
score-only, no-proxy, CMA-ES and DE. Report objective components and exact
segment-to-circle geometric clearance separately from the objective's sampled
soft penalties. This is a simulated proxy, not real UAV deployment.

4. Training audit: the recovered final curation selects all original 36 members
and changes order only. Verify the function implementations and distribution
parameters are identical by function ID, without ELA references. Consequently a
canonical-order membership-only ELA ablation is an identical training treatment;
do not invent a distinct selected set or claim a benefit from selection.
Reconstruct training from a strictly architecture-compatible historical model
and loss/Adam schedule. Record recipe and costs; native20-D uses a newly trained
dimension-specific anchor and must be called a retrained extension. Residual
transfer is separately identified if the original10-D residual is used.

5. Additional external BO: use a documented maintained implementation, same
instances, objective budget and repeat count as the third stage; freeze its
configuration before inspecting its results. Distinguish this from a meta-trained
transfer comparator, whose training assets must exist for valid reproduction.

Complete algorithm, source-to-equation mapping, active task branches, known
inactive counters, training provenance and evidence limits accompany results.

### Controlled training schedule, frozen before benchmark evaluation

Three anchors: d10 original function order, d10 curated order, d20 canonical
function-ID order; seed20271000,80epochs,batch64,population100,300NFE/trajectory,
Adam0.001,gradient accumulation4functions,clip10,offset refresh every20epochs.
Use recovered quality/diversity loss and select best **training** loss checkpoint.
Per-function random streams pair offsets, populations and dropout across orders.
There is one training initialization per treatment: evaluation-seed uncertainty
must not be presented as training-seed uncertainty. This is a matched function
order experiment, not evidence of different membership or selection superiority.
Main10-D checkpoint remains unchanged. Save training checkpoints, costs and
point-evaluation counts, including loss re-evaluation of parents/candidates.
Native20-D comparison: all24 COCO functions, instances101/102,10seeds per condition,
300NFE (same absolute budget) and600NFE (same30-times-dimension budget ratio as original10-D),
full with transferred frozen10-D residual, no-residual, same new anchor, CMA-ES,
DE. Report residual transfer as such, not as newly trained20-D residual.
Training-order comparison: all48 COCO10-D conditions,10seeds each, anchor-only
and full with each new anchor plus the original residual (fixed transfer).

### Native residual completion (before any20-D test evaluation)

In addition to explicitly labelled residual transfer, train a native20-D residual
with the recovered early-stop trainer:36training functions,10trajectories each,
300mainNFE and3800teacher evaluations/trajectory;1,296,000portfolio labels;
shuffle/subsample1.2M;the same seven residual-held-out function IDs;128hidden,
AdamW0.0008,weight_decay0.0001,batch4096,max400epochs,patience50,seed20260711.
Behavior policy is full ROOPF without residual, using the new20-D anchor.
Teacher evaluations never update online archives. Seven holdouts are NOT
whole-pipeline unseen tasks. Full native20-D uses this new residual; transferred
original residual is an extra comparator. No test data select weights.

### Released learned comparator, before its outcomes are inspected

Surr-RLDE author's released policy, upstream commit
7e1af60779e478ba701d903bcf517b4ccc8088f2, BSD-2-Clause source and policy hashes
archived under artifacts/external/surr_rlde. Same68external10-D conditions and
10seeds as Stage3,300actualNFE,NP100; no optimum information supplied. Upstream
checks termination before adding the last generation, so set internalmaxFEs200
and assert actual counted objective calls300. This also changes the internal
remaining-budget feature; report this adaptation, not an unmodified benchmark
claim. No policy retraining or test tuning. Source:
https://github.com/MetaEvo/Surr-RLDE (formerly GMC-DRL/Surr-RLDE).
