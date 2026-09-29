# Four-group complementarity protocol

This protocol is fixed before any four-group performance inspection. The training
continuation is governed separately by COMPLEMENTARITY_TRAINING_PROTOCOL.md.
Original release and completed revision evidence stay unchanged.

## Methods and exact control

A: the selected width200 offline anchor alone, loaded identically to F/F+R.
O: online-only, no pretrained checkpoint loaded or evaluated. Retain the six
runtime operators, six candidates each, the identical fixed seeded auxiliary
networks and ridge proxy, score formula, population and archive rules. Replace
the two anchor proposal positions by two independent uniform points within the
declared box, using a separate role-hashed RNG. All methods therefore compare
38 proposal slots where ranking is used. O chooses the best two distinct indices
by the no-residual score (stable index order for ties), with no anchor reservation,
readiness wait, or anchor margin. Duplicate coordinates are not deduplicated,
consistent with F. Both paid slots are available from NFE100. Uniform slots have
prior1/6 and do not update the six-operator success memory. On an odd final step,
O evaluates its best remaining scored index. The common36 generation/score
components must match F at the same state; complete trajectories can diverge.
This is a specifically defined online portfolio, not a claim to the best possible
online optimizer. Choosing uniform substitute slots is disclosed, not hidden.

F: unchanged unified no_residual candidate with one anchor slot and the fixed
readiness/margin rule. F+R: unchanged unified full candidate with a newly trained
residual corresponding to the SAME selected anchor. No threshold/loss changes.
F and F+R retain the previous formulas and constants, including .008 residual
correction. This study tests both overall complementarity and residual increment;
F success cannot be attributed to F+R or used to establish residual necessity.

## Model freeze and labels

All six anchors must finish the1000-epoch schedule and validation-only selection
before four-group comparison. Copy selected anchors, checksums, selection histories
and training identity into the new run. Existing models are not overwritten.
For each dimension and seed, collect38-point labels under F: 36 recipes x two
training instances x two populations, and36 x one validation instance x two
populations, at300NFE. These are training roles, distinct from confirmation roles.
Use existing full-pool feature definitions and teacher-blind state updates.
Train six residuals with the fixed rule. Freeze/checksum all six residuals before
comparison. All labels and fitting costs are counted separately.

## Fixed confirmation design

Three conditions: (10,300), (20,300), (20,600). Use all36 procedural configurations
(12 recipe families x3 scales), two new parameter instances and four populations
per instance, under split `complementarity_v1_confirmation`. No ELA/performance
filtering. The recipe families are shared with training and prior development;
these are new instance confirmation tests, NOT wholly unseen-family benchmarks.
Previously inspected CEC/BBOB remain prior development/expanded evidence. No
claim of fresh independent external validation is made by this experiment.

A/F/F+R each have3 matched trained seeds, sharing initialization and policy streams.
O has no training seed: evaluate it ONCE per task/population and reuse that same
outcome as the paired comparator, not as three independent replications. Per
condition: 2592 learned-method trajectories plus288 online trajectories=2880;
three conditions total8640 trajectories and3,456,000 main objective calls.
Initial100 points count in every method's budget. Four-population batching is
held fixed; the two objectives for a round are paid before the next update.
Known optima and teacher truth do not enter any main run.

Primary metric is the existing bounded normalized complete-rollout improvement
u=g/(1+g). Five declared effects per condition: F-A, F-O, (F+R)-A, (F+R)-O,
(F+R)-F. Use paired hierarchical bootstrap5000 over12 recipe families, the two
instances, and three anchor seeds; keep all scales/populations within a sampled
group. O's shared observations remain shared through every bootstrap draw.
For the five contrasts, report nominal95% intervals plus Bonferroni99% marginal
intervals (quantiles .005/.995), giving a nominal familywise95% statement within
each condition. There is no familywise claim across the three conditions.
Practical success for a contrast requires mean>=.005, adjusted lower endpoint>0,
and positive aggregate effects for each of the three anchor seeds. F (or F+R)
complementarity requires BOTH contrasts against A and O to pass. Residual value
additionally requires (F+R)-F to pass. Do not reinterpret failure as equivalence.

Report raw objective endpoints, full curves, per-seed effects, paired degradation
u(method)-u(comparator)<-.02, and decision counts. No function exclusions, best-seed
selection, or repeated extension of confirmation samples after looking at results.
Results can fail the hypothesis. Do not change O to secure a win for fusion.

## Timing and integrity

Count all main, label, training and validation queries. A small serial timing
panel uses fids0/24, instance0, group0, the10D300 and20D600 conditions, all four
methods and the same four-population batch:16 repeated batches,64 trajectories,
28,800 extra calls excluded from main N. Verify repeated points and values match.
Shared-host timing includes model construction/load and search, excludes Python
startup, instance construction and output I/O. Concurrent worker elapsed seconds
are not serial comparisons. Identity and data hashes, exact budgets, no teacher
calls and workers joined are required before marking the stage complete.
