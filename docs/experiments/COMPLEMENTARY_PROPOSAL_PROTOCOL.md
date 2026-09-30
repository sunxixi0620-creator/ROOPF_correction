# Complementary proposals: bounded mechanism pilot v1

Frozen before collecting this pilot's outcomes. This is a new research candidate,
not a replacement of the original unlocked checkpoint or support for residual.

Primary condition: dimension20, budget600, initial population100. All36 procedural
configurations (12 recipes x3 scales) retained. No benchmark/ELA curation.
Shared families mean new instances are not independent unseen-family validation.

## Data and modules

Collect frozen O trajectories, four populations per task, snapshots at search
steps0,35,70,105,140,175,210,245 (NFE100..590). Training uses two instances per
configuration; validation one; proposal check one. Roles have distinct task,
population and policy seeds under complementary_v1_{train,validation,check}.
End-to-end uses two further instances under complementary_v1_search, opened only
if the proposal gate passes. Check data never selects epochs.

Start each of three models from its already frozen selected long-training20D
anchor. Preserve width200. Add a context MLP (2d+4 ->200 ->2d) with zero final
layer. Inputs: two surrogate-selected O proposals relative to current best /10,
their clipped scores, budget remaining and stagnation. Correction=.5*tanh(output).
This jointly changes conditioning and training objective; no isolated claim about
either. Fixed bounds[-5,5] make this a procedural-domain pilot only.

During inference build O's original38 candidates, obtain its two intended choices,
replace only the two extra uniform proposals with new proposals and rerank38.
All36 operators, priors, archive, paid evaluation count and memory remain O's.
Two extra indices do not update operator memory. No residual/gate/reserved slot.
New proposal generator is online-context-dependent despite offline training:
there is no claimed standalone version obtained by zeroing its context.

## Training

Freeze behavior snapshots. Train generator plus context, Adam1e-4, gradient clip10,
at most60 epochs; validation every2 epochs; stop after8 validation checks without
improvement>1e-5. Batch all32 snapshots per training task, randomized72-task order.
Three seeds. No loss/width/epoch/threshold sweep. Evaluate in eval mode, training
uses inherited dropout. Validation maximizes exact marginal proposal utility.

Let s=observed population std (floor1e-8), t=min(current incumbent, true values
of O's two intended points). Candidate normalized values z=(f(a)-t)/s.
Loss=.05*softplus((-.05*log(mean(exp(-z/.05))))/.05).
This smooth objective gives gradients when exact marginal utility is zero.
Only generated training objectives supply gradients. Validation/check teacher
values are separately counted and cannot reach inference. Exact diagnostic
utility is max(0,(t-min(f(a1),f(a2)))/s), bounded by g/(1+g).
Store checkpoint identity, optimizer, histories and teacher costs.

## Gates and stopping

Proposal check compares new vs same-seed old anchor on identical O snapshots.
Bootstrap12 recipe clusters5000 times, averaging scales, instances, populations,
snapshots and seeds within recipe. Require mean advantage>=.001,95%CI lower>0,
and positive mean for every seed. This measures potential, not paid optimization.
If it fails, STOP; do not tune/relabel or open search set.

If it passes run A (frozen standalone anchor), O (once per case), Old (old anchor
with free selection), New (new contextual generator with free selection).
36 configurations x2 instances x4 populations, three learned seeds; O shared,
not counted as three independent runs. 2880 trajectories,1,728,000 paid calls.
No teachers in search. Store trajectories and exact NFE/point checks.
Primary New-O and New-Old contrasts: recipe-cluster bootstrap5000, Bonferroni97.5%
CIs for two comparisons; both mean>=.005, lower>0, all3 seed means>0 required.
New-A is a descriptive contextual fusion vs original standalone comparison, not
a proof that the new context-dependent module works standalone. Original F and
FR evidence remains as previously reported; no new residual training this round.

If search gate passes, report promising within-family development evidence;
freeze before proposing subsequent external confirmation. If it fails, stop.
Never select an easier condition after observing results. Record failures fully.

## Contracts and costs

Snapshot instrumentation must reproduce O exactly in points/trails/RNG. Old-mode
must reproduce existing FreeAllocation exactly. Teacher queries happen after O
rollouts. Verify odd NFE, no test gradients/teacher calls, finite gradients,
zero-context-head initial proposal equality. Record CPU/GPU time separately;
do not add overlapping worker wall times and call that elapsed/GPU hours.
