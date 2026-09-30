# Long-horizon value predictability v1

Bounded new learning problem, not retraining the old proposal/residual. Freeze
before outcomes. Fresh split long_value_v1:36configs x2 task instances x4 initial
populations, O20D/600NFE. Two prechosen steps0/105 (100/310 NFE).288 Base
trajectories,576 states. Retain only three simple menus Iso/Pop/Center,12 points
per state, identical directions/radii/novelty rules to region_potential_v1 with
new role-hashed RNGs. No offline neural proposals. For every eligible point,
replay exact prefix, keep first O point, force second point, then continue O to
600. Labels are terminal bounded utility(branch)-utility(Base), not immediate
improvement. Ineligible candidates never selected; no uncounted fitness queries.
At most288+6912 trajectories=4,320,000 calls; four114NFE contracts add456.
Save all case identities, snapshots, labels, branch trails/point hashes and costs.

Features computed solely from snapshot/point/novelty metadata, before branches.
Geo41:8 state summaries (remaining,stagnation,archive fraction,pop mean fitness
gap/pop std relative to archive std,best z,pop radius,centroid distance),8 point
summaries(radius,archive nearest distance,centroid distance,boundary fraction,
nearest fitness gap,5nearest mean gap,5nearest std,novelty),20 best-relative
coordinates/10,3 source flags,nominal radius,clipped flag. Fusion adds4 online
features:predicted gap to best,uncertainty,online score advantage over O's second
selection,and predicted gap to that selection. No fid/recipe/instance/population
ID or branch outcome/trajectory enters features. IDs used only for grouping and
fixed truth-independent tie breaking. Refit identical O surrogate on snapshot;
zero objective calls in feature extraction.

Three fixed outer folds:heldout recipes r%3=fold (4 recipes). Remaining8 sorted;
last2 validation,first6 train. All3 scales,instances,populations,stages of a recipe
stay together for each fitted model. Every recipe tested once. Train/validation
payloads exclude heldout states/labels. Feature standardization uses eligible
training rows only, std floor.001,z clip[-10,10]. All fitted models frozen before
heldout scoring. Overlapping outer training sets make bootstrap intervals only
development diagnostics, not final independent-test inference. Historical
family exposure still exists; no claim of untouched test families.

Two models Geo41->64->32->1 and Fusion45->64->32->1,SiLU,zero final layer.
3 initialization seeds per fold (18 models), no architecture/lr search.
Predict terminal utility*1000,unclipped MSE; Adam lr.001,weight_decay.0001,
batch256,max200 epochs. Initial zero-output checkpoint allowed. Every5epochs,
select by validation fixed-quota selected terminal gain, strict improvement
>1e-8;stop after4 nonimproving checks (20epochs). Retain all seeds, no test-based
model selection. Benchmark synthetic100 training steps CPU versus CUDA before
training; choose faster device only, record versions/timing. CPU training can
parallelize9 jobs per feature model; CUDA sequential18 models. All comparisons
use same chosen training device. Rollouts CPU32,1 thread, memory fallback24/16.

At each heldout state choose highest-scored eligible candidate. Within EACH
outer heldout fold choose exactly25% of states by that candidate's score; ties
use fixed role-hashed IDs. This is a fixed-quota OFFLINE ranking diagnostic,
requiring access to all heldout feature scores, not a causal deployable gate.
O proxy score=old second score-new candidate score. Zero-output constant model
uses identical hash ties. Random comparator is exact expected gain of uniformly
choosing exactlyK states then one eligible candidate, not a lucky random draw.
All rules are fixed without heldout labels. Also report forcing all states and
random candidates at Fusion's selected states as descriptive diagnostics.

Primary comparisons Fusion vs Base(0),Proxy,Random,Constant,Geo. Average across
3 model seeds,all576states (unselected states=0),12 recipe cluster bootstrap5000,
Bonferroni99% intervals for5 comparisons. Each requires mean>=.0001,lower>0,
all3 seed means>0. Recognition requires first4; extra online complementarity
requires all5. No final F>A/O claim from this diagnostic. Full search studies
require subsequent frozen protocol and fresh evaluation. If gate fails, stop
this fixed value model; no threshold/feature/loss/rate sweeps. If checkpoint0
selected, disclose and compare to constant; cannot call it learned evidence.
Original unlocked checkpoints untouched; this model is not original residual.
