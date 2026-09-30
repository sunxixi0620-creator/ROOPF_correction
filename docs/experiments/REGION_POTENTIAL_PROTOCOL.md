# Directional region potential v1 — freeze before outcomes

Question: can directions supplied by the frozen offline contextual proposal
produce novel points with long-term value beyond simple nonlearned exploration?
This is one bounded diagnostic, not a trained exploration policy or an oracle
600-NFE deployable method. No retraining, radius/step/model sweep this round.

Base: matched pure-online O (OnlinePortfolio), 20D/600 true calls, batch1.
New procedural split region_potential_v1, all36 configurations, instance0,
2 independent initial populations/policy streams:72 base trajectories.
Intervention steps0 and105 (100 and310 NFE already used). These are prechosen,
not selected by results or stagnation. Capture full state and policy RNG before
intervention. All continuations are O, without anchor/residual/gating.

Offline directions: use the three terminal_training_v1 selected contextual
proposals (updates0/20/10), inputs from current O state/intent/score. Subtract
current best from each of2 output points. For each direction use normalized-box
RMS radii .10 and .20 (coordinate RMS displacement1 and2 before clipping), giving
4 points per menu. Freeze all checkpoints. No new function labels enter proposal
generation. Fixed directions/radii are a diagnostic transformation of existing
proposals, not an already trained region generator.

Three nonlearned menus, also2 directions x2 identical radii:
Iso: independent Gaussian directions.
Pop: differences of two independently drawn pairs of current population points.
Center: box midpoint minus best; current population centroid minus best.
Use explicit role-hashed RNGs. These controls test random exploration, online
geometry, and simple center priors. They do not cover all nonlearned methods.

Clip points to[-5,5]. A point is eligible only if the source direction is nonzero
and its normalized-box RMS distance from EVERY previously evaluated point AND
every current O candidate is >=.05 (raw coordinate RMS .5). Full evaluated
prefix is initial100 plus completed2*step points, never future base points.
This defines geometric novelty, not proof of a different attraction basin.
Clipping changes actual radii; retain all eligibility/distance/clipping metadata.
No resampling or truth-based eligibility. Ineligible actions abstain and reuse
Base with0 extra calls; identical four-slot menus remain, availability may differ.

For each eligible point, replay from original population/policy seed; assert exact
state fingerprint and evaluated prefix. Keep O's first selected point; replace
second with exploratory point by forcing a spare uniform-slot index(-1 operator).
Continue O unchanged to600NFE. No hypothetical candidate fitness is queried first.
Controls and learned actions use the same intervention/memory convention. Record
immediate,+20,+60,and terminal utility changes, plus all points SHA. Save complete
Base coordinates and snapshots, branch best-value trails and fingerprints.

Per state/menu, diagnostic potential=max(0, four branch terminal utility gains).
The max uses terminal truth retrospectively and includes abstaining; report mean
of ALL four gains too (ineligible=0), so oracle headroom is not mistaken for a
usable selector. Report immediate benefit of the terminal-best branch, including
fraction of eventual wins with nonpositive immediate gain. Never select a new
model, radius, stage or direction set based on these results.

Primary: Offline potential minus Base and each of Iso/Pop/Center potential,
averaged over all3 model seeds,36configs,2populations,2stages. Baselines/control
rollouts reused across model seeds, not counted as independent observations.
12 recipe clusters bootstrap5000, Bonferroni98.75% intervals for4 contrasts.
Proceed to new training design only if all lower bounds>0, all3 seed means>0,
Offline-Base mean>=.001 and each Offline-control mean>=.0005. These are mechanism
feasibility thresholds, not proof of F>A/O or the paper's .005 practical gain.
If any fail, stop this directional-region route; no auto radius/rank/lr sweep.
Do not generalize failure to all possible exploration policies or offline learning.

CPU32 workers/1 thread, use preceding throughput evidence, memory fallback24/16
if headroom insufficient. Worker per(fid,pop), reuse Base/control across learned
seeds.72 Base + up to3456 branch rollouts, at most2,116,800calls; actual branch
calls depend on geometric eligibility. All branch rollouts are expensive label
cost, including replay prefixes. No off-rollout diagnostic fitness queries.
Contract: four114NFE runs(no hook/hook/no-op/forced) =456 additional calls.
Verify prefix/no-op exactness, first-slot retention, forced-point evaluation,
fixed model weights and zero uncounted objective calls. Keep original unlocked
weights untouched. Archive sources, parent hashes, cases, costs and all outcomes.
New instances of existing function families remain development evidence.
