# Complementarity study: frozen anchor-duration protocol

Authorized 2026-09-29: A = offline anchor only; O = online portfolio only;
F = anchor + online portfolio without residual; F+R = the same fusion with a
residual trained on full-pool labels collected under the selected matching anchor.
Previous experiment sources, checkpoints and results remain unchanged.

## First stage: duration, not a joint hyperparameter search

Continue all six existing training states (dimensions 10/20, seeds 0/1/2),
including Adam state, from their last trained epoch, not their selected epoch.
The 10D states ended at 80/30/80; the 20D states ended at 80/80/80. Preserve their
entire histories and selected models. Verify the source state manifests and copy
the states into the new run before execution. This is an explicitly extended
training protocol: remove the old early stopping rule and run to epoch 1000.
No test result controls training length. All three seeds must be retained.

Keep width200, batch16, Adam .001 (constant), clip10, 36 procedural tasks per
epoch and accumulation4: nine updates per epoch. Keep the shared-scale quality
loss and the exact role-hashed task/population/dropout streams from the existing
training code. Parameter instances are refreshed every epoch. 1000 epochs means
9000 updates per model, including the prefix, not 1000 independent datasets.
The recovered historical batch64/decaying-rate/default1000 recipe is provenance,
not the recipe being silently reproduced. Do not increase batch or change loss,
learning-rate schedule or width in this stage.

Validate every5 epochs on the same 36 generated validation tasks x2 populations,
using bounded complete-rollout improvement u=g/(1+g), g=max(0,(initial-best-final)/
initial_std). Select the best checkpoint over the prefix and extension, with
improvement tolerance1e-5; no best-seed selection. Explicit snapshots at
80/160/320/640/1000 supplement the complete histories. The resumed seed that
previously stopped at30 must first reach80; other epoch80 snapshots are the
actual terminal prefix states. Save model AND optimizer at milestones.

Costs distinguish prefix and extension, actual training point calls, validation
point calls and parameter updates. Exact replay assumes the recorded software
and arithmetic environment. GPU sharing and worker overlap preclude interpreting
summed process seconds as GPU hours. Report if validation still improves at1000;
the cap does not prove convergence. Training-source validation does not establish
unseen-family or external benchmark generalization.

## Downstream decisions and four-group study

Freeze selected anchors using only this generated validation evidence before
collecting new residual data or evaluating A/O/F/F+R. F and F+R use exactly the
same anchor as A. O must never load or call a pretrained anchor/residual, uses
all paid slots, and must not inherit an anchor-based warm-up. Its replacement
proposal slots and matching rules require executable contract checks and a
separate frozen four-group protocol before comparative performance is observed.

Residual architecture/target/training rule remain the previously fixed 27-input,
128-hidden incumbent-improvement classifier with full38 candidate labels, Adam
.001, batch4096, maximum60, validation patience8. New data and normalization are
collected separately under each selected anchor and dimension. No old residual
is relabeled as matched to the new anchor. No loss/threshold search is authorized
by a negative result.

Network enlargement is conditional, not automatic. First inspect duration curves
and fit on fixed training/validation probes. A width200/400 check, if justified,
needs its own pre-result budget/selection specification and must not be conflated
with the duration experiment. Mere small parameter count does not establish a
capacity bottleneck. Do not block reporting unfavorable four-group outcomes.

Previously inspected BBOB/CEC sources remain development/history evidence.
New random instances of the same recipe measure instance generalization, not
wholly unseen-family transfer. New final external-source claims require a source
audit and separate pre-result manifest; earlier completed tables are immutable.
