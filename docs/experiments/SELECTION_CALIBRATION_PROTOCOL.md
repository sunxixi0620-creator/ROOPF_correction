# Frozen-proposal selection calibration v1

New bounded research round,20D/600. Freeze three selected contextual proposal
models and all shared modules. No original residual, no gate/warm-up change.
Change only final candidate score; O's provisional intent used by the proposal
remains unchanged. Lower score is better. No proposal or online proxy retraining.

Correction = .5*tanh(MLP10->64 GELU->1), zero final layer initially,769 parameters.
Ten clipped[-10,10] features: original final score; predicted value relative to
incumbent / archive std; uncertainty / archive std; log prior; normalized distance
to best; nearest archive distance; remaining budget; stagnation; extra-proposal
indicator; original score rank/37. The correction applies to all38 candidates.
This is a NEW offline-trained calibration component, not evidence for the old
residual module. Magnitude .5, width64 and temperature .2 fixed; no sweep.

Collect current uncalibrated New behavior on selection_v1_train/validation
procedural splits, all36 configurations, two training instances and one validation
instance, four populations, three matched model seeds. Eight snapshots at
NFE100,170,240,310,380,450,520,590. Per seed2304 training/1152 validation states.
Post-rollout teacher queries38 candidates/state using separate task object;
no labels enter behavior. Total324 batches,777600 behavior calls,393984 labels.
States on a trajectory are correlated; these are shared families, not external
unseen-family validation. Prior diagnostic/search instances are not reused.

Train only correction network, Adam .001, batch128 states,80 epochs max,
validation every2, patience8 checks with improvement>1e-5. Three matched seeds.
Listwise cross entropy: target softmax(-clip((true_y-incumbent)/archive_std,-5,5)/.2);
prediction softmax(-(base_score+correction)/.2). Select checkpoint by actual
validation chosen-top2 bounded improvement over incumbent using archive std;
include epoch0 fallback. Training loss alone never selects a checkpoint.

Freeze all selected calibrators before opening selection_v1_search (two fresh
instances/configuration). Evaluate Base (current New), Cal (new selection), O.
Base/Cal use three matched proposal seeds; O evaluated once/case and shared as a
paired control. 2016 trajectories,1209600 true calls. No teachers. Also record
runtime, final selected extra fraction and trajectories. All methods600 exact.

Two primary Cal-Base and Cal-O comparisons, recipe-cluster bootstrap5000,
Bonferroni97.5% intervals, mean bounded final improvement>=.005, interval lower>0,
all3 seed means>0 required. Failure stops this round with full reporting; no
post-hoc threshold/feature/epoch variants. Success is within-family development
support only; further external confirmation requires another frozen protocol.

Before data collection verify original Base vs capture-only and zero calibration
bit-exact points/trails/RNG at600 and odd113 budgets; no teacher calls. Verify
proposal/shared hashes remain fixed. Source/data/checkpoint manifests and full
costs archived. Extra surrogate fitting for calibration features counts as runtime,
not objective calls. Offline label costs not hidden in paid search budget.
