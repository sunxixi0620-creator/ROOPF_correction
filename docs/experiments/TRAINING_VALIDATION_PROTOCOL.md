# Training sufficiency development protocol

Frozen before examining this study's generated validation endpoints. This is a
development study, not an external benchmark claim or historical reproduction.
The original final 10-D weights and previous evidence remain unchanged.

## Anchor continuation

Start both arms from exactly the epoch-80 resume state of
`retrain_d10_curated_20260928`, including Adam state and generator parameters.
Continue to epoch160, with one fixed training random stream. This measures
continuation from that trained state, not two from-scratch training recipes.
The same function order, parameter-refresh epochs, populations and dropout
streams are paired across arms. Keep lr0.001, batch64,36functions, accumulation4,
clip10, and300mainNFE. Diversity lambda is max(0,.5*(1-epoch/80)), independent
of stopping time. Thus the continuation has lambda0 in both arms. No claim is
made that an old80-epoch run used this same extended training schedule.

- legacy: separately standardize initial-parent and candidate outcomes, then
  subtract their minimum values.
- shared_scale: `(min(candidate_y)-min(initial_parent_y))/std(initial_parent_y)`.
  Detach the common parent scale. This aligns the quality term to observed
  improvement, but does not guarantee learned generalization.

Validation: all36 generated functions,2 fixed new instances per function,
4 initial populations per instance,300NFE. Seeds65000000+100*fid_index+instance
generate parameters on CPU once;66000000+100*fid_index+instance generate initial
populations. These are instance-held-out development tasks from known families,
not wholly unseen function families. Serialize parameters and populations so
device changes cannot silently regenerate tasks. Do not use BBOB/CEC outcomes.
Checkpoints at80,90,...160 are scored by the function-balanced mean of
`gain/(1+gain)`, where gain=(initial_best-final_best)/initial_std. Also retain
raw endpoints and normalized gains (assert gain>=-1e-5, then clamp rounding-sized
negative values to0). Select the best validation score,
tie favoring the earlier epoch. Training loss never selects the new checkpoint.

Confirmation: separate instances with parameter/population seeds67000000 and
68000000, identical36*2*4 layout. Evaluate only after both arms have completed
and their checkpoint choices are frozen. No retuning from confirmation results.
Compare both selected checkpoints, epoch80 start and original final anchor.
One continuation random stream is exploratory evidence; no multi-training-seed
stability claim. Additional training seeds require a separate confirmatory plan.

## Residual and auxiliary modules

Audit recovered full-pool labels against the actual second anchor candidate,
using each pool's baseline candidate_slot1. Keep incumbent and best-of-two
labels separate; quantify their disagreements, including the seven residual
holdout functions. Do not relabel missing/malformed pools by guessing. Frozen
residual predictions are evaluated for all three label definitions, without
training or changing checkpoints. This determines whether more training of the
same target addresses the intended decision.

On the same generated validation instances, compare frozen final full ROOPF,
zero gate logits plus existing success memory, and completely uniform operator
priors. Candidate generators and their RNG streams stay fixed; these tests
isolate the randomly initialized gate's contribution, not every auxiliary net.
No auxiliary network is trained merely because it exists. These observations
are development evidence and do not select an external-test winner.

All objective counts, source/checkpoint hashes and selection records are saved.
No changes to the original final algorithm, and no new test-suite tuning.
