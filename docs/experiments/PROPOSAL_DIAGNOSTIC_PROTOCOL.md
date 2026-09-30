# Frozen-proposal same-state diagnostic v1

No training, parameter tuning or new performance claim. Replay all prior O cases
(72 batches) and New cases (216 batches), four trajectories each, at20D/600.
Capture steps0,35,70,105,140,175,210,245. Require bit-exact points/trails against
verified prior artifacts BEFORE any teacher queries. Original checkpoints frozen.

On O states examine all three frozen new proposals and their matched old anchors;
on New states examine its own seed and matched old anchor. O behavior is replayed
once, not counted as three independent trajectories. Diagnostics involve432
seed/state batches x32 states =13,824 state records, not independent samples.

At each state preserve original O pool38 (36 shared +2uniform), replacing its two
extra points to form New38 and Old38. Recompute the unchanged online score using
identical archives. Teacher queries occur only after replay, on an independent
task object: original38 plus New2 plus Old2,42 points/state. No teacher feedback.
Expected replay calls691,200; diagnostic calls580,608; no new search scores.

Measure: new/old marginal potential beyond incumbent and O's intended two points;
actual chosen-pair benefit versus O's chosen pair; incremental oracle pool value
relative to O's full38; new-pool oracle-versus-selected gap; fraction of beneficial
new-proposal opportunities realized by the actual chosen pair; opportunity rate;
selection fraction; proposal pair distance, distance to shared36, clipping and
correction saturation. Bounded improvements use current-population std, and also
initial-population std for actual chosen-pair benefits. Oracle best-of-pool
represents immediate best-value utility only, not long-term two-point information.

Compare potential on O and New state distributions, and old/new on identical
states. Report by behavior, seed and search step. Bootstrap12 recipe clusters,
5000 resamples for descriptive95% intervals; seeds treated as fixed. No formal
acceptance test or external generalization claim from this retrospective audit.

Interpret patterns before choosing any redesign: lost potential on own states
supports state-distribution concerns; retained potential with missed improvements
supports selection concerns; tiny incremental full-pool oracle value supports
candidate redundancy. Causes may coexist. Do not equate selection frequency with
benefit or divide incompatible potential/performance metrics. Short continuation
branches are a subsequent conditional diagnostic only if immediate potential and
selection both look adequate; this run stops after audit/report, without training.
