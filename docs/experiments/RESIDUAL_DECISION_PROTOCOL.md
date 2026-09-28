# Frozen residual decision diagnostic protocol

Fresh36 generated-family instances, parameter seed74000000+100*family index, population seed+10000000, policy seed+20000000; four trajectories per instance and300NFE. Known families, not an unseen-family test. All five residual checkpoints are fixed from previous studies: original and both seeds of incumbent/second target. No tuning or retraining.

At every protected decision, compare the deployed full selection with a counterfactual selection recomputing both shortlist and gate without residual at the identical state. Evaluate the second anchor, actual choice and counterfactual choice as diagnostic teacher calls isolated from state and RNG. Record immediate true fitness and incumbent improvement; distinguish selected-point changes from full-policy final effects.

For each case/model, replay the entire run with residual disabled for exactly the earliest decision round in which any of the four batch trajectories has a changed selected point (chosen using points only, never teacher fitness). Residual enabled again thereafter. Assert exact prefix equality and that the substituted candidate equals the original-state counterfactual. Compare final outcomes. This measures removing one early intervention, not removing all residual, and cannot decompose total gain additively. If no point changes, record this and omit replay.

Check diagnostic-on/off exact point/trail/RNG parity for the first instance of every checkpoint before main execution. Final original weights unchanged. Teacher calls separately counted. Full/no-residual policies from the earlier study are not treated as identical-state causal comparisons. New decisions remain development diagnostics; further model development needs fresh confirmation data.
