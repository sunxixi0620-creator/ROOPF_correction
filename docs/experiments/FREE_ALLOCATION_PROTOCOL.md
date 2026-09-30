# Bounded allocation follow-up, 2026-09-30

The completed1000-epoch four-group study improved anchor validation but did not
establish either F or FR over O. Both were worse than O at20D600 under the declared
adjusted intervals. Residual contribution failed all three conditions. These
results are retained; no more training, capacity or residual search is initiated.

## Single new hypothesis

The mandatory anchor slot and admission gate may limit fusion. Compare a single
F_free variant: replace O's two uniform extra proposals by the same frozen trained
anchor proposals, rank all38 candidates using the unchanged no-residual score,
and evaluate the best two distinct indices. No forced anchor slot, readiness or
margin test. Other36 proposals, priors, proxy, archive, memory, initialization,
budgets and RNG streams match O. Negative operator IDs for anchor/uniform extras
both receive no six-operator memory update. This tests unrestricted allocation
as a whole, not an isolated causal estimate of the old first-slot reservation.

Parity checks must establish: anchor-free mode exactly reproduces O, the common
modules are identical, budgets are exact including odd budgets, extra proposals
equal the frozen anchor, and no residual/teacher query is used. Test fixtures do
not select formulas or evaluate comparative endpoint performance.

## Development diagnosis, then a conditional confirmation

First reuse the previous four-group tasks and archived A/O/F trajectories as
DEVELOPMENT evidence. They have already informed the new hypothesis and cannot
be called an independent test. Only run F_free: three conditions(10,300),(20,300),
(20,600), 36 configurations x2 instances x4 populations x3 anchor seeds,
2592 additional trajectories,1,036,800 main calls. Verify all reused data hashes,
parameter/population/policy identities and unchanged frozen model hashes.

Primary scale remains bounded normalized improvement u. Three contrasts per
condition: F_free-A, F_free-O, F_free-F. Paired bootstrap5000 over12 recipe
families, task instance and model seed; retain scales/populations. O remains one
shared comparator, not replicated independent trials. Report nominal95% and
Bonferroni98.3333% marginal intervals (quantiles .05/6,1-.05/6), nominal
familywise95% within a condition, not across conditions. Practical pass requires
mean>=.005, adjusted lower>0, and all three per-seed means>0.

The20D600 condition is the prespecified target because it exposed the allocation
problem. Only if ALL THREE target contrasts pass, perform one new-instance
confirmation, preserving all settings and all three conditions, with A/O/F/F_free
on split `allocation_v1_confirmation` (36x2x4; each learned method3 seeds, O once).
This would add8640 trajectories and3,456,000 calls. Freeze this split now. It is
instance generalization within shared recipes, not wholly unseen families. Do
not tune the method or increase sample size using these outcomes.

If the target gate fails, stop this follow-up, retain all outcomes and explain
whether allocation freedom improved F without demonstrating offline benefit
over O. A negative result does not authorize another operator, loss, threshold,
width, or training-length search. Full FR stays in the completed four-group
report; its negative result is not hidden or replaced by F_free.

Archive source/config/model identities and every new/reused case reference.
Main teacher calls must be zero. Diagnostic queries and any conditional new
confirmation calls are accounted separately. No fresh external superiority or
reliability claim follows merely from these generated-task experiments.
