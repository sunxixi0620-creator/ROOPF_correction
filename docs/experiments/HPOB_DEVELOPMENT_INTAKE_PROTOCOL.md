# HPO-B v3 development intake (before validation optimization)

Use the mirror explicitly linked by the official HPO-B README, fixed Hugging Face
revision `6b7a3fe1798b8058878f915e292a617be40f849b`. Download only meta-train,
meta-validation, initialization indices and README. Verify published LFS SHA-256.
Do not download meta-test responses during this intake.

The 16 official v3 search spaces are the common source/validation space IDs.
Infer held-out test **ID metadata only** within these spaces by subtracting
train and validation task IDs from the initialization-index table. This is not
a read of test inputs or responses and does not yet establish test eligibility.

Admission is independent of response performance: finite normalized numeric
inputs in [0,1], dimension 1..64, at least 105 distinct configurations for a
target, and at least five source tasks available in its search space. Preserve
all qualifying spaces and validation tasks; do not filter easy tasks or weak
transfer outcomes. Identical input rows are represented by their first original
index, chosen without response access. Report duplicate counts.

Produce two explicitly distinct manifests: (a) the official same-space split,
(b) a stricter source pool that excludes any source dataset ID appearing in
validation or inferred test IDs anywhere in the 16 spaces. The stricter manifest
is the default for developing a new-dataset claim. If insufficient sources
remain, report ineligibility rather than relaxing after seeing outcomes.
Numeric ID separation alone cannot rule out aliases or derived raw datasets;
that semantic dataset identity audit is still required before a final claim.

Only source responses may enter transfer training. Target optimizer interfaces
receive pending X, observed X and paid raw y; global target min/max belong only
to the separate report evaluator. Targets maximize accuracy; any minimizer must
negate observed values explicitly. Do not use the default HPOBHandler loader,
which reads the test table, or its full-table normalization in optimizer input.

Future development budget is 105 including five initial observations (not 105
plus initialization), five paired seed IDs. Use official initialization indices
when present and valid, otherwise a deterministic input-only permutation. This
intake freezes metadata rules, not an unimplemented learning algorithm. Final
method selection, source subsampling and test-run protocol must be frozen before
test-label access. No HPO optimization is claimed by completing this intake.
