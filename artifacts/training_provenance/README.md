# Recovered training evidence

The user identified `/home/skq/repos/ROOPF` as the submitted experiment project.
Its `ROOPF_AAAI_Reproducibility.zip` contains 24 files: inference code, frozen
checkpoints, benchmark definitions, reference results and reproduction scripts.
It does not contain anchor training entrypoints or generated training functions.
`submission_archive_check.json` compares archive members with the working tree.
The expected differences in the model are opt-in instrumentation on the experiment
branch; the original tagged snapshot remains the reference.

Separately, the user confirmed a Claude-created 36-function generator project.
Historical files located under `/home/skq/repos/xixi` are copied here only as
provenance evidence, not silently substituted for the submitted implementation.
`manifest.json` gives exact source paths and hashes.

The archived anchor and residual copies in the historical prospect tree match
our deployed checkpoint hashes exactly. The anchor training import
`exps/claude_create_trainsets.py` matches the generator's curated output hash.
The residual logging script explicitly names `outputs/trainset_generated_36.py`,
the original (pre-curation-order) module. Membership comparison finds the same
36 named functions in both modules; curation changes the order. This does not
recover the exact historical anchor training command or all its hyperparameters.
The current historical `main.py` imports a much-expanded NeurGO model, so it must
not be assumed to be the recipe for the deployed simple anchor.

The recovered audit retained 36 functions, including ten below its tau=0.9
threshold because its script fills the requested target size. Do not describe
all 36 as passing the distance threshold. This final streaming suite must not be
conflated with the earlier 420-to-150 procedural-pool pipeline without additional
lineage evidence. Full-pipeline test independence is not implied by ELA distance.

The residual metrics confirm 1,200,000 selected training/validation rows,
966,569 training rows, 233,431 validation rows and seven residual-held-out function
IDs. This is not a whole-pipeline holdout if the anchor used all 36 functions.
An interface check evaluated all 36 functions on two batches of 20 points and
verified raw finite outputs and finite nonzero gradients at the tested samples.
It does not certify all possible inputs or independent generalization.

The historical full-pool CSVs (about 1.4 GB) were located but are not copied here.
Offline runtime has not been reconstructed from timestamps; file timestamps
alone do not establish training wall time. Matched-suite retraining and high-D
ROOPF remain pending a verified compatible training recipe.
