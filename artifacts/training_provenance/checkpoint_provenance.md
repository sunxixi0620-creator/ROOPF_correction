# ROOPF Checkpoint Provenance

This record fixes the checkpoint lineage used by the final ROOPF experiments and
paper. The training source below is author-confirmed; the hashes make the deployed
artifacts and their archived copies independently checkable.

## Common training source

- Source implementation: `prospect_code/exps/claude_create_trainsets.py`
- Training suite: 36 generated functions from 12 landscape families
- Standard BBOB, CEC-style, HPO, and UAV test functions are not training tasks.

## Sequential training lineage

1. Train the anchor policy `pi_A` for 80 epochs on the audited 36-function suite.
2. Freeze the resulting anchor checkpoint.
3. Roll out that frozen anchor and the six runtime operators on the same 36
   functions, logging full candidate pools and true generated-function outcomes.
4. Train the residual selector `g_theta` from those logs and freeze the checkpoint
   selected by held-out-function validation.

The two checkpoints therefore share the same generated-function source, but they
are not independently trained in parallel. Their dependency is:

`F_train -> pi_A -> D_pool -> g_theta`.

## Frozen artifact hashes

- Anchor checkpoint: `prospect_code/exps/ckpt/neurgo_vparameter_d10.pth`
  - SHA-256: `a265d448ad4a2d191eed23dafa36e03e9475a7b95e824bb04d9a88715031a06d`
- Residual checkpoint:
  `prospect_results/paper_results_archive/v58_retrained_offline_w0008_summary/router_generated36_improved_best_earlystop.pt`
  - SHA-256: `42bac7c6792b6980ad149eff08ce4b2d93ce3ffba51d3d869dc0d1625b3a9d01`

The anchor copies in the root experiment directory, final v58 directory, and
`prospect_code` have the same SHA-256 digest.
