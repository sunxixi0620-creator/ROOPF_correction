# Terminal-reward training feasibility

Protocol: ../../experiments/TERMINAL_TRAINING_PROTOCOL.md.
Active run: results/terminal_training_20260930.
Read STATUS.json, PROGRESS.json and history.json for live progress. Scientific
results are valid only after COMPLETE.json and the final COSTS.json audit exist.

The frozen script starts or resumes an interrupted run using its verified state
and cached rollouts. Do NOT rerun it after COMPLETE.json exists: completed early
stopping is final for this protocol. Read completed outputs instead. There is no
permission for an additional update beyond a completed stop.

Three58,504-parameter proposals; fixed online selection; antithetic ES8 directions,
30updates max,16CPU workers. Validation every5 updates includes epoch-zero fallback.
Only terminal reward determines selection. Timing is an estimate, not convergence.
A failed validation gate stops automatically. Passing it triggers the preregistered
A/O/Initial/Terminal confirmation. No independent external benchmark is opened.

The initial artifact is a start snapshot only; parent proposal weights are in
artifacts/complementary_proposal_v1. Final model/cost reporting remains pending.
