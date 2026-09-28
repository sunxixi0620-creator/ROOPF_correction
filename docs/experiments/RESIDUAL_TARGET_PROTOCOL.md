# Residual target development protocol

Frozen before outcomes. Original final weights remain unchanged. Generated36 known families, new instances only; no BBOB/CEC selection. This does not establish unseen-family generalization.

- Fresh parameter seeds: train 71000000 (2 instances/family), validation 72000000 (1), confirmation 73000000 (1); add 100*sorted family index+instance. Population seed = parameter seed+10000000; policy seed = parameter seed+20000000. Four trajectories per case, 300 NFE.
- Offline behavior: frozen final anchor and no-residual ROOPF. Teacher evaluates both anchors and36 portfolio proposals, never feeds true values into policy. Check teacher/no-teacher exact parity before collection.
- Save38 candidate feature rows per pool, including anchors to support the deployed comparator. Both training arms use identical inputs, all phases, no class weighting. Targets: strict incumbent improvement (existing epsilon convention) versus strict fitness below second anchor. Labels include both anchor truth values, candidate fitness, phase and group identity. No teacher outcome is an input feature.
- Fit feature means/stds on training rows only. Original feature schema and architecture. Two training seeds (20260928,20260929), Adam lr .001, batch4096, max60 epochs, validation BCE early stop patience8, minimum improvement1e-5. Select within each objective/seed only; do not select between objectives using confirmation outcomes.
- Report held-out BCE/Brier, prevalence, constant-training-prior baseline, portfolio top5 precision, and calibration. Retain both seed results. Confirmation label metrics and optimization evaluate only after all models selected.
- Full optimization confirmation: original full, no_residual, and four newly trained residual variants; otherwise unchanged policy/gates/anchor. Same36 fresh instances and four trajectories per instance, no teacher calls. Describe mean-condition W/T/L and normalized bounded gain; no significance or generality claim from one instance per family.
- Main final weights frozen and hash checked; training datasets disjoint from previous training-validation study. Validation and confirmation data never used for gradient training. Record all budgets separately.

Preflight corrected seed namespaces before training or inspecting outcomes: parameter, population and policy now have disjoint numeric ranges. Partial setup/collection retained under results/residual_target_seed_preflight* and excluded from formal data.
