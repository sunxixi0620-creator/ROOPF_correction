# Ranking and improvement-magnitude development protocol

Frozen before outcomes. Reuse only train/validation cases from residual_target_study (72/36 instances). No previous confirmation or decision-audit cases are training data. Known generated36 families, not unseen families. Original final weights unchanged.

Same feature schema/MLP, training-only normalization, seeds20260928/20260929, Adam .001, batch128 complete pools of38 candidates, max40epochs, validation-objective early stopping patience6/min_delta1e-5. Both objectives include two anchors and36 portfolio candidates; no feature truth inputs.

- ranking: logistic pairwise loss on all703 unordered pairs in each pool, excluding equal-fitness pairs; positive score should mean lower fitness. Loss averaged over all non-tied pairs in each minibatch.
- magnitude: target0.5+0.5*g/(1+abs(g)), g=(second_anchor_fitness-candidate_fitness)/std(pool_fitness), scale floored1e-8. MSE on sigmoid score. This is a bounded improvement score, not a calibrated probability.
- Existing runtime sigmoid residual interface and all gate thresholds unchanged. Ranking scores also have no calibrated-event probability interpretation. This explicitly tests drop-in scores; failure may concern score/gate compatibility as well as objective quality.

Validation selection uses each objective's own loss, never confirmation outcomes. After all four selections freeze, generate confirmation tasks using parameter seed75000000+100*sorted family index, population+10000000, policy+20000000; four trajectories per instance,300NFE.

Evaluate full original, no_residual, both prior second-anchor checkpoints, both ranking checkpoints, both magnitude checkpoints. All eight methods use same fresh tasks. Also collect no_residual teacher pools on those tasks for held-out pairwise concordance, top1 regret and score metrics. Teacher values never update online states. Report both training seeds, no best-seed cherry-picking. This is a development round, not external benchmark proof.
