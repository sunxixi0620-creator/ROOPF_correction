# Method and protocol text for the controlled research candidate

Editorial scope: these passages describe the separately trained unified candidate. They do not describe the original frozen checkpoint under a new name. Component conclusions and external results must accompany them; implementation corrections and release comparisons are not methodological contributions.

## Evaluation allocation

We consider deterministic, bounded, unconstrained minimization under a fixed number of objective evaluations. A frozen policy proposes two anchor candidates at each round. A portfolio supplies six candidates from each of six operators: a fixed neural candidate map, current-to-pbest variation, elite covariance sampling, a trust-region operator, coordinate perturbation, and opposition restart. An online proxy, fitted only to previously evaluated points, ranks the candidates. The optimizer reserves one evaluation for the first anchor proposal and selects the second evaluation from the other anchor and the portfolio. This reservation concerns an evaluation slot at the current state; it does not preserve an independent anchor-only trajectory.

Each trajectory starts with 100 paid points. Coordinates are mapped from the declared bounds to a common internal box, and all competing methods receive the same objective interface. Neither the objective's name nor its known optimum enters the candidate's decision rule. The population retains the best 100 observations. When the archive exceeds 256 points, it retains the best 100 archive entries and the most recent 128 entries. These two sets may overlap; the implementation does not deduplicate them.

## Online scores

The proxy is an ensemble of five ridge regressions on constant, linear, squared-coordinate, and fixed random Fourier features. Coordinates are normalized to [-1,1]; target values are standardized using the current archive. The ridge coefficient is 0.002. The number of Fourier features is min(64,max(16,4d)). Each member is refitted analytically at every round. The ensemble mean gives the prediction, and the population standard deviation across members is augmented by 0.03 times the archive target standard deviation times distance to the nearest archive point. This spread is a heuristic exploration signal, not a calibrated confidence interval.

Let c and s denote the archive target mean and population standard deviation, with s floored at 1e-8. Let r be the fraction of evaluations remaining and h the number of non-improving rounds divided by the total possible rounds, clipped to [0,1]. Define d_best as Euclidean distance in normalized coordinates to the actual best archive point divided by sqrt(d), and d_archive as the unscaled Euclidean distance to the nearest archive point. The score minimized over the 38 candidates is

    a(x) = (mu(x)-c)/s - 0.3125*sigma(x)/s
           + (1-r)^2*(1-0.7*h)*d_best(x)
           - (0.15+0.35*r+0.35*h)*d_archive(x)
           - 0.03*log(max(prior(x),1e-8)).

Both anchors receive prior 1/6. Candidate priors come from the fixed state map and the updated operator-success memory. Scores are computed once on the complete candidate list and reused for the competing-slot decision. The selected simplified candidate uses this score directly and does not execute a residual predictor.

The residual ablation uses an offline MLP with 27 inputs and hidden width 128, trained to classify improvement over the incumbent. Its sigmoid output p is converted to a within-pool signal: z=(p-mean(p))/max(std(p),1e-6), using population standard deviation over all 38 candidates. The corrected score is a(x)-0.008*clip(z,-2.5,2.5). This target is not a calibrated probability of outperforming the competing anchor. The ablation tests its independent value rather than assuming that predictive accuracy implies optimization benefit.

## Admission and updates

The first anchor is always evaluated. For the second slot, the optimizer finds the minimum-score candidate among the second anchor and all 36 portfolio entries. A portfolio winner is admitted only after four rounds of paid predictions and when its score advantage exceeds 0.10+0.25e. Here e is the mean absolute prediction error, divided by the pre-observation archive standard deviation and clipped at 10, over the two paid points in each of the last at most eight rounds. Predictions are recorded before the corresponding true values are observed. Thus the earliest admission can occur after 108 evaluations, including initialization. The error statistic is conditional on adaptively selected observations and supplies no guarantee for unevaluated candidates.

After the two evaluations, the optimizer updates the archive, population, prediction-error window, stagnation counter, and operator-success memory. The memory is multiplied by 0.86 and receives rewards of 1 for improving the pre-round incumbent, 0.25 for improving the worst pre-round population value, and -0.05 otherwise; it is clipped to [-2,3]. Improvement tests require a decrease greater than 1e-12, and both rewards use the same pre-round population. An odd final evaluation is assigned to the first anchor. Extra full-pool objective evaluations are allowed only for offline labeling and never update online state.

The matched score-only control removes the readiness and margin tests while retaining the first anchor slot, identical proposals, and identical scores. The delayed controls additionally forbid portfolio admission before 70% of the total evaluation budget has been consumed. These controls distinguish timing and admission tests from the structural anchor-slot reservation.

## Training identity and source separation

We retain all 12 predetermined coefficient recipes at three scales, yielding 36 procedural configurations. The generator uses no external benchmark ELA, performance ranking, or test outcomes to select configurations. Training, validation, and development use separately keyed task and population streams. They share recipes, so development measures instance generalization within this generator. This controlled training source does not erase historical exposure to BBOB/CEC during earlier research.

We train three independent anchor seeds, preserving the architecture, with Adam at 0.001, batch size 16, gradient clipping at 10, and a maximum of 80 epochs. Gradients accumulate over four tasks, giving nine updates per 36-task epoch. The loss measures the best candidate's improvement relative to the best initial parent using the parent's shared standard deviation. Validation evaluates complete 300-evaluation rollouts every five epochs, includes the untrained checkpoint, and selects the best checkpoint. Training stops after at least 20 epochs and three validation checks without an improvement greater than 1e-5. An 80-epoch cap is a computational limit, not a claim of global convergence. Native 20-dimensional anchors are trained from scratch under the same recipe; 10-dimensional weights are not reused as 20-dimensional models.

For the development ablations, residual training uses full-pool incumbent-improvement labels collected under the matching frozen anchor and no-residual behavior policy. It uses unweighted binary cross-entropy, Adam at 0.001, a maximum of 60 epochs, and validation patience eight. Normalization is estimated from training inputs only. The state encoder, neural portfolio map, and neural prior retain fixed seeded initializations and are not described as offline-trained networks. Proxy coefficients are fitted online from the paid archive.

## Evaluation and interpretation

The six-way development comparison uses 36 configurations, two task instances, four populations, and three training seeds: 5,184 complete trajectories with 300 evaluations each. Its primary measure is u=g/(1+g), where g=max(0,(initial_best-final_best)/initial_std). This bounded normalized improvement is neither a probability nor a percentage reduction in objective value. Paired bootstrap intervals resample recipe families, instances, and training seeds while retaining the three scales and four populations within a sampled group.

The external protocol freezes the no-residual candidate before evaluation on the versioned opfunu implementation of all 12 CEC2022 functions. The conditions are 10 dimensions with 300 evaluations and 20 dimensions with 300 or 600 evaluations. Each method receives 30 runs per function and condition. For learned methods, three trained anchors each serve ten distinct optimizer runs; baseline runs are not duplicated to manufacture a training-replicate sample size. CMA-ES and GP-EI retain their native initialization protocols, fully counted within the same objective budget.

Training and external inference may overlap across dimensions or seeds. Each checkpoint is selected and hashed before its own external evaluations, while the training, validation-selection rule, and evaluation protocol remain fixed throughout. The aggregate native20 checkpoint manifest is assembled after all selections finish; it does not imply that every 20D checkpoint already existed before the first 10D result.

External comparisons use log10(max(final_value-known_optimum,1e-8)); the optimum is accessed for reporting only. Results are reported separately for each dimension/budget, with paired resampling of function, training/run group, and within-group run. The nominal 95% intervals are pointwise and are not adjusted for multiple comparisons. The finite suite and three training seeds limit generality. Neural inference uses float32 while objective evaluation and final reporting use float64. Costs distinguish online objective calls, online computation, offline training, validation, full-pool labels, implementation checks, and timing repeats. Shared-host runtime observations do not constitute exclusive-device benchmarks.

The evidence must separately establish overall fusion, residual contribution, and admission-rule value. A useful fusion result does not establish the other two claims. Component null or negative results are retained, and external outcomes do not trigger further loss, epoch, threshold, or per-function configuration searches in this protocol.
