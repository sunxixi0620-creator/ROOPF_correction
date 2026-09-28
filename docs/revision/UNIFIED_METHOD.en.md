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

A source audit performed after the 10D comparisons found reused optimum/shift data for six CEC2022 functions relative to previously examined opfunu CEC2017 classes. Two also reuse rotation arrays, although their composition formulas and parameters differ. This study therefore provides frozen external-suite evidence, not a wholly unseen landscape-family test. The audit does not remove functions from the main table or change the selected method. The new training generator does not import these benchmark data; historical author exposure is a separate limitation that is retained.

Training and external inference may overlap across dimensions or seeds. Each checkpoint is selected and hashed before its own external evaluations, while the training, validation-selection rule, and evaluation protocol remain fixed throughout. The aggregate native20 checkpoint manifest is assembled after all selections finish; it does not imply that every 20D checkpoint already existed before the first 10D result.

External comparisons use log10(max(final_value-known_optimum,1e-8)); the optimum is accessed for reporting only. Results are reported separately for each dimension/budget, with paired resampling of function, training/run group, and within-group run. The nominal 95% intervals are pointwise and are not adjusted for multiple comparisons. The finite suite and three training seeds limit generality. Neural inference uses float32 while objective evaluation and final reporting use float64. Costs distinguish online objective calls, online computation, offline training, validation, full-pool labels, implementation checks, and timing repeats. Shared-host runtime observations do not constitute exclusive-device benchmarks.

The evidence must separately establish overall fusion, residual contribution, and admission-rule value. A useful fusion result does not establish the other two claims. Component null or negative results are retained, and external outcomes do not trigger further loss, epoch, threshold, or per-function configuration searches in this protocol.

## Completed component results

The full candidate improved bounded utility over its matched anchor by 0.03734 (95% interval [0.02048, 0.05915]). Removing the residual retained an improvement of 0.03808 [0.02080, 0.05982]. The full-minus-no-residual effect was -0.00074 [-0.00254, 0.00061], failing the preregistered practical-contribution rule. The residual changed approximately 7.33 decisions per full-candidate trajectory on average: it was active, but its interventions did not establish a useful final optimization effect. We therefore retained the no-residual candidate for external evaluation, as specified before external results were available.

Earlier admission improved the no-residual candidate over its fixed-70% counterpart by 0.00857 [0.00506, 0.01261]. At 210 evaluations, its utility advantage over the anchor was already 0.02716 [0.01384, 0.04503]. This supports intervention after the 100-point initialization, rather than a claim about the first observations of a task. The development quality/risk criterion did not support an independent benefit from the admission tests.

## Completed external results

The following table gives mean log10-error advantage of the selected no-residual candidate over each comparator, with nominal pointwise 95% intervals. Positive values favor the candidate; intervals crossing zero do not establish equivalence.

| Dimension / budget | Same anchor | CMA-ES | GP-EI |
|---|---|---|---|
| 10 / 300 | 0.752 [0.399, 1.309] | -0.009 [-0.261, 0.314] | 0.223 [-0.187, 0.904] |
| 20 / 300 | 0.288 [0.068, 0.566] | -0.089 [-0.437, 0.233] | -0.012 [-0.327, 0.496] |
| 20 / 600 | 0.968 [0.516, 1.614] | 0.121 [-0.252, 0.646] | 0.600 [0.060, 1.540] |

All three conditions support improvement over the same anchor within this evaluation. The 20D/600 condition also supports improvement over GP-EI under the stated pointwise analysis. None establishes superiority over CMA-ES. The matched no-residual control without admission tests produced quality-difference intervals crossing zero in all conditions. Counts of predefined degradations relative to the anchor were 3 versus 2, 2 versus 3, and 0 versus 1 for the selected candidate versus that control, respectively, out of 360 trajectories per method and condition. No risk-reduction interval had a strictly positive lower endpoint. These results do not establish an independent reliability benefit from the admission tests. Both variants retain the first anchor slot; this experiment does not isolate that reservation or the offline policy against an otherwise matched purely online portfolio.

Paid-evaluation logs recorded an average of 16.19, 11.31, and 47.96 incumbent improvements from portfolio evaluations per trajectory in the three conditions. Under sequential accounting that credits the first anchor before the competing slot, the five largest portfolio improvements accounted for an average 58.2%, 50.9%, and 50.6% of total observed post-initialization improvement. These are descriptive event shares, not a causal decomposition of the final advantage over an anchor-only trajectory. Unevaluated candidates supply no counterfactual true values.

On the fixed serial timing panel (F1/F8, run 0), average total times for the candidate, anchor, CMA-ES, and GP-EI were 0.484, 0.127, 0.060, and 3.096 seconds at 10D/300, and 1.279, 0.362, 0.184, and 29.367 seconds at 20D/600. Timings include model construction/loading and search, exclude Python startup and objective-instance loading, and were collected on a shared host. The panel has two timings per method and condition, and is not an exclusive-device performance benchmark. All 20 repeats reproduced their original queried points and objective values exactly; their 9,000 additional objective calls are excluded from the main performance sample size.

The native20 anchors were all trained for 80 epochs and selected epochs 80, 80, and 75. Two best validation checkpoints lie at the training cap, so the study does not establish convergence at 80 epochs. Across the six 10D/20D anchor trainings, counted training evaluations totaled 123,840,000, with 1,987,200 validation evaluations. Residual data collection additionally used 194,400 behavior-policy evaluations and 2,462,400 full-pool teacher evaluations. Residual fitting reuses those labels without new objective calls. Main development and external evaluation used 3,715,200 objective calls over 10,584 trajectories. Implementation checks and extra timings are separately itemized in the cost records; process times from overlapping workers must not be added and described as GPU hours.

## Scope of the resulting claims

The results support online candidate supplementation of the same frozen anchor over the tested deterministic, unconstrained 10D/20D settings. They do not substantiate the original stronger narrative that both a residual model and an admission gate independently produce reliable improvements, nor establish general superiority over strong optimizers. The reused external landscape data, shared procedural recipes, three training seeds, implementation/precision choices, and pointwise comparisons limit generalization. No new noise, hard-constraint, UAV, or HPO results are claimed for this candidate. Original-checkpoint application results cannot be relabeled as evidence for the retrained candidate. These materials support a narrower empirical contribution and an explicit component analysis; they do not by themselves establish publication-level novelty.
