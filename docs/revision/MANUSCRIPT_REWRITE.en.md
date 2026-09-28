# ROOPF revision: evidence-bounded replacement text

> Superseded framing: the user has confirmed that the unlocked configuration is the final ROOPF method corresponding to the existing paper. The release-comparison abstract and results below are internal audit drafts, not replacement text for the paper. Use FINAL_METHOD_SUPPORT_PLAN.zh-CN.md as the current revision scope. The original three contributions remain the subject of the paper; unlocking is not a contribution.

Editorial status: 2026-09-28. These passages are proposed replacements for the corresponding sections of the submitted manuscript, using the selected `unlocked` configuration. They are not a complete camera-ready manuscript. The current evidence supports an internal paired comparison; external-baseline, component, training-suite, and UAV claims require provenance checks and renewed evaluation. Existing equations and references must be reconciled with the final source before submission.

## Proposed title

ROOPF: Anchor-Preserving Offline–Online Candidate Selection for Budgeted Black-Box Optimization

Editorial note: This title retains the project identifier while emphasizing the actual decision mechanism. Keeping the original title is also possible, provided “reliable” is explicitly operational rather than a formal guarantee.

## Abstract — replace the current abstract

Black-box optimization under a small evaluation budget requires deciding when to follow a learned search policy and when to evaluate an alternative suggested by current observations. We study this decision through ROOPF, which combines a frozen anchor policy, an online proxy fitted to evaluated points, and a frozen residual selector trained on offline candidate outcomes. A portfolio of six operators supplies alternative candidates. Following an anchor-only warm-up, the optimizer retains one anchor evaluation slot and allows a portfolio candidate to compete for the second slot through a margin-based selection rule. The revised configuration removes a rule that previously forced anchor selection on the CEC-style tasks, while retaining the existing checkpoints and remaining selection logic. In a paired 10-dimensional evaluation with 300 objective evaluations per trajectory, the revised configuration matches the original configuration's final values on 24 fixed BBOB instances and six unshifted analytic instances. On one shifted instance of each of six analytic functions, it improves all six function means, with 59 improved and one tied endpoint across 60 paired initializations. These results support removing the forced anchor restriction on the tested instances. They do not establish broad distributional robustness or a guarantee of improvement over an independent anchor-only run.

Editorial note: This is an honest abstract for the evidence currently available, not yet a strong resubmission claim. Replace the narrow release-comparison emphasis with validated mechanism and baseline results only after those experiments exist.

## Introduction — replace the problem framing and contribution paragraphs

Offline learned optimizers and online surrogate models supply different information. A frozen policy proposes points using search behavior acquired from prior tasks, whereas an online proxy ranks candidates using evaluations collected on the current objective. ROOPF first uses the anchor to accumulate evaluated observations and then permits guarded intervention during the remaining budget. This schedule does not use proxy-driven interventions to improve the earliest sparse-data phase. Instead, it raises a specific decision problem: once intervention is enabled, when should a proxy-ranked alternative receive an evaluation that would otherwise be allocated to the learned policy? Whether the offline improvement model remains useful at this stage is an empirical question requiring comparison with a protected proxy-only variant.

ROOPF represents this decision explicitly. The anchor proposes fallback candidates, a portfolio supplies alternatives, and the online proxy and residual selector provide evidence for choosing between them. The residual selector estimates incumbent-improvement evidence learned from offline candidate outcomes; it does not directly estimate the causal benefit of replacing the competing anchor. The selection rule uses these signals to regulate intervention while retaining an anchor evaluation slot.

The framework separates candidate generation, candidate ranking, and authorization of objective evaluations. Its main methodological object is the competing-slot decision, whose behavior can be inspected through the source of each evaluated candidate and the resulting paid observations. Our evaluation distinguishes fixed-instance regression checks from shifted-instance diagnostics and reports the effects of removing a forced anchor restriction. Establishing the independent contribution of the proxy, residual selector, and protection rule requires matched component comparisons under this revised configuration.

Editorial note: Replace the last sentence with the actual component findings once available. Do not claim the new configuration has already established all three contributions from the original paper.

## Method — replace the interpretation following Eq. (9)

During warm-up, all available evaluation slots follow the anchor. The threshold includes initialization: with 100 initial observations and a total budget of 300, intervention begins at 210 consumed evaluations, leaving at most 45 competing portfolio slots. Once 70% of the evaluation budget has been consumed, the first slot continues to evaluate an anchor proposal, while the second slot can evaluate either the competing anchor proposal or a portfolio candidate. If only one evaluation remains, it is allocated to the anchor. Candidate generation and scoring do not evaluate the objective; only admitted candidates consume the online objective-evaluation budget. Offline full-pool labeling is a separate training cost.

In the revised configuration, we disable the `structured_system_lock` option, which previously selected the competing anchor whenever the objective name began with `cecf`. The anchor checkpoint, residual checkpoint, candidate-pool width, warm-up schedule, and remaining selection rules are retained. This change enables the existing comparison mechanism on these tasks; it is not a newly trained policy or a coordinate-equivariant optimizer. Other task-category-dependent rules remain in the implementation and must be documented alongside the configuration.

Anchor preservation refers to the allocation of an evaluation slot at the current search state. It does not preserve an independent anchor-only trajectory. An admitted portfolio point can alter the population and archive, thereby changing subsequent anchor proposals. The selection margin and residual tests are empirical safeguards rather than calibrated confidence bounds or a proof of non-degradation. Likewise, the residual output concerns improvement over the incumbent under its training distribution, not a calibrated probability that a portfolio candidate outperforms the competing anchor.

Editorial note: Keep the original numerical gate definition only after reconciling it with the executable branches. The no-proxy/no-residual/score-only variants need explicit definitions; labels alone are insufficient.

## Experimental setup — replace protocol claims for the new paired study

We compare the original and revised configurations on three groups of 10-dimensional minimization problems: 24 BBOB functions with the distributed fixed transformations, six CEC-style analytic functions with zero shift, and the same six analytic functions with one sampled shift per function. The analytic group is not an official CEC competition suite. Each condition uses ten paired initial populations of 100 points, with a total budget of 300 objective evaluations per trajectory, including initialization.

Initial populations are generated from seeds 20261000 through 20261009 and reused across configurations. For each analytic function indexed by j, its shift is sampled within the implemented function-specific range using seed 20260925+j. Model construction and optimization use separate fixed RNG resets in the experiment driver; these initial-population seeds should not be interpreted as independently sampled function instances. The archived driver checks that the optimizer's evaluation count and an objective-wrapper counter both equal 300 evaluations per trajectory.

The primary outcome for this comparison is the final best objective value. We report function means and paired endpoint win/tie/loss counts, with strict numerical equality defining ties. These are descriptive comparisons. Repeated initializations on one function instance do not provide the same evidence as repeated independently sampled instances. This protocol also differs from the original manuscript's 100-trajectory protocol and is reported separately.

## Results — new internal comparison subsection

Disabling the forced anchor restriction leaves all 240 paired BBOB endpoints and all 60 unshifted analytic endpoints unchanged. On the shifted analytic instances, it lowers the mean final objective for all six functions. Across their 60 paired initializations, 59 endpoints improve, one ties, and none worsen. The mean reductions range from 7.62% on Ackley to 79.30% on Rosenbrock. These reductions compare the revised configuration with the original configuration on the same shifted instances; they are not comparisons with external optimizers.

The result identifies a restriction that was harmful on the tested shifted cases. It does not isolate the contribution of the residual model, establish superiority over external baselines, or show that removing the restriction is sufficient for general coordinate robustness. Equality on the fixed BBOB instances is also consistent with the fact that the removed `cecf`-specific restriction did not apply to them. The archived paired table contains endpoints, so this analysis does not claim verified equality of entire search trajectories.

## Statistical reporting — replace the “5%” description

Where retained for historical comparison, the practical-difference diagnostic uses a mixed absolute–relative threshold of 0.05 max(1, |m_V|), where m_V is the comparator mean. For comparator means with magnitude below one, this criterion applies an absolute tolerance of 0.05. It should therefore not be interpreted as a uniform 5% relative-difference test. Formal uncertainty assessment should accompany effect sizes and account for the hierarchy of functions, instances, and optimization repetitions.

## Limitations and conclusion — replace the final interpretation

ROOPF exposes a candidate-level interface between a frozen search policy and evidence fitted to the current objective. The revised configuration permits this interface to operate on the CEC-style tasks by removing a forced anchor-selection rule. The available paired results show improvements on the six tested shifted analytic instances while preserving final values on the tested fixed and unshifted instances.

The present evidence is limited to 10 dimensions, a 300-evaluation budget, and one shifted instance per analytic function. The evaluated instances have informed development decisions and are not a substitute for an independently frozen confirmation set. The implementation also retains task-category-dependent logic. In addition, full reproduction of the offline training procedure and the original application experiments requires assets beyond the currently available inference package. These limitations constrain claims about generalization, component necessity, and application transfer. Preserving an anchor slot remains a structural property of the method, not a guarantee that its final outcome will dominate an anchor-only optimizer.

## Existing material requiring reconciliation before submission

- Do not relabel old external-baseline or ablation tables as results of the revised configuration.
- Resolve the shift configuration behind the original analytic tables before describing them as shifted or unshifted measurements.
- Audit training-task generation and benchmark-guided curation before retaining claims of fully unseen test functions or a independently validated training-suite contribution.
- Update Fig. 2 and the algorithm to distinguish anchor-slot preservation from a preserved counterfactual trajectory.
- Restore and rerun UAV objectives before retaining application-transfer claims in the revised abstract or conclusion.
- Disclose online computational overhead and offline labeling/training costs separately from online objective calls.
