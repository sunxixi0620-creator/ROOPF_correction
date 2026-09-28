# Evidence-aligned manuscript material

These passages describe the selected final ROOPF and its controlled supplementary studies. They do not compare historical software releases or frame a configuration change as a contribution. Fill numerical tables from `FINAL_RESULTS_20260928.zh-CN.md` only after its completion checks pass.

## Contribution framing

We investigate how a frozen learned optimizer can be combined with task-specific online information under a limited objective-evaluation budget. ROOPF supplies a portfolio of candidate points, uses an online proxy and offline incumbent-improvement evidence to score them, and retains an anchor evaluation slot while applying an explicit comparison rule to the competing slot. Our contribution is the concrete candidate-level allocation mechanism and its empirical analysis, rather than a guarantee that surrogate-guided intervention cannot degrade search.

We provide a reproducible 36-function generated training resource and disclose how benchmark landscape summaries entered its audit. The recovered final curation preserves all 36 function members and changes their order; consequently, we distinguish a function-order control from a membership-selection ablation. We do not claim that this curation demonstrates superior function membership or completely benchmark-independent development.

## Motivation and warm-up

The online proxy is fitted during search, but its recommendations do not affect true evaluations during the initial anchor-only phase. This phase accumulates task observations before intervention is enabled. The default boundary is a design choice with task-dependent tradeoffs. Direct pool diagnostics show that late-stage proxy reliability differs substantially between the tested landscapes, so neither universal late-stage accuracy nor universal benefit from a protective margin is assumed.

## Interpretation of the selector

The residual predictor is trained to identify candidates that improve the incumbent. This label differs from improvement over the competing anchor candidate. Its sigmoid output is used as ranking evidence; our calibration audit does not support interpreting it as a calibrated probability of a beneficial replacement. Retaining an anchor slot preserves a source of proposals, but changes in the other slot affect the subsequent population and therefore the future anchor proposals. An independent anchor-only trajectory is not protected by a pathwise non-degradation theorem.

The implementation appendix reports the complete score, comparison margins, rescue conditions, active task-category branches, archive updates and counter behavior. In the selected configuration, the historical-evidence counter required by the veto condition is not updated; thus that veto must not be described as an empirically active safeguard. This disclosure is essential to matching the manuscript to the executable final method.

Only the anchor and residual predictor load trained checkpoints in the released final evaluation path. The state encoder, neural portfolio module and gating network retain their fixed-seed initial parameters. The method description must distinguish these initial parameters from learned offline knowledge and must not imply that every frozen network has been trained.

## Position relative to existing work

MetaBO meta-trains acquisition functions for transfer within Bayesian optimization. ROOPF instead studies an explicit competition between frozen anchor proposals and a candidate portfolio, with a retained anchor slot. This is a distinction in the implemented decision structure, not a claim to originate offline-to-online transfer. [Volpp et al., ICLR 2020](https://iclr.github.io/build/virtual/poster_ryeYpJSKwr.html).

Surr-RLDE learns surrogate objectives to reduce the cost of training an RL policy that configures DE mutation operators. ROOPF's task-specific proxy is fitted to observations from the current objective, while its residual signal is learned from offline candidate outcomes. The supplementary comparison uses the authors' released Surr-RLDE policy with explicit budget adaptation and independently counted evaluations. [Ma et al., GECCO 2025](https://arxiv.org/abs/2503.18060), [released implementation](https://github.com/MetaEvo/Surr-RLDE).

## Protocol and scope

We distinguish development/regression instances, frozen-configuration evaluation on new benchmark instances and complex landscapes, and native-dimensional controlled retraining. New instances do not undo prior use of their benchmark family in configuration development or landscape auditing. The native20-D study trains dimension-specific components and separately evaluates transfer of the original10-D residual. It is not a claim that a10-D checkpoint directly handles20-D inputs. One training initialization is used per reconstructed treatment; uncertainty over search initializations must not be described as uncertainty over training seeds.

## Mechanism and application limitations

Fixed-width operator controls and single-intervention replays reveal heterogeneous component effects. We retain all evaluated conditions and report negative as well as positive differences. Descriptive concentration of observed incumbent decreases is not an additive causal decomposition of improvement over anchor-only.

The UAV experiments optimize a sampled soft-penalty path-planning objective. We additionally evaluate continuous segment-to-circle clearance and boundary compliance at the returned point. A lower objective does not imply geometric feasibility, and these proxy experiments do not establish physical deployment safety.

## Evidence-aligned conclusion

The experiments support the use of online candidate information to improve the tested frozen anchor in several regimes. They do not establish consistent superiority over strong independent optimizers, universal benefit from every operator or the protective gate, calibrated replacement probabilities, or guaranteed non-degradation. These limits should appear in the main discussion and conclusion rather than being hidden behind an aggregate of favorable tasks.

## Replacement abstract supported by the completed comparisons

Expensive black-box optimization requires allocating a limited number of objective evaluations between proposals informed by prior experience and alternatives informed by current observations. We study this allocation through ROOPF, which combines a frozen learned anchor, an online proxy, a six-operator candidate portfolio, and offline incumbent-improvement evidence. Following an anchor-only observation phase, one evaluation slot retains an anchor proposal while an explicit comparison rule controls the competing slot. We provide matched component ablations, candidate-level diagnostics, single-intervention replays, and a disclosed training and evaluation protocol. In frozen-configuration 10-dimensional tests, ROOPF improves the mean endpoint over the same anchor in 46 of 48 COCO conditions and all 20 tested CEC-style hybrid/composition conditions. Native 20-dimensional controlled retraining likewise improves 46 of 48 conditions at both tested budgets. However, strong CMA-ES and GP-EI configurations outperform ROOPF on most tested 10-dimensional conditions, and the benefits of the protective gate and residual component vary by task. The results support task-dependent improvement of the tested anchor through online information, while delimiting claims of general optimizer superiority and guaranteed reliability.

Editorial note: “improves” above refers to strict condition-mean comparisons, not a multiple-comparison-adjusted significance claim. The 20 complex functions use the recorded third-party CEC2017 implementation; describe that implementation precisely in the experimental section. Native 20-D results use newly trained, separately labelled checkpoints and do not replace the fixed 10-D main method.
