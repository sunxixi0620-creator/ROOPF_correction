# R06: Event-triggered, action-reviewed evidence fusion

## Combined repair

The full37-candidate scoring and the separate protected Anchor are retained. R05, the original baseline, neural weights, and earlier raw records are preserved.

The first request is triggered by an available protected alternative and evidence conflict, reliability contrast, or stagnation after5 warmup rounds, no later than active round25. Each executed nonrelease action receives a local5-round progress review. A second request is reserved for trajectories with completed intervention feedback; it is not spent on another unrelated action-only epoch. Each batch has at most two request epochs. A second intervention is still locally reviewed before the45-round active window ends. A failed API call falls back without automatic paid retries.

Online utility is actual competing-point improvement over the previous incumbent, normalized by observed archive spread, minus a fixed0.001 evaluation-opportunity cost. Beating a poor Anchor without improving the incumbent is no longer rewarded. Observed5-round search progress is supplied separately and explicitly not called causal action advantage. Rank reliability uses at most16 recent observed pairs with Beta(2,2) smoothing; selected-pair bias remains disclosed.

Offline cases are rebuilt on36 generated training functions and4 seeds each. At most four informative states per episode are selected by preregistered windows. Each state uses the exact current37-candidate policy and feedback-dependent parameters. Counterfactual branches share the complete original prefix, population and RNG, execute one action and return to release, with1/5/10-round best-value labels. The same seven heldout generated functions remain separate from retrieval. The35 test functions never supply case labels.

The provider contract remains JSON-object output with thinking disabled and complete trajectory IDs. The LLM receives compact precomputed action evidence and matching-policy case summaries rather than repeated full vectors and obsolete case labels. Full37 vectors are retained in local logs and cryptographically bound to every request. Incomplete outputs remain invalid; no claim is made that a third-party service cannot truncate or fail.

## Validation and limits

Contract checks are explicitly mocked local unit tests, not experimental results. They verify event round5, review round10, actual use of prior-action observations, incumbent-aligned reward, immutable feedback, and pure-online independence.

The full experiment remains35 functions x100 trajectories, D10,300 evaluations including initial100, with original-equivalent Agent-off, same-structure deterministic current-case controller, and real DeepSeek API controller. The actual original100 populations and solver RNGs are reused and verified. Exact objective-response caching is allowed; LLM decisions are not substituted by mock/retrieval outputs.

The global paid-request cap is600. Provider credits have no assumed currency. BBOB20 remains separate from primary analysis; CEC structured locks remain. This is a combined repair on a development-inspected benchmark, not a blind test or an individual-factor causal ablation. A successful full run is not proof of improvement.

Completion requires all35 functions, all100 trajectories, actual API-to-execution matching, full-state-to-compact-request identity, causal feedback snapshots, independently recalculated incumbent utilities, and local review coverage of every executed action. Final reports and complete raw archives are published on the independent R06 baseline only after audits and credential checks.
