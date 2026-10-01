# V2 engineering amendment, before inspecting aggregate method outcomes

Parent: STRONG_ONLINE_BASELINES_PROTOCOL.md. Tasks, initialization, budget,
kernels, acquisition, trust-region rules and inference remain identical.

V1 was halted after two TuRBO ModelFittingErrors. Six complete GP runs (1,800
calls) are retained. Twenty case jobs had started. Fourteen uncommitted/failed
jobs lack per-step logs; their total actual calls are bounded by [140,4,200],
not exactly reconstructible. Do not invent an exact total. V1 contracts used 48
calls. A deterministic reproduction on the first failed job used 165 calls,
which are recorded separately and not used as a performance result.

At the saved 165-observation state, L-BFGS-B from previous fitted parameters
returns ABNORMAL after 3 steps. Reinitializing the same GP on the same data
permits 100 iterations with a finite improved negative marginal likelihood
(-3.574718 vs -3.495014). This is evidence about fitting numerics, not about
optimization success. No target queries are used to compare these fits.

V2 first attempts exactly the V1 warm fit; if BoTorch raises ModelFittingError,
construct the same model with default starting parameters and fit again with
the same 100-step cap and existing retry rules. Log every recovery. On further
failure, retain the case as failed with its paid data. Never delete failures or
silently use random search. Thus completed V1 cases are valid imports: their
successful paths are unchanged. Store each original data hash in the new case.

Add an atomic full state checkpoint after every paid batch/point and support
exact resumption. Record source identity, paid x/y, GP parameters, trust state,
RNG identity and complete trace. The parent records each case success/failure
instead of blocking indefinitely in executor shutdown before printing errors.

V2 contracts: 24 paid calls for two small trajectories compared to V1, 12 paid
calls for an 11-to-12 exact resume check, zero calls for failure-state recovery
and posterior replay. Full final table still has 144 trajectories / 43,200 calls;
1,800 come from V1 and 41,400 are newly required by V2. Interrupted V1 overhead,
165 diagnostic calls and 84 combined V1/V2 contract calls are reported separately.
No new prior training and no additional W trajectories.

The final conclusion stays pending until all cases and score replays pass.
