# Supplementary experiment results

Development-instance evidence, not independent confirmation. Full method is the final ROOPF; comparators are controlled ablations or schedules.

60 trajectories, each 300 objective evaluations. Bootstrap intervals condition on each fixed instance and resample paired initializations; they do not establish family-level generalization. No multiplicity-adjusted significance claims are made.

| Group | Comparator | Warm-up | Function mean W/T/L | Paired endpoint W/T/L |
|---|---|---:|---|---|
| bbob | anchor_only | 0.7 | 1/1/0 | 1/5/0 |
| shifted | anchor_only | 0.7 | 2/0/0 | 6/0/0 |
| bbob | no_proxy | 0.7 | 1/1/0 | 1/5/0 |
| shifted | no_proxy | 0.7 | 2/0/0 | 6/0/0 |
| bbob | no_residual | 0.7 | 0/1/1 | 1/3/2 |
| shifted | no_residual | 0.7 | 0/2/0 | 0/6/0 |
| bbob | score_only | 0.7 | 0/1/1 | 0/3/3 |
| shifted | score_only | 0.7 | 1/1/0 | 1/4/1 |

W/T/L is full/default against comparator. Tiny exact floating-point differences are counted, so assess paired_summary.csv effect sizes and intervals before interpreting counts.

If parallel_execution.json exists, these runtimes are scheduling diagnostics, not controlled timing comparisons. Runtime is instrumented CPU batch wall time with observer overhead removed; component timers are nested. Amortized seconds are throughput, not isolated single-trajectory latency. CEC residual routing is disabled by the selected final implementation. f20 is excluded for predeclared objective stochasticity/batch coupling.
