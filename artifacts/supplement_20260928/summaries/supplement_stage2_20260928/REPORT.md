# Supplementary experiment results

Development-instance evidence, not independent confirmation. Full method is the final ROOPF; comparators are controlled ablations or schedules.

720 trajectories, each 300 objective evaluations. Bootstrap intervals condition on each fixed instance and resample paired initializations; they do not establish family-level generalization. No multiplicity-adjusted significance claims are made.

| Group | Comparator | Warm-up | Function mean W/T/L | Paired endpoint W/T/L |
|---|---|---:|---|---|
| bbob | full | 0.3333333333333333 | 3/0/0 | 65/1/24 |
| shifted | full | 0.3333333333333333 | 0/0/3 | 4/0/86 |
| bbob | full | 0.5 | 2/0/1 | 35/31/24 |
| shifted | full | 0.5 | 0/0/3 | 2/0/88 |
| bbob | full | 0.9 | 2/1/0 | 24/60/6 |
| shifted | full | 0.9 | 3/0/0 | 85/0/5 |

W/T/L is full/default against comparator. Tiny exact floating-point differences are counted, so assess paired_summary.csv effect sizes and intervals before interpreting counts.

If parallel_execution.json exists, these runtimes are scheduling diagnostics, not controlled timing comparisons. Runtime is instrumented CPU batch wall time with observer overhead removed; component timers are nested. Amortized seconds are throughput, not isolated single-trajectory latency. CEC residual routing is disabled by the selected final implementation. f20 is excluded for predeclared objective stochasticity/batch coupling.
