# Supplementary experiment results

Development-instance evidence, not independent confirmation. Full method is the final ROOPF; comparators are controlled ablations or schedules.

5250 trajectories, each 300 objective evaluations. Bootstrap intervals condition on each fixed instance and resample paired initializations; they do not establish family-level generalization. No multiplicity-adjusted significance claims are made.

| Group | Comparator | Warm-up | Function mean W/T/L | Paired endpoint W/T/L |
|---|---|---:|---|---|
| bbob | anchor_only | 0.7 | 7/15/1 | 158/531/1 |
| zero | anchor_only | 0.7 | 0/6/0 | 0/180/0 |
| shifted | anchor_only | 0.7 | 6/0/0 | 180/0/0 |
| bbob | no_proxy | 0.7 | 7/15/1 | 150/531/9 |
| zero | no_proxy | 0.7 | 0/6/0 | 0/180/0 |
| shifted | no_proxy | 0.7 | 6/0/0 | 177/0/3 |
| bbob | no_residual | 0.7 | 6/15/2 | 75/549/66 |
| zero | no_residual | 0.7 | 0/6/0 | 0/180/0 |
| shifted | no_residual | 0.7 | 0/6/0 | 0/180/0 |
| bbob | score_only | 0.7 | 0/20/3 | 19/655/16 |
| zero | score_only | 0.7 | 2/4/0 | 3/174/3 |
| shifted | score_only | 0.7 | 0/3/3 | 7/162/11 |

W/T/L is full/default against comparator. Tiny exact floating-point differences are counted, so assess paired_summary.csv effect sizes and intervals before interpreting counts.

If parallel_execution.json exists, these runtimes are scheduling diagnostics, not controlled timing comparisons. Runtime is instrumented CPU batch wall time with observer overhead removed; component timers are nested. Amortized seconds are throughput, not isolated single-trajectory latency. CEC residual routing is disabled by the selected final implementation. f20 is excluded for predeclared objective stochasticity/batch coupling.
