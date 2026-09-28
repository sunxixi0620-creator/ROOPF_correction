# Frozen post-development benchmark expansion

3400 trajectories; 300 true objective evaluations each; ten repetitions per method and case.

COCO uses new BBOB instances 101/102. opfunu uses its versioned hybrid/composition classes. This is not a claim of entire unseen primitive families or official CEC competition conformance.

| Suite | Comparator | Cases | ROOPF mean W/T/L |
|---|---|---:|---|
| coco | anchor_only | 48 | 46/2/0 |
| coco | cma_es | 48 | 5/0/43 |
| coco | de | 48 | 40/0/8 |
| coco | random | 48 | 46/0/2 |
| cec2017 | anchor_only | 20 | 20/0/0 |
| cec2017 | cma_es | 20 | 1/0/19 |
| cec2017 | de | 20 | 14/0/6 |
| cec2017 | random | 20 | 17/0/3 |

The strong improvement over the frozen anchor coexists with frequent losses to CMA-ES. No claim of broad baseline superiority is supported by these counts. Runtime records are parallel-execution diagnostics, not uncontended latency measurements. Per-case means and deviations are in per_case_methods.csv; no averages of unnormalized objectives across functions are used.
