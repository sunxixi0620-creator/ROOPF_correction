# r01 P4b controller pilot

## Protocol

- benchmark: BBOB f1, f8, f15;
- dimension: 10;
- population size: 100;
- true objective budget: 300 NFE;
- trajectories per function: 10;
- seed: 20260630;
- comparison: original ROOPF versus `--agentic-mode p4b`.

## Results

| Function | ROOPF mean ± std | P4b mean ± std | Outcome |
|---|---:|---:|---|
| f1 | 4.3283228e-05 ± 2.4425075e-05 | 4.3283228e-05 ± 2.4425075e-05 | tie |
| f8 | 1.8804614e-02 ± 1.5114039e-02 | 1.8804614e-02 ± 1.5114039e-02 | tie |
| f15 | 1.8423447e-02 ± 1.3048544e-02 | 1.8423447e-02 ± 1.3048544e-02 | tie |

All runs terminated at exactly 300 true objective evaluations.

## Interpretation

The controller changed operator admission counts but did not change final performance on this pilot.
This round is retained as a mechanism baseline, not a performance-improved baseline. The next round
should learn operator advantage from counterfactual candidate labels and control an evaluation slot
directly under the protected safety gate.
