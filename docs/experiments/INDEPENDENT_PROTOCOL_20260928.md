# Frozen post-development expansion (10 dimensions)

Before any full expansion results are examined, freeze:

- COCO 2.8.2: all 24 noiseless BBOB functions, instances 101 and 102, dimension10.
- opfunu 1.0.4: all 20 CEC2017 hybrid/composition classes F10..F29 according to
  that package's numbering. This is a versioned third-party implementation,
  not asserted to be the official competition implementation. All 20 classes
  passed deterministic repeated-value and known-optimum checks before runs.
- Ten repetitions per case, seeds 20265000..20265009, same for every method.
- Methods: final ROOPF, identical frozen anchor, pycma4.5.0 CMA-ES, SciPy DE,
  and uniform random search. This extension does not replace the missing matched
  BO/learned-baseline expansion; do not claim coverage of every baseline family.
- Every trajectory exactly300 paid objective evaluations. ROOPF/anchor/DE share
  100 initial points; DE uses two generations with mutation(.5,1), crossover.7,
  no polishing. CMA-ES uses population10, domain-center mean, sigma.2*width,
  30 generations, no early termination. Random uses300 uniform points.
- CPU one thread per worker,12 independent workers. Wall times collected here
  are not used as uncontended timing comparisons. Each optimizer trajectory
  is run individually; raw observations in ROOPF use float32 while external
  implementations evaluate float64. No algorithm uses optimal-point metadata.
- No tuning after the development ablations. Function-name category mapping
  preserves the final method's existing BBOB/CEC logic. All cases, including
  failures, are retained. Original checkpoint SHA-256 values remain unchanged.

New COCO instances address instance generalization, not previously unseen BBOB
families. Hybrid/composition tests expand landscape coverage, but we do not assert
that all constituent primitives were absent from training-suite ELA references.
The post-development protocol does not retroactively eliminate historical
benchmark-guided development. This extension is10-D; no unsupported dimensional
extrapolation is claimed.

Smoke tests used a separate seed20269999 and COCO instance999; the opfunu hybrid
F10 was also used for interface smoke testing. Declare that overlap; there was
no performance-based tuning from the smoke outcome.

Sources: https://coco-platform.org/getting-started/ ;
https://opfunu.readthedocs.io/en/stable/pages/cec_based.html .
Exact package versions and source hashes are saved in each run's protocol.json.
