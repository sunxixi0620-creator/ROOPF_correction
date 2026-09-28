# Completed frozen external evaluation archive

This archive belongs to the separately trained unified research candidate. The original `final_unlocked` configuration and its checkpoint files are unchanged. Development selected `no_residual` before external performance was available; neither these results nor the source-overlap audit selected another method.

## Contents

- Fifteen `d*_b*_<method>.zip` files contain all 5,400 external trajectories: three dimension/budget conditions, five methods, 360 trajectories per ZIP (12 functions × 30 runs). Each case retains its content and identity manifest as well as its raw data. There are no teacher queries in these external runs.
- `anchor20_0.pt`, `anchor20_1.pt`, and `anchor20_2.pt` are native 20D models selected on generated validation tasks before their respective external evaluations. Their selected epochs are 80, 80, and 75. The 10D anchor and unused residual payloads reside in `../unified_revision_v1/`; no external method executes the residual predictor.
- `native20_task_parameters.pt` stores 2,916 generated parameter sets: 36 tasks for each of 80 training epochs, plus 36 validation tasks. The parameter roles are shared across the three model seeds. Population streams and all splits are defined by the archived generator and identity, not inferred from file names.
- `sources_and_timing.zip` contains the native20/external computation sources, frozen protocols, analysis helpers, and timing artifacts used for this run.
- `manifest.json` records the ZIP and payload checksums and the complete external source/configuration identity. Earlier archive provenance files record intermediate backups, not independent replications or extra performance samples.

Completed numerical tables, plots, checkpoint histories, scheduler metadata, verification and costs are under [`docs/revision/unified_external`](../../docs/revision/unified_external/RESULTS.zh-CN.md). See [final conclusions](../../docs/revision/UNIFIED_CONCLUSIONS.zh-CN.md), [reviewer correspondence](../../docs/revision/UNIFIED_REVIEW_CLOSURE.zh-CN.md), and [reproduction instructions](../../docs/experiments/REPRODUCE_UNIFIED.md).

## Reading the data correctly

`results.csv` is one row per main trajectory. Final objective values and known-optimum errors are reported in float64. The known optimum is a reporting reference and is not supplied to the optimizer. Learned methods share 100-point initializations within each matched case, while CMA-ES and GP-EI retain their native budget-counted starts. For classical baseline rows, `initial_best` and `initial_scale` refer to the first 100 paid queries as a reporting reference; they do not mean that the baseline started with a 100-point initial design. The bounded utility and anchor-degradation comparisons apply to the matched learned methods.

Raw per-case `seconds` include shared-host contention and, for some baseline workers, a documented pause. They are not fair serial speed measurements. The separate 20-repeat timing panel consumes 9,000 additional evaluations, reproduces its corresponding main trajectories exactly, and does not increase the main sample size. Timing includes model load/construction and optimization, but excludes Python startup, objective-instance load, cache verification, and output writing. Concurrent process times are not additive GPU hours.

All 12 versioned opfunu CEC2022 functions remain in the table. Source inspection found reused optimum/shift arrays with previously examined CEC2017 classes for six 10D functions, including shared rotations for two composition functions whose formulas differ. This is frozen external-suite validation, not evidence that every tested landscape family was wholly unseen throughout the research history. See `SOURCE_OVERLAP.json` in the report directory. The new training generator does not use those benchmark arrays or ELA selection.

External effect intervals are nominal pointwise 95% intervals without multiple-comparison correction. Function counts use a practical log-error tie threshold of 0.01; they are not per-function significance tests. The same 12 functions at different dimension/budget conditions are not 36 independent families. All cases, including null or unfavorable comparisons, are retained.
