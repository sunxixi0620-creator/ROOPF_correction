# Completed long-training and four-group archive

This archive contains the six selected1000-epoch-budget anchors, six matched
residuals, milestone model/optimizer states, residual label data, raw four-group
trajectories, and source snapshot. The original released weights are unchanged.
`manifest.json` records61 payload checksums; the README is later documentation.

There are8640 main trajectories: A/F/FR have three matched anchors; O has only
864 independent trajectories across the three conditions and is shared as the
comparator, not duplicated independent runs. All100 initialization evaluations
count in every method's budget. Cases use new instances of the same procedural
recipes, not wholly unseen-family benchmarks. Teacher data are used only for
residual training, not main online decisions.

Training snapshots contain full state at80/160/320/640/1000 epochs; the selected
model may be from an intermediate validation epoch.10D seed1 resumed from30,
whereas the other five resumed from80. Adam states and original random roles are
preserved; the old early stopping rule is explicitly removed in this extension.
See `../complementarity_training_start_v1` for verified starting states.

Results, histories, serial timing and costs are under
[`docs/revision/complementarity/four_group`](../../docs/revision/complementarity/four_group/RESULTS.zh-CN.md).
Interpretation is in [the conclusions](../../docs/revision/complementarity/CONCLUSIONS.zh-CN.md).
The64 serial timing trajectories are repeats and do not increase main N. Their
28,800 additional calls are counted separately. Concurrent process seconds are
not fair serial runtime comparisons or additive GPU hours.

The negative F/FR-vs-O and residual findings are retained. Any subsequent
allocation experiment is a separately identified candidate, not a relabeling
of these completed results.
