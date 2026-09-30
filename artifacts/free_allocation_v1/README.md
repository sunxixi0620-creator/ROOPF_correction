# Completed bounded allocation follow-up

Nine case ZIPs contain648 four-population batches:2592 new Free trajectories
over10D300,20D300,20D600, three frozen anchor seeds and36 configurations with two
instances each. This adds1,036,800 objective calls. `sources.zip` records the
four new frozen source/protocol files; their unchanged dependencies and reused
anchor/A/O/F cases remain in `../complementarity_v1` and its source manifest.

Every new case retains raw queried points, observed best-value trajectories,
task parameter tensors, decision records and its identity/content manifest.
There are no main teacher queries. The anchor-free implementation reproduces
the previously frozen O exactly in contract tests. The whole38-candidate pool
competes for two slots; learned anchor proposals replace O's two uniform extras.

The previous four-group data informed this follow-up, so these results are
development evidence. O is shared as a comparator rather than counted three
times as independent data. The predefined20D600 joint gate failed, and the
conditional new-instance confirmation was not run. That is the declared stop
rule, not an incomplete stage. The20D300 positive result is retained without
changing the primary target after observing results.

See [conclusions and plots](../../docs/revision/free_allocation/CONCLUSIONS.zh-CN.md)
and [complete comparisons](../../docs/revision/free_allocation/development/RESULTS.zh-CN.md).
`manifest.json` records10 payload hashes. Package verification additionally
checks ZIP CRCs; the README and plotting helper are later reporting materials.
