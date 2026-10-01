# HPO-B-derived raw-dataset grouping: metadata freeze

This supplementary cohort is derived from the already downloaded **official
meta-train and meta-validation tables only**. It is not the official HPO-B-v3
test protocol, and results must not be labeled as official test performance.
The official test response file remains undownloaded. No optimizer has consumed
any HPO-B target response or used a target performance summary before this split.

Map HPO-B second-level keys via the OpenML task API to raw dataset IDs. Combine
identical raw IDs in every search space. Additionally conservatively group
credit-approval (29) with Australian (40509), and the three autoUniv-au7 variants
(1552,1553,1554). The official metadata identifies common credit-approval
provenance and au7 generation/drift lineage. These are protective groupings,
not claims of byte-identical tables. Exact-name or checksum aliases would also
be grouped; the current metadata has none beyond equal raw IDs. Other unrecognized
derived-data relationships remain a limitation.

Use every available group. Sort groups by SHA-256 of the literal string
`hpob_raw_group_v1|` followed by the group ID. First floor(0.6*N) are source,
next floor(0.2*N) development, all remaining final-confirmation. Do not re-roll
the salt or rearrange groups based on coverage or observed performance.

Within each (search space, raw group), retain the lowest numerical OpenML task
ID available in the two allowed files, before looking at y. Distinct X rows
retain the first row. Admit numeric finite [0,1] inputs, dimension 1..64, at
least 105 distinct configurations. A target space must have at least five
eligible **distinct source groups**, rather than counting task-ID aliases as
independent sources. Log all exclusions. Keep all qualifying targets; no
selection on difficulty, transfer similarity or method outcomes.

Budget 105 including five initial observations, five seeds. Algorithm/proxy
training reads source roles only; configuration selection uses development
roles only. Confirmation y may be extracted into a separate evaluator only
after the algorithm/configuration is frozen. Metadata and X can be used for
admission, but final target response statistics cannot. This metadata freeze
does not yet authorize an unfrozen method to evaluate confirmation responses.

The cohort is an independent-pipeline supplement and must coexist with a clear
account of official benchmark protocol differences. Reusing public labels is
normal for HPO-B, but historical loading of JSON for metadata is disclosed;
we claim no model/configuration use of held-out responses, not cryptographic
inaccessibility. A mature transfer reference (e.g. RGPE) should establish that
source information helps against a fitted online GP before training another
ROOPF prior. No new learned module is warranted solely by constructing this split.
