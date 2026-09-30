# Real-data HPO prior pilot v1

Development pilot only; joint gate failed. Original ROOPF checkpoints unchanged. The source mean prior is not the original anchor/residual network. Data is sklearn's bundled digits; official provenance links are in the frozen protocol. Source/target image and class-pair indices, raw local data, all 105 SVC objective tables, source prior, 945 search traces, timings, code identity and reports are included.

Extract `real_hpo_prior_v1.zip` at repository root. Check `MANIFEST.json` before using archived outputs. The ZIP includes the original dataset snapshot and task split, so no download is needed. Source checksum checks deliberately reject running a modified algorithm against these results. Environment: Python recorded in identity.json, numpy/scipy versions also recorded, scikit-learn 1.7.2, matplotlib for plots.

Replay without additional classifier fitting:

```bash
.venv/bin/python scripts/real_hpo_prior.py verify --workers 24
.venv/bin/python scripts/real_hpo_prior.py report
.venv/bin/python scripts/finalize_real_hpo.py
```

To recompute from scratch, use a separate checkout and move aside the extracted `results/real_hpo_prior_20260930` directory first; then run phases `freeze`, `source`, `contracts`, `target`, `search`, `verify`, `report` in that order. This performs 26,880 actual SVC fits. The 30,240 online table lookups are separate from, and do not replace, that research cost. Source p is frozen before target generation. No posthoc variant selection.

Read `docs/revision/real_hpo_prior/NEXT_DECISION.zh-CN.md` for limitations and stopping decision. Pair clusters share some target images; their intervals are not independent-dataset confirmation. The initial four random observations already meet the quality target in 85.9% of cases.
