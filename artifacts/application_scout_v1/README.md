# Public application source scout

Seven exploratory TuneControl objective calls, no optimizer comparison or neural training. The unmodified official 0.1.0 wheel and its embedded MIT license are retained. PyPI download SHA256 was checked against publisher metadata. Sources, limitations and candidate recommendations are in `docs/revision/application_search/SEARCH_AND_RECOMMENDATION.zh-CN.md`.

`smoke_tunecontrol.py` loads the wheel from its own directory and writes a result JSON there. It was run under existing Python3.10 solely as a compatibility probe; official support requiresPython3.12. Repeat in an isolated supported environment before scientific use. Existing ROOPF dependencies were not changed. The preserved run result is in `docs/revision/application_search/TUNECONTROL_SMOKE.json`.

No Furuta or safe-control-gym performance results are included. Furuta fetch/clone attempts encountered network connection resets. No conclusion about optimizer success, task difficulty, or hardware safety follows from this scout.
