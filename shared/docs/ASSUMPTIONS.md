# ASSUMPTIONS

1. `projects/rubric_reward/runs/` is the canonical historical run store for this repository.
3. Legacy external run links are archived under `ARCHIVE/legacy_snapshot/` and are not canonical.
4. Any missing metric field in logs should be treated as `unknown`, not as zero.
5. Reproducibility requires preserving `config.json`, `code.diff`, `metrics.jsonl`, and batch/train logs when available.
6. This repository is prepared for GitHub while large raw artifacts remain local-first and ignored by default.
