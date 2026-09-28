# VERIFICATION CHECKLIST

## Reproducibility
- [x] Historical runs are available under `projects/rubric_reward/runs/`.
- [x] Config snapshots are available under `projects/rubric_reward/configs/run_configs/2026/`.
- [x] Run indexing script generates reproducible run registry files.

## Extendability
- [x] Method categories organized under `src/co_scientist/rubric_reward/` (grpo, sdpo, refinement, etc.).
- [x] Source-of-truth trainer code remains under `src/co_scientist/rubric_reward/`.
- [x] Config templates exist under `projects/rubric_reward/configs/templates/`.

## Traceability
- [x] Path migration documented in `shared/docs/migration_map.md`.
- [x] Legacy structure preserved under `ARCHIVE/legacy_snapshot/`.
- [x] Experiment protocol documented in `shared/docs/EXPERIMENTS.md`.

## GitHub Readiness
- [x] `.gitignore` excludes heavy artifacts by default.
- [x] Agent handoff doc is provided (`shared/docs/AGENT.md`).
- [x] Status report is provided (`shared/docs/STATUS.md`).
