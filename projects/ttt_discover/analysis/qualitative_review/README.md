# Qualitative Review Samples

Manually-curated reference plans and generated samples used for human
eyeballing during signal design iteration.

Contents:
- `manually_picked_{0,1,2}.txt` — reference plans chosen by hand as
  high-quality examples
- `ref_like_0.948.txt` — a generated plan that scored 0.948 (looks like
  a reference)
- `highfaith_0.463.txt` — a generated plan that scored 0.463 (mid-tier)
- `clean_highfaith.txt` — clean, high-faithfulness sample

This is **not** an automated test set — it's a library of samples that
researchers read manually to sanity-check rubric behavior and calibrate
grader prompts.

For automated perturbation-based validation, see `../signal_validity/`.
