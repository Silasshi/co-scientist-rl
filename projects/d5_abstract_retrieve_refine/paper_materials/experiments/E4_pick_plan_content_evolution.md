# E4 — PICK plan (teacher distribution) content evolution

PICK plans = teacher samples (with critique in prompt) selected for SDPO training. Source: `runs/2026_04_27_mu_v4/buffer.jsonl`. (μ-v4 trainer picks 1 plan per iter for the `prev_critique` slot of next iter — these are the teacher's response to the previous critique.)

## Per-iter content markers

| Iter | n | PUCT | β(s) | J_β | adv form | word salad | XML leak | mixed script | avg chars |
|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| 0 | 8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 4364 |
| 1 | 8 | 8/8 | 8/8 | 7/8 | 0/8 | 0/8 | 0/8 | 0/8 | 4669 |
| 2 | 8 | 1/8 | 7/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 4895 |
| 3 | 8 | 8/8 | 7/8 | 4/8 | 0/8 | 0/8 | 0/8 | 0/8 | 6057 |
| 4 | 8 | 8/8 | 8/8 | 5/8 | 0/8 | 0/8 | 0/8 | 0/8 | 7167 |
| 5 | 8 | 2/8 | 3/8 | 2/8 | 0/8 | 1/8 | 0/8 | 3/8 | 6829 |
| 6 | 8 | 3/8 | 1/8 | 0/8 | 0/8 | 2/8 | 2/8 | 5/8 | 9490 |
| 7 | 8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 7/8 | 8994 |

## Key observation

From daemon final report (Opus subagent reading critic_responses): iter 1-3 PICK plans contain PUCT, J_β, adv form, Q=max child. iter 4 starts Goodhart-bloat (LC-15 categories, RSA-2048). iter 5-7 word salad (Brobdingnag, Mudra method, baryonic merging) + recursive meta-critique.
