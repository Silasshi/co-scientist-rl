# E5 — EVAL plan (student distribution) content evolution

EVAL plans = student samples generated WITHOUT critique in prompt (`goal + slim_oracle` only). These are the production-channel plans. Source: `runs/2026_04_27_mu_v4/eval_rollouts.jsonl`.

## Per-iter content markers (8 plans/iter)

| Iter | n | PUCT | β(s) | J_β | adv form | word salad | XML leak | mixed script | avg chars |
|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| 0 | 8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 4232 |
| 1 | 8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 4501 |
| 2 | 8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 5294 |
| 3 | 8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 5778 |
| 4 | 8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 6980 |
| 5 | 8 | 0/8 | 0/8 | 0/8 | 0/8 | 3/8 | 1/8 | 1/8 | 8137 |
| 6 | 8 | 0/8 | 0/8 | 0/8 | 0/8 | 1/8 | 2/8 | 5/8 | 8841 |
| 7 | 8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 6/8 | 10144 |

## Key observation

**PUCT formula 0/8 across ALL iters (0-7)**, despite teacher PICK plans containing it at iter 1-3. This is the F4 critique-token-blindness diagnostic in concrete form. Word-salad/XML-leak/mixed-script counts ramp from 0 at iter 0-4 to high at iter 5+, matching the F3 collapse curve.
