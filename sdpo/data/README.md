# sdpo/data

Same format and protocol as [`grpo/data`](../../grpo/data/README.md). Run names map to the original run ids in [`runs.csv`](runs.csv).

The key comparison is `sdpo` vs `grpo_control`. Both are trained by **the same script** (`sdpo/train.py`); in the control the distillation term is disabled (`use_sdpo=False`). Both run 107 batches and are evaluated at checkpoint b106 on the same 640 × 8 eval set.

The `external_critic/` variant has **no data**: it was implemented and unit-tested but never trained.
