# src/co_scientist/ — Python Package

## Import convention

```python
# Shared utilities (cross-direction)
from co_scientist.shared.api_profiles import create_service_client
from co_scientist.shared.eval_core import build_grader_prompt, compute_rubric_reward_from_xml
from co_scientist.shared.openrouter_client import OpenRouterClient
from co_scientist.shared.seven_signal_reward import compute_seven_signal_reward, SignalReward

# Direction-specific trainers
from co_scientist.rubric_reward.grpo.best_ver import main       # D1
from co_scientist.ibt.train_ibt import main                     # D2
from co_scientist.ttt_discover.train_7signal_vector import main  # D3

# Eval scripts (shared)
from co_scientist.eval.eval_bon import main
```

## Directory map

| Directory | Direction | Contents |
|---|---|---|
| `shared/` | Cross-direction | api_profiles, eval_core, openrouter_client, seven_signal_reward |
| `rubric_reward/` | D1 | grpo/, sdpo/, reward_shaping/, multiturn/, refinement/, selector/ |
| `ibt/` | D2 | train_ibt, train_ibt_v2, baselines, self_calibration |
| `ttt_discover/` | D3 | train_7signal_vector |
| `eval/` | Shared | eval_bon, eval_cpr, eval_double, eval_ibt, eval_multiturn, eval_rag, eval_reference, eval_selector, eval_sota |

## Legacy code

`trainers/` still exists with the original file layout. New development should use the direction-specific paths above. The old `trainers/` paths will be removed after validation that all references are updated.
