import asyncio
import logging
from pathlib import Path
import sys

import chz

# Allow direct script execution without requiring manual PYTHONPATH setup.
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from co_scientist.rubric_reward.sdpo.train_sdpo import Config as SDPOConfig
from co_scientist.rubric_reward.sdpo.train_sdpo import main as run_sdpo_training


logger = logging.getLogger(__name__)


def _sanity_log_path(base_log_path: str) -> str:
    path = Path(base_log_path)
    suffix = "(sanity check)"
    if path.name.endswith(suffix):
        return str(path)
    return str(path.with_name(f"{path.name}{suffix}"))


async def main(config: SDPOConfig):
    """
    GRPO-only sanity run entrypoint.

    It reuses the SDPO training pipeline but force-disables SDPO terms:
    - use_sdpo = False
    - sdpo_grpo_mix_lambda = 1.0 (pure GRPO advantage path)
    - sdpo_teacher_reg_alpha = 0.0 (no teacher interpolation)
    - keeps grader/token/runtime settings identical to the input config
      for fair SDPO-vs-GRPO comparison.
    """
    sanity_config = chz.replace(
        config,
        log_path=_sanity_log_path(str(config.log_path)),
        use_sdpo=False,
        sdpo_grpo_mix_lambda=1.0,
        sdpo_teacher_reg_alpha=0.0,
        grader_prompt_mode=config.grader_prompt_mode,
        fast_grader_mode=config.fast_grader_mode,
        max_tokens=config.max_tokens,
        grader_max_tokens=config.grader_max_tokens,
        max_batches=config.max_batches,
        format_retry_enabled=config.format_retry_enabled,
        format_retry_max_attempts=config.format_retry_max_attempts,
    )
    logger.info(
        "Running GRPO sanity check with overrides: log_path=%s, use_sdpo=%s, lambda=%s, "
        "teacher_reg_alpha=%s, grader_prompt_mode=%s, fast_grader_mode=%s, "
        "max_tokens=%s, grader_max_tokens=%s, max_batches=%s, format_retry_enabled=%s, "
        "format_retry_max_attempts=%s",
        sanity_config.log_path,
        sanity_config.use_sdpo,
        sanity_config.sdpo_grpo_mix_lambda,
        sanity_config.sdpo_teacher_reg_alpha,
        sanity_config.grader_prompt_mode,
        sanity_config.fast_grader_mode,
        sanity_config.max_tokens,
        sanity_config.grader_max_tokens,
        sanity_config.max_batches,
        sanity_config.format_retry_enabled,
        sanity_config.format_retry_max_attempts,
    )
    await run_sdpo_training(sanity_config)


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
