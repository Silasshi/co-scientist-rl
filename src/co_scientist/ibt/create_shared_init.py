"""Create a shared LoRA initialization checkpoint for fair comparison.

All training variants load from this checkpoint so they start with
identical LoRA weights.

Usage:
  python create_shared_init.py api_profile=NEW
"""

import logging
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import chz
from co_scientist.shared.api_profiles import create_service_client
from tinker_cookbook import checkpoint_utils

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    base_url: str | None = None
    api_profile: str | None = "NEW"
    policy_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    output_path: str = "/home/silas/co-scientist-project/runs/2026/4/ibt/1"


def main(config: Config):
    logging.basicConfig(level=logging.INFO)

    service_client = create_service_client(
        base_url=config.base_url, api_profile=config.api_profile,
    )

    logger.info(f"Creating LoRA training client: {config.policy_model}, rank={config.lora_rank}")
    training_client = service_client.create_lora_training_client(
        base_model=config.policy_model, rank=config.lora_rank,
    )

    logger.info(f"Saving initial checkpoint to {config.output_path}")
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="shared_init",
        log_path=config.output_path,
        kind="state",
        loop_state={"goal": -1, "global_step": 0},
    )
    logger.info("Done. All variants should load from this checkpoint.")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
