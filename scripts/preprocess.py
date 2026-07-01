"""Build the dataset cache (F0 + phonemes shards / processed splits) and exit — no training.

Reuses the SAME config composition + create_dataloader path as training, so
`+experiment=<mode>` preprocesses exactly what that mode's run would, for every current and
future experiment. `max_train_clips` is a cumulative target: existing train shards are reused,
only missing ones are F0'd (pay-as-you-go). CPU + disk bound (pyworld + espeak); no GPU.

    python scripts/preprocess.py +experiment=5M
    python scripts/preprocess.py dataset.max_train_clips=200000  # arbitrary row-count target (cumulative)
    python scripts/preprocess.py preprocess_splits='[train]'     # train shards only
"""
import logging
from dotenv import load_dotenv

load_dotenv()

import hydra
from omegaconf import DictConfig

from naturalspeech2.data.loaders import create_dataloader

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    requested = cfg.get("preprocess_splits") or [
        cfg.dataset.train_split, cfg.dataset.dev_split, cfg.dataset.test_split
    ]
    # Train FIRST (it builds the vocab; dev/test assert it exists), deduped, order otherwise preserved.
    splits = sorted(dict.fromkeys(requested), key=lambda s: s != cfg.dataset.train_split)
    logger.info(f"Preprocessing splits (train-first): {splits}")

    for split in splits:
        logger.info(f"=== Preprocessing split '{split}' ===")
        create_dataloader(cfg, split, num_workers=0)  # __init__ builds/loads shards; loader is discarded

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
