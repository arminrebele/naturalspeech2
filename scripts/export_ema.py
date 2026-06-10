"""Re-serialize a crash-recovery ckpt.pt's EMA weights into ema_final.safetensors.

A finished run writes ema_final.safetensors post-loop; a crashed or early-stopped run does not,
even though ckpt.pt already holds the EMA shadow weights. This extracts them into the
inference-loadable safetensors format — the one post-loop file artifact a partial run misses
(ema_best is written in-loop; the final eval only logs metrics, not a file). Pure re-serialization:
build the model from the checkpoint's own cfg, overlay EMA, save. No data, no GPU, no eval — output
matches a finished run's ema_final byte-for-byte (same swap_in + save path as train.py).

Example:
    # in container
    python scripts/export_ema.py \
        --checkpoint models/checkpoints/ckpt.pt \
        --output models/checkpoints/ema_final.safetensors
"""
import argparse
from pathlib import Path

import torch

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.eval.runner import atomic_save_safetensors
from naturalspeech2.model import NaturalSpeech2Model
from naturalspeech2.paths import CHECKPOINTS_DIR
from naturalspeech2.utils.ema import EMA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINTS_DIR / "ckpt.pt",
        help="Crash-recovery checkpoint to read EMA weights from (default: checkpoints/ckpt.pt).",
    )
    parser.add_argument(
        "--output", type=Path, default=CHECKPOINTS_DIR / "ema_final.safetensors",
        help="Destination safetensors path (default: checkpoints/ema_final.safetensors).",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "ema" not in ckpt:
        raise KeyError(
            f"{args.checkpoint} holds no 'ema' state (EMA was disabled for this run) — nothing to export."
        )

    # Rebuild from the checkpoint's own cfg (self-contained — no Hydra/vocab file), mirroring
    # train.py's resume. latent_stats_path → None: the persistent latent_mean/std buffers come from
    # ckpt['model'] below, so skip the redundant (and possibly absent) stats-file load.
    model_cfg_dict = ckpt["model_cfg"]
    model_cfg_dict["encodec"]["latent_stats_path"] = None
    model = NaturalSpeech2Model(
        model_cfg_from_omegaconf(model_cfg_dict),
        token_vocabulary_size=ckpt["token_vocabulary_size"],
        sampling_rate=ckpt["sampling_rate"],
    )
    model.load_state_dict(ckpt["model"])

    ema = EMA(model)
    ema.load_state_dict(ckpt["ema"])

    # Overlay EMA shadow onto trainable params (buffers/frozen stay live) — the exact op train.py
    # runs for its post-loop ema_final write.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ema.swap_in(model):
        atomic_save_safetensors(model, args.output)
    print(f"Wrote {args.output} (EMA weights from iter {ckpt['iter_num']})")


if __name__ == "__main__":
    main()
