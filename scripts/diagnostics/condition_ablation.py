"""Conditioning ablation — does the denoiser USE the phoneme content, or only the speaker prompt?

Splits cause from victim. We already know the output varies by speaker (prompt path works); this asks
the dual: hold the prompt fixed and perturb the content `condition`, re-sampling from identical noise.
  content_rmse   = ‖sample(real) − sample(zeroed condition)‖   how much content moves the output
  speaker_rmse   = ‖sample(real) − sample(swapped prompt)‖     how much speaker moves the output
  content/speaker ratio ≪ 1  ⇒ the denoiser ignores content (problem is downstream of the aligner).
  ratio ~ O(1)               ⇒ the denoiser uses content → a bad alignment is the suspect, not the wiring.

Same seed before every sample() ⇒ differences are purely the intervention (not noise). Real condition
is the GT-aligned one from the training forward (return_diffusion_inputs). Read-only.

Usage:
  python scripts/diagnostics/condition_ablation.py checkpoint=.../ckpt.pt num_samples=8 sampling_steps=150
"""
import logging
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

# Load environment variables from .env file (e.g. WANDB_API_KEY)
load_dotenv()

from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH
from naturalspeech2.paths import PROJECT_ROOT
from naturalspeech2.utils.utils import setup_file_logger
from naturalspeech2.utils.warning_filters import install_warning_filters

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _common import load_model_and_loader

logger = logging.getLogger("condition_ablation")


def masked_rmse(a, b, mask) -> torch.Tensor:
    m = mask.to(a.dtype)
    diff2 = ((a - b) ** 2) * m
    denom = (m.sum() * a.shape[-1]).clamp(min=1)
    return (diff2.sum() / denom).sqrt()


@torch.no_grad()
def ablate_batch(model, batch, device, seed, sampling_steps) -> dict:
    fwd = {k: batch[k].to(device) for k in
           ("audio", "audio_lengths", "phoneme_tokens", "phoneme_tokens_mask", "phoneme_tokens_lengths", "pitch")}
    _, di = model(
        fwd["audio"], fwd["audio_lengths"],
        fwd["phoneme_tokens"], fwd["phoneme_tokens_mask"], fwd["phoneme_tokens_lengths"],
        fwd["pitch"], return_diffusion_inputs=True,
    )
    cond = di["condition_target"]
    cmask = di["target_latents_mask"]
    penc, pmask = di["prompt_encodings"], di["prompt_encodings_mask"]
    target = di["target_latents"]
    B, Ft, _ = cond.shape

    def sample(c, pe, pm):
        torch.manual_seed(seed)   # identical z_T across variants → differences are the intervention only
        return model.diffusion_model.sample(c, cmask, pe, pm, sampling_steps=sampling_steps)

    cond_zero = torch.zeros_like(cond)
    perm = torch.randperm(Ft, device=device)
    cond_shuf = (cond[:, perm, :]) * cmask.to(cond.dtype)         # temporally scramble content

    real = sample(cond, penc, pmask)
    metrics = {
        "ablation/recon_rmse": masked_rmse(real, target, cmask),                 # sample(real) vs GT
        "ablation/content_rmse_zero": masked_rmse(real, sample(cond_zero, penc, pmask), cmask),
        "ablation/content_rmse_shuffle": masked_rmse(real, sample(cond_shuf, penc, pmask), cmask),
    }
    if B > 1:
        flip = torch.roll(torch.arange(B, device=device), 1)     # same content, neighbor's speaker
        metrics["ablation/speaker_rmse"] = masked_rmse(real, sample(cond, penc[flip], pmask[flip]), cmask)

    audio = {
        "real": model.encodec.decode_from_latents(real),
        "zeroed": model.encodec.decode_from_latents(sample(cond_zero, penc, pmask)),
    }
    return {"metrics": metrics, "audio": audio, "frame_lengths": cmask.squeeze(-1).sum(dim=1)}


def render(scalars, audio_rows, sr, cfg, run_name, step) -> None:
    logger.info(f"=== Conditioning ablation (step {step}) ===")
    for k in sorted(scalars):
        logger.info(f"  {k}: {scalars[k]:.4f}")
    if "ablation/speaker_rmse" in scalars and scalars["ablation/speaker_rmse"] > 1e-6:
        ratio = scalars["ablation/content_rmse_zero"] / scalars["ablation/speaker_rmse"]
        logger.info(f"  ablation/content_over_speaker = {ratio:.3f}  (<<1 ⇒ denoiser ignores content)")

    if not cfg.wandb.log:
        return
    wandb.init(project=cfg.wandb.project, name=run_name, group=cfg.wandb.group,
               notes=cfg.wandb.notes, tags=list(cfg.wandb.tags),
               config=OmegaConf.to_container(cfg, resolve=True))
    wandb.define_metric("eval/snapshot_step")
    wandb.define_metric("ablation/*", step_metric="eval/snapshot_step")
    payload = dict(scalars)
    if "ablation/speaker_rmse" in scalars and scalars["ablation/speaker_rmse"] > 1e-6:
        payload["ablation/content_over_speaker"] = (
            scalars["ablation/content_rmse_zero"] / scalars["ablation/speaker_rmse"])
    table = wandb.Table(columns=["sample", "real", "zeroed_condition"])
    for i, (real_wav, zero_wav) in enumerate(audio_rows):
        table.add_data(i, wandb.Audio(real_wav, sample_rate=sr), wandb.Audio(zero_wav, sample_rate=sr))
    payload["ablation/audio"] = table
    payload["eval/snapshot_step"] = step
    wandb.log(payload)
    wandb.finish()
    logger.info(f"Logged to wandb: {cfg.wandb.project}/{run_name}.")


@hydra.main(config_path="../../config", config_name="diagnostics", version_base=None)
def main(cfg: DictConfig) -> None:
    tag = Path(cfg.checkpoint).stem
    setup_file_logger(logger, PROJECT_ROOT / "logs" / "diagnostics" / f"ablation_{tag}.log", root=True)
    logging.captureWarnings(True)
    install_warning_filters()
    assert torch.cuda.is_available(), "diagnostics require an NVIDIA GPU"

    model, loader, _, sr, step = load_model_and_loader(cfg)
    logger.info(f"Loaded {cfg.checkpoint} (step {step}); ablating up to {cfg.num_samples} samples from {cfg.split}.")

    sums, count, audio_rows = {}, 0, []
    for batch in loader:
        out = ablate_batch(model, batch, cfg.device, cfg.seed, cfg.sampling_steps)
        for k, v in out["metrics"].items():
            sums[k] = sums.get(k, 0.0) + v.item()
        count += 1
        for b in range(min(2, out["audio"]["real"].shape[0])):
            if len(audio_rows) >= cfg.num_samples:
                break
            t = int(out["frame_lengths"][b].item()) * ENCODER_HOP_LENGTH   # valid samples (trim padding)
            real = out["audio"]["real"][b].squeeze(0).float().cpu().numpy()
            zero = out["audio"]["zeroed"][b].squeeze(0).float().cpu().numpy()
            audio_rows.append((real[:t], zero[:t]))
        if len(audio_rows) >= cfg.num_samples:
            break

    scalars = {k: v / count for k, v in sums.items()}
    render(scalars, audio_rows, sr, cfg, run_name=cfg.wandb.run_name or f"ablation_{tag}_step{step}", step=step)


if __name__ == "__main__":
    main()
