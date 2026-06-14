"""Alignment diagnostic — three-way heatmap (prior / learned-alone / posterior) + Viterbi overlay.

Decomposes the aligner so you can tell *cause from victim*: the posterior alone is ambiguous (a
near-diagonal could be a good alignment OR the prior dominating). Plotting the LEARNED scores alone
(pre-prior) against the prior alone disambiguates:
  - learned ≈ uniform / learned≈prior  → aligner inert, prior carries everything (fix the aligner:
    scale / blank / fs↔bin).
  - learned sharp + content-dependent    → aligner is learning → bottleneck is downstream.

Quantitative companions (pooled over valid frames, logged as scalars so checkpoints line up):
  align/learned_peak        mean max-prob of softmax(learned) — vs align/uniform_baseline (1/P).
  align/learned_vs_prior_tv mean total-variation(learned, prior) ∈ [0,1] — 0 ⇒ riding the prior.
  align/posterior_peak      mean max-prob of the posterior that actually drives durations.

Duration-sanity companions (per-utterance) — the peaks above are BLIND to a degenerate path that
stays confident per-frame while dwelling hundreds of frames on a few "sink" phonemes and skipping
the rest (smears the conditioning → wordless audio). These read the Viterbi step-widths directly:
  align/dur_skip_frac       frac of phonemes given 0 frames — degenerate alignment skips content.
  align/dur_cv              coeff of variation of durations — 0 ⇒ uniform, ≫1 ⇒ few sinks dominate.
  align/dur_max_frac        frac of an utterance's frames in its single longest phoneme (sink capture).

Read-only. Usage (architecture from the checkpoint, like eval_checkpoint):
  python scripts/diagnostics/alignment_heatmap.py checkpoint=.../ckpt.pt num_samples=8 split=dev
"""
import logging
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wandb
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

# Load environment variables from .env file (e.g. WANDB_API_KEY)
load_dotenv()

from naturalspeech2.modules.aligner import compute_beta_binomial_prior, maximum_path_indices
from naturalspeech2.paths import PROJECT_ROOT
from naturalspeech2.utils.utils import setup_file_logger
from naturalspeech2.utils.warning_filters import install_warning_filters

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _common import load_model_and_loader

logger = logging.getLogger("alignment_heatmap")


@torch.no_grad()
def alignment_matrices(model, batch, device) -> dict:
    """Run mel + phoneme encoder + aligner net up to the three [B, F, P] distributions + Viterbi path."""
    audio = batch["audio"].to(device)
    audio_lengths = batch["audio_lengths"].to(device)
    tokens = batch["phoneme_tokens"].to(device)
    tok_mask = batch["phoneme_tokens_mask"].to(device)
    tok_lengths = batch["phoneme_tokens_lengths"].to(device)

    audio_encodings, frame_mask, frame_lengths = model.log_mel_spectrogram_generator(audio, audio_lengths)
    phoneme_encodings = model.phoneme_encoder(tokens, tok_mask)

    learned_scores = model.aligner.aligner_net(audio_encodings, frame_mask, phoneme_encodings, tok_mask)  # [B,F,P]
    # Mirror Aligner.forward: mask invalid phoneme columns pre-softmax with large-negative.
    mask_value = -torch.finfo(learned_scores.dtype).max
    col_mask = rearrange(tok_mask.bool(), 'b p 1 -> b 1 p')
    learned_scores = learned_scores.masked_fill(~col_mask, mask_value)

    prior_lp = compute_beta_binomial_prior(
        frame_lengths, tok_lengths, audio_encodings.shape[1], phoneme_encodings.shape[1],
        w=model.aligner.prior_w,
    )
    learned_lp = learned_scores.log_softmax(dim=-1)
    posterior_lp = (learned_lp + prior_lp).log_softmax(dim=-1)
    path = maximum_path_indices(learned_lp + prior_lp, frame_lengths, tok_lengths)  # [B,F]

    return {
        "learned": learned_lp.exp().float(),    # [B,F,P] per-frame distribution from the net alone
        "prior": prior_lp.exp().float(),        # [B,F,P] beta-binomial prior alone
        "posterior": posterior_lp.exp().float(),# [B,F,P] what drives durations
        "path": path,                           # [B,F] Viterbi phoneme index per frame
        "frame_lengths": frame_lengths,
        "phoneme_lengths": tok_lengths,
    }


def accumulate_scalars(mats, acc) -> None:
    """Pool peak-prob + learned-vs-prior TV over valid frames into acc (sum, count)."""
    B, F = mats["path"].shape
    fl = mats["frame_lengths"]
    frame_valid = (torch.arange(F, device=fl.device)[None, :] < fl[:, None]).float()  # [B,F]
    n = frame_valid.sum()
    learned, prior, posterior = mats["learned"], mats["prior"], mats["posterior"]
    acc["n"] += n
    acc["learned_peak"] += (learned.max(dim=-1).values * frame_valid).sum()
    acc["posterior_peak"] += (posterior.max(dim=-1).values * frame_valid).sum()
    acc["prior_peak"] += (prior.max(dim=-1).values * frame_valid).sum()
    acc["learned_vs_prior_tv"] += (0.5 * (learned - prior).abs().sum(dim=-1) * frame_valid).sum()
    acc["P"] = learned.shape[-1]

    # Duration-degeneracy (per-utterance): bin the Viterbi path into frames-per-phoneme, then read
    # skip / spread / sink-capture. Pooled by utterance (n_utt), not frames — these are per-utt props.
    P = learned.shape[-1]
    pl = mats["phoneme_lengths"]
    phon_valid = (torch.arange(P, device=fl.device)[None, :] < pl[:, None])               # [B,P] bool
    durations = torch.zeros(B, P, device=fl.device).scatter_add_(                          # [B,P] frames/phoneme
        1, mats["path"], frame_valid)                                                      # padded frames add 0 (src=0)
    mean_dur = fl.float() / pl.float()                                                     # [B] = F/P
    var = (durations.square() * phon_valid).sum(dim=1) / pl.float() - mean_dur.square()
    acc["n_utt"] += B
    acc["dur_cv"] += (var.clamp_min(0.0).sqrt() / mean_dur).sum()
    acc["dur_skip_frac"] += (((durations == 0) & phon_valid).sum(dim=1).float() / pl.float()).sum()
    acc["dur_max_frac"] += (durations.amax(dim=1) / fl.float()).sum()


def plot_sample(mats, b, text, step) -> plt.Figure:
    F = int(mats["frame_lengths"][b].item())
    P = int(mats["phoneme_lengths"][b].item())
    path = mats["path"][b, :F].cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, key in zip(axes, ("prior", "learned", "posterior")):
        m = mats[key][b, :F, :P].cpu().numpy().T  # [P, F]
        im = ax.imshow(m, aspect="auto", origin="lower", interpolation="nearest", cmap="magma", vmin=0.0)
        ax.plot(range(F), path, color="cyan", lw=0.8, alpha=0.75)
        ax.set_title(f"{key}  (F={F}, P={P})")
        ax.set_xlabel("frame")
        ax.set_ylabel("phoneme")
        fig.colorbar(im, ax=ax, fraction=0.025)
    title = f"step {step}" + (f" — {text[:80]}" if text else "")
    fig.suptitle(title)
    return fig


def render(scalars, figures, cfg, run_name, step) -> None:
    logger.info(f"=== Alignment scalars (step {step}) ===")
    for k in sorted(scalars):
        logger.info(f"  {k}: {scalars[k]:.4f}")

    if cfg.out_dir is not None:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(figures):
            fig.savefig(out / f"alignment_step{step}_sample{i}.png", dpi=110)
        logger.info(f"Saved {len(figures)} PNG(s) under {out}.")

    if not cfg.wandb.log:
        return
    wandb.init(project=cfg.wandb.project, name=run_name, group=cfg.wandb.group,
               notes=cfg.wandb.notes, tags=list(cfg.wandb.tags),
               config=OmegaConf.to_container(cfg, resolve=True))
    wandb.define_metric("eval/snapshot_step")
    wandb.define_metric("align/*", step_metric="eval/snapshot_step")
    payload = dict(scalars)
    payload.update({f"alignment/sample_{i}": wandb.Image(fig) for i, fig in enumerate(figures)})
    payload["eval/snapshot_step"] = step
    wandb.log(payload)
    wandb.finish()
    logger.info(f"Logged to wandb: {cfg.wandb.project}/{run_name}.")


@hydra.main(config_path="../../config", config_name="diagnostics", version_base=None)
def main(cfg: DictConfig) -> None:
    tag = Path(cfg.checkpoint).stem
    setup_file_logger(logger, PROJECT_ROOT / "logs" / "diagnostics" / f"align_{tag}.log", root=True)
    logging.captureWarnings(True)
    install_warning_filters()
    assert torch.cuda.is_available(), "diagnostics require an NVIDIA GPU"
    torch.manual_seed(cfg.seed)

    model, loader, _, _, step = load_model_and_loader(cfg)
    logger.info(f"Loaded {cfg.checkpoint} (step {step}); drawing {cfg.num_samples} from {cfg.split}.")

    acc = {k: torch.zeros((), device=cfg.device) for k in
           ("n", "n_utt", "learned_peak", "posterior_peak", "prior_peak", "learned_vs_prior_tv",
            "dur_cv", "dur_skip_frac", "dur_max_frac")}
    figures, n_plotted = [], 0
    for batch in loader:
        mats = alignment_matrices(model, batch, cfg.device)
        accumulate_scalars(mats, acc)
        for b in range(mats["path"].shape[0]):
            if n_plotted >= cfg.num_samples:
                break
            text = batch["text"][b] if "text" in batch else None
            figures.append(plot_sample(mats, b, text, step))
            n_plotted += 1
        if n_plotted >= cfg.num_samples:
            break

    n = acc["n"].clamp(min=1)
    n_utt = acc["n_utt"].clamp(min=1)
    scalars = {
        "align/learned_peak": (acc["learned_peak"] / n).item(),
        "align/posterior_peak": (acc["posterior_peak"] / n).item(),
        "align/prior_peak": (acc["prior_peak"] / n).item(),
        "align/learned_vs_prior_tv": (acc["learned_vs_prior_tv"] / n).item(),
        "align/uniform_baseline": 1.0 / acc["P"],
        "align/dur_skip_frac": (acc["dur_skip_frac"] / n_utt).item(),
        "align/dur_cv": (acc["dur_cv"] / n_utt).item(),
        "align/dur_max_frac": (acc["dur_max_frac"] / n_utt).item(),
    }
    render(scalars, figures, cfg, run_name=cfg.wandb.run_name or f"align_{tag}_step{step}", step=step)


if __name__ == "__main__":
    main()
