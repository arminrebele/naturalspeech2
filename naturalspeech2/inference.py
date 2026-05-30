"""Library-level inference entry points for NaturalSpeech2.

Wraps the model-level primitives behind text-friendly, file-friendly helpers
used by the CLI, the training-loop eval block, notebooks, and downstream
projects (`fakeinversion-speech` planned).

Three-layer architecture (see .claude/plans/inference.md §2):

    Layer 1: naturalspeech2.modules.diffusion_model.DiffusionModel.sample
    Layer 2: naturalspeech2.model.NaturalSpeech2Model.generate
    Layer 3: this module — load_inference_model, generate_audio,
             compute_inference_data_loss
"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taF
from einops import rearrange
from omegaconf import DictConfig
from safetensors.torch import load_file

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.model import NaturalSpeech2Model
from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH, SAMPLING_RATE
from naturalspeech2.paths import CONFIG_DIR, DATA_DIR

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------------

def _load_default_cfg() -> DictConfig:
    """Compose the project default Hydra config (config/config.yaml + groups).

    Used when `load_inference_model(cfg=None)`. Hydra's GlobalHydra singleton
    is cleared before initializing to support repeated calls from the same
    process (notebook re-runs, CLI scripts that load multiple checkpoints).
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config")


def _resolve_vocab_path(cfg: DictConfig) -> Path:
    """Resolve token_vocabulary_path from cfg, mirroring DatasetWrapper's resolution.

    Vocab is a dataset-level artifact: one file per dataset, shared across splits.
    """
    explicit = cfg.dataset.token_vocabulary_path
    if explicit is not None:
        return Path(explicit)
    return DATA_DIR / cfg.dataset.name / "token_vocabulary.json"


def _load_audio(
    src: torch.Tensor | np.ndarray | Path | str,
    target_sr: int = SAMPLING_RATE,
    device: str | torch.device = "cuda",
    *,
    orig_sr: int | None = None,
    prompt_seconds_warning_threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load + normalize a reference clip → ([1, T] float32 on device, [1] long length).

    Accepts path / numpy / tensor. For paths, the source sample rate is read
    from the file header. For numpy / tensor inputs, the source sample rate
    is whatever the caller passes as `orig_sr`; if `orig_sr is None`, the
    input is assumed to already be at `target_sr` (the common case for
    pre-resampled training-loop slices). Resamples via
    `torchaudio.functional.resample`, mean-collapses multi-channel to mono.

    If `prompt_seconds_warning_threshold` is set, emits a warning when the
    final clip is shorter than that (in seconds). The threshold is the model's
    training prompt length — see plan §4.2.1.
    """
    if isinstance(src, (str, Path)):
        data, sr = sf.read(str(src), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        audio = torch.from_numpy(data)
        src_sr = sr
    elif isinstance(src, np.ndarray):
        audio = torch.from_numpy(src.astype(np.float32, copy=False))
        src_sr = orig_sr if orig_sr is not None else target_sr
    elif isinstance(src, torch.Tensor):
        audio = src.to(torch.float32)
        src_sr = orig_sr if orig_sr is not None else target_sr
    else:
        raise TypeError(
            f"reference_audio must be Path/str/np.ndarray/torch.Tensor, got {type(src).__name__}"
        )

    # Defensive: collapse [C, T] or [T, C] → [T]. The dataset path is always
    # mono by the time we see it, but external callers may hand in stereo wavs.
    # Square shapes ([T, T]) are pathological; pick `dim=0` to match the more
    # common stereo layout, since "is this 1-sample wav" is a non-real case.
    if audio.dim() == 2:
        collapse_dim = 0 if audio.shape[0] <= audio.shape[1] else 1
        audio = audio.mean(dim=collapse_dim)
    elif audio.dim() != 1:
        raise ValueError(f"reference_audio must be 1D or 2D, got {audio.dim()}D")

    if src_sr != target_sr:
        audio = taF.resample(audio, src_sr, target_sr)

    audio = rearrange(audio, "t -> 1 t").to(device, non_blocking=True)
    lengths = torch.tensor([audio.shape[-1]], dtype=torch.long, device=device)

    if prompt_seconds_warning_threshold is not None:
        clip_seconds = audio.shape[-1] / target_sr
        if clip_seconds < prompt_seconds_warning_threshold:
            logger.warning(
                f"Reference clip is {clip_seconds:.2f}s; model was trained on "
                f"{prompt_seconds_warning_threshold:.1f}s prompts. "
                "Quality may degrade for clips significantly shorter than training prompt length."
            )

    return audio, lengths


def _phonemize_to_tokens(
    text: str,
    tokenizer: PhonemeTokenizer,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """text → ([1, P] long tokens on device, [1, P, 1] bool mask, [1] int length).

    `tokenizer(str)` runs espeak then maps phonemes → IDs in one call (see
    `PhonemeTokenizer.__call__`).
    """
    tokens_list = tokenizer(text)
    tokens = torch.tensor(tokens_list, dtype=torch.long, device=device)
    tokens = rearrange(tokens, "p -> 1 p")
    P = tokens.shape[-1]
    mask = torch.ones((1, P, 1), dtype=torch.bool, device=device)
    lengths = torch.tensor([P], dtype=torch.long, device=device)
    return tokens, mask, lengths


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

@torch.no_grad()
def load_inference_model(
    checkpoint_path: Path | str,
    cfg: DictConfig | None = None,
    device: str = "cuda",
) -> NaturalSpeech2Model:
    """Build the model from cfg, load weights, move to device, set eval mode.

    `cfg=None` falls back to composing the project default
    (`config/config.yaml` with all `defaults:` groups resolved via Hydra).

    The best-only safetensors checkpoint stores EMA-averaged weights as its
    primary state_dict, so the loader picks them up transparently.

    Attaches a phonemizing `PhonemeTokenizer` at `model._inference_tokenizer`
    so downstream `generate_audio()` calls don't need it threaded through.
    """
    if cfg is None:
        cfg = _load_default_cfg()

    model_cfg = model_cfg_from_omegaconf(cfg.model)
    sampling_rate = cfg.dataloader.sampling_rate

    vocab_path = _resolve_vocab_path(cfg)
    if not vocab_path.is_file():
        raise FileNotFoundError(
            f"Token vocabulary not found at {vocab_path}. The vocabulary is "
            "built during the first preprocessing run; check that the dataset "
            "is preprocessed and cfg.dataset.* points to the right place."
        )

    tokenizer = PhonemeTokenizer(token_vocabulary_path=str(vocab_path), with_backend=True)
    token_vocabulary_size = tokenizer.token_vocabulary_size

    model = NaturalSpeech2Model(
        model_cfg,
        token_vocabulary_size=token_vocabulary_size,
        sampling_rate=sampling_rate,
    )
    # safetensors.torch.load_model runs _remove_duplicate_names over the model's
    # state_dict, which errors on buffers whose storage isn't covered by a single
    # name (torchaudio's spectrogram `window`, encodec's LSTM `_flat_weights`).
    # load_file + plain load_state_dict copies by name and is indifferent to
    # storage sharing — the load-side counterpart of the _save_safetensors
    # clone-on-save workaround in scripts/train.py.
    model.load_state_dict(load_file(str(checkpoint_path)))
    model.to(device)
    model.eval()

    # Attach inference helpers — non-Parameter / non-Buffer attributes so they
    # don't pollute state_dict, don't affect compilation, don't affect forward.
    model._inference_tokenizer = tokenizer
    model._inference_sampling_rate = sampling_rate

    return model


@torch.no_grad()
def generate_audio(
    model: NaturalSpeech2Model,
    reference_audio: torch.Tensor | np.ndarray | Path | str,
    target_text: str,
    sampling_steps: int = 150,
) -> tuple[np.ndarray, int]:
    """Reference audio + text → synthesized audio.

    Layer-3 wrapper around `model.generate()`. Phonemization + tokenization
    happen inside via the cached tokenizer attached to the model. Caller hands
    in a text string; never sees phonemes.

    `reference_audio` accepts a file path (any soundfile-readable format),
    numpy array, or torch tensor. Auto-resampled to the model's training
    sample rate. Auto-batched to [1, T] float32 on the model's device.

    Returns (audio_np, valid_length) — caller trims `audio_np[:valid_length]`
    before writing to disk for B=1 invocations.

    Requires `model._inference_tokenizer` (and ideally `_inference_sampling_rate`)
    to be set; `load_inference_model` does this. The training loop sets it once
    at setup so the eval block can call generate_audio uniformly.
    """
    tokenizer = getattr(model, "_inference_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            "model._inference_tokenizer is not set. Either obtain the model via "
            "load_inference_model(), or attach a PhonemeTokenizer (with_backend=True) "
            "at model._inference_tokenizer before calling generate_audio()."
        )

    device = next(model.parameters()).device
    sampling_rate = getattr(model, "_inference_sampling_rate", SAMPLING_RATE)
    # Recover trained prompt_seconds from model.prompt_frames so the OOD
    # warning fires when the user-provided clip is shorter than what the model
    # has seen during training (plan §4.2.1).
    prompt_seconds = (model.prompt_frames * ENCODER_HOP_LENGTH) / sampling_rate

    ref_audio, ref_lengths = _load_audio(
        reference_audio,
        target_sr=sampling_rate,
        device=device,
        prompt_seconds_warning_threshold=prompt_seconds,
    )
    tokens, tokens_mask, tokens_lengths = _phonemize_to_tokens(
        target_text, tokenizer, device=device
    )

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        generated_audio, audio_lengths = model.generate(
            reference_audio=ref_audio,
            reference_audio_lengths=ref_lengths,
            phoneme_tokens=tokens,
            phoneme_tokens_mask=tokens_mask,
            phoneme_tokens_lengths=tokens_lengths,
            sampling_steps=sampling_steps,
        )

    audio_np = generated_audio[0].detach().cpu().to(torch.float32).numpy()
    valid_length = int(audio_lengths[0].item())
    return audio_np, valid_length


@torch.no_grad()
def compute_inference_data_loss(
    model: NaturalSpeech2Model,
    batch: dict,
    sampling_steps_sweep: tuple[int, ...] = (150, 300, 600, 1000),
) -> dict[int, float]:
    """Latent-space diagnostic. For each n_steps in the sweep:

      1. Forward pass with return_diffusion_inputs=True → recover GT
         condition_target + prompt_encodings + GT z₀ (in normalized space).
      2. Call model.diffusion_model.sample(condition=..., sampling_steps=n_steps).
      3. Masked MSE between sampled z₀ and GT z₀.

    Returns {n_steps: loss_float}. Bypasses Encodec decoding entirely —
    measures solver fidelity directly in latent space, isolating "is the
    diffusion sampler producing the right latents" from decoder behavior.

    Lifted from scripts/train.py:839-867. Accepts any batch dict the dataloader
    produces — not restricted to the overfit batch. Non-tensor batch fields
    (e.g. `text: list[str]`) are filtered out before the model forward.
    """
    device = next(model.parameters()).device

    # Move only tensor batch fields to device; pass-through non-tensor metadata
    # (text) without trying to .to() it.
    b = {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        _, diff_inputs = model(
            audio=b["audio"],
            audio_lengths=b["audio_lengths"],
            phoneme_tokens=b["phoneme_tokens"],
            phoneme_tokens_mask=b["phoneme_tokens_mask"],
            phoneme_tokens_lengths=b["phoneme_tokens_lengths"],
            pitch=b["pitch"],
            return_diffusion_inputs=True,
        )

    z0_true = diff_inputs["target_latents"].float()
    z_mask = diff_inputs["target_latents_mask"].to(z0_true.dtype)
    valid_scalars = z_mask.sum().clamp(min=1.0) * model.diffusion_model.latent_dim

    losses: dict[int, float] = {}
    for n_steps in sampling_steps_sweep:
        z0_sampled = model.diffusion_model.sample(
            condition=diff_inputs["condition_target"],
            condition_mask=diff_inputs["target_latents_mask"],
            prompt_encodings=diff_inputs["prompt_encodings"],
            prompt_encodings_mask=diff_inputs["prompt_encodings_mask"],
            sampling_steps=n_steps,
        )
        diff_sq = (z0_sampled.float() - z0_true) ** 2
        loss = (diff_sq * z_mask).sum() / valid_scalars
        losses[n_steps] = loss.item()

    return losses
