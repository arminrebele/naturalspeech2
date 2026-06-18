"""Library-level inference entry points for NaturalSpeech2.

Text/file-friendly wrappers over the model primitives, used by the CLI, training-loop
eval block, notebooks, and downstream projects. Three layers:
    1. DiffusionModel.sample          (sampler)
    2. NaturalSpeech2Model.generate   (full pipeline)
    3. this module — load_inference_model, generate_audio, compute_inference_data_loss
"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taF
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.model import NaturalSpeech2Model
from naturalspeech2.modules.encodec import ENCODER_HOP_LENGTH, SAMPLING_RATE
from naturalspeech2.paths import CONFIG_DIR, DATA_DIR
from naturalspeech2.utils.utils import create_mask_from_lengths

logger = logging.getLogger(__name__)

# Per-phoneme duration ceiling (seconds) for the inference safety guard. Anchored on articulatory
# reality — no single phoneme in natural speech approaches this (sustained vowels top out ~1–2 s),
# while a mis-firing (under-trained / OOD) duration predictor blows up to tens of seconds.
# Legitimate (≤~2 s) and pathological (≫) sit an order of magnitude apart, so the exact value is
# insensitive: this generous default rejects nothing real yet still catches runaway-frame OOMs.
MAX_SECONDS_PER_PHONEME = 4.0


# ----------------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------------

def _load_default_cfg() -> DictConfig:
    """Compose the default Hydra config (config/config.yaml + groups); used when cfg=None.
    Clears Hydra's GlobalHydra singleton first → supports repeated calls in one process."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config")


def _resolve_vocab_path(cfg: DictConfig) -> Path:
    """Resolve token_vocabulary_path from cfg (mirrors DatasetWrapper). Vocab is
    dataset-level: one file per dataset, shared across splits."""
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
    slice_seconds: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load + normalize a reference clip → ([1, T] float32 on device, [1] long length).

    Accepts path / numpy / tensor. Path: src sr from file header. numpy/tensor: src sr =
    orig_sr, or target_sr if None (pre-resampled training slices). Resamples, mono-collapses.
    prompt_seconds_warning_threshold set → warn if the clip is shorter (model's training prompt length).
    slice_seconds set → keep only the leading slice_seconds (no-op if already shorter).
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

    # Collapse [C, T] or [T, C] → [T] for external stereo wavs (dataset path is already mono).
    # Square [T, T] is pathological; dim=0 matches the common stereo layout.
    if audio.dim() == 2:
        collapse_dim = 0 if audio.shape[0] <= audio.shape[1] else 1
        audio = audio.mean(dim=collapse_dim)
    elif audio.dim() != 1:
        raise ValueError(f"reference_audio must be 1D or 2D, got {audio.dim()}D")

    if src_sr != target_sr:
        audio = taF.resample(audio, src_sr, target_sr)

    if slice_seconds is not None:
        audio = audio[: int(slice_seconds * target_sr)]   # leading window; no-op if shorter

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """text → ([1, P] long tokens on device, [1, P, 1] bool mask).
    tokenizer(str) runs espeak then maps phonemes → IDs in one call."""
    tokens_list = tokenizer(text)
    tokens = torch.tensor(tokens_list, dtype=torch.long, device=device)
    tokens = rearrange(tokens, "p -> 1 p")
    P = tokens.shape[-1]
    mask = torch.ones((1, P, 1), dtype=torch.bool, device=device)
    return tokens, mask


def _check_inference_limits(n_phonemes: int, ref_frames: int, rope_max_seq_len: int,
                            *, prompt_sliced: bool) -> None:
    """Boundary validation for one (text, reference) pair → loud, clean errors instead of cryptic
    downstream failures: empty text → length-0 conv crash; sub-frame audio → empty cross-attn → NaN;
    over-ceiling sequence → RoPE-cache shape mismatch."""
    fps = SAMPLING_RATE // ENCODER_HOP_LENGTH
    if n_phonemes < 1:
        raise ValueError("Empty text: phonemization produced 0 tokens.")
    if n_phonemes > rope_max_seq_len:
        raise ValueError(f"Text too long: {n_phonemes} phonemes exceed the model ceiling "
                         f"({rope_max_seq_len}); split into shorter segments.")
    if ref_frames < 1:
        raise ValueError("Reference audio too short to yield a single Encodec frame (~13 ms).")
    if ref_frames > rope_max_seq_len:
        fix = "reduce prompt_seconds" if prompt_sliced else "pass prompt_seconds= to slice it"
        raise ValueError(f"Reference too long: {ref_frames} frames exceed the model ceiling "
                         f"({rope_max_seq_len} ≈ {rope_max_seq_len // fps}s) — {fix}.")


def _resolve_pretrained(checkpoint: Path | str, cfg: DictConfig | None) -> tuple[Path, DictConfig | None]:
    """Local path → returned unchanged. Otherwise treat as a Hugging Face repo id and download the
    inference bundle from the Hub. Expected repo layout: ema_best.safetensors, config.yaml (a resolved
    OmegaConf dump), token_vocabulary.json."""
    p = Path(checkpoint)
    if p.exists():
        return p, cfg
    from huggingface_hub import hf_hub_download
    repo = str(checkpoint)
    weights = Path(hf_hub_download(repo, "ema_best.safetensors"))
    if cfg is None:
        cfg = OmegaConf.load(hf_hub_download(repo, "config.yaml"))
        cfg.dataset.token_vocabulary_path = hf_hub_download(repo, "token_vocabulary.json")
    return weights, cfg


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

@torch.no_grad()
def load_inference_model(
    checkpoint_path: Path | str,
    cfg: DictConfig | None = None,
    device: str = "cuda",
) -> NaturalSpeech2Model:
    """Build model from cfg, load weights, move to device, eval mode.

    checkpoint_path = local .safetensors path, or a Hugging Face repo id (downloaded on demand,
    bringing its own config + token vocab). cfg=None + local path → compose the project default
    (config/config.yaml). The best-only safetensors stores EMA weights as its state_dict, so they
    load transparently. Attaches a PhonemeTokenizer at model._inference_tokenizer so generate_audio()
    needn't thread it through.
    """
    checkpoint_path, cfg = _resolve_pretrained(checkpoint_path, cfg)
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
    # safetensors.torch.load_model runs _remove_duplicate_names, which errors on buffers whose
    # storage isn't covered by one name (torchaudio `window`, encodec LSTM `_flat_weights`).
    # load_file + load_state_dict copies by name, storage-sharing-indifferent — load-side
    # counterpart of the _save_safetensors clone-on-save workaround.
    model.load_state_dict(load_file(str(checkpoint_path)))
    model.to(device)
    model.eval()

    # Attach inference helpers — plain attributes (not Parameter/Buffer) → don't touch
    # state_dict, compilation, or forward.
    model._inference_tokenizer = tokenizer
    model._inference_sampling_rate = sampling_rate

    return model


@torch.no_grad()
def generate_audio(
    model: NaturalSpeech2Model,
    reference_audio: torch.Tensor | np.ndarray | Path | str,
    target_text: str,
    prompt_seconds: float | None = None,
    sampling_steps: int = 150,
    max_seconds_per_phoneme: float | None = MAX_SECONDS_PER_PHONEME,
    on_overflow: str = "raise",
) -> tuple[np.ndarray, int]:
    """Reference audio + text → synthesized audio (Layer-3 wrapper over model.generate()).

    Phonemization + tokenization happen inside via the model's cached tokenizer; caller passes
    text, never phonemes. reference_audio = path / numpy / tensor, auto-resampled to the training
    sample rate, auto-batched to [1, T] float32 on the model's device.
    prompt_seconds set → slice the reference to that leading window (longer prompts can improve
    quality; None = full clip). max_seconds_per_phoneme + on_overflow guard duration-predictor
    blow-ups (see model.generate). Returns (audio_np, valid_length); trim audio_np[:valid_length].
    Requires model._inference_tokenizer (set by load_inference_model / the training loop).
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
    # Recover trained prompt length from model.prompt_frames → OOD warning when the
    # reference clip is shorter than training prompts.
    trained_prompt_seconds = (model.prompt_frames * ENCODER_HOP_LENGTH) / sampling_rate

    ref_audio, ref_lengths = _load_audio(
        reference_audio,
        target_sr=sampling_rate,
        device=device,
        prompt_seconds_warning_threshold=trained_prompt_seconds,
        slice_seconds=prompt_seconds,
    )
    tokens, tokens_mask = _phonemize_to_tokens(
        target_text, tokenizer, device=device
    )

    ref_frames = (ref_audio.shape[-1] + ENCODER_HOP_LENGTH - 1) // ENCODER_HOP_LENGTH  # shape, not .item() → no sync
    _check_inference_limits(tokens.shape[-1], ref_frames, model.rope_max_seq_len,
                            prompt_sliced=prompt_seconds is not None)
    cap = (round(max_seconds_per_phoneme * sampling_rate / ENCODER_HOP_LENGTH)
           if max_seconds_per_phoneme is not None else None)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        generated_audio, audio_lengths = model.generate(
            reference_audio=ref_audio,
            reference_audio_lengths=ref_lengths,
            phoneme_tokens=tokens,
            phoneme_tokens_mask=tokens_mask,
            sampling_steps=sampling_steps,
            max_frames_per_phoneme=cap,
            on_overflow=on_overflow,
        )

    audio_np = generated_audio[0].detach().cpu().to(torch.float32).numpy()
    valid_length = int(audio_lengths[0].item())
    return audio_np, valid_length


@torch.no_grad()
def generate_audio_batch(
    model: NaturalSpeech2Model,
    reference_audios: list,
    target_texts: list,
    prompt_seconds: float | None = None,
    sampling_steps: int = 150,
    max_seconds_per_phoneme: float | None = MAX_SECONDS_PER_PHONEME,
    on_overflow: str = "raise",
) -> list:
    """Batched generate_audio: (ref clips, texts) → list of trimmed audio np, INPUT order.
    Zero-pads prompts/tokens to batch max (+ masks), one model.generate, trims each by its
    predicted length. model.generate is batch-native; this generalizes the [1,T] wrapper.
    prompt_seconds / max_seconds_per_phoneme / on_overflow: see generate_audio."""
    assert len(reference_audios) == len(target_texts), "reference/text count mismatch"
    tokenizer = getattr(model, "_inference_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("model._inference_tokenizer is not set (see generate_audio).")
    device = next(model.parameters()).device
    sampling_rate = getattr(model, "_inference_sampling_rate", SAMPLING_RATE)
    trained_prompt_seconds = (model.prompt_frames * ENCODER_HOP_LENGTH) / sampling_rate

    refs, ref_lens, tok_lists = [], [], []
    for ref, text in zip(reference_audios, target_texts):
        audio, _ = _load_audio(ref, target_sr=sampling_rate, device=device,
                               prompt_seconds_warning_threshold=trained_prompt_seconds,
                               slice_seconds=prompt_seconds)
        refs.append(audio[0])
        ref_lens.append(audio.shape[-1])   # == length.item(), but from shape → no CPU↔GPU sync
        tok_lists.append(tokenizer(text))

    for toks, ref_len in zip(tok_lists, ref_lens):
        ref_frames = (ref_len + ENCODER_HOP_LENGTH - 1) // ENCODER_HOP_LENGTH
        _check_inference_limits(len(toks), ref_frames, model.rope_max_seq_len,
                                prompt_sliced=prompt_seconds is not None)
    cap = (round(max_seconds_per_phoneme * sampling_rate / ENCODER_HOP_LENGTH)
           if max_seconds_per_phoneme is not None else None)

    B = len(refs)
    T_ref = max(ref_lens)
    P = max(len(t) for t in tok_lists)

    reference_audio = torch.zeros((B, T_ref), dtype=refs[0].dtype, device=device)
    for i, a in enumerate(refs):
        reference_audio[i, : a.shape[0]] = a
    reference_audio_lengths = torch.tensor(ref_lens, dtype=torch.long, device=device)

    phoneme_tokens = torch.zeros((B, P), dtype=torch.long, device=device)   # pad id 0 (<pad>)
    for i, toks in enumerate(tok_lists):
        phoneme_tokens[i, : len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
    phoneme_lengths = torch.tensor([len(t) for t in tok_lists], dtype=torch.long, device=device)
    phoneme_tokens_mask = create_mask_from_lengths(phoneme_lengths, P)      # [B, P, 1]

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        generated_audio, audio_lengths = model.generate(
            reference_audio=reference_audio,
            reference_audio_lengths=reference_audio_lengths,
            phoneme_tokens=phoneme_tokens,
            phoneme_tokens_mask=phoneme_tokens_mask,
            sampling_steps=sampling_steps,
            max_frames_per_phoneme=cap,
            on_overflow=on_overflow,
        )

    audio_np = generated_audio.detach().cpu().to(torch.float32).numpy()     # [B, T]
    lengths = audio_lengths.detach().cpu().tolist()
    return [audio_np[i, : int(lengths[i])] for i in range(B)]


@torch.no_grad()
def compute_inference_data_loss(
    model: NaturalSpeech2Model,
    batch: dict,
    sampling_steps_sweep: tuple[int, ...] = (150, 300, 600, 1000),
) -> dict[int, float]:
    """Latent-space diagnostic. Per n_steps in the sweep:
      1. forward(return_diffusion_inputs=True) → GT condition_target, prompt_encodings, GT z₀.
      2. diffusion_model.sample(condition=..., sampling_steps=n_steps).
      3. masked MSE(sampled z₀, GT z₀).
    Returns {n_steps: loss}. Bypasses Encodec decode → measures solver fidelity in latent
    space (isolates sampler from decoder). Accepts any batch dict; non-tensor fields (text) filtered.
    """
    device = next(model.parameters()).device

    # Move only tensor fields to device; pass non-tensor metadata (text) through.
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
