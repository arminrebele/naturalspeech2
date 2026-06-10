"""Objective TTS eval metrics — consumer-agnostic, off the autograd path.

WER (intelligibility) via HuBERT-Large CTC, no LM (`facebook/hubert-large-ls960-ft`):
no-LM CTC transcribes acoustics faithfully instead of letting a decoder LM paper over
mispronunciations. Lazy module-level singleton on the idle 2nd GPU (off the training
card's budget), FP32, no_grad. Standard word-level Levenshtein(ref,hyp)/len(ref),
shared ref/hyp normalization. SIM-o (speaker similarity) = WavLM-Large-SV embedding
cosine (s3prl WavLM frontend + vendored ECAPA head; see `compute_sim_o`).
"""
import re
import logging
from functools import lru_cache

import numpy as np
import torch
import torchaudio.functional as taF

from naturalspeech2.utils.utils import pack_by_budget

logger = logging.getLogger(__name__)

_ASR_NAME = "facebook/hubert-large-ls960-ft"
_ASR_SR = 16000
_MIN_SAMPLES = _ASR_SR // 4   # <0.25 s → degenerate (early-training garbage); skip → NaN


_METRIC_DEVICE_OVERRIDE: str | None = None


def resolve_metric_device(value: str) -> str:
    """Resolve a config `metric_device`. 'auto' → idle 2nd GPU if present, else CPU — never the
    training card unless named explicitly (eval is infrequent; CPU beats stealing train VRAM).
    Any other value passes through verbatim (e.g. 'cuda:0', 'cuda:1', 'cpu')."""
    if value != "auto":
        return value
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return "cuda:1"
    return "cpu"


def set_metric_device(device: str | None) -> None:
    """Pin the ASR device; call once at startup before the first WER. Clears the cached ASR
    singleton so a re-pin re-homes it. None → revert to the resolved 'auto' default.
    Daemon sets its own card ('cuda:0' under CUDA_VISIBLE_DEVICES=1 == physical GPU1)."""
    global _METRIC_DEVICE_OVERRIDE
    _METRIC_DEVICE_OVERRIDE = device
    _load_asr.cache_clear()
    _load_sv.cache_clear()


def _metric_device() -> str:
    """Override if pinned (config/daemon), else the resolved 'auto' default (idle 2nd GPU / CPU)."""
    return _METRIC_DEVICE_OVERRIDE if _METRIC_DEVICE_OVERRIDE is not None else resolve_metric_device("auto")


@lru_cache(maxsize=1)
def _load_asr():
    from transformers import HubertForCTC, Wav2Vec2Processor
    device = _metric_device()
    proc = Wav2Vec2Processor.from_pretrained(_ASR_NAME)
    model = HubertForCTC.from_pretrained(_ASR_NAME).to(device).eval()
    return proc, model, device


def _to_numpy(audio) -> np.ndarray:
    return audio.detach().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio)


def _prep_16k(audio, src_sr: int) -> np.ndarray:
    """numpy/torch @ src_sr → 16 kHz float32 numpy (gen audio is numpy; reference may be torch)."""
    wav = torch.from_numpy(np.ascontiguousarray(_to_numpy(audio))).float()
    if src_sr != _ASR_SR:
        wav = taF.resample(wav, src_sr, _ASR_SR)
    return wav.numpy()


def _norm_text(s: str) -> list[str]:
    """Lowercase, strip punct, split → words. Applied to BOTH ref+hyp (HuBERT-CTC emits
    uppercase/no-punct → raw compare inflates WER). Digits kept (MLS/VCTK text pre-verbalized)."""
    return re.sub(r"[^\w\s']", " ", s.lower()).split()


def _word_error_rate(ref_words: list[str], hyp_words: list[str]) -> float:
    """Standard WER = word-level Levenshtein(ref, hyp) / len(ref). NaN if ref empty."""
    if not ref_words:
        return float("nan")
    prev = list(range(len(hyp_words) + 1))
    for i, rw in enumerate(ref_words, start=1):
        cur = [i] + [0] * len(hyp_words)
        for j, hw in enumerate(hyp_words, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(ref_words)


def _degenerate(audio) -> bool:
    arr = _to_numpy(audio)
    return arr.size < _MIN_SAMPLES or not np.isfinite(arr).all()


@torch.no_grad()
def transcribe(audio, src_sr: int = 24000) -> str:
    proc, model, device = _load_asr()
    wav = _prep_16k(audio, src_sr)
    iv = proc(wav, sampling_rate=_ASR_SR, return_tensors="pt").input_values.to(device)
    return proc.batch_decode(model(iv).logits.argmax(-1))[0]


@torch.no_grad()
def compute_wer(gen_audio, target_text: str, src_sr: int = 24000) -> tuple[float, str]:
    """(WER, raw transcription) of ASR(gen_audio) vs target_text. (NaN, "") on degenerate audio
    (the NaN drops out of np.nanmean instead of crashing/skewing). The returned `hyp` is the RAW
    ASR output (pre-normalization) so callers can surface literally what the ASR emitted."""
    if _degenerate(gen_audio):
        return float("nan"), ""
    hyp = transcribe(gen_audio, src_sr)
    return _word_error_rate(_norm_text(target_text), _norm_text(hyp)), hyp


@torch.no_grad()
def compute_wer_batch(gen_audios: list, target_texts: list, src_sr: int = 24000,
                      batch_samples: int = 1_600_000) -> list[tuple[float, str]]:
    """Batched compute_wer: lists of (generated audio, target text) → list of (WER, raw hyp), INPUT
    order. Degenerate clips → (NaN, "") with no forward (drops out of the mean). Valid clips are
    length-sorted + packed (batch·max_16k_samples ≤ batch_samples); each group runs one padded
    HuBERT-CTC forward WITH an attention mask, and each sample's logits are sliced to its own CTC
    length before decode so padded-frame tokens never leak. Matches per-clip compute_wer up to a
    sub-frame conv-boundary effect at clip ends (trailing silence → CTC blank → no hyp change)."""
    assert len(gen_audios) == len(target_texts), "audio/text count mismatch"
    n = len(gen_audios)
    results: list = [None] * n
    valid = [i for i in range(n) if not _degenerate(gen_audios[i])]
    valid_set = set(valid)
    for i in range(n):
        if i not in valid_set:
            results[i] = (float("nan"), "")
    if not valid:
        return results

    proc, model, device = _load_asr()
    wavs = [_prep_16k(gen_audios[i], src_sr) for i in valid]   # 16 kHz float32 numpy, valid-order
    for group in pack_by_budget([len(w) for w in wavs], batch_samples):
        enc = proc([wavs[g] for g in group], sampling_rate=_ASR_SR,
                   return_tensors="pt", padding=True)
        input_values = enc.input_values.to(device)
        attention_mask = enc.attention_mask.to(device)
        logits = model(input_values, attention_mask=attention_mask).logits          # [b, T, V]
        out_lens = model._get_feat_extract_output_lengths(attention_mask.sum(-1)).tolist()
        preds = logits.argmax(-1)                                                    # [b, T]
        for pos, g in enumerate(group):
            hyp = proc.decode(preds[pos, : int(out_lens[pos])])
            i = valid[g]
            results[i] = (_word_error_rate(_norm_text(target_texts[i]), _norm_text(hyp)), hyp)
    return results


# ----------------------------------------------------------------------------
# SIM-o (speaker similarity) — WavLM-Large-SV via the s3prl recipe
# ----------------------------------------------------------------------------

_SV_CKPT_REPO = "Dongchao/UniAudio"          # HF mirror of UniSpeech's wavlm_large_finetune.pth
_SV_CKPT_FILE = "wavlm_large_finetune.pth"
# Backbone key prefixes that MUST load (strict=False). A wrong frontend silently skips the whole
# transformer/CNN stack → this asserts the catastrophic case, NOT `missing == []` (benign aux
# buffers may legitimately be missing).
_SV_BACKBONE_PREFIXES = (
    "feature_extract.model.encoder.layers.",
    "feature_extract.model.feature_extractor.",
)


@lru_cache(maxsize=1)
def _load_sv():
    """Lazy WavLM-Large-SV singleton (ECAPA head + s3prl WavLM frontend) on the metric device.
    Loads the fine-tuned backbone via the s3prl recipe — NOT a transformers frontend swap, which
    silently ships pretrained weights. The scoped load-assertion guards that backbone load."""
    from huggingface_hub import hf_hub_download
    from naturalspeech2.eval.ecapa_tdnn import ECAPA_TDNN_SMALL

    device = _metric_device()
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    ckpt_path = hf_hub_download(_SV_CKPT_REPO, _SV_CKPT_FILE)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    # Log once (first load) to pin expectations against the silent-skip bug. unexpected keys are
    # IGNORED by load_state_dict (they never enter the forward), so with no backbone key missing
    # (asserted below) they're harmless fine-tune leftovers — names logged so the one-off is
    # self-documenting rather than an opaque count.
    logger.info(f"[sim_o] WavLM-SV loaded: {len(missing)} missing, {len(unexpected)} unexpected "
                f"(unexpected={unexpected}; missing[:8]={missing[:8]}).")
    backbone_missing = [k for k in missing if k.startswith(_SV_BACKBONE_PREFIXES)]
    assert not backbone_missing, (
        f"WavLM-SV backbone weights did not load — frontend/checkpoint mismatch. "
        f"e.g. {backbone_missing[:5]}"
    )
    return model.to(device).eval(), device


@torch.no_grad()
def speaker_embedding(audio, src_sr: int = 24000) -> torch.Tensor:
    """WavLM-Large-SV speaker embedding for one clip → [1, D] on the metric device. FP32, no
    autocast. Fixed per reference prompt → cache once (build) instead of re-embedding every eval."""
    model, device = _load_sv()
    wav = torch.from_numpy(_prep_16k(audio, src_sr)).to(device).unsqueeze(0)       # [1, T]
    return model(wav)


@torch.no_grad()
def compute_sim_o(gen_audio, ref_embedding: torch.Tensor, src_sr: int = 24000) -> float:
    """SIM-o = cosine of WavLM-Large-SV speaker embeddings, generated vs. reference prompt.
    ref_embedding = the prompt's cached speaker_embedding() (fixed per ref). NaN on degenerate
    generated audio (drops out of the mean). cosine_similarity normalizes internally (matches F5-TTS)."""
    if _degenerate(gen_audio):
        return float("nan")
    return torch.cosine_similarity(speaker_embedding(gen_audio, src_sr), ref_embedding).item()
