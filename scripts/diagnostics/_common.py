"""Shared loading for the read-only diagnostics (alignment_heatmap, condition_ablation).

Builds the model from a checkpoint's stored cfg (resume rule) + a split dataloader. Read-only —
never writes checkpoints/eval state. Architecture comes from the checkpoint, NOT cfg.model.
"""
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.data.loaders import create_dataloader
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.model import NaturalSpeech2Model


def _resolve_checkpoint(cfg) -> tuple:
    """(full_state_dict, model_cfg_dict, vocab_size | None, sampling_rate, step).
      .pt          → live weights + stored model_cfg / vocab / sr / iter_num.
      .safetensors → EMA weights; metadata from model_config= or the sibling ckpt.pt.
    """
    path = Path(cfg.checkpoint)
    assert path.is_file(), f"checkpoint not found: {path}"

    if path.suffix == ".pt":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        return (ckpt["model"], ckpt["model_cfg"], ckpt["token_vocabulary_size"],
                ckpt["sampling_rate"], ckpt["iter_num"])

    assert path.suffix == ".safetensors", f"unsupported checkpoint type {path.suffix!r} (.pt / .safetensors)"
    shadow = load_file(str(path))                                  # full EMA state dict
    sibling = path.with_name("ckpt.pt")
    if cfg.model_config is not None:
        model_cfg_dict = OmegaConf.to_container(OmegaConf.load(cfg.model_config), resolve=True)
        return shadow, model_cfg_dict, None, cfg.dataloader.sampling_rate, 10 ** 9
    assert sibling.is_file(), (
        f"{path.name} has no sibling ckpt.pt and no model_config given — pass model_config=<yaml> "
        f"(and token_vocab=<json> if not using the local dataset vocab).")
    meta = torch.load(sibling, map_location="cpu", weights_only=True)
    return (shadow, meta["model_cfg"], meta["token_vocabulary_size"],
            meta["sampling_rate"], meta["iter_num"])


def load_model_and_loader(cfg) -> tuple:
    """Returns (model.eval() on cfg.device, dataloader, dataset, sampling_rate, step)."""
    state, model_cfg_dict, vocab_size, sr, step = _resolve_checkpoint(cfg)

    split = {"dev": cfg.dataset.dev_split, "test": cfg.dataset.test_split,
             "train": cfg.dataset.train_split}[cfg.split]
    nw = cfg.setup.eval_daemon.num_workers
    bsd = cfg.setup.eval_daemon.batch_size_divisor
    loader, ds = create_dataloader(cfg, split, cfg.token_vocab, num_workers=nw, batch_size_divisor=bsd)

    tok_path = cfg.token_vocab or str(ds.token_vocabulary_path)
    n_vocab = len(json.loads(Path(tok_path).read_text()))
    if vocab_size is None:
        vocab_size = n_vocab
    else:
        assert vocab_size == n_vocab, f"checkpoint vocab_size={vocab_size} != token vocab len {n_vocab} ({tok_path})"

    model = NaturalSpeech2Model(model_cfg_from_omegaconf(model_cfg_dict),
                                token_vocabulary_size=vocab_size, sampling_rate=sr).to(cfg.device).eval()
    model.load_state_dict(state, strict=True)
    model._inference_tokenizer = PhonemeTokenizer(token_vocabulary_path=tok_path, with_backend=True)
    model._inference_sampling_rate = sr
    return model, loader, ds, sr, step
