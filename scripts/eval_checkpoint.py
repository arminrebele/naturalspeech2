"""Standalone checkpoint benchmark — runs the in-train eval block on any saved checkpoint.

Same metrics as the live eval (held-out + train-subset loss, WER, SIM-o, sample audio), decoupled
from a training run via naturalspeech2.eval.runner.run_decoupled_eval. Read-only: never writes
ema_best/eval_state. Accepts either checkpoint artifact:

  ckpt.pt            self-describing (model_cfg / vocab / sr / iter_num) → both -live and EMA metrics.
  ema_*.safetensors  EMA shadow only → EMA metrics; metadata from the sibling ckpt.pt, or pass
                     model_config= (+ token_vocab=) for a checkpoint downloaded without one.

Usage (eval POLICY is the normal Hydra cfg — override on the CLI like training):
  python scripts/eval_checkpoint.py checkpoint=models/checkpoints/main_training/ckpt.pt
  python scripts/eval_checkpoint.py checkpoint=.../ema_best.safetensors setup.num_audio_refs=200
  python scripts/eval_checkpoint.py checkpoint=ema_final.safetensors model_config=m.yaml token_vocab=v.json

Prereq: with eval_train=true (default) the train split must be preprocessed (the eval estimates a
train-subset loss, like the in-loop eval); eval_train=false → held-out only (dev+test), needing just
those splits + an existing/`token_vocab=` vocabulary. Intended to run when the training card is free
(defaults to cuda:0; ASR/SV on the 2nd GPU or CPU). Compile is off — one-shot doesn't amortize it.
"""
import json
import logging
import random
import time
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from dotenv import load_dotenv

# Load environment variables from .env file (e.g. WANDB_API_KEY)
load_dotenv()

from naturalspeech2.config.schema import model_cfg_from_omegaconf
from naturalspeech2.data.loaders import create_dataloader
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
from naturalspeech2.eval import ipc, resolve_metric_device, set_metric_device
from naturalspeech2.eval.runner import AudioClip, build_fixed_refs_data, run_decoupled_eval
from naturalspeech2.model import LossWrapper, NaturalSpeech2Model
from naturalspeech2.paths import PROJECT_ROOT
from naturalspeech2.utils.utils import setup_file_logger
from naturalspeech2.utils.warning_filters import install_warning_filters

logger = logging.getLogger("eval_checkpoint")


def load_checkpoint(cfg) -> tuple:
    """Resolve weights + metadata from cfg.checkpoint. Returns
    (live_trainable | None, shadow_trainable, model_cfg_dict, vocab_size | None, sampling_rate, step).
      .pt          → live (full state dict) + EMA shadow + stored model_cfg / vocab / sr / iter_num.
      .safetensors → EMA shadow only (live=None); metadata from model_config= or the sibling ckpt.pt.
    """
    path = Path(cfg.checkpoint)
    assert path.is_file(), f"checkpoint not found: {path}"

    if path.suffix == ".pt":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        # ckpt['model'] is the full state dict; _load_trainable filters it to named_parameters.
        return (ckpt["model"], ckpt["ema"]["shadow"], ckpt["model_cfg"],
                ckpt["token_vocabulary_size"], ckpt["sampling_rate"], ckpt["iter_num"])

    assert path.suffix == ".safetensors", f"unsupported checkpoint type {path.suffix!r} (.pt / .safetensors)"
    shadow = load_file(str(path))                              # full model state dict = EMA weights
    sibling = path.with_name("ckpt.pt")
    if cfg.model_config is not None:
        # Downloaded checkpoint with an external config: vocab derived from the token file later, sr
        # from the dataloader, step past all loss-warmups (final weights).
        model_cfg_dict = OmegaConf.to_container(OmegaConf.load(cfg.model_config), resolve=True)
        return None, shadow, model_cfg_dict, None, cfg.dataloader.sampling_rate, 10 ** 9
    if sibling.is_file():
        meta = torch.load(sibling, map_location="cpu", weights_only=True)
        return (None, shadow, meta["model_cfg"], meta["token_vocabulary_size"],
                meta["sampling_rate"], meta["iter_num"])
    raise FileNotFoundError(
        f"{path.name} has no sibling ckpt.pt and no model_config given. Pass model_config=<yaml> "
        f"(and token_vocab=<json> if not using the local dataset vocab) to benchmark it.")


def render(report, cfg, run_name: str) -> None:
    """Console scalar summary + (cfg.wandb.log) a one-shot wandb run + (cfg.out_dir) a JSON/wav dump.
    report.new_best is intentionally ignored — the benchmark never writes checkpoints."""
    logger.info(f"=== Benchmark results (snapshot_step={report.snapshot_step}) ===")
    for k in sorted(report.scalars):
        logger.info(f"  {k}: {report.scalars[k]:.4f}")

    if cfg.out_dir is not None:
        ipc.write_results(run_dir=Path(cfg.out_dir), report=report)   # serializes AudioClip cells → wav
        logger.info(f"Dumped JSON + wav artifacts under {Path(cfg.out_dir) / 'results'}.")

    if not cfg.wandb.log:
        return
    wandb.init(project=cfg.wandb.project, name=run_name, group=cfg.wandb.group,
               notes=cfg.wandb.notes, tags=list(cfg.wandb.tags),
               config=OmegaConf.to_container(cfg, resolve=True))
    # Eval metrics chart against the true snapshot step (matches the trainer's eval x-axis), so
    # several checkpoints benchmarked into one project line up on a step curve.
    wandb.define_metric("eval/snapshot_step")
    wandb.define_metric("Evaluation: *", step_metric="eval/snapshot_step")
    payload = dict(report.scalars)
    for title, table in report.audio_tables.items():
        wt = wandb.Table(columns=table["columns"])
        for row in table["rows"]:
            wt.add_data(*[wandb.Audio(c.waveform, sample_rate=c.sample_rate) if isinstance(c, AudioClip)
                          else c for c in row])
        payload[title] = wt
    payload["eval/snapshot_step"] = report.snapshot_step
    wandb.log(payload)
    logger.info(f"Logged to wandb: {cfg.wandb.project}/{run_name}.")
    wandb.finish()


@hydra.main(config_path="../config", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    tag = Path(cfg.checkpoint).stem
    setup_file_logger(logger, PROJECT_ROOT / "logs" / "benchmarks" / f"eval_{tag}.log", root=True)
    logging.captureWarnings(True)
    install_warning_filters()
    logger.info(f"Benchmarking checkpoint: {cfg.checkpoint}")

    device = cfg.device
    assert torch.cuda.is_available(), "checkpoint benchmark requires an NVIDIA GPU"
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    set_metric_device(resolve_metric_device(cfg.setup.metric_device))   # ASR/SV off the eval card

    live, shadow, model_cfg_dict, vocab_size, sr, step = load_checkpoint(cfg)

    # Loaders mirror the daemon's batch_size_divisor so run_decoupled_eval's grad_accum compensation
    # matches the bucket batch sizes (the loss is invariant to the micro-batch split either way).
    # The train loader (only it can BUILD the vocab) is built only when needed — train-subset loss or
    # trained-on refs; else dev/test require an existing vocab (token_vocab= or the dataset default).
    nw = cfg.setup.eval_daemon.num_workers
    bsd = cfg.setup.eval_daemon.batch_size_divisor
    need_train = cfg.eval_train or ("fixed_train_refs" in cfg.setup.audio_tables)
    train_loader, train_ds = None, None
    if need_train:
        train_loader, train_ds = create_dataloader(cfg, cfg.dataset.train_split, cfg.token_vocab,
                                                   num_workers=nw, batch_size_divisor=bsd)
    tok_path = cfg.token_vocab or (str(train_ds.token_vocabulary_path) if train_ds else None)
    dev_loader, dev_ds = create_dataloader(cfg, cfg.dataset.dev_split, tok_path, num_workers=nw, batch_size_divisor=bsd)
    tok_path = cfg.token_vocab or str(dev_ds.token_vocabulary_path)
    test_loader, test_ds = create_dataloader(cfg, cfg.dataset.test_split, tok_path, num_workers=nw, batch_size_divisor=bsd)

    n_vocab = len(json.loads(Path(tok_path).read_text()))
    if vocab_size is None:
        vocab_size = n_vocab
        logger.info(f"Derived token_vocabulary_size={vocab_size} from {tok_path} (no checkpoint metadata) "
                    f"— verify it matches the checkpoint's training vocab.")
    else:
        assert vocab_size == n_vocab, f"checkpoint vocab_size={vocab_size} != token vocab len {n_vocab} ({tok_path})"

    # Architecture from the checkpoint's stored cfg (resume rule); eager (one-shot → skip compile).
    model = NaturalSpeech2Model(model_cfg_from_omegaconf(model_cfg_dict),
                                token_vocabulary_size=vocab_size, sampling_rate=sr).to(device).eval()
    model._inference_tokenizer = PhonemeTokenizer(token_vocabulary_path=tok_path, with_backend=True)
    model._inference_sampling_rate = sr

    loss_wrapper = LossWrapper(
        loss_weights=OmegaConf.to_container(cfg.model.loss_weights, resolve=True),
        loss_warmup_steps=OmegaConf.to_container(cfg.model.loss_warmup_steps, resolve=True),
        loss_warmup_hold_steps=OmegaConf.to_container(cfg.model.loss_warmup_hold_steps, resolve=True),
    ).to(device)

    # Fixed, seeded refs — identical draw to the trainer/daemon for a given seed + splits +
    # num_audio_refs, so two checkpoints share one eval set. Build only the sets audio_tables needs.
    do_wer = "wer" in cfg.setup.eval_metrics
    do_sim_o = "sim_o" in cfg.setup.eval_metrics
    n_refs = cfg.setup.num_audio_refs
    prompt_samples_len = int(cfg.model.prompt_seconds * sr)
    val_refs, train_refs = [], []
    if "fixed_val_refs" in cfg.setup.audio_tables:
        val_refs = build_fixed_refs_data([dev_ds, test_ds], n_refs, prompt_samples_len, random.Random(cfg.seed),
                                         sampling_rate=sr, compute_gt_wer=do_wer, compute_sim_emb=do_sim_o)
    if "fixed_train_refs" in cfg.setup.audio_tables:
        train_refs = build_fixed_refs_data([train_ds], n_refs, prompt_samples_len, random.Random(cfg.seed),
                                           sampling_rate=sr, compute_gt_wer=do_wer, compute_sim_emb=do_sim_o)

    t0 = time.perf_counter()
    report = run_decoupled_eval(
        model=model, loss_model=model, loss_wrapper=loss_wrapper,
        train_loader=train_loader, dev_loader=dev_loader, test_loader=test_loader,
        live_trainable=live, shadow_trainable=shadow,
        val_refs=val_refs, train_refs=train_refs, val_datasets=[dev_ds, test_ds],
        cfg=cfg, device=device, prompt_seconds=cfg.model.prompt_seconds,
        sampling_rate=sr, snapshot_step=step, prev_best_val_loss=float("inf"),
        eval_train=cfg.eval_train,
    )
    logger.info(f"Eval done in {time.perf_counter() - t0:.0f}s "
                f"({'EMA-only' if live is None else 'live + EMA'}).")
    render(report, cfg, run_name=cfg.wandb.run_name or f"{tag}_step{step}")


if __name__ == "__main__":
    main()
