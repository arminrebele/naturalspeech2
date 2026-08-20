"""Filesystem IPC between the trainer and the decoupled eval daemon.

Why files, not multiprocessing.Queue: the daemon is a *fresh* `subprocess.Popen`
(`CUDA_VISIBLE_DEVICES=1` → clean CUDA context, no fork-after-init corruption), so the two
processes share no Python runtime. Atomic-rename files in a RAM-backed dir (`/dev/shm`) carry
everything, survive a daemon restart, and never pickle CUDA tensors or wandb objects.

Channels (under one run dir):
  daemon_init.pt   trainer→daemon, once   handshake: model_cfg, vocab size, sr, resolved cfg
  snapshot.pt      trainer→daemon, /eval  {step, live, shadow} trainable weights (CPU)
  snapshot.marker  trainer→daemon, /eval  latest step (coalescing trigger)
  results/         daemon→trainer, /eval  eval_<step>.json (+ eval_<step>/*.wav) → drained+deleted
  control          trainer→daemon, once   {"shutdown": True, "final_step": N}
eval_state.json lives in the PERSISTENT checkpoints dir (survives /dev/shm wipe on reboot) and
is daemon-owned: {best_val_loss, best_step}.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from naturalspeech2.eval.runner import AudioClip, EvalReport

_SNAPSHOT = "snapshot.pt"
_MARKER = "snapshot.marker"
_INIT = "daemon_init.pt"
_CONTROL = "control"
_RESULTS = "results"


# ----------------------------------------------------------------------------
# Run dir + atomic primitives
# ----------------------------------------------------------------------------

def resolve_run_dir(snapshot_dir_cfg, run_tag: str, fresh: bool = False) -> Path:
    """Resolve the RAM-backed run dir. cfg None → /dev/shm/ns2_eval/<run_tag> (falls back to
    /tmp if /dev/shm is absent). Created here so both sides agree on the path. fresh=True
    deletes a pre-existing run dir first (trainer-side only — the daemon receives the path
    via --run-dir and must never wipe it)."""
    if snapshot_dir_cfg:
        root = Path(snapshot_dir_cfg)
    else:
        shm = Path("/dev/shm")
        root = (shm if shm.is_dir() else Path("/tmp")) / "ns2_eval"
    run_dir = root / run_tag
    if fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / _RESULTS).mkdir(parents=True, exist_ok=True)
    return run_dir


def _atomic_replace(write_fn, final: Path) -> None:
    """write_fn(tmp_path) then atomically rename onto `final` (single-writer per file)."""
    tmp = final.with_name(final.name + ".tmp")
    write_fn(tmp)
    os.replace(tmp, final)


def _atomic_torch_save(obj, final: Path) -> None:
    _atomic_replace(lambda tmp: torch.save(obj, tmp), final)


def _atomic_json_dump(obj, final: Path) -> None:
    _atomic_replace(lambda tmp: tmp.write_text(json.dumps(obj)), final)


# ----------------------------------------------------------------------------
# Handshake
# ----------------------------------------------------------------------------

def write_daemon_init(run_dir: Path, payload: dict) -> None:
    """Static, once at spawn: {model_cfg, token_vocabulary_size, sampling_rate, cfg_container, ...}."""
    _atomic_torch_save(payload, run_dir / _INIT)


def read_daemon_init(run_dir: Path) -> dict:
    return torch.load(run_dir / _INIT, map_location="cpu", weights_only=True)


# ----------------------------------------------------------------------------
# Weight snapshots (trainer write / daemon read), coalesced via the marker
# ----------------------------------------------------------------------------

def write_snapshot(run_dir: Path, step: int, live_trainable: dict, shadow_trainable: dict) -> None:
    """Write the latest trainable-weight snapshot + bump the marker. Inputs are CPU tensors
    (trainer clones on its thread; this runs on the writer thread). Overwrites — the marker keeps
    only the newest step, so a lagging daemon skips intermediates."""
    _atomic_torch_save(
        {"step": int(step), "live": live_trainable, "shadow": shadow_trainable},
        run_dir / _SNAPSHOT,
    )
    _atomic_replace(lambda tmp: tmp.write_text(str(int(step))), run_dir / _MARKER)


def read_marker(run_dir: Path) -> int | None:
    """Latest snapshot step, or None if none written yet."""
    marker = run_dir / _MARKER
    if not marker.is_file():
        return None
    text = marker.read_text().strip()
    return int(text) if text else None


def read_snapshot(run_dir: Path) -> dict | None:
    """Load the latest snapshot onto CPU → {step, live, shadow}. The embedded step is authoritative
    (robust to a replace racing the marker read)."""
    snap = run_dir / _SNAPSHOT
    if not snap.is_file():
        return None
    return torch.load(snap, map_location="cpu", weights_only=True)


# ----------------------------------------------------------------------------
# Eval results (daemon write / trainer drain)
# ----------------------------------------------------------------------------

def _serialize_row(row: list, eval_dir: Path, rel_prefix: str) -> list:
    """Row cells → JSON-safe: AudioClip → wav file + {"__audio__": relpath}; scalars inline."""
    out = []
    for ci, cell in enumerate(row):
        if isinstance(cell, AudioClip):
            rel = f"{rel_prefix}_c{ci}.wav"
            sf.write(str(eval_dir / rel), np.asarray(cell.waveform), cell.sample_rate)
            out.append({"__audio__": rel})
        else:
            out.append(cell)
    return out


def write_results(run_dir: Path, report: EvalReport) -> None:
    """Write one eval's results: wavs under results/eval_<step>/, then the JSON (atomically, last)
    so the trainer's glob only ever sees a complete record."""
    results_dir = run_dir / _RESULTS
    step = report.snapshot_step
    eval_dir = results_dir / f"eval_{step}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    tables = {}
    for ti, (title, table) in enumerate(report.audio_tables.items()):
        rows = [_serialize_row(row, eval_dir, f"t{ti}_r{ri}") for ri, row in enumerate(table["rows"])]
        tables[title] = {"columns": table["columns"], "rows": rows}

    payload = {
        "step": step,
        "scalars": report.scalars,
        "audio_tables": tables,
        "new_best": report.new_best,
        "best_val_loss": report.best_val_loss,
    }
    _atomic_json_dump(payload, results_dir / f"eval_{step}.json")


def drain_results(run_dir: Path) -> list[dict]:
    """Load all pending eval results, oldest step first. Audio cells resolved to absolute wav paths
    ({"__audio__": "/abs/path.wav"}) for the trainer to wrap as wandb.Audio; each record carries
    "_eval_dir" so the caller cleans the wavs via cleanup_eval_dir AFTER reading them (wandb.Audio
    reads the file at construction — deleting earlier FileNotFounds it). The json is dropped here so
    the record isn't re-drained."""
    results_dir = run_dir / _RESULTS
    jsons = sorted(results_dir.glob("eval_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    drained = []
    for jp in jsons:
        payload = json.loads(jp.read_text())
        eval_dir = results_dir / f"eval_{payload['step']}"
        for table in payload["audio_tables"].values():
            for row in table["rows"]:
                for cell in row:
                    if isinstance(cell, dict) and "__audio__" in cell:
                        cell["__audio__"] = str(eval_dir / cell["__audio__"])
        payload["_eval_dir"] = str(eval_dir)
        drained.append(payload)
        jp.unlink()   # drop the json now so the record isn't re-drained; wavs cleaned by the caller
    return drained


def result_exists(run_dir: Path, step: int) -> bool:
    """True once the daemon has written (and the trainer hasn't yet drained) the eval result for
    `step`. Lets shutdown detect final-eval completion by the durable work product, not a blind timer."""
    return (run_dir / _RESULTS / f"eval_{step}.json").is_file()


def cleanup_eval_dir(eval_dir) -> None:
    """Delete a drained eval's wav dir (bounds /dev/shm). Call AFTER consuming the wav paths —
    wandb.Audio reads the file at construction, so deleting earlier FileNotFounds it."""
    eval_dir = Path(eval_dir)
    if eval_dir.is_dir():
        for w in eval_dir.glob("*.wav"):
            w.unlink()
        eval_dir.rmdir()


# ----------------------------------------------------------------------------
# Persistent daemon-owned eval state (best tracking)
# ----------------------------------------------------------------------------

def load_eval_state(path: Path) -> dict:
    """{best_val_loss, best_step}; defaults when absent. Persistent (checkpoints dir) → survives a
    /dev/shm wipe on full-run resume."""
    if Path(path).is_file():
        return json.loads(Path(path).read_text())
    return {"best_val_loss": float("inf"), "best_step": -1}


def save_eval_state(path: Path, state: dict) -> None:
    _atomic_json_dump(state, Path(path))


# ----------------------------------------------------------------------------
# Shutdown control
# ----------------------------------------------------------------------------

def signal_shutdown(run_dir: Path, final_step: int) -> None:
    _atomic_json_dump({"shutdown": True, "final_step": int(final_step)}, run_dir / _CONTROL)


def read_control(run_dir: Path) -> dict | None:
    ctrl = run_dir / _CONTROL
    return json.loads(ctrl.read_text()) if ctrl.is_file() else None
