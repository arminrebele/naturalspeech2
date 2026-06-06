#!/usr/bin/env python3
"""Optuna search over the aligner's unreported scalar hyperparameters.

Tunes {temperature, prior_w, blank_logit} — the few alignment scalars the alignment paper leaves
unspecified — to MINIMIZE the held-out forward_sum + bin loss. A small continuous space, which is
what makes Optuna (TPE) the right tool here rather than a grid or the per-site screen.

Each trial is a fresh `train.py +experiment=aligner_trial` subprocess (fresh CUDA context + compile
cache) with the sampled scalars overridden via `model.aligner.*`. The trial dumps per-eval held-out
aligner losses to a per-trial JSONL (setup.aligner_trial_out); the objective averages forward_sum+bin
over the converged tail. Study state persists to SQLite so a run is resumable/inspectable.

Run on the LOCKED aligner architecture (after the attn_channels 80->512 distance-dim change), at the
zero-dropout default. Alignment quality (monotonic durations, intelligible overfit audio) is a manual
check on the winning scalars — the scalar objective here is necessary, not sufficient.

  python scripts/tuning/run_aligner_optuna.py --n-trials 30
  python scripts/tuning/run_aligner_optuna.py --n-trials 30 --max-iters 8000 --override wandb.log=false
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import optuna

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "scripts" / "train.py"

# Aligner scalar search space (edit ranges here). temperature = cosine-attention scale (effective
# 2x on cos; base.yaml sits at 12); prior_w = Beta-Binomial prior strength (base 1.0); blank_logit =
# CTC blank-vs-label calibration in ForwardSumLoss (base -1.0).
SEARCH_SPACE = {
    "temperature": dict(low=4.0, high=24.0, log=False),
    "prior_w":     dict(low=0.5, high=4.0,  log=True),
    "blank_logit": dict(low=-5.0, high=1.0, log=False),
}

TAIL_FRAC = 0.4   # average the held-out metric over the last TAIL_FRAC of a trial's eval points


def tail_objective(jsonl_path: Path, tail_frac: float = TAIL_FRAC) -> float:
    """Mean held-out (forward_sum + bin) over the converged tail of a trial's eval dump. Trials run
    on the real train split → estimate_loss chains dev+test into one pooled 'val' set per record."""
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"no eval records in {jsonl_path} — did the trial run + eval?")
    records.sort(key=lambda r: r["step"])
    tail = records[int(len(records) * (1.0 - tail_frac)):] or records[-1:]
    vals = [r["val"]["forward_sum_loss"] + r["val"]["bin_loss"] for r in tail]
    return sum(vals) / len(vals)


def make_objective(workdir: Path, max_iters: int, extra: list):
    def objective(trial: optuna.Trial) -> float:
        params = {name: trial.suggest_float(name, **spec) for name, spec in SEARCH_SPACE.items()}
        out_path = workdir / f"trial_{trial.number:04d}.jsonl"
        overrides = [
            "+experiment=aligner_trial",
            f"setup.max_iters={max_iters}",
            f"setup.aligner_trial_out={out_path}",
            f"wandb.run_name=aligner_opt_{trial.number:04d}",
            "wandb.group=aligner-optuna",
            *[f"model.aligner.{k}={v}" for k, v in params.items()],
            *extra,
        ]
        print(f"[trial {trial.number}] {params}")
        subprocess.run([sys.executable, str(TRAIN), *overrides], check=True, cwd=REPO)
        return tail_objective(out_path)

    return objective


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--max-iters", type=int, default=5000, help="train steps per trial")
    ap.add_argument("--study-name", default="aligner_scalars")
    ap.add_argument("--storage", default=None, help="Optuna storage URL; default sqlite under --workdir")
    ap.add_argument("--workdir", type=Path, default=REPO / "research" / "aligner_optuna")
    ap.add_argument("--seed", type=int, default=42, help="TPE sampler seed (reproducible suggestions)")
    ap.add_argument("--override", nargs="*", default=[], help="extra Hydra overrides for every trial")
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{args.workdir / 'study.db'}"
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed), load_if_exists=True,
    )
    study.optimize(make_objective(args.workdir, args.max_iters, args.override), n_trials=args.n_trials)

    print("\n" + "=" * 72)
    print(f"BEST aligner scalars over {len(study.trials)} trial(s) "
          f"(held-out forward_sum+bin = {study.best_value:.4f}) — paste into config/model/base.yaml:aligner:")
    for key, val in study.best_params.items():
        print(f"  {key}: {val:.5g}")
    print("=" * 72)


if __name__ == "__main__":
    main()
