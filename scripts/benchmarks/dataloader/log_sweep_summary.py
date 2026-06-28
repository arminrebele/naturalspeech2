"""Dedicated W&B 'sweep' run for the dataloader benchmark: combined sweep log + cross-config comparison
table. Invoked by run_benchmark_dataloader.sh AFTER the num_workers loop (also on an early sweep-abort),
so the one artifact that aggregates every config — including the shell orchestration lines and which
config stopped the sweep — lives in W&B too, not just the local log.

Light by design: no torch / dataset / model imports, so it is safe to run right after a RAM/thrash abort.
Paths come from env (LOG_FILE, SWEEP_RESULTS_FILE, both absolute); W&B project/group from the same Hydra
config the per-config runs used (so the sweep run lands in the same group)."""
import os
import json
import shutil
import logging

import hydra
from omegaconf import DictConfig
import wandb
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Column order for the comparison table; keys match the result rows written by the benchmark (success)
# and the shell (abort). Missing keys → None → empty cell.
TABLE_COLUMNS = ["num_workers", "setting", "valid_steps", "starvation_rate", "iter_time_p95",
                 "cpu_gpu_ratio_p95", "status"]


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def log_sweep_summary(cfg: DictConfig):
    if not cfg.wandb.log:
        logger.info("wandb.log=false → skipping sweep-summary upload.")
        return

    log_file = os.environ.get("LOG_FILE")
    results_file = os.environ.get("SWEEP_RESULTS_FILE")
    setting = "otf" if cfg.dataloader.resample_on_the_fly else "pre"

    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,          # shell sets this to sweep_<tag>
        group=cfg.wandb.group,
        job_type="sweep-summary",
        config={"setting": setting},
    )

    # Cross-config comparison table from the per-config result rows.
    rows = []
    if results_file and os.path.isfile(results_file):
        with open(results_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    if rows:
        rows.sort(key=lambda r: r.get("num_workers") or 0)
        table = wandb.Table(columns=TABLE_COLUMNS)
        for r in rows:
            table.add_data(*[r.get(c) for c in TABLE_COLUMNS])
        wandb.log({"sweep/comparison": table})
        logger.info(f"Logged comparison table ({len(rows)} configs).")
    else:
        logger.warning(f"No per-config result rows at {results_file} — comparison table skipped.")

    # Combined sweep log → Files tab. Copied into run.dir near the end so the snapshot captures the whole
    # sweep (all per-config output + the shell orchestration/abort lines); synced on finish.
    if log_file and os.path.isfile(log_file):
        shutil.copy(log_file, os.path.join(run.dir, "dataloader_benchmark_sweep.log"))
        logger.info(f"Attached combined sweep log ({log_file}).")
    else:
        logger.warning(f"Combined log not found at {log_file} — nothing to attach.")

    wandb.finish()


if __name__ == "__main__":
    log_sweep_summary()
