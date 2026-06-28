from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent    # NATURALSPEECH2/naturalspeech2
PROJECT_ROOT = PACKAGE_ROOT.parent                # NATURALSPEECH2

DATA_DIR = PROJECT_ROOT / "data"

MODELS_DIR = PROJECT_ROOT / "models"
ENCODEC_24KHZ_DIR = MODELS_DIR / "encodec_24khz"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

LOGS_DIR = PROJECT_ROOT / "logs"

CONFIG_DIR = PROJECT_ROOT / "config"


# Logs and checkpoints share one <group>/<run_name> layout (fully mirrored) so a run's logs sit beside
# its checkpoints under the same identity. group = cfg.wandb.group, run_name = cfg.run_name. Trainer and
# eval daemon both derive these from the same cfg → they agree on the paths.
def run_checkpoint_dir(group: str, run_name: str) -> Path:
    """Per-run checkpoint subdir CHECKPOINTS_DIR/<group>/<run_name> — holds ckpt.pt + ckpt_bak.pt +
    ema_* + eval_state.json. Isolated per run so one lineage can never clobber another's resume point."""
    return CHECKPOINTS_DIR / group / run_name


def run_log_dir(group: str, run_name: str) -> Path:
    """Per-run log subdir LOGS_DIR/<group>/<run_name> — holds <run_name>.log (trainer + submodules) and
    eval_<run_name>.log (eval daemon). Mirrors run_checkpoint_dir so logs and checkpoints stay paired."""
    return LOGS_DIR / group / run_name