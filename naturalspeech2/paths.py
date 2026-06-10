from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent    # NATURALSPEECH2/naturalspeech2
PROJECT_ROOT = PACKAGE_ROOT.parent                # NATURALSPEECH2

DATA_DIR = PROJECT_ROOT / "data"

MODELS_DIR = PROJECT_ROOT / "models"
ENCODEC_24KHZ_DIR = MODELS_DIR / "encodec_24khz"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

CONFIG_DIR = PROJECT_ROOT / "config"


def run_checkpoint_dir(log_name: str) -> Path:
    """Per-run checkpoint subdir CHECKPOINTS_DIR/<log_name> — isolates each run lineage's
    ckpt.pt + ema_* artifacts so a diagnostic run can't clobber the main run's resume point.
    Trainer and eval daemon both derive it from cfg.setup.log_name → they agree on the path."""
    return CHECKPOINTS_DIR / log_name