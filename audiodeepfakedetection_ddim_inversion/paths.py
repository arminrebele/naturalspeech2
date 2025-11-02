from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
VCTK_DIR = DATA_DIR / "vctk"
VCTK_PROCESSED_DIR = DATA_DIR / "vctk_processed"
MODELS_DIR = PROJECT_ROOT / "models"

