from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent    # Audiodeepfakedetection-DDIM-Inversion/audiodeepfakedetection_ddim_inversion
PROJECT_ROOT = PACKAGE_ROOT.parent                # Audiodeepfakedetection-DDIM-Inversion

DATA_DIR = PROJECT_ROOT / "data"
VCTK_DIR = DATA_DIR / "vctk"
VCTK_PROCESSED_DIR = DATA_DIR / "vctk_processed"

MODELS_DIR = PROJECT_ROOT / "models"
ENCODEC_24KHZ_DIR = MODELS_DIR / "encodec_24khz"

CONFIG_DIR = PACKAGE_ROOT / "config"
TOKEN_VOCABULARY_PATH = DATA_DIR / "token_vocabulary.json"