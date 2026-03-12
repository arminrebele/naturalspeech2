from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent    # NATURALSPEECH2/naturalspeech2
PROJECT_ROOT = PACKAGE_ROOT.parent                # NATURALSPEECH2

DATA_DIR = PROJECT_ROOT / "data"

MODELS_DIR = PROJECT_ROOT / "models"
ENCODEC_24KHZ_DIR = MODELS_DIR / "encodec_24khz"

CONFIG_DIR = PACKAGE_ROOT / "config"