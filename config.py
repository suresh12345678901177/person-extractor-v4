"""
config.py
==========
Central configuration for PERSON_EXTRACTOR_V4. Every tunable setting
lives here, not scattered through the codebase - per the blueprint's
Design Principles (no hidden magic numbers in library code... except
where a validator's own threshold is genuinely local to that one file,
e.g. LengthValidator's MIN_CHARS, which is documented inline there).
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"
DATASETS_DIR = BASE_DIR / "datasets"
BENCHMARK_DIR = DATASETS_DIR / "benchmark"
TRAINING_DIR = DATASETS_DIR / "training"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

DEFAULT_CONFIG = {
    "detection": {
        "use_regex": True,
        "use_dictionary": True,
        "use_spacy": True,
    },
    "classification": {
        "use_ml_classifier": True,
        "model_path": "models/lightgbm/person_classifier.txt",
    },
    "feedback": {
        "enabled": True,  # log REVIEW-bucket candidates for active learning
    },
    "decision": {
        "strict_corroboration": True,  # False = comparison mode, see cli.py --loose-gate
    },
    "export": {
        "default_format": "csv",  # csv | json | txt (report)
    },
    "extraction": {
        # Which src.extraction.registry.EXTRACTOR_REGISTRY entries Pipeline
        # runs. Only "person" exists today; this is the extension point for
        # future extraction types (phone numbers, ID numbers, ...) - see
        # src/extraction/base_extractor.py. Pipeline currently only ever
        # runs the first entry (single-extractor-per-run, for now).
        "enabled_types": ["person"],
    },
}
