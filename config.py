
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
        # spaCy pipeline SpacyDetector loads (python -m spacy download <name>).
        # Changing it changes what gets detected - measure it first (see
        # README, 2026-09-25 NER-model probe).
        "spacy_model": "en_core_web_sm",
        # Worker processes spaCy may use WITHIN one document (nlp.pipe
        # n_process). Only kicks in when the document spans 2+ of
        # SpacyDetector's 400K-char chunks, i.e. files over ~400KB.
        # Measured 2026-09-24 on a 5.4MB file: 4 processes = 2.6x faster,
        # byte-identical entities. Default 1: scan_directory.py already
        # runs one worker PER FILE across all cores, so also splitting
        # inside each file there would oversubscribe the CPU. Raise it
        # (cli.py --spacy-processes) for single very large files.
        "spacy_processes": 1,
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
    "language_filter": {
        # Skips detection entirely on documents that aren't natural-
        # language English prose (non-English text, legal/EULA
        # boilerplate, phone-settings screens, ...) - see
        # src/preprocessing/language_filter.py for why this exists and
        # how the threshold was calibrated. Off switch kept here in case
        # a future extraction type needs to run on non-English text.
        "enabled": True,
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
