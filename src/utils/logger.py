"""
src.utils.logger
=================
Centralized logging configuration. Library code must never use print();
every module obtains a logger via get_logger(__name__).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def configure_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> None:
    """Configure root logging once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-34s | %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger("person_extractor_v4")
    root.setLevel(level)
    root.propagate = False

    if not root.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(f"person_extractor_v4.{name}")
