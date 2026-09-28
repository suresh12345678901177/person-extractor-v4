"""
src.utils.logger
=================
Centralized logging configuration. Library code must never use print();
every module obtains a logger via get_logger(__name__).
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from pathlib import Path

_CONFIGURED = False

# The project's logs/ folder, whatever the working directory. The default
# used to be the relative "logs": modules call get_logger() at import time,
# which configures logging before cli.py/scan_directory.py/server.py reach
# their own configure_logging(BASE_DIR / "logs") - a no-op by then - so a
# run started from another folder logged into a logs/ folder there.
PROJECT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

# pipeline.log is rotated at startup once it passes MAX_LOG_BYTES, keeping
# LOG_BACKUPS older files (pipeline.log.1 newest). Added 2026-09-28: it had
# grown to 1 GB unbounded - one full real-case scan writes ~59 MB.
MAX_LOG_BYTES = 50 * 1024 * 1024
LOG_BACKUPS = 3


def _rotate_at_startup(path: Path, max_bytes: int = MAX_LOG_BYTES, backups: int = LOG_BACKUPS) -> None:
    """Shift path -> path.1 -> ... -> path.<backups> (dropping the oldest)
    if path is at least max_bytes. Done once at startup, not with
    RotatingFileHandler: scan_directory.py's worker processes append to the
    same file, and a mid-run rename of a file other processes hold open
    fails on Windows. If another process (a running server or scan) has the
    log open, the rename fails the same way and rotation waits for the next
    start."""
    try:
        if path.stat().st_size < max_bytes:
            return
    except FileNotFoundError:
        return
    try:
        for i in range(backups - 1, 0, -1):
            older = path.with_name(f"{path.name}.{i}")
            if older.exists():
                os.replace(older, path.with_name(f"{path.name}.{i + 1}"))
        os.replace(path, path.with_name(f"{path.name}.1"))
    except OSError:
        pass


def configure_logging(log_dir: str | Path = PROJECT_LOG_DIR, level: int = logging.INFO) -> None:
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

        # Workers (scan_directory.py, spaCy n_process) start after the main
        # process and must not rotate the file it is already writing to.
        if multiprocessing.current_process().name == "MainProcess":
            _rotate_at_startup(log_dir / "pipeline.log")
        file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(f"person_extractor_v4.{name}")
