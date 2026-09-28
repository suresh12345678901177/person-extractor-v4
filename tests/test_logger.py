"""Tests for pipeline.log rotation at startup (src.utils.logger)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.utils.logger as logger_module
from src.utils.logger import _rotate_at_startup


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_log_under_the_limit_is_left_alone(tmp_path):
    log = tmp_path / "pipeline.log"
    _write(log, "small")
    _rotate_at_startup(log, max_bytes=100, backups=3)
    assert log.read_text(encoding="utf-8") == "small"
    assert not (tmp_path / "pipeline.log.1").exists()


def test_log_over_the_limit_becomes_the_first_backup(tmp_path):
    log = tmp_path / "pipeline.log"
    _write(log, "x" * 20)
    _rotate_at_startup(log, max_bytes=10, backups=3)
    assert not log.exists()
    assert (tmp_path / "pipeline.log.1").read_text(encoding="utf-8") == "x" * 20


def test_backups_shift_and_the_oldest_is_dropped(tmp_path):
    log = tmp_path / "pipeline.log"
    _write(log, "current" * 5)
    for i, text in ((1, "one"), (2, "two"), (3, "three")):
        _write(tmp_path / f"pipeline.log.{i}", text)
    _rotate_at_startup(log, max_bytes=10, backups=3)
    assert (tmp_path / "pipeline.log.1").read_text(encoding="utf-8") == "current" * 5
    assert (tmp_path / "pipeline.log.2").read_text(encoding="utf-8") == "one"
    assert (tmp_path / "pipeline.log.3").read_text(encoding="utf-8") == "two"
    assert not (tmp_path / "pipeline.log.4").exists()


def test_missing_log_is_not_an_error(tmp_path):
    _rotate_at_startup(tmp_path / "pipeline.log", max_bytes=10, backups=3)
    assert list(tmp_path.iterdir()) == []


def test_log_held_open_elsewhere_is_left_for_the_next_start(tmp_path, monkeypatch):
    # On Windows a file another process has open can't be renamed.
    log = tmp_path / "pipeline.log"
    _write(log, "x" * 20)

    def locked(src, dst):
        raise PermissionError("in use by another process")

    monkeypatch.setattr(logger_module.os, "replace", locked)
    _rotate_at_startup(log, max_bytes=10, backups=3)
    assert log.read_text(encoding="utf-8") == "x" * 20
