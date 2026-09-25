"""Tests for scripts/retrain_from_feedback.py's exclusion of labels sourced
from the files the evaluations score against."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from retrain_from_feedback import _is_benchmark_source


def test_official_benchmark_files_are_excluded():
    assert _is_benchmark_source(r"C:\repo\datasets\benchmark\benchmark_real_660.txt")
    assert _is_benchmark_source("/repo/datasets/benchmark/benchmark_001.txt")


def test_held_out_unseen_benchmark_is_excluded():
    # Regression test (2026-09-25): the folder check only knew "benchmark",
    # so a label from the held-out datasets/benchmark_unseen/ would have
    # been trained on - one of its names was already in pending_review.jsonl.
    assert _is_benchmark_source(r"C:\repo\datasets\benchmark_unseen\benchmark_unseen_names.txt")


def test_copy_of_a_benchmark_file_elsewhere_is_excluded():
    assert _is_benchmark_source(r"C:\Users\someone\Downloads\benchmark_real_675.txt")


def test_ordinary_source_files_are_kept():
    assert not _is_benchmark_source(r"C:\Users\someone\Downloads\meeting_notes.txt")
    assert not _is_benchmark_source(r"C:\cases\benchmark_notes\chat.txt")
