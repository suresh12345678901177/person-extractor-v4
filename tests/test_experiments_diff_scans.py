"""Tests for experiments/diff_scans.py's before/after comparison."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import diff_scans


def _write(path: Path, rows: list[tuple[str, str]]) -> Path:
    path.write_text(
        "name,total_occurrences,decision,num_source_files,source_files\n"
        + "".join(f"{n},3,{d},1,a.txt\n" for n, d in rows),
        encoding="utf-8",
    )
    return path


def test_transitions_cover_every_state_change_and_ignore_unchanged(tmp_path):
    before = diff_scans.load(_write(tmp_path / "b.csv", [("Read", "accepted"), ("Ada Lovelace", "accepted"),
                                                          ("Block", "review"), ("Gone", "review")]))
    after = diff_scans.load(_write(tmp_path / "a.csv", [("Read", "review"), ("Ada Lovelace", "accepted"),
                                                         ("Block", "accepted"), ("New Name", "accepted")]))
    assert diff_scans.transitions(before, after) == {
        "accepted->review": ["read"],
        "review->accepted": ["block"],
        "review->absent": ["gone"],
        "absent->accepted": ["new name"],
    }
