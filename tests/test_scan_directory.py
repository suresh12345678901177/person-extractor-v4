"""Tests for scripts/scan_directory.py's --exclude matching."""
import sys
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scan_directory import _matching_exclude


def test_exclude_matches_file_name_anywhere_case_insensitive():
    rel = Path(PureWindowsPath(r"Archives\Samsung_A25\report.xml").as_posix())
    assert _matching_exclude(rel, ["REPORT.XML"]) == "REPORT.XML"
    assert _matching_exclude(rel, ["*.xml"]) == "*.xml"


def test_exclude_matches_relative_path_globs_with_either_slash():
    rel = Path("cache/sub/tmp.txt")
    assert _matching_exclude(rel, ["cache/*"]) == "cache/*"
    assert _matching_exclude(rel, [r"cache\*"]) == r"cache\*"


def test_exclude_leaves_other_files_alone():
    assert _matching_exclude(Path("notes/report_final.txt"), ["report.xml", "cache/*"]) is None
