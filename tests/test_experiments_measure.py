"""Tests for experiments/measure.py's ledger bookkeeping (not the
measurements themselves, which run the full pipeline)."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import measure


def test_scan_counts_keeps_only_numeric_count_fields(tmp_path):
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "metric,value\nroot_folder,C:\\some\\case\\folder\nunique_names_accepted,236\n"
        "unique_names_review,334\nextraction_seconds,2600.5\ndiscovery_seconds,7.0\n",
        encoding="utf-8",
    )
    counts = measure.scan_counts(summary)
    assert counts == {"unique_names_accepted": 236.0, "unique_names_review": 334.0,
                      "extraction_seconds": 2600.5, "discovery_seconds": 7.0}
    assert "root_folder" not in counts  # a case path must never reach the committed ledger


def test_ledger_is_rebuilt_from_results_in_time_order(tmp_path, monkeypatch):
    monkeypatch.setattr(measure, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(measure, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "results").mkdir()
    for rid, when, f1 in (("exp01", "2026-09-26T00:00:00Z", 0.94), ("baseline", "2026-09-25T00:00:00Z", 0.93)):
        (tmp_path / "results" / f"{rid}.json").write_text(json.dumps({
            "id": rid, "measured_utc": when, "verdict": "kept",
            "benchmark": {"accepted": {"P": 0.96, "R": 0.91, "F1": f1}},
            "scan": {"unique_names_accepted": 236.0, "unique_names_review": 334.0,
                     "extraction_seconds": 100.0, "discovery_seconds": 5.0},
        }), encoding="utf-8")

    measure.rebuild_ledger()

    rows = list(csv.DictReader((tmp_path / "ledger.csv").open(encoding="utf-8")))
    assert [r["id"] for r in rows] == ["baseline", "exp01"]
    assert rows[1]["acc_F1"] == "0.94" and rows[0]["scan_seconds"] == "105.0"
    assert "| baseline |" in (tmp_path / "ledger.md").read_text(encoding="utf-8")
