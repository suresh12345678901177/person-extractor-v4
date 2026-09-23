"""Unit tests for scripts/check_feedback_privacy.py (independent-audit
finding #8's guardrail against real-case content in
datasets/feedback/*.jsonl).

Written after this checker caught a REAL, live finding on first run: 5
records in the actual (uncommitted) datasets/feedback/confirmed_labels.jsonl
matched the "ProDiscover" pattern - genuine casework content (a real
WhatsApp export with real phone numbers and a real name) that had not yet
reached git history. These tests use only synthetic content."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SPEC = importlib.util.spec_from_file_location(
    "check_feedback_privacy", Path(__file__).resolve().parents[1] / "scripts" / "check_feedback_privacy.py"
)
check_feedback_privacy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_feedback_privacy)

find_risks = check_feedback_privacy.find_risks
load_patterns = check_feedback_privacy.load_patterns

PATTERNS = load_patterns(check_feedback_privacy.PATTERNS_PATH)


def test_real_case_fingerprint_is_flagged():
    lines = [
        '{"text": "Alex Rao", "context_text": "Cardholder: Alex Rao | file: CASEFILE.txt"}',
    ]
    findings = find_risks(lines, PATTERNS, "test.jsonl")
    assert findings
    assert "record #1" in findings[0]


def test_synthetic_benchmark_shape_is_not_flagged():
    """The exact "Cardholder: X | Card: ... | Bank: ..." shape used by
    this project's own legitimate synthetic benchmark data must NOT be
    flagged just for looking superficially similar to real casework -
    only an actual case-identifier pattern should trigger."""
    lines = [
        '{"text": "Femi Novak", "context_text": "Cardholder: Femi Novak | Card: 1409 | Bank: DBS | Balance: $9312 | Status: pending"}',
    ]
    findings = find_risks(lines, PATTERNS, "test.jsonl")
    assert findings == []


def test_public_book_context_is_not_flagged():
    lines = [
        '{"text": "Alpert", "context_text": "An imprint of Penguin Random House LLC 375 Hudson Street New York"}',
    ]
    findings = find_risks(lines, PATTERNS, "test.jsonl")
    assert findings == []


def test_emoji_and_non_ascii_context_does_not_crash():
    """Regression test for a real bug this checker had on its first real
    run: subprocess.run(..., text=True) decodes with the platform locale
    default (cp1252 on Windows), which crashes outright on real
    context_text containing emoji/curly quotes. find_risks() itself
    operates on already-decoded strings, so this specifically guards the
    JSON-parsing path against non-ASCII content, not the subprocess path
    (see test_check_feedback_privacy_staged_diff_handles_utf8 below for
    that)."""
    lines = [
        '{"text": "X", "context_text": "\\ud83d\\udd12 Messages are end-to-end encrypted \\u2014 ProDiscover Forensics"}',
    ]
    findings = find_risks(lines, PATTERNS, "test.jsonl")
    assert findings  # "ProDiscover" still matches even inside emoji-bearing text
    assert "record #1" in findings[0]


def test_malformed_json_line_is_skipped_not_crashed():
    lines = ["not valid json at all", '{"text": "Fine", "context_text": "benign text"}']
    findings = find_risks(lines, PATTERNS, "test.jsonl")
    assert findings == []


def test_non_string_fields_are_skipped_not_crashed():
    lines = ['{"text": "Fine", "context_text": null}', '{"text": 123, "context_text": ["not", "a", "string"]}']
    findings = find_risks(lines, PATTERNS, "test.jsonl")
    assert findings == []


def test_case_id_prefix_pattern_is_flagged():
    lines = ['{"text": "X", "context_text": "source file CASEID_chat.txt"}']
    findings = find_risks(lines, PATTERNS, "test.jsonl")
    assert findings


def test_staged_diff_handles_utf8_without_crashing(tmp_path, monkeypatch):
    """End-to-end regression test for the real encoding bug: a staged
    addition containing genuine emoji/curly-quote bytes must not crash
    subprocess.run's decoding (this exact shape - a WhatsApp
    "end-to-end encrypted" system message - is what the real finding
    looked like)."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)

    target = repo / "confirmed_labels.jsonl"
    target.write_text(
        '{"text": "X", "context_text": "\U0001f512 Messages are end-to-end encrypted — benign"}\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "confirmed_labels.jsonl"], cwd=repo, check=True)

    monkeypatch.setattr(check_feedback_privacy, "BASE_DIR", repo)
    lines = check_feedback_privacy._staged_added_lines(target)

    assert len(lines) == 1
    assert "end-to-end encrypted" in lines[0]
