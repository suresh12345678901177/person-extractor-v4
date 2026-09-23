"""
scripts/check_feedback_privacy.py
=====================================
Guardrail against a future accident, not a fix to anything currently
wrong (independent-audit finding #8): datasets/feedback/*.jsonl is
git-tracked (unlike output/, which .gitignore excludes entirely) and
stores `context_text` - a real snippet of the source line surrounding
every logged candidate (see FeedbackRecord.context_text). A manual audit
(2026-09-23) confirmed it's currently clean - sourced from
datasets/benchmark/ and public-domain book/paper text only, matching
README's "Known limitations" claim - but nothing previously stopped a
future session from accidentally committing a real case snippet into
version control the way this exact class of content has already been
kept OUT of confirmed_labels.jsonl by manual discipline alone (see
README: "634 items were real corporate-fraud case content and were
deliberately left to the case's own reviewer").

Checks every added record's `text`/`context_text` fields against
case-identifier patterns in
assets/privacy_guard/case_identifier_patterns.txt (a plain-text asset,
not a hardcoded list - same convention as every other assets/*.txt file,
so starting a new case just means adding a line there).

Usage:
    python scripts/check_feedback_privacy.py            # check newly staged additions (for a pre-commit hook)
    python scripts/check_feedback_privacy.py --all       # check the ENTIRE current file(s) - one-off verification / CI
    python scripts/check_feedback_privacy.py --all --files path/to/other.jsonl

Exit code 0 = clean. Exit code 1 = a likely real-case identifier was
found - the commit is blocked (when run as the pre-commit hook in
.git/hooks/pre-commit) until a human reviews it.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PATTERNS_PATH = BASE_DIR / "assets" / "privacy_guard" / "case_identifier_patterns.txt"
DEFAULT_TARGET_FILES = [
    BASE_DIR / "datasets" / "feedback" / "confirmed_labels.jsonl",
    BASE_DIR / "datasets" / "feedback" / "pending_review.jsonl",
]
CHECKED_FIELDS = ("text", "context_text")


def load_patterns(path: Path) -> list[re.Pattern]:
    if not path.exists():
        return []
    patterns = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(re.compile(line, re.IGNORECASE))
    return patterns


def _staged_added_lines(target_file: Path) -> list[str]:
    """Every line newly ADDED by the current git staging area for
    `target_file` (unified diff `+` lines, excluding the `+++` file
    header) - never touches lines that were already committed, so this
    doesn't re-flag history on every run, only what's about to be added."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "-U0", "--", str(target_file)],
            cwd=BASE_DIR, capture_output=True, timeout=15, check=True,
            # Explicit UTF-8, not text=True's locale default (cp1252 on
            # Windows) - real context_text can contain emoji/curly quotes
            # that crash a cp1252 decode outright (found via this exact
            # hook run against the real contaminated records below).
            encoding="utf-8", errors="replace",
        )
    except Exception as exc:  # noqa: BLE001 - never let the checker itself crash a commit
        print(f"WARNING: could not read staged diff for {target_file.name}: {exc}", file=sys.stderr)
        return []

    added = []
    for line in result.stdout.splitlines():
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
    return added


def _all_lines(target_file: Path) -> list[str]:
    if not target_file.exists():
        return []
    return target_file.read_text(encoding="utf-8", errors="replace").splitlines()


def find_risks(lines: list[str], patterns: list[re.Pattern], source_label: str) -> list[str]:
    """Returns human-readable findings - deliberately WITHOUT echoing the
    matched text itself (the checker's own output must not become a new
    leak vector, e.g. into a CI log)."""
    findings = []
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # not a JSON record on its own (e.g. a diff context artifact) - not this checker's job
        if not isinstance(record, dict):
            continue
        for field in CHECKED_FIELDS:
            value = record.get(field)
            if not isinstance(value, str):
                continue
            for pattern in patterns:
                if pattern.search(value):
                    findings.append(
                        f"{source_label}: record #{i} field '{field}' matches pattern "
                        f"'{pattern.pattern}' - looks like a real case identifier, not "
                        f"benchmark/public-domain text. Review before committing."
                    )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                         help="Check the full current content of the target file(s), not just staged additions.")
    parser.add_argument("--files", nargs="*", type=Path, default=None,
                         help="Override the target file(s) (default: datasets/feedback/*.jsonl)")
    args = parser.parse_args()

    patterns = load_patterns(PATTERNS_PATH)
    if not patterns:
        print(f"WARNING: no patterns loaded from {PATTERNS_PATH} - checker is a no-op.", file=sys.stderr)

    target_files = args.files if args.files else DEFAULT_TARGET_FILES

    all_findings: list[str] = []
    for target_file in target_files:
        lines = _all_lines(target_file) if args.all else _staged_added_lines(target_file)
        all_findings.extend(find_risks(lines, patterns, target_file.name))

    if all_findings:
        print("=" * 78)
        print("PRIVACY GUARDRAIL: possible real-case content in a git-tracked feedback file")
        print("=" * 78)
        for finding in all_findings:
            print(f"  - {finding}")
        print()
        print("If this is a false positive (e.g. new legitimate benchmark/synthetic data")
        print("that happens to match a pattern), either adjust the pattern in")
        print(f"  {PATTERNS_PATH.relative_to(BASE_DIR)}")
        print("or, if you've manually confirmed the content is safe, bypass with")
        print("  git commit --no-verify   (only after a human has actually looked)")
        return 1

    scope = "entire file(s)" if args.all else "staged additions"
    print(f"check_feedback_privacy: clean ({scope}, {len(target_files)} file(s) checked, "
          f"{len(patterns)} pattern(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
