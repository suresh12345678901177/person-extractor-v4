"""
experiments/diff_scans.py
===========================
The real-corpus half of the acceptance gate: diffs two scan_directory.py
*_names_*.csv files (before / after a candidate change) and reports how
many names moved between ACCEPTED, REVIEW and absent.

Case content never reaches the console, the ledger or git: the console and
--counts-json get COUNTS only. Every changed name, with a short context line
from its first source file, goes to --review-out - keep that under output/
(gitignored) - for the owner to read locally.

Usage:
    python experiments/diff_scans.py --before <names.csv> --after <names.csv> \
        --case-dir <case folder> --review-out output/self_upgrade/<id>_review.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

SNIPPET_CHARS = 160


def load(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as fh:
        return {r["name"].lower(): r for r in csv.DictReader(fh)}


def transitions(before: dict[str, dict], after: dict[str, dict]) -> dict[str, list[str]]:
    """{'accepted->review': [key, ...], ...} for every name whose state changed."""
    out: dict[str, list[str]] = collections.defaultdict(list)
    for key in sorted(set(before) | set(after)):
        b = before[key]["decision"] if key in before else "absent"
        a = after[key]["decision"] if key in after else "absent"
        if a != b:
            out[f"{b}->{a}"].append(key)
    return dict(out)


def _snippet(name: str, files: list[str], index: dict[str, list[Path]]) -> tuple[str, str]:
    from src.core.document_factory import DocumentFactory
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    for fname in files:
        for path in index.get(fname, []):
            result = DocumentFactory().create(str(path))
            if not result.success:
                continue
            text = result.document.full_text
            m = pattern.search(text)
            if m:
                start = max(0, m.start() - SNIPPET_CHARS // 2)
                return " ".join(text[start:m.end() + SNIPPET_CHARS // 2].split()), str(path)
    return "", ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, default=None, help="Needed for context snippets")
    parser.add_argument("--review-out", type=Path, default=None, help="Local CSV of changed names (keep under output/)")
    parser.add_argument("--counts-json", type=Path, default=None)
    args = parser.parse_args()

    before, after = load(args.before), load(args.after)
    moves = transitions(before, after)
    counts = {
        "before": dict(collections.Counter(r["decision"] for r in before.values())),
        "after": dict(collections.Counter(r["decision"] for r in after.values())),
        "transitions": {k: len(v) for k, v in moves.items()},
    }
    print(json.dumps(counts, indent=2))
    if args.counts_json:
        args.counts_json.write_text(json.dumps(counts, indent=2), encoding="utf-8")

    if args.review_out and moves:
        import logging
        logging.disable(logging.CRITICAL)
        index: dict[str, list[Path]] = collections.defaultdict(list)
        if args.case_dir:
            for p in args.case_dir.rglob("*"):
                if p.is_file():
                    index[p.name].append(p)
        args.review_out.parent.mkdir(parents=True, exist_ok=True)
        with args.review_out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["transition", "name", "occurrences_before", "occurrences_after", "context", "file"])
            for move, keys in sorted(moves.items()):
                for key in keys:
                    row = after.get(key) or before[key]
                    files = row["source_files"].split("; ")
                    context, path = _snippet(row["name"], files, index) if index else ("", "")
                    writer.writerow([move, row["name"], before.get(key, {}).get("total_occurrences", 0),
                                     after.get(key, {}).get("total_occurrences", 0), context, path])
        print(f"review list (local only): {args.review_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
