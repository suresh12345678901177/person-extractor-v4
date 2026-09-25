"""
scripts/extend_gold_first_name_mentions.py
=============================================
Makes every benchmark gold file follow ONE labeling convention: a person
mentioned by first name alone ("Wendy already logged it", "Hi Rohan,",
"Ashwin's brother") is a person mention.

Before 2026-09-24 the files disagreed: benchmark_001/002 counted first-
name-only mentions (see scripts/extend_benchmark_gold_labels.py - the
same rule, applied to 001 only), every other file didn't. So the
pipeline correctly finding "Wendy" (a gold person, Wendy Zhao) was
scored as a false positive - 121 of the 182 accepted "false positives"
were exactly this. Counting them is the standard NER convention and what
an investigator needs (every reference to a person matters).

Rule, applied to the CLEANED text (what the pipeline and gold offsets
both use - NOT the raw file; they differ in most benchmark files):
for every gold person with a 2+ token name, the first non-title token is
their first name; every word-boundary occurrence of it not already inside
a gold span becomes a gold mention. Titles/initials are never used as
first names. Every addition was reviewed in context before this script
was written; the only exclusion and the only roster fix are listed
explicitly below with their reason, not inferred.

Each gold file is backed up to datasets/benchmark*/backups/ first.
Idempotent: re-running adds nothing.

Usage:
    python scripts/extend_gold_first_name_mentions.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.io.text_reader import TextReader
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.cleaner import clean_text

BENCHMARK_DIRS = [BASE_DIR / "datasets" / "benchmark", BASE_DIR / "datasets" / "benchmark_unseen"]

# Occurrences that match the rule but are NOT a reference to the person -
# reviewed in context. Matched against the text around the occurrence.
EXCLUDE_CONTEXTS = {
    # A TV show title; the text itself says the character merely shares
    # the name with the real Kavya Bhandari.
    "benchmark_004.txt": ["The Kavya Diaries"],
}

# People missing from a file's gold roster entirely - every occurrence of
# the full name is added. Reviewed in context.
ROSTER_ADDITIONS = {
    # Call-log party in the same rows, same "Name <phone>" format, as
    # gold-labeled people ("Marcus Delaney 14045551190 Branko Osvalt
    # 14045552210"); also a gold person in benchmark_real_660.txt.
    "benchmark_real_675.txt": ["Marcus Delaney"],
}


def _first_names(mentions: list[dict], kb: KnowledgeBase) -> set[str]:
    names = set()
    for m in mentions:
        tokens = [t for t in m["text"].split()
                  if not kb.is_title(t.rstrip(".")) and not kb.is_honorific(t.rstrip("."))]
        if len(tokens) >= 2 and len(tokens[0].rstrip(".")) > 1:
            names.add(tokens[0])
    return names


def _excluded(text: str, start: int, end: int, contexts: list[str]) -> bool:
    """True if [start, end) lies inside an occurrence of an excluded phrase."""
    return any(
        m.start() <= start and end <= m.end()
        for phrase in contexts
        for m in re.finditer(re.escape(phrase), text)
    )


def extend(txt_path: Path, gold_path: Path, kb: KnowledgeBase, stamp: str) -> tuple[int, dict]:
    text = clean_text(TextReader().read(str(txt_path)).document.full_text)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    mentions = gold["mentions"]
    spans = [(m["start"], m["end"]) for m in mentions]

    def free(s: int, e: int) -> bool:
        return not any(s < ge and gs < e for gs, ge in spans)

    added: dict[str, int] = {}
    new: list[dict] = []
    targets = [(name, "first name") for name in sorted(_first_names(mentions, kb))]
    targets += [(name, "roster") for name in ROSTER_ADDITIONS.get(txt_path.name, [])]
    for name, _kind in targets:
        for match in re.finditer(rf"\b{re.escape(name)}\b", text):
            s, e = match.span()
            if not free(s, e):
                continue
            if _excluded(text, s, e, EXCLUDE_CONTEXTS.get(txt_path.name, [])):
                continue
            new.append({"start": s, "end": e, "text": name})
            spans.append((s, e))
            added[name] = added.get(name, 0) + 1

    if not new:
        return 0, added

    backup_dir = gold_path.parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    shutil.copy2(gold_path, backup_dir / f"{gold_path.stem}_pre_first_names_{stamp}.json")

    mentions.extend(new)
    mentions.sort(key=lambda m: m["start"])
    gold["mention_count"] = len(mentions)
    gold["first_name_convention_note"] = (
        "2026-09-24: extended by scripts/extend_gold_first_name_mentions.py - every standalone "
        "first-name mention of a gold person is a gold mention (the convention benchmark_001 "
        "already used). See that script for the reviewed exclusions/roster additions."
    )
    gold_path.write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(new), added


def main() -> None:
    kb = KnowledgeBase.load(BASE_DIR / "assets")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for bench_dir in BENCHMARK_DIRS:
        for txt_path in sorted(bench_dir.glob("*.txt")):
            gold_path = txt_path.with_name(txt_path.stem + "_gold.json")
            if not gold_path.exists():
                continue
            count, added = extend(txt_path, gold_path, kb, stamp)
            detail = ", ".join(f"{k} x{v}" for k, v in sorted(added.items(), key=lambda kv: -kv[1]))
            print(f"{txt_path.name}: +{count} mention(s){' - ' + detail if detail else ''}")


if __name__ == "__main__":
    main()
