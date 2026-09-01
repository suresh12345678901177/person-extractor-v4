"""
scripts/build_unseen_gold_labels.py
======================================
Builds datasets/benchmark_unseen/benchmark_unseen_names_gold.json by
mechanically finding every occurrence of each canonical name in
benchmark_unseen_names.txt via regex word-boundary search - same
"never hand-count offsets" principle as build_benchmark_from_markup.py,
adapted for a file with no ** markup since every mention here is a
plain, unadorned occurrence of a known canonical name.

Titled forms are searched FIRST and their matched ranges are excluded
from the later bare-name search, so "Dr. Ilinca Prodescu" is recorded
as one titled mention (title included in the span, same convention
used in benchmark_003's gold labels) rather than also double-counting
the "Ilinca Prodescu" substring inside it as a second mention.

This benchmark is intentionally kept OUT of datasets/benchmark/ (the
official suite the README's documented precision/recall/F1 numbers are
computed from) - every name in it is verified absent from
assets/first_names.txt, assets/last_names.txt, and every other asset
list, specifically to measure the pipeline's accuracy WITHOUT any
dictionary support, as a separate, standalone diagnostic. See
scripts/evaluate_unseen_names.py to run it.

Usage:
    python scripts/build_unseen_gold_labels.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BENCH_DIR = BASE_DIR / "datasets" / "benchmark_unseen"
TXT_PATH = BENCH_DIR / "benchmark_unseen_names.txt"
GOLD_PATH = BENCH_DIR / "benchmark_unseen_names_gold.json"

# Titled forms searched first (title text included in the gold span,
# matching benchmark_003's existing convention), then bare forms.
TITLED_NAMES = [
    "Dr. Ilinca Prodescu",
    "Mr. Ranjodh Aulakh",
]
BARE_NAMES = [
    "Zeyra Voskuijlen",
    "Baltasar Nkemelu",
    "Ilinca Prodescu",
    "Casimir Ostrowski",
    "Thandeka Mabuza",
    "Fionnuala Devereux",
    "Ranjodh Aulakh",
    "Esperanza Quilodran",
    "Bartholomew Fenwick",
    "Zsofia Lakatos",
]


def _find_all(text: str, name: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(rf"\b{re.escape(name)}\b", text)]


def build() -> None:
    text = TXT_PATH.read_text(encoding="utf-8")

    claimed: list[tuple[int, int]] = []
    mentions: list[dict] = []

    for name in TITLED_NAMES:
        for start, end in _find_all(text, name):
            claimed.append((start, end))
            mentions.append({"start": start, "end": end, "text": name})

    def _inside_claimed(start: int, end: int) -> bool:
        return any(start >= cs and end <= ce for cs, ce in claimed)

    for name in BARE_NAMES:
        for start, end in _find_all(text, name):
            if _inside_claimed(start, end):
                continue  # already covered by a titled mention above
            claimed.append((start, end))
            mentions.append({"start": start, "end": end, "text": name})

    mentions.sort(key=lambda m: m["start"])

    GOLD_PATH.write_text(json.dumps({
        "source_txt_file": str(TXT_PATH.relative_to(BASE_DIR)),
        "note": (
            "Every person name in this file is verified absent from "
            "assets/first_names.txt, assets/last_names.txt, and every "
            "other assets/*.txt list - built specifically to measure "
            "detection accuracy without dictionary support."
        ),
        "mention_count": len(mentions),
        "mentions": mentions,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {GOLD_PATH} ({len(mentions)} gold mentions)")
    for name in TITLED_NAMES + BARE_NAMES:
        count = sum(1 for m in mentions if m["text"] == name)
        print(f"  {name:<24} {count} mention(s)")


if __name__ == "__main__":
    build()
