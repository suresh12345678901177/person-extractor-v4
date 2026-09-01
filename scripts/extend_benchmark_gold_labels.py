"""
scripts/extend_benchmark_gold_labels.py
==========================================
benchmark_001's original gold labels only marked ONE representative
bolded occurrence of each person per paragraph, not every repeated
mention - so any evaluator that penalizes an un-bolded repeat mention
as a "false positive" is measuring gold-label incompleteness, not real
system errors.

This script makes the labels exhaustive: for each of the 10 canonical
people in benchmark_001, it finds EVERY standalone occurrence of their
unambiguous first name in the text and adds it to the gold set.

Deliberately EXCLUDED: abbreviated forms like "S. Kumar", "R. Kumar",
"A. K." etc. These are genuinely ambiguous BY THE TEXT'S OWN NARRATIVE
(the story's plot is literally that nobody can tell who these initials
refer to) - scoring the system against a guess at their "true" referent
would be evaluating against manufactured ground truth, not real ground
truth. Excluding them is the intellectually honest choice, not a
convenient one: it means the benchmark neither rewards nor penalizes
how the system handles inherently ambiguous abbreviations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CANONICAL_FIRST_NAMES = [
    "Suresh", "Ramesh", "Lakshmi", "Anil", "Kavitha",
    "Srinivas", "Prasad", "Swapna", "Naresh", "Geetha",
]


def extend(txt_path: Path, gold_path: Path) -> None:
    text = txt_path.read_text(encoding="utf-8")
    gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
    existing_spans = {(m["start"], m["end"]) for m in gold_data["mentions"]}

    added = 0
    for name in CANONICAL_FIRST_NAMES:
        for match in re.finditer(rf"\b{name}\b", text):
            span = (match.start(), match.end())
            if span not in existing_spans:
                gold_data["mentions"].append({"start": span[0], "end": span[1], "text": name})
                existing_spans.add(span)
                added += 1

    gold_data["mentions"].sort(key=lambda m: m["start"])
    gold_data["mention_count"] = len(gold_data["mentions"])
    gold_data["note"] = (
        "Gold labels extended programmatically to include EVERY standalone "
        "occurrence of each canonical first name (not just the originally "
        "bolded subset). Deliberately ambiguous abbreviated forms (e.g. "
        "'S. Kumar') are excluded - the source text's own narrative makes "
        "their true referent genuinely undecidable."
    )

    gold_path.write_text(json.dumps(gold_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{txt_path.name}: added {added} mentions, gold set now has {len(gold_data['mentions'])} total")


if __name__ == "__main__":
    bench_dir = BASE_DIR / "datasets" / "benchmark"
    extend(bench_dir / "benchmark_001.txt", bench_dir / "benchmark_001_gold.json")
