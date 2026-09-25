"""
scripts/build_benchmark_from_markup.py
=========================================
Converts a **bold-marked** source text into a clean .txt benchmark file
plus a gold_labels.json of exact (start, end, text) person-mention
spans, by stripping the ** markers and recording where each marked span
lands in the stripped output.

This is the reliable, mechanical way benchmark_001 was built - it
avoids manual char-offset counting, which is the #1 way hand-built NER
benchmarks end up with silently wrong gold labels.

Usage:
    python scripts/build_benchmark_from_markup.py <marked_source.md> <output_name>
    python scripts/build_benchmark_from_markup.py <marked_source.md> <output_name> --output-dir datasets/evaluation_holdout/snapshots

Produces:
    datasets/benchmark/<output_name>.txt
    datasets/benchmark/<output_name>_gold.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def strip_markup(raw: str) -> tuple[str, list[dict]]:
    plain_chunks: list[str] = []
    gold_spans: list[dict] = []
    i = 0
    out_pos = 0
    n = len(raw)

    while i < n:
        if raw[i:i + 2] == "**":
            j = raw.index("**", i + 2)
            name = raw[i + 2:j]
            gold_spans.append({"start": out_pos, "end": out_pos + len(name), "text": name})
            plain_chunks.append(name)
            out_pos += len(name)
            i = j + 2
        else:
            plain_chunks.append(raw[i])
            out_pos += 1
            i += 1

    return "".join(plain_chunks), gold_spans


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TXT + gold spans from **marked** source text")
    parser.add_argument("marked_source", type=Path)
    parser.add_argument("output_name")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "datasets" / "benchmark",
                        help="Directory for the generated TXT and *_gold.json files")
    args = parser.parse_args()

    source_path = args.marked_source
    output_name = args.output_name

    raw = source_path.read_text(encoding="utf-8")
    plain_text, gold_spans = strip_markup(raw)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / f"{output_name}.txt"
    gold_path = out_dir / f"{output_name}_gold.json"

    txt_path.write_text(plain_text, encoding="utf-8")
    gold_path.write_text(json.dumps({
        "source_markup_file": str(source_path),
        "mention_count": len(gold_spans),
        "mentions": gold_spans,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {txt_path} ({len(plain_text)} chars)")
    print(f"Wrote {gold_path} ({len(gold_spans)} gold mentions)")


if __name__ == "__main__":
    main()
