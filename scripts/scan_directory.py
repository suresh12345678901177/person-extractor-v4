"""
scripts/scan_directory.py
============================
Recursive offline batch extraction over an arbitrary folder tree.

Give it any directory - a case folder, a forensic export root, whatever
- and it walks every subdirectory (no depth limit) looking for *.txt
and *.pdf files, runs each one through the same Pipeline the CLI uses,
and writes three fresh, timestamped CSVs (never overwriting a previous
run):

  1. <name>_files_<timestamp>.csv   - one row per file: how many names
     were found in THAT file, the list of those names, how long that
     file took, and its status.
  2. <name>_names_<timestamp>.csv   - the case-wide rollup: every unique
     person found across all files, combined the same way cli.py's
     batch mode combines them (same-person mentions across files merge
     into one row with a total occurrence count and every source file
     that mentioned them).
  3. <name>_summary_<timestamp>.csv - run totals (files found, skipped,
     names found, total time), persisted so they survive after the
     terminal is gone.

PDF text extraction only reads the embedded text layer (via pypdf) -
a scanned/image-only PDF with no text layer yields nothing and is
flagged distinctly in the files CSV ("no extractable text - scanned/
image PDF, needs OCR"), never silently reported as "0 names found" as
if it had been genuinely read.

A run summary (files found, skipped counts, total names, total time)
prints to the console and is also written to the summary CSV so it's
never lost once the terminal scrolls away.

Usage:
    python scripts/scan_directory.py --input-dir "C:\\path\\to\\case_folder"
    python scripts/scan_directory.py --input-dir "C:\\path\\to\\case_folder" --output "C:\\path\\to\\results"
    python scripts/scan_directory.py --input-dir "C:\\path\\to\\case_folder" --include-review
"""
from __future__ import annotations

import argparse
import copy
import csv
import fnmatch
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import DEFAULT_CONFIG
from src.candidate.boundary_refiner import refine_combined_counts
from src.core.models import IDENTITY_RESOLUTION_CAVEAT, ExtractionResult
from src.feedback.feedback_store import FeedbackStore
from src.io.dispatcher import ReaderDispatcher
from src.io.pdf_reader import PdfReader
from src.pipeline.orchestrator import Pipeline
from src.utils.logger import configure_logging, get_logger
from src.utils.provenance import build_run_provenance, provenance_summary_line
from src.review.llm_reviewer import resolve_status

logger = get_logger("scripts.scan_directory")

# Every format a reader handles (src/io/dispatcher.py) - one list, so a new
# reader is scanned automatically.
SUPPORTED_EXTENSIONS = ReaderDispatcher().supported_extensions()

# --- Parallel file processing (2026-09-18) ---
# Each worker process gets its OWN Pipeline instance, loaded once per
# worker (not once per file) via _init_worker, exactly mirroring the
# existing single-process rationale ("spaCy + the trained classifier
# load once... since a large case folder can hold thousands of .txt
# files") - just amortized across N worker processes instead of one.
# A loaded spaCy Language object cannot be cheaply shared across process
# boundaries (no shared memory, and it isn't designed to be pickled per
# call), so a process pool - not a thread pool - each initializing its
# own Pipeline is the correct shape here, not naive
# Pipeline-sharing-across-processes. Verified before parallelizing: the
# only cross-file mutable state in the original loop is the `combined`
# dict built by _merge() in the MAIN process after each file's own
# result comes back - workers never touch it, so no shared-state
# synchronization is needed on the worker side at all, only ordered
# collection of each worker's already-self-contained ExtractionResult.
_worker_pipeline: Pipeline | None = None


def _init_worker(config: dict, base_dir_str: str) -> None:
    global _worker_pipeline
    configure_logging(Path(base_dir_str) / "logs")
    _worker_pipeline = Pipeline(config=config, base_dir=Path(base_dir_str))


def _run_one_in_worker(input_path_str: str) -> tuple[str, ExtractionResult | None, float, str | None]:
    """Runs in a worker process. Returns (path, result-or-None, seconds,
    exception-message-or-None) - never raises, mirroring the original
    per-file try/except in the sequential loop (a single corrupt/
    unreadable file must never abort the whole scan, worker or not)."""
    global _worker_pipeline
    file_start = time.perf_counter()
    try:
        result = _worker_pipeline.run(input_path_str)
        return (input_path_str, result, time.perf_counter() - file_start, None)
    except Exception as exc:  # noqa: BLE001 - see docstring
        return (input_path_str, None, time.perf_counter() - file_start, str(exc))


def _build_config(args: argparse.Namespace) -> dict:
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.no_feedback_log:
        config["feedback"]["enabled"] = False
    if args.loose_gate:
        config["decision"]["strict_corroboration"] = False
    if args.no_language_filter:
        config["language_filter"]["enabled"] = False
    return config


def _matching_exclude(rel_path: Path, patterns: list[str]) -> str | None:
    """The first --exclude glob matching this file's name or its path
    relative to --input-dir (forward slashes, case-insensitive), else None."""
    rel = rel_path.as_posix().lower()
    name = rel_path.name.lower()
    for pattern in patterns:
        pat = pattern.replace("\\", "/").lower()
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat):
            return pattern
    return None


def _discover_input_files(root: Path) -> tuple[list[Path], int]:
    """Recursively walks every subdirectory under root, arbitrarily
    deep, and returns every *.txt/*.pdf file (case-insensitive
    extension) found, sorted for a deterministic run order, plus a
    count of unsupported files encountered along the way (reported,
    never processed)."""
    input_files: list[Path] = []
    other_count = 0
    for path in root.rglob("*"):
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                input_files.append(path)
            else:
                other_count += 1
    input_files.sort()
    return input_files, other_count


def _pdf_has_extractable_text(path: Path) -> bool:
    """Cheap standalone re-read (via pypdf directly, not the full NLP
    pipeline) used ONLY to explain a 0-names PDF result: distinguishes
    "genuinely read the text, found no names" from "this PDF has no
    text layer at all" (scanned/image-only - needs OCR, out of scope
    here) so the files CSV never reports the latter as a plain, no
    further comment "0 names found" as if real text had been read."""
    doc_result = PdfReader().read(path)
    if not doc_result.success or doc_result.document is None:
        return False
    return len(doc_result.document.full_text.strip()) > 0


def _fresh_csv_path(output_dir: Path, prefix: str, timestamp: str) -> Path:
    path = output_dir / f"{prefix}_{timestamp}.csv"
    suffix = 1
    while path.exists():
        path = output_dir / f"{prefix}_{timestamp}_{suffix}.csv"
        suffix += 1
    return path


def run_scan(input_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    config = _build_config(args)
    if args.loose_gate:
        print("*** --loose-gate: COMPARISON MODE (accepts spaCy/context-only evidence) ***")

    # Computed once for the whole run (not per worker, not per file) -
    # git state and the model file are identical across every worker
    # process in this run, so one lookup in the main process is enough.
    # See src/utils/provenance.py (independent-audit finding #3).
    model_path = None
    if config.get("classification", {}).get("use_ml_classifier", True):
        model_path = BASE_DIR / config.get("classification", {}).get(
            "model_path", "models/lightgbm/person_classifier.txt"
        )
    provenance = build_run_provenance(BASE_DIR, model_path)
    provenance["spacy_model"] = config["detection"].get("spacy_model") if config["detection"].get("use_spacy", True) else None
    # The optional LLM reviewer (src/review/llm_reviewer.py) is resolved once
    # here, not by every worker (each would load and check the model), and
    # the verdict is passed to them in the config.
    llm_settings = config.get("llm_reviewer", {})
    if llm_settings:
        status = resolve_status(llm_settings)
        llm_settings["resolved"] = {"active": status.active, "reason": status.reason}
        provenance["llm_reviewer"] = status.reason
    print(f"  Provenance: {provenance_summary_line(provenance)}")
    if llm_settings:
        print(f"  LLM reviewer: {provenance['llm_reviewer']}")

    print(f"Scanning '{input_dir}' recursively for supported files ({', '.join(SUPPORTED_EXTENSIONS)}) ...")
    discover_start = time.perf_counter()
    input_files, other_count = _discover_input_files(input_dir)
    discover_seconds = time.perf_counter() - discover_start
    excluded: list[tuple[Path, str]] = []
    if args.exclude:
        kept = []
        for path in input_files:
            pattern = _matching_exclude(path.relative_to(input_dir), args.exclude)
            if pattern:
                excluded.append((path, pattern))
            else:
                kept.append(path)
        input_files = kept
        print(f"  Excluded by --exclude: {len(excluded)} file(s)"
              + (f", e.g. {', '.join(str(p.relative_to(input_dir)) for p, _ in excluded[:5])}" if excluded else ""))

    print(f"  Found {len(input_files)} supported file(s) "
          f"({other_count} other file(s) skipped, unsupported type) "
          f"in {discover_seconds:.2f}s")

    if not input_files:
        print("No supported files found under that path - nothing to do.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_tag = input_dir.name or "scan"

    files_csv_path = _fresh_csv_path(output_dir, f"{folder_tag}_files", timestamp)
    names_csv_path = _fresh_csv_path(output_dir, f"{folder_tag}_names", timestamp)
    summary_csv_path = _fresh_csv_path(output_dir, f"{folder_tag}_summary", timestamp)

    # workers=1 keeps the original single-process behavior verbatim (one
    # Pipeline instance for the whole run) - the safe fallback/debugging
    # path. workers>1 amortizes one Pipeline load per WORKER instead
    # (see _init_worker), still not once per file.
    pipeline = Pipeline(config=config, base_dir=BASE_DIR) if args.workers <= 1 else None

    combined: dict[str, dict] = {}

    # Only receives records from parallel workers (feedback.defer_writes);
    # the sequential path still logs inside its own Pipeline as before.
    feedback_store = FeedbackStore(BASE_DIR / "datasets" / "feedback")

    def _merge(person, decision: str, source_filename: str) -> None:
        key = person.normalized_text
        entry = combined.setdefault(key, {
            "display": person.display_text, "occurrences": 0,
            "files": [], "decision": "review",
        })
        entry["occurrences"] += person.occurrence_count
        if source_filename not in entry["files"]:
            entry["files"].append(source_filename)
        if decision == "accepted":
            entry["decision"] = "accepted"

    file_rows: list[dict] = []
    failures = 0
    skipped_non_english = 0
    skipped_binary = 0
    skipped_no_text = 0
    total_names_found = 0
    scan_start = time.perf_counter()

    def _iter_results():
        """Yields (input_path, result_or_None, seconds, exc_or_None) for
        every file, in input order, from either a direct sequential call
        (workers<=1) or an ordered pool.imap over worker processes
        (workers>1) - imap (not imap_unordered) deliberately preserves
        input order so file_rows/progress-printing/CSV-row-order are
        byte-identical to the sequential path regardless of worker count,
        which is what makes a before/after correctness diff meaningful."""
        if args.workers <= 1:
            for input_path in input_files:
                file_start = time.perf_counter()
                try:
                    result = pipeline.run(str(input_path))
                    yield input_path, result, time.perf_counter() - file_start, None
                except Exception as exc:  # a single corrupt/unreadable file must never abort the whole scan
                    logger.exception("Unhandled error processing %s", input_path)
                    yield input_path, None, time.perf_counter() - file_start, str(exc)
            return

        # Workers build feedback records but never write them - the main
        # process does the one deduplicated write below (see
        # FeedbackStore.build_records for the race this avoids).
        worker_config = copy.deepcopy(config)
        worker_config["feedback"]["defer_writes"] = True

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(worker_config, str(BASE_DIR)),
        ) as pool:
            path_strs = [str(p) for p in input_files]
            for path_str, result, seconds, exc in pool.imap(_run_one_in_worker, path_strs):
                if exc is not None:
                    logger.error("Unhandled error processing %s: %s", path_str, exc)
                yield Path(path_str), result, seconds, exc

    with files_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "status", "num_names", "names_found", "seconds_taken"])
        for path, pattern in excluded:
            writer.writerow([str(path.relative_to(input_dir)), f"skipped: excluded by --exclude {pattern}", 0, "", "0.0000"])

        for i, (input_path, result, seconds, exc) in enumerate(_iter_results(), start=1):
            rel_path = input_path.relative_to(input_dir)
            if exc is not None:
                writer.writerow([str(rel_path), f"ERROR: {exc}", 0, "", f"{seconds:.4f}"])
                failures += 1
                continue

            if not result.success:
                writer.writerow([str(rel_path), f"FAILED: {result.error}", 0, "", f"{seconds:.4f}"])
                failures += 1
                continue

            if result.feedback_records:
                feedback_store.append_new(list(result.feedback_records))

            if result.model_info.get("skipped_reason") == "non_english":
                ratio = result.model_info.get("english_word_ratio")
                writer.writerow([str(rel_path), f"skipped: non-English (ratio={ratio})", 0, "", f"{seconds:.4f}"])
                skipped_non_english += 1
                continue

            if result.model_info.get("skipped_reason") == "binary_content":
                kind = result.model_info.get("binary_format", "")
                writer.writerow([str(rel_path), f"skipped: binary content ({kind}) - not text", 0, "", f"{seconds:.4f}"])
                skipped_binary += 1
                continue

            persons = list(result.persons)
            if args.include_review:
                persons = persons + list(result.review_persons)

            if (not persons and input_path.suffix.lower() == ".pdf"
                    and not _pdf_has_extractable_text(input_path)):
                writer.writerow([str(rel_path), "skipped: no extractable text (scanned/image PDF, needs OCR)",
                                  0, "", f"{seconds:.4f}"])
                skipped_no_text += 1
                continue

            names_list = "; ".join(p.display_text for p in persons)
            writer.writerow([str(rel_path), "ok", len(persons), names_list, f"{seconds:.4f}"])
            total_names_found += len(persons)

            source_filename = input_path.name
            for person in result.persons:
                _merge(person, "accepted", source_filename)
            for person in result.review_persons:
                _merge(person, "review", source_filename)

            if i % 25 == 0 or i == len(input_files):
                print(f"  [{i}/{len(input_files)}] {rel_path} -> {len(persons)} name(s) ({seconds:.2f}s)")

    total_seconds = time.perf_counter() - scan_start

    combined = refine_combined_counts(combined)
    ordered = sorted(combined.values(), key=lambda e: e["occurrences"], reverse=True)

    with names_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "total_occurrences", "decision", "num_source_files", "source_files"])
        for e in ordered:
            writer.writerow([e["display"], e["occurrences"], e["decision"], len(e["files"]), "; ".join(e["files"])])

    accepted_count = sum(1 for e in ordered if e["decision"] == "accepted")
    review_count = sum(1 for e in ordered if e["decision"] == "review")
    files_processed_ok = len(input_files) - failures - skipped_non_english - skipped_no_text - skipped_binary
    total_files_in_folder = len(input_files) + len(excluded) + other_count
    # Independent-audit finding #1 (2026-09-23): the coverage gap between
    # "files in the folder" and "files that actually contributed a name"
    # was previously only reconstructable by hand from the other rows
    # below. Computed and printed unconditionally so it can't be missed -
    # see README's front-page coverage caveat, computed from this exact
    # formula.
    coverage_pct_of_all_files = (
        100.0 * files_processed_ok / total_files_in_folder if total_files_in_folder else 0.0
    )
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["root_folder", str(input_dir)])
        writer.writerow(["run_timestamp", timestamp])
        writer.writerow(["total_files_in_folder", total_files_in_folder])
        writer.writerow(["input_files_found", len(input_files) + len(excluded)])
        writer.writerow(["files_excluded", len(excluded)])
        writer.writerow(["unsupported_files_skipped", other_count])
        writer.writerow(["files_processed_ok", files_processed_ok])
        writer.writerow(["files_skipped_non_english", skipped_non_english])
        writer.writerow(["files_skipped_no_extractable_text", skipped_no_text])
        writer.writerow(["files_skipped_binary_content", skipped_binary])
        writer.writerow(["files_failed", failures])
        writer.writerow(["coverage_pct_of_all_files", f"{coverage_pct_of_all_files:.2f}"])
        for key, value in provenance.items():
            writer.writerow([f"provenance_{key}", value])
        writer.writerow(["note_identity_resolution", IDENTITY_RESOLUTION_CAVEAT])
        writer.writerow(["names_found_per_file_summed", total_names_found])
        writer.writerow(["unique_names_case_wide", len(ordered)])
        writer.writerow(["unique_names_accepted", accepted_count])
        writer.writerow(["unique_names_review", review_count])
        writer.writerow(["discovery_seconds", f"{discover_seconds:.4f}"])
        writer.writerow(["extraction_seconds", f"{total_seconds:.4f}"])
        writer.writerow(["total_seconds", f"{discover_seconds + total_seconds:.4f}"])
        writer.writerow(["files_csv", str(files_csv_path)])
        writer.writerow(["names_csv", str(names_csv_path)])

    print()
    print("=" * 78)
    print("SCAN SUMMARY")
    print("=" * 78)
    print(f"  Root folder             : {input_dir}")
    print(f"  Total files in folder   : {total_files_in_folder}")
    print(f"  Supported files found   : {len(input_files) + len(excluded)}")
    if excluded:
        print(f"  Excluded by --exclude   : {len(excluded)}")
    print(f"  Unsupported files skipped : {other_count}")
    print(f"  Files processed OK      : {files_processed_ok}")
    print(f"  Files skipped (non-English/boilerplate) : {skipped_non_english}")
    print(f"  Files skipped (no extractable text - scanned/image PDF) : {skipped_no_text}")
    print(f"  Files skipped (binary content, e.g. Android binary XML) : {skipped_binary}")
    print(f"  Files failed            : {failures}")
    print(f"  COVERAGE: names were extracted from {coverage_pct_of_all_files:.1f}% of all "
          f"{total_files_in_folder} files in this folder (see README 'Scope' section for why)")
    print(f"  Names found (per-file, summed) : {total_names_found}")
    print(f"  NOTE: {IDENTITY_RESOLUTION_CAVEAT}")
    print(f"  Unique names (case-wide, combined) : {len(ordered)} "
          f"({sum(1 for e in ordered if e['decision'] == 'accepted')} accepted, "
          f"{sum(1 for e in ordered if e['decision'] == 'review')} review)")
    print(f"  Total time taken        : {total_seconds:.2f}s "
          f"(+ {discover_seconds:.2f}s discovery)")
    print()
    print(f"  Per-file CSV  : {files_csv_path}")
    print(f"  Combined CSV  : {names_csv_path}")
    print(f"  Summary CSV   : {summary_csv_path}")

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recursively scan a folder for supported files (.txt, .pdf, .csv, .json, .html, .xml, .docx, .eml, ...) and run PERSON_EXTRACTOR_V4 on each one."
    )
    parser.add_argument("--input-dir", required=True, help="Folder to scan recursively for supported files.")
    parser.add_argument("--output", default=None,
                         help="Output folder for the two result CSVs (default: ./output/scans)")
    parser.add_argument("--include-review", action="store_true",
                         help="Also include REVIEW-bucket (weak/ambiguous) names alongside ACCEPTED "
                              "ones in both CSVs. Without this flag, the per-file names list and the "
                              "combined rollup only count ACCEPTED names.")
    parser.add_argument("--no-feedback-log", action="store_true",
                         help="Don't log REVIEW-bucket candidates for active learning this run")
    parser.add_argument("--loose-gate", action="store_true",
                         help="COMPARISON MODE, not for normal use - see cli.py --loose-gate")
    parser.add_argument("--no-language-filter", action="store_true",
                         help="Disable the English-language gate (on by default) that skips "
                              "files which aren't natural-language English prose - see "
                              "cli.py --no-language-filter")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                         help="Skip files whose name or path (relative to --input-dir) matches this "
                              "glob, case-insensitive. Repeatable, e.g. --exclude report.xml "
                              "--exclude \"*/cache/*\". Excluded files are still listed in the per-file "
                              "CSV and counted in the summary.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                         help="Number of worker processes for parallel file processing "
                              "(default: CPU count). Each worker loads its own spaCy model + "
                              "classifier once at startup, not once per file. Pass 1 to force "
                              "the original single-process behavior.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        parser.error(f"--input-dir must be an existing folder: '{args.input_dir}'")

    output_dir = Path(args.output) if args.output else BASE_DIR / "output" / "scans"

    configure_logging(BASE_DIR / "logs")
    return run_scan(input_dir, output_dir, args)


if __name__ == "__main__":
    raise SystemExit(main())
