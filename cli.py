"""
cli.py
=======
Command-line entry point for PERSON_EXTRACTOR_V4.

Usage:
    python cli.py --input path/to/file.txt
    python cli.py --input path/to/file.txt --format json --output result.json
    python cli.py --input path/to/file.txt --explain
    python cli.py --evaluate                      (runs the labeled benchmark suite)
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from config import BASE_DIR, DEFAULT_CONFIG
from src.candidate.boundary_refiner import refine_combined_counts
from src.export.csv_exporter import CsvExporter
from src.export.json_exporter import JsonExporter
from src.export.report_exporter import ReportExporter
from src.pipeline.orchestrator import Pipeline
from src.utils.logger import configure_logging, get_logger

EXPORTERS = {
    "csv": CsvExporter,
    "json": JsonExporter,
}


def _write_names_only_csv(output_dir: Path, rows: list[tuple[str, int, str]], include_decision: bool) -> Path:
    """Writes a plain (name, occurrences[, decision]) CSV with a fresh
    timestamped filename - never overwrites a previous run's file, per
    --names-only's whole purpose: one command, one new CSV, nothing else.
    The decision column is opt-in (--include-review) so the default
    2-column shape stays exactly what --names-only has always promised
    ("nothing else") for anyone already parsing it by fixed column count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"names_{timestamp}.csv"
    # datetime.now() is second-resolution; guard the extremely unlikely
    # case of two runs landing in the same second.
    suffix = 1
    while path.exists():
        path = output_dir / f"names_{timestamp}_{suffix}.csv"
        suffix += 1

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "occurrences", "decision"] if include_decision else ["name", "occurrences"])
        for name, count, decision in rows:
            writer.writerow([name, count, decision] if include_decision else [name, count])
    return path


def _print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _print_section(title: str) -> None:
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def run_extraction(input_path: str, output_dir: Path, args: argparse.Namespace, config: dict):
    """Runs the pipeline on one input file, prints/exports its report,
    and returns the ExtractionResult (result.success indicates outcome)
    so batch mode can aggregate accepted persons across files."""
    logger = get_logger("cli")
    pipeline = Pipeline(config=config, base_dir=BASE_DIR)
    result = pipeline.run(input_path)

    if not result.success:
        logger.error("Extraction failed for %s: %s", input_path, result.error)
        print(f"FAILED: {result.error}", file=sys.stderr)
        return result

    if getattr(args, "names_only", False):
        return result  # caller writes the plain names CSV itself, nothing else

    _print_header("PERSON_EXTRACTOR_V4 - EXTRACTION RESULT")
    print(f"Source file : {input_path}")

    if result.model_info.get("skipped_reason") == "non_english":
        print(f"\n*** SKIPPED: not English-language prose "
              f"(english_word_ratio={result.model_info.get('english_word_ratio')}) - "
              f"likely non-English legal/UI boilerplate, not evidentiary text. "
              f"Use --no-language-filter to force extraction anyway. ***\n")

    _print_section("TIME TAKEN")
    for st in result.statistics.stage_timings:
        print(f"  {st.stage_name:<28} {st.seconds:>8.4f} sec")
    print(f"  {'TOTAL':<28} {result.statistics.total_seconds:>8.4f} sec")

    _print_section("MODEL CONFIGURATION & ACCURACY")
    print(f"  Detectors enabled          : {', '.join(result.model_info.get('detectors_enabled', []))}")
    ml_loaded = result.model_info.get("ml_classifier_loaded", False)
    print(f"  ML classifier loaded       : {ml_loaded}")
    if ml_loaded:
        acc = result.model_info.get("ml_classifier_validation_accuracy", "n/a")
        print(f"  ML classifier accuracy*    : {acc}")
        print(f"  *synthetic distant-supervision validation accuracy - NOT real-world")
        print(f"   accuracy. Run 'python cli.py --evaluate' for real end-to-end")
        print(f"   precision/recall/F1 against hand-labeled benchmark text.")
    else:
        print(f"  (no trained model found - run scripts/train_model.py; "
              f"pipeline is running rules+spaCy only)")

    _print_section(f"ACCEPTED PERSONS ({len(result.persons)})")
    for person in result.persons:
        print(f"\n  NAME            : {person.display_text}")
        print(f"  OCCURRENCES     : {person.occurrence_count} time(s)")
        print(f"  BEST CONFIDENCE : {person.best_confidence:.2f}")
        print(f"  LOCATIONS       :")
        for loc in person.locations[:10]:
            print(f"      - {loc}")
        if len(person.locations) > 10:
            print(f"      ... and {len(person.locations) - 10} more")
        if args.explain:
            print(f"  WHY DETECTED (first mention):")
            for e in person.mentions[0].state.evidence:
                sign = "+" if e.score >= 0 else ""
                print(f"      {e.label:<42} {sign}{e.score:.2f}  [{e.source}]")

    if result.review_persons:
        _print_section(f"FLAGGED FOR REVIEW ({len(result.review_persons)}) - weak/ambiguous evidence")
        for person in result.review_persons:
            print(f"\n  NAME            : {person.display_text}")
            print(f"  OCCURRENCES     : {person.occurrence_count} time(s)")
            print(f"  BEST CONFIDENCE : {person.best_confidence:.2f}")
            if args.explain:
                print(f"  WHY FLAGGED:")
                for e in person.mentions[0].state.evidence:
                    sign = "+" if e.score >= 0 else ""
                    print(f"      {e.label:<42} {sign}{e.score:.2f}  [{e.source}]")

    _print_section("SUMMARY")
    print(f"  Unique persons accepted : {len(result.persons)}")
    print(f"  Unique persons in review: {len(result.review_persons)}")
    print(f"  Rejected candidate spans: {len(result.rejected)}")
    caveat = result.model_info.get("identity_resolution_caveat")
    if caveat:
        print(f"  NOTE: {caveat}")

    if result.review_persons:
        from src.feedback.feedback_store import FeedbackStore
        pending_count = len(FeedbackStore(BASE_DIR / "datasets" / "feedback").load_pending())
        if pending_count:
            print(f"\n  {pending_count} item(s) awaiting your confirmation for active learning.")
            print(f"  Run: python scripts/label_feedback.py")

    # Exports
    report_path = ReportExporter().export(result, output_dir / "report.txt")
    written = [report_path]
    if args.format:
        exporter = EXPORTERS[args.format]()
        path = exporter.export(
            list(result.persons), list(result.review_persons), list(result.rejected),
            output_dir / f"result.{args.format}", input_path,
            run_info=result.model_info,
        )
        written.append(path)

    print()
    print("Exported files:")
    for p in written:
        print(f"  - {p}")

    return result


def _build_config(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if getattr(args, "no_feedback_log", False):
        config["feedback"]["enabled"] = False
    if getattr(args, "loose_gate", False):
        config["decision"]["strict_corroboration"] = False
    if getattr(args, "no_language_filter", False):
        config["language_filter"]["enabled"] = False
    config["detection"]["spacy_processes"] = getattr(args, "spacy_processes", 1)
    return config


def run_evaluate(args: argparse.Namespace) -> int:
    from src.evaluation.evaluator import run_benchmark

    pipeline = Pipeline(config=_build_config(args), base_dir=BASE_DIR)

    _print_header("PERSON_EXTRACTOR_V4 - END-TO-END BENCHMARK EVALUATION")
    if args.loose_gate:
        print("*** --loose-gate: COMPARISON MODE (accepts spaCy/context-only evidence) ***")
    classifier_metrics = getattr(getattr(pipeline.extractor, "classifier", None), "training_metrics", {})
    if classifier_metrics:
        print("LightGBM internal validation (synthetic/weak-supervision data; not end-to-end accuracy):")
        print(f"  accuracy {classifier_metrics.get('validation_accuracy', 'n/a')} | "
              f"precision {classifier_metrics.get('validation_precision', 'n/a')} | "
              f"recall {classifier_metrics.get('validation_recall', 'n/a')} | "
              f"F1 {classifier_metrics.get('validation_f1', 'n/a')}")
    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else BASE_DIR / "datasets" / "benchmark"
    print(f"Running the full pipeline against every labeled file in {benchmark_dir} ...")

    report = run_benchmark(pipeline, benchmark_dir)

    if not report.file_results:
        print(f"No labeled TXT/gold pairs found in {benchmark_dir}.")
        return 1

    for fr in report.file_results:
        _print_section(f"FILE: {fr.name}")
        a, c = fr.accepted_metrics, fr.candidate_metrics
        print(f"  Gold mentions               : {a.gold_total}")
        print(f"  -- ACCEPTED only (automatic extraction quality) --")
        print(f"    Predicted / TP / FP / FN  : {a.predicted_total} / {a.true_positives} / {a.false_positives} / {a.false_negatives}")
        print(f"    Precision / Recall / F1   : {a.precision:.4f} / {a.recall:.4f} / {a.f1:.4f}")
        print(f"  -- ACCEPTED + REVIEW (candidate-discovery quality) --")
        print(f"    Predicted / TP / FP / FN  : {c.predicted_total} / {c.true_positives} / {c.false_positives} / {c.false_negatives}")
        print(f"    Precision / Recall / F1   : {c.precision:.4f} / {c.recall:.4f} / {c.f1:.4f}")
        if fr.missed_mentions:
            print(f"  Missed even w/ review       : {fr.missed_mentions}")
        if fr.false_positive_texts:
            print(f"  False positives (accepted)  : {fr.false_positive_texts}")

    _print_section("OVERALL (all benchmark files combined)")
    n_files = len(report.file_results)
    print(f"  Sample-size caveat: F1/precision/recall below are computed on N={n_files} "
          f"benchmark files. Treat any swing under ~2 percentage points as noise at this "
          f"sample size, not a real change - the same 0.02 tolerance "
          f"retrain_from_feedback.py's own promotion guardrail uses. See README's "
          f"'Independent-audit findings' section for why the benchmark is this small "
          f"and what would need to change to shrink this caveat.")
    a, c = report.overall_accepted, report.overall_candidate
    print(f"  ACCEPTED only (automatic extraction quality):")
    print(f"    Precision : {a.precision:.4f}")
    print(f"    Recall    : {a.recall:.4f}")
    print(f"    F1        : {a.f1:.4f}")
    print(f"    Accuracy  : {a.accuracy:.4f}  (TP / (TP+FP+FN), standard NER convention)")
    print(f"  ACCEPTED + REVIEW (candidate-discovery quality - what a human sees, not what's auto-trusted):")
    print(f"    Precision : {c.precision:.4f}")
    print(f"    Recall    : {c.recall:.4f}")
    print(f"    F1        : {c.f1:.4f}")
    print(f"    Accuracy  : {c.accuracy:.4f}  (TP / (TP+FP+FN), standard NER convention)")

    if report.domain_accepted:
        _print_section("BY DOCUMENT-TYPE DOMAIN (see evaluator.py's FILE_DOMAINS)")
        for domain in sorted(report.domain_accepted):
            da, dc = report.domain_accepted[domain], report.domain_candidate[domain]
            print(f"  {domain} (gold: {da.gold_total}):")
            print(f"    ACCEPTED only      : P {da.precision:.4f} / R {da.recall:.4f} / F1 {da.f1:.4f}")
            print(f"    ACCEPTED + REVIEW  : P {dc.precision:.4f} / R {dc.recall:.4f} / F1 {dc.f1:.4f}")
        if "uncategorized" not in report.domain_accepted:
            known_domains = {"prose", "chat_or_call_log", "email", "system_log"}
            missing = known_domains - set(report.domain_accepted)
            if missing:
                print(f"  (no benchmark file yet for: {', '.join(sorted(missing))} - "
                      f"not measured, not assumed zero)")

    if report.shape_recall_accepted:
        _print_section("BY CANDIDATE SHAPE (recall only - see RecallBreakdown docstring)")
        for shape in sorted(report.shape_recall_accepted):
            ra = report.shape_recall_accepted[shape]
            rc = report.shape_recall_candidate[shape]
            print(f"  {shape:<20} accepted-only recall {ra.recall:.4f} ({ra.hits}/{ra.total})"
                  f"  |  +review recall {rc.recall:.4f} ({rc.hits}/{rc.total})")

    return 0


def _strip_quotes(line: str) -> str:
    """Strips one layer of matching leading/trailing quote characters.
    Windows Explorer's and PowerShell's "Copy as path" always wraps the
    copied path in double quotes - pasted straight into a manifest, that
    otherwise makes every single path silently fail its existence check
    with no visible reason why (Path('"C:\\\\foo.txt"') is never a real
    file, quotes included)."""
    if len(line) >= 2 and line[0] == line[-1] and line[0] in ('"', "'"):
        return line[1:-1]
    return line


def _looks_like_manifest(path: str) -> bool:
    """True if every non-comment, non-blank line in the file is itself a
    path to an existing file - i.e. this is a manifest listing other
    files to process, not a document to extract names from. Real text
    documents (chat logs, prose, etc.) essentially never satisfy this by
    coincidence: their lines are natural-language content, not
    filesystem paths, so this check is a reliable, cheap auto-detector
    that saves having to remember --input vs --input-list."""
    try:
        # utf-8-sig (not plain utf-8): strips a leading BOM if present -
        # real Windows-tool exports often write one, and a bare "utf-8"
        # decode leaves it stuck on the FIRST line's text, silently
        # breaking that line's path check (and nothing else's) with no
        # visible error.
        text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return False

    content_lines = [
        _strip_quotes(ln.strip()) for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not content_lines or len(content_lines) > 200:
        return False

    for line in content_lines:
        if len(line) > 260:  # longer than any real filesystem path can be
            return False
        try:
            if not Path(line).exists():
                return False
        except OSError:
            return False
    return True


def _looks_like_path(line: str) -> bool:
    """Structural-only check ('is this shaped like a filesystem path'),
    deliberately NOT checking existence - used by _auto_fix_manifest to
    decide whether a manifest line is a path merely pointing at the
    wrong location (worth searching for) vs. real document prose that
    just happens to fail an existence check for unrelated reasons."""
    return bool(re.match(r"^[A-Za-z]:[\\/]", line) or line.startswith("/") or line.startswith("\\\\"))


def _find_file_near(filename: str, search_root: Path, max_files_scanned: int = 20000) -> Path | None:
    """Looks for a file named `filename`, first directly inside
    search_root (the common case: the manifest and its referenced files
    sit in the same folder, only the path baked into the manifest is
    stale), then falls back to a bounded recursive walk under
    search_root. Bounded so a manifest sitting near a huge directory
    tree can't trigger a runaway scan."""
    direct = search_root / filename
    if direct.exists():
        return direct

    scanned = 0
    for root, _dirs, files in os.walk(search_root):
        for f in files:
            scanned += 1
            if scanned > max_files_scanned:
                return None
            if f == filename:
                return Path(root) / f
    return None


def _auto_fix_manifest(path: str) -> bool:
    """Rewrites a manifest in place when its listed paths point to
    filenames that don't exist at the listed location but DO exist
    elsewhere near the manifest itself - e.g. the manifest says
    '...\\DF-08-12-2026\\EF\\x.txt' but the actual file is sitting right
    next to the manifest in '...\\DF-05-01-2026\\EF\\', a stale path
    baked in at export time (seen repeatedly on real ProDiscover
    exports). Only rewrites the file when EVERY originally-missing line
    gets resolved; otherwise leaves it untouched and reports exactly
    what's still missing, so a genuine gap (files not yet provided at
    all) is never silently papered over."""
    manifest = Path(path)
    raw_lines = manifest.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    search_root = manifest.parent

    new_lines: list[str] = []
    fixes: list[tuple[str, str]] = []
    unresolved: list[str] = []

    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(raw_line)
            continue

        line = _strip_quotes(stripped)
        if Path(line).exists():
            # Quotes-only issue (e.g. "Copy as path") - no search needed,
            # just drop the quote characters themselves.
            if line != stripped:
                fixes.append((stripped, line))
            new_lines.append(line)
            continue

        if not _looks_like_path(line):
            new_lines.append(raw_line)
            continue

        found = _find_file_near(Path(line).name, search_root)
        if found is None:
            unresolved.append(line)
            new_lines.append(raw_line)
        else:
            fixes.append((stripped, str(found)))
            new_lines.append(str(found))

    if unresolved:
        print(f"Could not auto-fix manifest '{path}' - still missing:", file=sys.stderr)
        for line in unresolved:
            print(f"  - {line}  (no file named '{Path(line).name}' found under {search_root})", file=sys.stderr)
        return False

    if not fixes:
        return False  # nothing needed fixing - not actually a stale-path manifest

    manifest.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"Auto-fixed {len(fixes)} stale path(s) in manifest '{path}':")
    for old, new in fixes:
        print(f"  {old}\n    -> {new}")
    return True


def _read_input_list(list_path: str) -> list[str]:
    """Reads a manifest file of input paths, one per line. Blank lines
    and lines starting with '#' are ignored. Missing files are reported
    and skipped rather than aborting the whole batch."""
    manifest = Path(list_path)
    if not manifest.exists():
        raise FileNotFoundError(f"--input-list manifest not found: {list_path}")

    paths: list[str] = []
    for raw_line in manifest.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        line = _strip_quotes(stripped)
        if not Path(line).exists():
            print(f"WARNING: skipping missing file listed in manifest: {line}", file=sys.stderr)
            continue
        paths.append(line)
    return paths


def _common_case_number(input_paths: list[str]) -> str | None:
    """Extracts the shared leading digit-run (e.g. '930204590' out of
    '930204590_chat.txt') common to every input file's stem, so the
    combined batch output can be named after the actual case number
    instead of the generic 'all_names' - only fires when EVERY file in
    the batch shares the same numeric case ID; returns None (falls back
    to 'all_names.csv') for anything else, e.g. a manifest that mixes
    files from more than one case."""
    case_numbers: set[str] = set()
    for p in input_paths:
        match = re.match(r"^(\d+)_", Path(p).stem)
        if not match:
            return None
        case_numbers.add(match.group(1))
    return case_numbers.pop() if len(case_numbers) == 1 else None


def run_batch(input_paths: list[str], args: argparse.Namespace, config: dict, base_output: Path) -> int:
    failures = 0
    # Aggregates persons across every file in the batch, keyed by
    # normalized (lowercased) name, so the same person mentioned in
    # multiple source files collapses into one consolidated entry.
    # Includes REVIEW-bucket persons alongside ACCEPTED ones (not just
    # ACCEPTED as before) - each entry carries its own "decision" so a
    # human can still see which names are auto-trusted vs need a glance,
    # rather than review-bucket names being invisible from this combined
    # view entirely. "accepted" wins if a name is accepted in ANY file,
    # same merge rule AggregatedPerson.decision already uses within one
    # file - see refine_combined_counts' docstring.
    combined: dict[str, dict] = {}

    def _merge(person, decision: str, source_filename: str) -> None:
        key = person.normalized_text
        entry = combined.setdefault(key, {
            "display": person.display_text,
            "occurrences": 0,
            "files": [],
            "decision": "review",
        })
        entry["occurrences"] += person.occurrence_count
        if source_filename not in entry["files"]:
            entry["files"].append(source_filename)
        if decision == "accepted":
            entry["decision"] = "accepted"

    for input_path in input_paths:
        stem = Path(input_path).stem
        output_dir = base_output / stem
        result = run_extraction(input_path, output_dir, args, config)
        if not result.success:
            failures += 1
            continue
        source_filename = Path(input_path).name
        for person in result.persons:
            _merge(person, "accepted", source_filename)
        for person in result.review_persons:
            _merge(person, "review", source_filename)

    # Batch-level boundary refinement: a noise variant that didn't
    # dominate within any single file (e.g. "Testing Corvin Hale"
    # appearing twice in one email while "Corvin Hale" is common
    # case-wide across other files) still gets merged here, using
    # combined counts across every file in this batch.
    combined = refine_combined_counts(combined)

    ordered = sorted(combined.values(), key=lambda e: e["occurrences"], reverse=True)

    if getattr(args, "names_only", False):
        source = ordered if args.include_review else [e for e in ordered if e["decision"] == "accepted"]
        rows = [(e["display"], e["occurrences"], e["decision"]) for e in source]
        names_path = _write_names_only_csv(base_output, rows, include_decision=args.include_review)
        print(f"Names extracted : {len(rows)}")
        print(f"Saved to        : {names_path}")
        return 1 if failures else 0

    case_number = _common_case_number(input_paths)
    names_filename = f"{case_number}_output.csv" if case_number else "all_names.csv"
    names_path = base_output / names_filename
    names_path.parent.mkdir(parents=True, exist_ok=True)
    with names_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "total_occurrences", "decision", "source_files"])
        for e in ordered:
            writer.writerow([e["display"], e["occurrences"], e["decision"], "; ".join(e["files"])])

    _print_header("BATCH SUMMARY")
    print(f"  Files processed         : {len(input_paths)}")
    print(f"  Succeeded               : {len(input_paths) - failures}")
    print(f"  Failed                  : {failures}")
    print(f"  Unique names (combined) : {len(ordered)} "
          f"({sum(1 for e in ordered if e['decision'] == 'accepted')} accepted, "
          f"{sum(1 for e in ordered if e['decision'] == 'review')} review)")
    print(f"  Combined names file     : {names_path}")

    _print_section(f"ALL NAMES FOUND ACROSS {len(input_paths)} FILE(S) ({len(ordered)})")
    if ordered:
        name_width = max(len(e["display"]) for e in ordered)
        name_width = max(name_width, len("NAME"))
        print(f"  {'NAME':<{name_width}}  {'OCCURRENCES':>11}  DECISION   SOURCE FILE(S)")
        for e in ordered:
            print(f"  {e['display']:<{name_width}}  {e['occurrences']:>11}  {e['decision']:<9}  {'; '.join(e['files'])}")
    else:
        print("  (no persons found in any file)")

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PERSON_EXTRACTOR_V4 CLI")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input", help="Path to an input .txt file. If every line in the file "
                                               "is itself an existing file path, it's auto-detected as "
                                               "a manifest and batch-processed - no need for --input-list.")
    input_group.add_argument("--input-list", help="Path to a manifest file listing one input .txt "
                                                     "path per line; each is processed into its own "
                                                     "subfolder under --output. (--input now "
                                                     "auto-detects manifests too - this flag is kept "
                                                     "for explicitness/scripting.)")
    parser.add_argument("--output", default=None,
                         help="Output directory (default: ./output). With --input-list, each input "
                              "file gets its own subfolder named after its filename stem.")
    parser.add_argument("--format", choices=["csv", "json"], default="csv",
                         help="Structured export format in addition to the always-generated report.txt")
    parser.add_argument("--explain", action="store_true",
                         help="Print full evidence/explanation for each person")
    parser.add_argument("--evaluate", action="store_true",
                         help="Run the labeled benchmark suite and report precision/recall/F1")
    parser.add_argument("--benchmark-dir", default=None,
                        help="Directory of labeled TXT + *_gold.json pairs for --evaluate "
                             "(default: datasets/benchmark)")
    parser.add_argument("--no-feedback-log", action="store_true",
                         help="Don't log REVIEW-bucket candidates for active learning this run")
    parser.add_argument("--names-only", action="store_true",
                         help="Write ONLY a plain (name, occurrences) CSV, nothing else - no "
                              "report.txt, no evidence. A fresh timestamped file is written every "
                              "run, so repeated runs never overwrite each other's output.")
    parser.add_argument("--include-review", action="store_true",
                         help="Only affects --names-only: also include REVIEW-bucket names (not "
                              "just ACCEPTED) in the plain names CSV, each tagged with a 'decision' "
                              "column. Without this flag, --names-only stays accepted-only exactly "
                              "as before. The regular (non-names-only) batch combined-names output "
                              "and report.txt/terminal output already include both ACCEPTED and "
                              "REVIEW by default, always labeled with a decision column - this flag "
                              "has no effect there.")
    parser.add_argument("--auto-fix-manifest", action="store_true",
                         help="If --input looks like a manifest but its listed paths don't exist, "
                              "search for files with the same names near the manifest itself and "
                              "rewrite the manifest to point to them before running. Only rewrites "
                              "when EVERY listed file can be found this way; otherwise leaves the "
                              "manifest untouched and reports exactly what's still missing.")
    parser.add_argument("--loose-gate", action="store_true",
                         help="COMPARISON MODE, not for normal use: accepts a name based on "
                              "spaCy/context/repetition evidence alone, without requiring a "
                              "dictionary or title match. Trades precision for recall - use "
                              "alongside the default (strict) mode to compare results before "
                              "deciding which one you actually want.")
    parser.add_argument("--no-language-filter", action="store_true",
                         help="Disable the English-language gate (on by default) that skips "
                              "documents which aren't natural-language English prose - "
                              "non-English text, translated legal/EULA boilerplate, etc. - "
                              "instead of running detection on them and producing false "
                              "positives from cross-language word collisions.")
    parser.add_argument("--spacy-processes", type=int, default=1,
                         help="Run spaCy on up to N CPU processes within each document "
                              "(default 1). Only helps files over ~400KB; results are identical "
                              "either way. Try 4 for a single very large file.")
    args = parser.parse_args()
    if args.spacy_processes < 1:
        parser.error("--spacy-processes must be >= 1")

    if args.output and Path(args.output).exists() and Path(args.output).is_file():
        parser.error(
            f"--output must be a FOLDER, not a file: '{args.output}'\n"
            f"The tool creates report.txt/result.csv INSIDE the folder you give it - "
            f"it shouldn't end in .txt or any other file extension. For example:\n"
            f"  --output \"C:\\path\\to\\output\"          (correct - a folder)\n"
            f"  --output \"C:\\path\\to\\output\\results.txt\"  (wrong - looks like a file)"
        )

    configure_logging(BASE_DIR / "logs")

    if args.names_only:
        # --names-only means "nothing but the CSV" - the pipeline's own
        # per-stage INFO logging still goes to logs/pipeline.log in full
        # (nothing is lost for debugging), but console output is raised
        # to WARNING+ so a terminal run prints only the final CSV path.
        import logging
        root_logger = logging.getLogger("person_extractor_v4")
        for handler in root_logger.handlers:
            if not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.WARNING)

    if args.evaluate:
        return run_evaluate(args)

    if not args.input and not args.input_list:
        parser.error("--input or --input-list is required unless --evaluate is used")

    if args.input and not Path(args.input).exists():
        parser.error(f"--input file not found: '{args.input}'")

    config = _build_config(args)
    if args.loose_gate:
        print("*** --loose-gate: COMPARISON MODE (accepts spaCy/context-only evidence) ***")

    manifest_path = args.input_list
    if not manifest_path and args.input:
        if not _looks_like_manifest(args.input) and args.auto_fix_manifest:
            _auto_fix_manifest(args.input)
        if _looks_like_manifest(args.input):
            print(f"'{args.input}' looks like a manifest of file paths (every line is an existing "
                  f"file), not a document - switching to batch mode automatically.")
            manifest_path = args.input

    if manifest_path:
        try:
            input_paths = _read_input_list(manifest_path)
        except FileNotFoundError as exc:
            parser.error(str(exc))
        if not input_paths:
            print("No valid input files found in manifest.", file=sys.stderr)
            return 1
        if args.output:
            base_output = Path(args.output)
        else:
            base_output = BASE_DIR / "output" / Path(manifest_path).stem
        return run_batch(input_paths, args, config, base_output)

    output_dir = Path(args.output) if args.output else BASE_DIR / "output"
    result = run_extraction(args.input, output_dir, args, config)
    if not result.success:
        return 1

    if args.names_only:
        rows = [(p.display_text, p.occurrence_count, "accepted") for p in result.persons]
        if args.include_review:
            rows += [(p.display_text, p.occurrence_count, "review") for p in result.review_persons]
            rows.sort(key=lambda r: r[1], reverse=True)
        names_path = _write_names_only_csv(output_dir, rows, include_decision=args.include_review)
        print(f"Names extracted : {len(rows)}")
        print(f"Saved to        : {names_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
