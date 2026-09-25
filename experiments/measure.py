"""
experiments/measure.py
=========================
One measurement for the self-upgrade experiment ledger - the same numbers,
measured the same way, for the baseline and every candidate change:

  - test suite result
  - datasets/benchmark/ (N=7 files): accepted-only and accepted+review
    P/R/F1, per-domain F1, per-shape recall, accepted false positives on
    benchmark_structured_dump_005.txt
  - datasets/benchmark_unseen/ (held out - measured, never tuned on)
  - per-stage pipeline timing on the benchmark files (median of 3 passes)
  - real-time latency: every line of benchmark_real_660.txt sent through
    server.py's HTTP handler as one conversation
  - hardware / device in use

Results go to experiments/results/<id>.json; experiments/ledger.csv and
ledger.md are rebuilt from every results file each run.

The full real-case scan (scripts/scan_directory.py, ~45 min) is run
separately so it never overlaps the timing measurements, then attached
with --scan-summary. Only its COUNTS are recorded - never names, paths or
any other case content (this ledger is committed).

Feedback logging is off throughout: measuring must not add names to
datasets/feedback/pending_review.jsonl.

Usage:
    python experiments/measure.py --id baseline --description "..."
    python experiments/measure.py --id baseline --scan-summary <summary.csv> --skip-measure
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import DEFAULT_CONFIG
from src.evaluation.evaluator import run_benchmark
from src.pipeline.orchestrator import Pipeline
from src.utils.provenance import build_run_provenance

EXPERIMENTS_DIR = BASE_DIR / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
BENCHMARK_DIR = BASE_DIR / "datasets" / "benchmark"
UNSEEN_DIR = BASE_DIR / "datasets" / "benchmark_unseen"
LATENCY_FILE = BENCHMARK_DIR / "benchmark_real_660.txt"
TIMING_PASSES = 3

# Numeric fields taken from a scan_directory.py summary CSV - counts only.
SCAN_FIELDS = (
    "total_files_in_folder", "files_processed_ok", "files_failed", "files_excluded",
    "unique_names_accepted", "unique_names_review", "discovery_seconds", "extraction_seconds",
)

LEDGER_COLUMNS = [
    "id", "measured_utc", "git_commit", "git_dirty", "model_sha", "device", "tests",
    "acc_P", "acc_R", "acc_F1", "rev_P", "rev_R", "rev_F1",
    "shape_single", "shape_multi", "shape_dict", "shape_unseen",
    "dom_chat_F1", "dom_email_F1", "dom_prose_F1", "dump005_acc_FP",
    "unseen_acc_P", "unseen_acc_R",
    "bench_seconds", "server_median_ms", "server_p95_ms",
    "scan_seconds", "scan_accepted", "scan_review",
    "verdict", "description",
]


def _quiet_console() -> None:
    """Every stage logs at INFO; keep the file log, quiet the console."""
    for handler in logging.getLogger("person_extractor_v4").handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.WARNING)


def _config() -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["feedback"]["enabled"] = False
    return config


def _metrics(m) -> dict:
    return {"P": m.precision, "R": m.recall, "F1": m.f1,
            "TP": m.true_positives, "FP": m.false_positives, "FN": m.false_negatives, "gold": m.gold_total}


def benchmark_results(pipeline: Pipeline, bench_dir: Path) -> dict:
    report = run_benchmark(pipeline, bench_dir)
    return {
        "files": len(report.file_results),
        "accepted": _metrics(report.overall_accepted),
        "accepted_plus_review": _metrics(report.overall_candidate),
        "domains_accepted": {d: _metrics(m) for d, m in report.domain_accepted.items()},
        "domains_accepted_plus_review": {d: _metrics(m) for d, m in report.domain_candidate.items()},
        "shape_recall_accepted": {s: b.as_dict() for s, b in report.shape_recall_accepted.items()},
        "shape_recall_accepted_plus_review": {s: b.as_dict() for s, b in report.shape_recall_candidate.items()},
        "per_file_accepted": {f.name: _metrics(f.accepted_metrics) for f in report.file_results},
    }


def stage_timing(pipeline: Pipeline) -> dict:
    """Median over TIMING_PASSES of each stage's time summed across all
    benchmark files, plus the median total."""
    files = sorted(BENCHMARK_DIR.glob("*.txt"))
    passes: list[dict[str, float]] = []
    for _ in range(TIMING_PASSES):
        stages: dict[str, float] = {}
        total = 0.0
        for path in files:
            result = pipeline.run(path)
            for st in result.statistics.stage_timings:
                stages[st.stage_name] = stages.get(st.stage_name, 0.0) + st.seconds
            total += result.statistics.total_seconds
        stages["TOTAL"] = total
        passes.append(stages)
    names = sorted({k for p in passes for k in p})
    return {name: round(statistics.median(p.get(name, 0.0) for p in passes), 4) for name in names}


def server_latency(pipeline: Pipeline) -> dict:
    from server import make_server
    from src.realtime.service import RealtimeService

    server = make_server(RealtimeService(pipeline), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/sessions/latency/messages"
    lines = [ln for ln in LATENCY_FILE.read_text(encoding="utf-8").split("\n") if ln.strip()]
    timings_ms = []
    try:
        for line in lines:
            body = json.dumps({"text": line}).encode("utf-8")
            request = urllib.request.Request(url, data=body, method="POST",
                                             headers={"Content-Type": "application/json"})
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=60) as response:
                response.read()
            timings_ms.append((time.perf_counter() - started) * 1000)
    finally:
        server.shutdown()
        server.server_close()
    ordered = sorted(timings_ms)
    return {
        "messages": len(ordered),
        "median_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 2),
        "max_ms": round(ordered[-1], 2),
    }


def run_tests() -> dict:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
        cwd=BASE_DIR, capture_output=True, encoding="utf-8", errors="replace",
    )
    summary = next((ln for ln in reversed(proc.stdout.splitlines()) if " in " in ln and ("passed" in ln or "failed" in ln)), "")
    return {"ok": proc.returncode == 0, "summary": summary.strip("= "), "seconds": round(time.perf_counter() - started, 1)}


def hardware() -> dict:
    info = {"cpu_logical_cores": os.cpu_count(), "device_used": "cpu"}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
            capture_output=True, encoding="utf-8", timeout=15, check=True,
        ).stdout.strip()
        info["gpu"] = out
    except Exception:  # noqa: BLE001 - no NVIDIA driver is a valid answer
        info["gpu"] = None
    try:
        import spacy
        info["spacy_gpu_available"] = bool(spacy.prefer_gpu())
        if info["spacy_gpu_available"]:
            spacy.require_cpu()  # measuring the pipeline as configured, not switching it
    except Exception:  # noqa: BLE001
        info["spacy_gpu_available"] = False
    return info


def scan_counts(summary_csv: Path) -> dict:
    with summary_csv.open(encoding="utf-8") as fh:
        rows = {r["metric"]: r["value"] for r in csv.DictReader(fh)}
    return {k: float(rows[k]) for k in SCAN_FIELDS if k in rows}


def _flatten(result: dict) -> dict:
    b = result.get("benchmark", {})
    acc, rev = b.get("accepted", {}), b.get("accepted_plus_review", {})
    shapes = b.get("shape_recall_accepted", {})
    doms = b.get("domains_accepted", {})
    unseen = result.get("unseen", {}).get("accepted", {})
    scan = result.get("scan", {})
    prov = result.get("provenance", {})
    server = result.get("server_latency", {})
    return {
        "id": result["id"],
        "measured_utc": result.get("measured_utc", ""),
        "git_commit": (prov.get("git_commit") or "")[:10],
        "git_dirty": prov.get("git_dirty_file_count", ""),
        "model_sha": prov.get("model_sha256_short", ""),
        "device": result.get("hardware", {}).get("device_used", ""),
        "tests": result.get("tests", {}).get("summary", ""),
        "acc_P": acc.get("P", ""), "acc_R": acc.get("R", ""), "acc_F1": acc.get("F1", ""),
        "rev_P": rev.get("P", ""), "rev_R": rev.get("R", ""), "rev_F1": rev.get("F1", ""),
        "shape_single": shapes.get("single_token", {}).get("recall", ""),
        "shape_multi": shapes.get("multi_token", {}).get("recall", ""),
        "shape_dict": shapes.get("dictionary_known", {}).get("recall", ""),
        "shape_unseen": shapes.get("unseen_name", {}).get("recall", ""),
        "dom_chat_F1": doms.get("chat_or_call_log", {}).get("F1", ""),
        "dom_email_F1": doms.get("email", {}).get("F1", ""),
        "dom_prose_F1": doms.get("prose", {}).get("F1", ""),
        "dump005_acc_FP": b.get("per_file_accepted", {}).get("benchmark_structured_dump_005.txt", {}).get("FP", ""),
        "unseen_acc_P": unseen.get("P", ""), "unseen_acc_R": unseen.get("R", ""),
        "bench_seconds": result.get("stage_timing", {}).get("TOTAL", ""),
        "server_median_ms": server.get("median_ms", ""), "server_p95_ms": server.get("p95_ms", ""),
        "scan_seconds": round(scan["extraction_seconds"] + scan.get("discovery_seconds", 0), 1) if scan else "",
        "scan_accepted": int(scan["unique_names_accepted"]) if scan else "",
        "scan_review": int(scan["unique_names_review"]) if scan else "",
        "verdict": result.get("verdict", ""),
        "description": result.get("description", ""),
    }


def rebuild_ledger() -> None:
    results = sorted(
        (json.loads(p.read_text(encoding="utf-8")) for p in RESULTS_DIR.glob("*.json")),
        key=lambda r: r.get("measured_utc", ""),
    )
    rows = [_flatten(r) for r in results]
    with (EXPERIMENTS_DIR / "ledger.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    shown = ["id", "git_commit", "acc_P", "acc_R", "acc_F1", "rev_P", "rev_R", "shape_unseen",
             "dump005_acc_FP", "unseen_acc_P", "unseen_acc_R", "bench_seconds", "server_median_ms",
             "scan_seconds", "scan_accepted", "scan_review", "verdict"]
    lines = [
        "# Self-upgrade experiment ledger",
        "",
        "Generated by `experiments/measure.py` from `experiments/results/*.json` - do not edit by hand.",
        "Benchmark numbers are on N=7 files (`datasets/benchmark/`); swings under ~2pp are noise.",
        "`benchmark_unseen` is held out: measured, never tuned on. Scan columns are counts from a",
        "full real-case scan (no case content is recorded here).",
        "",
        "| " + " | ".join(shown) + " |",
        "|" + "---|" * len(shown),
    ]
    lines += ["| " + " | ".join(str(row[c]) for c in shown) + " |" for row in rows]
    lines += ["", "## Descriptions", ""]
    lines += [f"- **{row['id']}** ({row['measured_utc']}): {row['description']}" for row in rows]
    (EXPERIMENTS_DIR / "ledger.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True, help="Experiment id, e.g. baseline, exp01-calibration")
    parser.add_argument("--description", default=None)
    parser.add_argument("--verdict", default=None, help="baseline | kept | rolled back | needs owner decision")
    parser.add_argument("--scan-summary", type=Path, default=None,
                        help="scan_directory.py *_summary_*.csv to attach (counts only)")
    parser.add_argument("--skip-measure", action="store_true",
                        help="Only attach --scan-summary / update fields and rebuild the ledger")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{args.id}.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {"id": args.id}
    if args.description is not None:
        result["description"] = args.description
    if args.verdict is not None:
        result["verdict"] = args.verdict

    if not args.skip_measure:
        _quiet_console()
        config = _config()
        model_path = BASE_DIR / config["classification"]["model_path"]
        result["measured_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        result["provenance"] = build_run_provenance(BASE_DIR, model_path)
        result["hardware"] = hardware()
        print("tests ...", flush=True)
        result["tests"] = run_tests()
        print(f"  {result['tests']['summary']}", flush=True)
        pipeline = Pipeline(config=config, base_dir=BASE_DIR)
        print("benchmark ...", flush=True)
        result["benchmark"] = benchmark_results(pipeline, BENCHMARK_DIR)
        print("benchmark_unseen ...", flush=True)
        result["unseen"] = benchmark_results(pipeline, UNSEEN_DIR)
        print("stage timing ...", flush=True)
        result["stage_timing"] = stage_timing(pipeline)
        print("server latency ...", flush=True)
        result["server_latency"] = server_latency(pipeline)

    if args.scan_summary is not None:
        result["scan"] = scan_counts(args.scan_summary)

    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rebuild_ledger()
    print(json.dumps(_flatten(result), indent=2))
    return 0 if result.get("tests", {}).get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
