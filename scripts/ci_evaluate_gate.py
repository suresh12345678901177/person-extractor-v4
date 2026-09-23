"""
scripts/ci_evaluate_gate.py
==============================
CI-only pass/fail wrapper around `--evaluate` (independent-audit finding
#7 / Tier 3b): the two real production regressions documented in
README's "Real measured results" (2026-09-16, 2026-09-22) were both
caught only because someone remembered to run `--evaluate` by hand
afterward - the same failure mode `retrain_from_feedback.py`'s own
promotion guardrail (MAX_ACCEPTABLE_SHAPE_RECALL_DROP) already exists to
close for a single retrain. This applies the identical idea at the CI
level so it doesn't depend on a human remembering: fails the build if
accepted-only F1 on datasets/benchmark/ drops more than
MAX_ACCEPTABLE_F1_DROP below MIN_ACCEPTED_ONLY_F1.

MIN_ACCEPTED_ONLY_F1 is a floor, not a moving baseline read from a state
file - deliberately, so CI doesn't need a committed "last known F1"
number that could itself drift out of sync. Per README's own sample-size
caveat (N=7 benchmark files, printed by `--evaluate` itself), bump this
constant by hand when a real, verified improvement raises the number -
never lower it to make a failing build pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import DEFAULT_CONFIG
from src.evaluation.evaluator import run_benchmark
from src.pipeline.orchestrator import Pipeline

# Current live accepted-only F1 (2026-09-23 closing rescan): 0.8633.
# Floor set 0.02 below that - the same tolerance
# retrain_from_feedback.py's MAX_ACCEPTABLE_SHAPE_RECALL_DROP uses -
# so normal sample-size noise (see the --evaluate sample-size caveat)
# doesn't fail CI, but a real regression of the size already seen twice
# in this project's history does.
MIN_ACCEPTED_ONLY_F1 = 0.84


def main() -> int:
    pipeline = Pipeline(config=DEFAULT_CONFIG, base_dir=BASE_DIR)
    report = run_benchmark(pipeline, BASE_DIR / "datasets" / "benchmark")

    if not report.file_results:
        print("CI EVALUATE GATE: FAIL - no benchmark files found in datasets/benchmark/.")
        return 1

    f1 = report.overall_accepted.f1
    n_files = len(report.file_results)
    print(f"CI EVALUATE GATE: accepted-only F1 = {f1:.4f} (N={n_files} benchmark files, "
          f"floor = {MIN_ACCEPTED_ONLY_F1:.4f})")

    if f1 < MIN_ACCEPTED_ONLY_F1:
        print(f"CI EVALUATE GATE: FAIL - F1 {f1:.4f} is below the floor {MIN_ACCEPTED_ONLY_F1:.4f}. "
              f"This blocks the build - see README's 'Real measured results' for the process this "
              f"exists to enforce (a real-corpus rescan is still required before trusting a claimed "
              f"GAIN, per the 2026-09-22 process lesson - this gate only catches regressions).")
        return 1

    print("CI EVALUATE GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
