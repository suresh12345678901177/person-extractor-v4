"""
src.utils.timing
==================
Dedicated timing utilities. This exists as its own module (rather than
folded into the logger) because "time taken for the task" is an explicit
output requirement, not just an internal debugging aid - StageTimer
produces the StageTiming objects that flow all the way into the final
CLI report and JSON export.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from src.core.models import StageTiming
from src.utils.logger import get_logger

logger = get_logger("utils.timing")


class Stopwatch:
    """Simple wall-clock stopwatch. `elapsed` is safe to read while running."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def reset(self) -> None:
        self._start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start


@contextmanager
def timed_stage(stage_name: str, collector: list[StageTiming]) -> Iterator[None]:
    """Context manager that times a pipeline stage, logs start/end, and
    appends a StageTiming to `collector` - the single source of truth
    for the "time taken per stage" reporting used by the CLI and exports."""
    logger.info("STAGE START: %s", stage_name)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("STAGE ERROR: %s", stage_name)
        raise
    finally:
        elapsed = time.perf_counter() - start
        collector.append(StageTiming(stage_name=stage_name, seconds=elapsed))
        logger.info("STAGE END: %s (%.4fs)", stage_name, elapsed)
