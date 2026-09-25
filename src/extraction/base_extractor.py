"""
src.extraction.base_extractor
================================
The extension point this project is built around: one `Extractor`
subclass owns everything needed to pull ONE kind of entity (person
names today; phone numbers, ID numbers, etc. later) out of cleaned
text - its own detectors, validators, features, classifier, and
decision logic, if it needs any of those at all.

Deliberately a SINGLE abstract method, not a rigid detect/validate/
classify/decide skeleton. `PersonExtractor` needs all of those stages;
a future phone-number or ID extractor plausibly needs only a regex
detector and a format check, with no knowledge base and no ML step -
forcing it to implement unused abstract methods just to satisfy an
interface built around the person pipeline's shape would be the wrong
kind of abstraction. `Pipeline` (src.pipeline.orchestrator) only ever
needs to call `extract(...)` and get an `ExtractionResult` back; how a
given extractor gets there is entirely its own business.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.core.models import ExtractionResult, PipelineStatistics, SourceFormat
from src.preprocessing.line_indexer import LineIndex


class Extractor(ABC):
    def __init__(self, config: dict[str, Any], base_dir: str | Path) -> None:
        self.config = config
        self.base_dir = Path(base_dir)

    @abstractmethod
    def extract(
        self,
        cleaned_text: str,
        line_index: LineIndex,
        input_path: str,
        stats: PipelineStatistics,
        source_format: SourceFormat | None = None,
    ) -> ExtractionResult:
        """Runs this extractor's own stage sequence over already-read,
        already-cleaned document text and returns a fully-populated
        ExtractionResult.

        `stats` is a single PipelineStatistics object shared with the
        calling Pipeline, which has already timed its own io_read_document
        and preprocessing stages into it - implementations should keep
        appending their own stage timings to `stats.stage_timings` (e.g.
        via `src.utils.timing.timed_stage`) the same way, but must NEVER
        set `stats.total_seconds` themselves. Pipeline sets that once,
        immediately after this method returns, so it always reflects the
        whole run (io + preprocessing + extraction) rather than just this
        extractor's slice - important once more than one extractor can be
        timed into the same run.
        """
        raise NotImplementedError
