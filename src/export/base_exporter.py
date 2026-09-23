"""
src.export.base_exporter
===========================
Base interface for exporters. Each exporter writes the pipeline's
grouped-by-person results (accepted, review, rejected) to a file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.core.models import AggregatedPerson, CandidateResult


class BaseExporter(ABC):
    extension: str = ""

    @abstractmethod
    def export(
        self,
        persons: list[AggregatedPerson],
        review_persons: list[AggregatedPerson],
        rejected: list[CandidateResult],
        output_path: str | Path,
        source_path: str = "",
        run_info: dict | None = None,
    ) -> Path:
        """`run_info` is `ExtractionResult.model_info` (or an equivalent
        dict) - includes the provenance fields from
        `src.utils.provenance.build_run_provenance` (git commit/dirty
        state, model file identity, scan timestamp). Optional and
        defaults to None so existing callers that only care about the
        person rows are unaffected - see independent-audit finding #3."""
        raise NotImplementedError

    @staticmethod
    def _person_row(person: AggregatedPerson, source_path: str) -> dict:
        best_mention = max(person.mentions, key=lambda m: m.state.final_confidence)
        locations = "; ".join(str(loc) for loc in person.locations)
        reasons = "; ".join(
            f"{e.label} ({e.source})" for e in best_mention.state.evidence
        )
        return {
            "source_file": source_path,
            "person_name": person.display_text,
            "occurrence_count": person.occurrence_count,
            "decision": person.decision.value if person.decision else "",
            "best_confidence": best_mention.state.final_confidence,
            "locations": locations,
            "why_detected": reasons,
        }
