"""src.export.json_exporter - writes grouped-by-person results to JSON,
including full per-mention evidence traces for explainability."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.models import AggregatedPerson, CandidateResult
from src.export.base_exporter import BaseExporter


class JsonExporter(BaseExporter):
    extension = "json"

    def export(
        self,
        persons: list[AggregatedPerson],
        review_persons: list[AggregatedPerson],
        rejected: list[CandidateResult],
        output_path: str | Path,
        source_path: str = "",
        run_info: dict | None = None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "source_file": source_path,
            "provenance": run_info or {},
            "accepted": [self._full_person(p, source_path) for p in persons],
            "review": [self._full_person(p, source_path) for p in review_persons],
            "rejected": [self._rejected_row(c, source_path) for c in rejected],
        }

        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    def _full_person(self, person: AggregatedPerson, source_path: str) -> dict:
        row = self._person_row(person, source_path)
        row["mentions"] = [
            {
                "text": m.candidate.text,
                "location": str(m.candidate.location) if m.candidate.location else None,
                "confidence": m.state.final_confidence,
                "decision": m.state.decision.value if m.state.decision else None,
                "detectors": list(m.candidate.detector_names),
                "evidence": [
                    {"source": e.source, "label": e.label, "score": e.score}
                    for e in m.state.evidence
                ],
            }
            for m in person.mentions
        ]
        return row

    def _rejected_row(self, candidate: CandidateResult, source_path: str) -> dict:
        c, s = candidate.candidate, candidate.state
        return {
            "source_file": source_path,
            "text": c.text,
            "location": str(c.location) if c.location else None,
            "rejection_reason": s.rejection_reason or "",
        }
