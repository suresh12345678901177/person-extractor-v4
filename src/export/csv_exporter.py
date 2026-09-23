"""src.export.csv_exporter - writes grouped-by-person results to CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.core.models import AggregatedPerson, CandidateResult
from src.export.base_exporter import BaseExporter

_FIELDS = [
    "source_file", "person_name", "occurrence_count", "decision",
    "best_confidence", "locations", "why_detected",
]


class CsvExporter(BaseExporter):
    extension = "csv"

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

        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FIELDS)
            writer.writeheader()
            for person in list(persons) + list(review_persons):
                writer.writerow(self._person_row(person, source_path))

        if run_info:
            # A per-row CSV has no room for run-level metadata without
            # breaking its fixed schema - a sidecar is the least invasive
            # fit (independent-audit finding #3). Same directory, same
            # basename, so it's never separated from the CSV it describes.
            sidecar_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
            sidecar_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")

        return output_path
