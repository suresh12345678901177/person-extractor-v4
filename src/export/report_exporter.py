"""
src.export.report_exporter
=============================
Human-readable, clearly-labeled plain-text report - built specifically
to satisfy the "clear labeling" output requirement in a format that's
easy to skim without opening a CSV/JSON viewer: every person gets a
labeled block showing occurrence count, every location, and the exact
reason each mention was accepted.
"""

from __future__ import annotations

from pathlib import Path

from src.core.models import AggregatedPerson, CandidateResult, ExtractionResult
from src.utils.provenance import provenance_summary_line


class ReportExporter:
    extension = "txt"

    def export(self, result: ExtractionResult, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        lines.append("=" * 78)
        lines.append("PERSON_EXTRACTOR_V4 - EXTRACTION REPORT")
        lines.append("=" * 78)
        lines.append(f"Source file        : {result.source_path}")
        lines.append(f"Status              : {'SUCCESS' if result.success else 'FAILED'}")
        if result.model_info.get("git_commit"):
            lines.append(f"Provenance          : {provenance_summary_line(result.model_info)}")
            lines.append("  (a stale result from before a code/model fix is now self-identifying -")
            lines.append("   see README's 'Independent-audit findings' section)")
        if not result.success:
            lines.append(f"Error               : {result.error}")
            output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return output_path

        stats = result.statistics
        lines.append("")
        lines.append("-" * 78)
        lines.append("TIME TAKEN")
        lines.append("-" * 78)
        for st in stats.stage_timings:
            lines.append(f"  {st.stage_name:<32} {st.seconds:>8.4f} sec")
        lines.append(f"  {'TOTAL':<32} {stats.total_seconds:>8.4f} sec")

        lines.append("")
        lines.append("-" * 78)
        lines.append("SUMMARY")
        lines.append("-" * 78)
        lines.append(f"  Unique persons found (accepted)   : {len(result.persons)}")
        lines.append(f"  Unique persons flagged for review : {len(result.review_persons)}")
        caveat = result.model_info.get("identity_resolution_caveat")
        if caveat:
            lines.append(f"  NOTE: {caveat}")
        lines.append(f"  Total mentions (accepted+review)  : "
                      f"{sum(p.occurrence_count for p in result.persons) + sum(p.occurrence_count for p in result.review_persons)}")
        lines.append(f"  Rejected candidate spans          : {len(result.rejected)}")
        if result.model_info:
            lines.append("")
            lines.append("  Model configuration:")
            for k, v in result.model_info.items():
                lines.append(f"    {k:<34}: {v}")

        lines.append("")
        lines.append("-" * 78)
        lines.append(f"ACCEPTED PERSONS ({len(result.persons)})")
        lines.append("-" * 78)
        for person in result.persons:
            lines.extend(self._person_block(person))

        if result.review_persons:
            lines.append("")
            lines.append("-" * 78)
            lines.append(f"FLAGGED FOR REVIEW ({len(result.review_persons)}) - weak/ambiguous evidence")
            lines.append("-" * 78)
            for person in result.review_persons:
                lines.extend(self._person_block(person))

        if result.low_confidence_persons:
            lines.append("")
            lines.append("-" * 78)
            lines.append(
                f"LOW-CONFIDENCE, AUTO-REJECTED ({len(result.low_confidence_persons)}) - "
                f"spot-check only, NOT counted as accepted/review"
            )
            lines.append("-" * 78)
            lines.append(
                "  These are name-shaped spans with ZERO dictionary, title, or spaCy "
                "corroboration - automatically rejected outright rather than sent for "
                "review, to keep precision high. Most of these are genuinely not names. "
                "Shown here only so a real person the dictionary doesn't know can still "
                "be caught by manual review - they do not appear in any CSV/JSON export "
                "or count toward the accepted/review totals above."
            )
            for person in result.low_confidence_persons:
                lines.extend(self._person_block(person))

        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    def _person_block(self, person: AggregatedPerson) -> list[str]:
        lines = []
        lines.append("")
        lines.append(f"  NAME            : {person.display_text}")
        lines.append(f"  OCCURRENCES     : {person.occurrence_count} time(s)")
        lines.append(f"  DECISION        : {person.decision.value if person.decision else 'n/a'}")
        lines.append(f"  BEST CONFIDENCE : {person.best_confidence:.2f}")
        lines.append(f"  LOCATIONS       :")
        for loc in person.locations:
            lines.append(f"      - {loc}")
        lines.append(f"  WHY DETECTED (first mention's evidence):")
        if person.mentions:
            for e in person.mentions[0].state.evidence:
                sign = "+" if e.score >= 0 else ""
                lines.append(f"      {e.label:<42} {sign}{e.score:.2f}  [{e.source}]")
        return lines
