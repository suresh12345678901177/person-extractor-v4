"""
src.candidate.candidate_aggregator
=====================================
Groups individual CandidateResult mentions into one AggregatedPerson per
UNIQUE normalized surface form - this is what produces the "how many
times found" and "all locations" output requirements.

Deliberately grouped by EXACT normalized text (case-insensitive), not by
any attempt at name/alias resolution ("Suresh" vs "Dr. Suresh Kumar" vs
"S. Kumar" are kept as three separate entries). Two reasons:
  1. Honesty: alias/coreference resolution is a hard NLP problem this
     rules engine does not attempt - silently merging surface forms
     would overstate the system's capability.
  2. Safety: merging different surface forms risks conflating two
     DIFFERENT people who happen to share a first name, which is a
     worse error for a "no false positives" tool than under-merging.
This is documented in the README as a deliberate, not accidental, scope
boundary.
"""

from __future__ import annotations

from src.core.models import AggregatedPerson, CandidateResult


def aggregate_by_surface_form(candidates: list[CandidateResult]) -> list[AggregatedPerson]:
    """Group candidates by normalized_text (case-insensitive key), most
    frequent surface-form casing wins as the display text."""
    groups: dict[str, list[CandidateResult]] = {}

    for c in candidates:
        key = c.candidate.normalized_text.lower()
        groups.setdefault(key, []).append(c)

    aggregated: list[AggregatedPerson] = []
    for key, mentions in groups.items():
        surface_forms: dict[str, int] = {}
        for m in mentions:
            surface_forms[m.candidate.normalized_text] = surface_forms.get(m.candidate.normalized_text, 0) + 1
        display_text = max(surface_forms.items(), key=lambda kv: kv[1])[0]

        mentions_sorted = sorted(mentions, key=lambda m: m.candidate.start)
        aggregated.append(AggregatedPerson(
            normalized_text=key,
            display_text=display_text,
            mentions=mentions_sorted,
        ))

    # Most-frequent-first ordering makes the CLI report immediately
    # useful without extra sorting by the caller.
    aggregated.sort(key=lambda p: p.occurrence_count, reverse=True)
    return aggregated
