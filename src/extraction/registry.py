"""
src.extraction.registry
==========================
Maps a short config/CLI-facing name ("person", eventually "phone_number",
"id_number", ...) to the Extractor class that implements it. This is the
actual "add a new extraction type easily" mechanism the whole
src.extraction package exists for: a new extractor is one new class
(src.extraction.base_extractor.Extractor subclass) plus one new line
here - nothing else in this file, orchestrator.py, or person_extractor.py
needs to change.
"""

from __future__ import annotations

from src.extraction.base_extractor import Extractor
from src.extraction.person_extractor import PersonExtractor

EXTRACTOR_REGISTRY: dict[str, type[Extractor]] = {
    "person": PersonExtractor,
}


def get_extractor(name: str) -> type[Extractor]:
    try:
        return EXTRACTOR_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(EXTRACTOR_REGISTRY))
        raise ValueError(f"Unknown extraction type '{name}' - available: {available}") from None
