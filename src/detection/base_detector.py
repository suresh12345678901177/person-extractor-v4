"""
src.detection.base_detector
=============================
Base interface for the plugin-based detection layer. Every detector
consumes a page of text and produces zero or more Detection objects.
New detectors can be added without modifying existing ones - they just
register with DetectorManager (open/closed principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.models import Detection, DetectorName


class BaseDetector(ABC):
    """A detector finds candidate person-name spans in a block of text."""

    name: DetectorName

    @abstractmethod
    def detect(self, text: str, page_index: int) -> list[Detection]:
        """Return all Detection spans found in `text`. Must not raise for
        malformed/empty input - return an empty list instead."""
        raise NotImplementedError
