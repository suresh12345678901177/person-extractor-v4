"""
src.detection.detector_manager
================================
DetectorManager loads the set of enabled detectors (per config) and runs
all of them against a page of text, collecting every Detection produced.
New detectors register here without any other code needing to change
(open/closed principle, per the blueprint's Plugin Architecture).
"""

from __future__ import annotations

from typing import Any

from src.core.models import Detection
from src.detection.base_detector import BaseDetector
from src.detection.dictionary_detector import DictionaryDetector
from src.detection.regex_detector import RegexDetector
from src.knowledge.knowledge_base import KnowledgeBase
from src.utils.logger import get_logger

logger = get_logger("detection.detector_manager")


class DetectorManager:
    def __init__(self, config: dict[str, Any], knowledge_base: KnowledgeBase) -> None:
        self.detectors: list[BaseDetector] = []
        detection_cfg = config.get("detection", {})

        if detection_cfg.get("use_regex", True):
            self.detectors.append(RegexDetector())

        if detection_cfg.get("use_dictionary", True):
            self.detectors.append(DictionaryDetector(knowledge_base))

        if detection_cfg.get("use_spacy", True):
            try:
                from src.detection.spacy_detector import SpacyDetector
                self.detectors.append(SpacyDetector())
            except RuntimeError as exc:
                logger.warning(
                    "use_spacy is enabled but SpacyDetector could not be "
                    "loaded (%s) - continuing without it.", exc
                )

        logger.info("DetectorManager initialized with detectors: %s",
                    [d.name.value for d in self.detectors])

    def detect_all(self, text: str, page_index: int) -> list[Detection]:
        results: list[Detection] = []
        for detector in self.detectors:
            try:
                results.extend(detector.detect(text, page_index))
            except Exception:  # noqa: BLE001 - one bad detector must not kill the run
                logger.exception("Detector %s failed on page %d", detector.name, page_index)
        return results
