"""
src.preprocessing.line_indexer
=================================
Converts a character offset into a 1-indexed (line, column) pair. This
exists specifically to satisfy the "clear location labeling" output
requirement: a raw character offset like "start=4821" means nothing to
someone reviewing results, but "line 47, column 12" lets them jump
straight to the spot in a text editor.

Built as a proper index (not repeated linear scans) so it stays fast
even on large files: LineIndex.build() is O(n) once per document, and
every subsequent locate() call is O(log n) via binary search.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from src.core.models import TextLocation


@dataclass(slots=True)
class LineIndex:
    """Precomputed index of line-start offsets for one document's text."""
    _line_start_offsets: list[int]

    @classmethod
    def build(cls, text: str) -> "LineIndex":
        offsets = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                offsets.append(i + 1)
        return cls(_line_start_offsets=offsets)

    def locate(self, char_start: int, char_end: int) -> TextLocation:
        """Return the 1-indexed line/column for a character span. Uses
        bisect for O(log n) lookup regardless of document size."""
        line_idx = bisect.bisect_right(self._line_start_offsets, char_start) - 1
        line_idx = max(0, min(line_idx, len(self._line_start_offsets) - 1))
        line_start_offset = self._line_start_offsets[line_idx]
        column = char_start - line_start_offset + 1
        return TextLocation(
            char_start=char_start,
            char_end=char_end,
            line_number=line_idx + 1,
            column_number=column,
        )
