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
from dataclasses import dataclass, field

from src.core.models import TextLocation


@dataclass(slots=True)
class LineIndex:
    """Precomputed index of line-start offsets for one document's text."""
    _line_start_offsets: list[int]
    # Char offset where each PDF page starts in the joined text, and the
    # 0-based line index of that offset - so a position can be reported as
    # "page P, line L of that page". Empty for single-stream text.
    _page_start_offsets: list[int] = field(default_factory=list)
    _page_first_line: list[int] = field(default_factory=list)

    @classmethod
    def build(cls, text: str, page_start_offsets: list[int] | None = None) -> "LineIndex":
        offsets = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                offsets.append(i + 1)
        index = cls(_line_start_offsets=offsets)
        if page_start_offsets:
            index._page_start_offsets = list(page_start_offsets)
            index._page_first_line = [index._line_idx(p) for p in page_start_offsets]
        return index

    def _line_idx(self, char_offset: int) -> int:
        line_idx = bisect.bisect_right(self._line_start_offsets, char_offset) - 1
        return max(0, min(line_idx, len(self._line_start_offsets) - 1))

    def locate(self, char_start: int, char_end: int) -> TextLocation:
        """Return the 1-indexed line/column (and PDF page, when pages are
        known) for a character span. Uses bisect for O(log n) lookup
        regardless of document size."""
        line_idx = self._line_idx(char_start)
        column = char_start - self._line_start_offsets[line_idx] + 1
        if not self._page_start_offsets:
            return TextLocation(
                char_start=char_start, char_end=char_end,
                line_number=line_idx + 1, column_number=column,
            )
        page_idx = max(0, bisect.bisect_right(self._page_start_offsets, char_start) - 1)
        return TextLocation(
            char_start=char_start, char_end=char_end,
            line_number=line_idx - self._page_first_line[page_idx] + 1,
            column_number=column,
            page_number=page_idx + 1,
        )
