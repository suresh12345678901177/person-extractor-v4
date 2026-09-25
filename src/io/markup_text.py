"""
src.io.markup_text
=====================
HTML (and malformed-XML fallback) to plain text, standard library only.

Readers ONLY read (see base_reader.py): this is format conversion, not
extraction. The one thing it must get right for extraction is structure:
block elements become line breaks and table cells are separated by
CELL_SEPARATOR, so two names in adjacent cells or paragraphs can never be
glued into one fake name - the same rule the detectors already enforce
for newlines and 3+ space column padding.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

#: Joins table cells / CSV fields. '|' is not a letter, so no detector can
#: extend a name across it.
CELL_SEPARATOR = " | "

_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "object", "iframe"}
_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "body", "br", "caption", "dd", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "hr", "html", "li", "main", "nav", "ol", "p", "pre", "section", "table", "tbody",
    "thead", "tfoot", "title", "tr", "ul",
}
_CELL_TAGS = {"td", "th"}
_SPACES_RE = re.compile(r"[ \t\r\f\v ]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _CELL_TAGS:
            self._parts.append(CELL_SEPARATOR)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(_SPACES_RE.sub(" ", data.replace("\n", " ")))

    def text(self) -> str:
        lines = []
        for line in "".join(self._parts).split("\n"):
            line = line.strip()
            # Drop separators left dangling at a row's edges ("| Ana | Bo |").
            while line.startswith(CELL_SEPARATOR.strip()):
                line = line[1:].strip()
            while line.endswith(CELL_SEPARATOR.strip()):
                line = line[:-1].strip()
            lines.append(line)
        return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    parser.feed(markup)
    parser.close()
    return parser.text()
