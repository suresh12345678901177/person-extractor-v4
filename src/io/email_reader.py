"""
src.io.email_reader
=====================
Reader for .eml (RFC 822 email exports). Emits the people-bearing headers
(From, To, Cc, Bcc, Reply-To, Sender) plus Subject and Date as
"Header: value" lines - "From: Priyanka Deshmukh <p.d@example.com>" is
exactly the shape the pipeline already handles in pasted email text
(benchmark_004) - followed by the message body: the plain-text part when
there is one, otherwise the HTML part converted to visible text.
Attachments are not opened. Outlook .msg is a different (binary) format
and is not supported.
"""

from __future__ import annotations

from email import policy
from email.parser import BytesParser
from pathlib import Path

from src.core.models import Document, Page, SourceFormat
from src.io.base_reader import BaseReader
from src.io.markup_text import html_to_text

_HEADERS = ("From", "Sender", "Reply-To", "To", "Cc", "Bcc", "Subject", "Date")


class EmailReader(BaseReader):
    supported_extensions = ("eml",)

    def _read(self, path: str) -> Document:
        message = BytesParser(policy=policy.default).parsebytes(Path(path).read_bytes())
        lines = []
        for header in _HEADERS:
            for value in message.get_all(header) or []:
                value = " ".join(str(value).split())
                if value:
                    lines.append(f"{header}: {value}")

        body = ""
        part = message.get_body(preferencelist=("plain", "html"))
        if part is not None:
            content = part.get_content()
            body = html_to_text(content) if part.get_content_type() == "text/html" else content
        full_text = "\n".join(lines) + ("\n\n" + body.strip() if body.strip() else "")
        return Document(
            source_path=path, source_format=SourceFormat.EML,
            pages=(Page(index=0, text=full_text),), full_text=full_text,
            metadata={"location_note": "line N of the headers + body text"},
        )
