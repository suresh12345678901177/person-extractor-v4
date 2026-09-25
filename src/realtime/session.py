"""
src.realtime.session
=======================
One live conversation (chat thread, call transcript, message stream) fed
to the extractor a message at a time.

Why a session instead of running each message on its own: much of this
pipeline's evidence is document-wide - repetition corroboration
(CorroborationValidator, DecisionEngine), boundary refinement's
cross-candidate counts, location rescue. A single chat line has none of
that context, so "Femi Novak" mentioned in five separate messages would
never accumulate the repetition that gets it ACCEPTED from a file. A
session keeps a rolling window of recent messages so every new message
is judged with the same context it would have had inside an exported
chat file.

Cost stays bounded per message: each message runs through the detectors
(spaCy is the expensive part) exactly ONCE, when it arrives - its
detections are cached with it and reused, offset-shifted, for every
later window. Only the window cap (max_window_chars) bounds the cheap
remaining work (candidate building, refinement, regex repetition counts).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

from src.core.models import AggregatedPerson, Detection


@dataclass(slots=True)
class _Message:
    text: str
    detections: list[Detection]  # offsets relative to this message's own text


class ConversationSession:
    def __init__(self, session_id: str, max_window_chars: int) -> None:
        self.session_id = session_id
        self.max_window_chars = max_window_chars
        self.message_count = 0
        # normalized name -> running totals across the whole session
        # (NOT just the current window - the roster outlives trimming).
        self.roster: dict[str, dict] = {}
        self._messages: deque[_Message] = deque()
        self._window_chars = 0
        # Chars (incl. joining newlines) of messages trimmed off the front.
        # window offset + this = a mention's position in the WHOLE
        # conversation, stable across trimming - the roster's mention key.
        self.window_start = 0

    def add(self, text: str, detections: list[Detection]) -> None:
        """Append a message (already cleaned) with its own detections,
        then drop the oldest messages until the window fits the cap. The
        newest message is always kept, even if it alone exceeds the cap."""
        self._messages.append(_Message(text, detections))
        self._window_chars += len(text) + 1
        self.message_count += 1
        while len(self._messages) > 1 and self._window_chars > self.max_window_chars:
            dropped = self._messages.popleft()
            self._window_chars -= len(dropped.text) + 1
            self.window_start += len(dropped.text) + 1

    def window(self) -> tuple[str, list[Detection], int]:
        """Returns (window_text, window_detections, newest_message_offset).
        Messages are joined with '\\n' - every detector already refuses to
        cross a newline, so a message boundary is a hard name boundary,
        exactly like a line in an exported chat file."""
        parts: list[str] = []
        detections: list[Detection] = []
        offset = 0
        newest_offset = 0
        for message in self._messages:
            parts.append(message.text)
            detections.extend(
                replace(d, start=d.start + offset, end=d.end + offset) for d in message.detections
            )
            newest_offset = offset
            offset += len(message.text) + 1
        return "\n".join(parts), detections, newest_offset

    @property
    def window_chars(self) -> int:
        return max(0, self._window_chars - 1)

    @property
    def window_messages(self) -> int:
        return len(self._messages)

    def record(
        self, accepted: tuple[AggregatedPerson, ...], review: tuple[AggregatedPerson, ...], message_index: int,
    ) -> list[str]:
        """Fold one extraction's results into the roster. The results can
        include earlier mentions re-judged because the newest message
        added repetition evidence for their name (see service.py), so
        mentions are tracked by conversation position: a re-judged
        mention updates its own entry instead of being counted twice, and
        an ACCEPTED mention is never demoted later (e.g. when the window
        trims away the repetitions that earned it). Returns names that
        became ACCEPTED for the first time with this message."""
        newly_accepted: list[str] = []
        for decision, persons in (("accepted", accepted), ("review", review)):
            for person in persons:
                entry = self.roster.setdefault(person.normalized_text, {
                    "name": person.display_text,
                    "decision": "review",
                    "first_message": message_index,
                    "last_message": message_index,
                    "_seen": set(),
                    "_accepted": set(),
                })
                entry["last_message"] = message_index
                positions = {self.window_start + m.candidate.start for m in person.mentions}
                entry["_seen"] |= positions
                if decision == "accepted":
                    entry["_accepted"] |= positions
                    if entry["decision"] != "accepted":
                        entry["decision"] = "accepted"
                        newly_accepted.append(entry["name"])
        return newly_accepted

    def roster_rows(self) -> list[dict]:
        rows = []
        for entry in self.roster.values():
            row = {k: v for k, v in entry.items() if not k.startswith("_")}
            row["mentions"] = len(entry["_seen"])
            row["accepted_mentions"] = len(entry["_accepted"])
            rows.append(row)
        return sorted(rows, key=lambda r: (-r["mentions"], r["name"]))
