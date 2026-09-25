"""
src.realtime.service
=======================
Keeps one fully-loaded Pipeline in memory and serves extraction requests
against it - the real-time counterpart to cli.py/scan_directory.py,
which pay the ~11-13s model/knowledge-base load on every invocation.

Transport-agnostic on purpose: returns plain dicts, so server.py (HTTP)
is a thin layer and tests can drive this class directly.

Thread safety: the pipeline's validators/decision engine hold per-
document caches, so it is NOT safe to run two documents through it at
once. Every request takes one lock - requests are served one at a time,
which is fine for this workload (a message takes milliseconds once the
models are loaded) and far cheaper than a Pipeline per thread.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict

from src.core.models import AggregatedPerson, CandidateResult, PipelineStatistics
from src.extraction.person_extractor import PersonExtractor
from src.pipeline.orchestrator import Pipeline
from src.preprocessing.cleaner import clean_text
from src.preprocessing.line_indexer import LineIndex
from src.realtime.session import ConversationSession

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


class RealtimeService:
    def __init__(
        self,
        pipeline: Pipeline,
        max_window_chars: int = 200_000,
        max_sessions: int = 100,
    ) -> None:
        if not isinstance(pipeline.extractor, PersonExtractor):
            raise TypeError("RealtimeService sessions require the 'person' extractor")
        self.pipeline = pipeline
        self.extractor: PersonExtractor = pipeline.extractor
        self.max_window_chars = max_window_chars
        self.max_sessions = max_sessions
        self._lock = threading.Lock()
        # Insertion/recency-ordered so the least recently used session is
        # evicted first once max_sessions is reached (bounded memory).
        self._sessions: OrderedDict[str, ConversationSession] = OrderedDict()

    # --- one-shot -----------------------------------------------------

    def extract(self, text: str, include_review: bool = True) -> dict:
        """Extract names from one standalone piece of text. No state is
        kept between calls - use add_message() for a conversation."""
        with self._lock:
            started = time.perf_counter()
            result = self.pipeline.run_text(text, source_label="<realtime:extract>")
            return {
                "accepted": [_person_json(p, 0, with_location=True) for p in result.persons],
                "review": [_person_json(p, 0, with_location=True) for p in result.review_persons]
                if include_review else [],
                "skipped_reason": result.model_info.get("skipped_reason"),
                "seconds": round(time.perf_counter() - started, 4),
            }

    # --- conversations ------------------------------------------------

    def add_message(self, session_id: str, text: str) -> dict:
        """Add the next message of a conversation and return the names
        found IN THAT MESSAGE, judged with the recent conversation as
        context (see src/realtime/session.py)."""
        with self._lock:
            started = time.perf_counter()
            session = self._get_or_create(session_id)

            message = clean_text(text)
            detections = self.extractor.detector_manager.detect_all(message, page_index=0)
            session.add(message, detections)

            window, window_detections, message_offset = session.window()

            def _new_message_and_affected(candidates: list[CandidateResult]) -> list[CandidateResult]:
                # The new message's candidates, PLUS earlier mentions of the
                # same names: the new message just raised those names'
                # document-wide repetition count, which can flip an earlier
                # mention's decision (e.g. a high-ML mention that needed 4+
                # repetitions). No other earlier mention's evidence changed,
                # so nothing else needs re-judging.
                new_texts = {
                    c.candidate.normalized_text.lower() for c in candidates if c.candidate.start >= message_offset
                }
                return [
                    c for c in candidates
                    if c.candidate.start >= message_offset or c.candidate.normalized_text.lower() in new_texts
                ]

            result = self.extractor.extract(
                window,
                LineIndex.build(window),
                f"session:{session_id}",
                PipelineStatistics(),
                precomputed_detections=window_detections,
                candidate_filter=_new_message_and_affected,
                # Full names accepted earlier in the conversation vouch for a
                # first name in this message (name propagation), even once
                # those messages have scrolled out of the window.
                known_person_names=[e["name"] for e in session.roster.values() if e["decision"] == "accepted"],
            )

            message_index = session.message_count
            newly_accepted = session.record(result.persons, result.review_persons, message_index)
            return {
                "session_id": session_id,
                "message_index": message_index,
                # Only this message's mentions - re-judged earlier ones show
                # up via newly_accepted and GET /sessions/<id>.
                "accepted": _people_in_message(result.persons, message_offset),
                "review": _people_in_message(result.review_persons, message_offset),
                "newly_accepted": newly_accepted,
                "skipped_reason": result.model_info.get("skipped_reason"),
                "window_messages": session.window_messages,
                "seconds": round(time.perf_counter() - started, 4),
            }

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return {
                "session_id": session_id,
                "messages": session.message_count,
                "window_messages": session.window_messages,
                "window_chars": session.window_chars,
                "names": session.roster_rows(),
            }

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _get_or_create(self, session_id: str) -> ConversationSession:
        session = self._sessions.get(session_id)
        if session is None:
            session = ConversationSession(session_id, self.max_window_chars)
            self._sessions[session_id] = session
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(session_id)
        return session


def _people_in_message(persons: tuple[AggregatedPerson, ...], message_offset: int) -> list[dict]:
    """Keep only mentions inside the newest message (results can also
    carry re-judged earlier mentions); drop names left with none."""
    rows = []
    for person in persons:
        mentions = [m for m in person.mentions if m.candidate.start >= message_offset]
        if mentions:
            rows.append(_person_json(person, message_offset, mentions=mentions))
    return rows


def _person_json(
    person: AggregatedPerson, offset: int, with_location: bool = False, mentions: list | None = None,
) -> dict:
    """offset shifts spans so they're relative to the text the caller
    sent (the message), not the internal conversation window."""
    mentions = person.mentions if mentions is None else mentions
    spans = []
    for mention in mentions:
        span = {
            "text": mention.candidate.text,
            "start": mention.candidate.start - offset,
            "end": mention.candidate.end - offset,
        }
        if with_location and mention.candidate.location is not None:
            span["line"] = mention.candidate.location.line_number
            span["column"] = mention.candidate.location.column_number
        spans.append(span)
    return {
        "name": person.display_text,
        "mentions": len(mentions),
        "confidence": round(max(m.state.final_confidence for m in mentions), 4),
        "spans": spans,
    }
