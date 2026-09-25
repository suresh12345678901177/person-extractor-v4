"""
server.py
===========
Real-time mode: a small local HTTP service that loads the extraction
pipeline ONCE and then answers requests in milliseconds, instead of
paying the ~11-13s model/knowledge-base startup on every cli.py run.

Standard library only (http.server) - no new dependencies, still fully
offline. Binds to 127.0.0.1 by default so it is only reachable from this
machine; the text sent to it is case material, so do not expose it on a
network interface without putting authentication in front of it.

Usage:
    python server.py                      # http://127.0.0.1:8765
    python server.py --port 9000 --no-feedback-log

Endpoints (all JSON):
    GET    /health                        -> {"status": "ok", ...}
    POST   /extract                       {"text": "...", "include_review": true}
           One standalone piece of text, no memory between calls.
    POST   /sessions/<id>/messages        {"text": "..."}
           Next message of a conversation. Names are judged with the
           recent conversation as context, so a name repeated across
           messages builds up evidence the way it would in a chat export.
           The session is created on first use.
    GET    /sessions/<id>                 -> running name list for the conversation
    DELETE /sessions/<id>                 -> forget the conversation

Example (PowerShell):
    Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/sessions/chat1/messages `
        -ContentType 'application/json' -Body '{"text": "Femi Novak called again."}'
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import BASE_DIR, DEFAULT_CONFIG
from src.pipeline.orchestrator import Pipeline
from src.realtime.service import SESSION_ID_RE, RealtimeService
from src.utils.logger import configure_logging, get_logger

logger = get_logger("server")

_MESSAGES_ROUTE = re.compile(r"^/sessions/([^/]+)/messages/?$")
_SESSION_ROUTE = re.compile(r"^/sessions/([^/]+)/?$")


class _RequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class ExtractionRequestHandler(BaseHTTPRequestHandler):
    # Set on the class by make_server() - one shared service per process.
    service: RealtimeService
    max_body_bytes: int = 10 * 1024 * 1024

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        self._handle(self._route_get)

    def do_POST(self) -> None:  # noqa: N802
        self._handle(self._route_post)

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle(self._route_delete)

    # --- routing --------------------------------------------------------

    def _route_get(self, path: str) -> tuple[HTTPStatus, dict]:
        if path in ("/health", "/health/"):
            return HTTPStatus.OK, {
                "status": "ok",
                "ml_classifier_loaded": self.service.extractor.classifier is not None,
                "detectors": [d.name.value for d in self.service.extractor.detector_manager.detectors],
            }
        match = _SESSION_ROUTE.match(path)
        if match:
            session = self.service.get_session(self._session_id(match))
            if session is None:
                raise _RequestError(HTTPStatus.NOT_FOUND, "No such session")
            return HTTPStatus.OK, session
        raise _RequestError(HTTPStatus.NOT_FOUND, f"Unknown path: {path}")

    def _route_post(self, path: str) -> tuple[HTTPStatus, dict]:
        if path in ("/extract", "/extract/"):
            body = self._read_json()
            include_review = body.get("include_review", True)
            if not isinstance(include_review, bool):
                raise _RequestError(HTTPStatus.BAD_REQUEST, "'include_review' must be true or false")
            return HTTPStatus.OK, self.service.extract(self._text(body), include_review=include_review)
        match = _MESSAGES_ROUTE.match(path)
        if match:
            session_id = self._session_id(match)
            return HTTPStatus.OK, self.service.add_message(session_id, self._text(self._read_json()))
        raise _RequestError(HTTPStatus.NOT_FOUND, f"Unknown path: {path}")

    def _route_delete(self, path: str) -> tuple[HTTPStatus, dict]:
        match = _SESSION_ROUTE.match(path)
        if match:
            session_id = self._session_id(match)
            if not self.service.delete_session(session_id):
                raise _RequestError(HTTPStatus.NOT_FOUND, "No such session")
            return HTTPStatus.OK, {"deleted": session_id}
        raise _RequestError(HTTPStatus.NOT_FOUND, f"Unknown path: {path}")

    # --- helpers --------------------------------------------------------

    def _handle(self, route) -> None:
        path = self.path.split("?", 1)[0]
        try:
            status, payload = route(path)
        except _RequestError as exc:
            status, payload = exc.status, {"error": exc.message}
        except Exception:  # noqa: BLE001 - one bad request must never kill the server
            logger.exception("Unhandled error serving %s %s", self.command, path)
            status, payload = HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Internal error - see logs/pipeline.log"}
        self._send(status, payload)

    def _read_json(self) -> dict:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise _RequestError(HTTPStatus.LENGTH_REQUIRED, "Content-Length header required")
        try:
            length = int(length_header)
        except ValueError:
            raise _RequestError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length") from None
        if length > self.max_body_bytes:
            raise _RequestError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"Body over {self.max_body_bytes} bytes - use cli.py/scan_directory.py for large files",
            )
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _RequestError(HTTPStatus.BAD_REQUEST, "Body must be UTF-8 JSON") from None
        if not isinstance(body, dict):
            raise _RequestError(HTTPStatus.BAD_REQUEST, "Body must be a JSON object")
        return body

    @staticmethod
    def _text(body: dict) -> str:
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise _RequestError(HTTPStatus.BAD_REQUEST, "'text' must be a non-empty string")
        return text

    @staticmethod
    def _session_id(match: re.Match) -> str:
        session_id = match.group(1)
        if not SESSION_ID_RE.match(session_id):
            raise _RequestError(
                HTTPStatus.BAD_REQUEST, "Session id must be 1-128 characters of A-Z a-z 0-9 _ . -",
            )
        return session_id

    def _send(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - http.server signature
        logger.debug("%s - %s", self.address_string(), format % args)


def make_server(
    service: RealtimeService, host: str, port: int, max_body_bytes: int = 10 * 1024 * 1024,
) -> ThreadingHTTPServer:
    handler = type(
        "BoundExtractionRequestHandler",
        (ExtractionRequestHandler,),
        {"service": service, "max_body_bytes": max_body_bytes},
    )
    return ThreadingHTTPServer((host, port), handler)


def main() -> int:
    parser = argparse.ArgumentParser(description="PERSON_EXTRACTOR_V4 real-time HTTP service")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (default 127.0.0.1 = this machine only).")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-feedback-log", action="store_true",
                        help="Don't log REVIEW-bucket names to datasets/feedback/pending_review.jsonl.")
    parser.add_argument("--no-language-filter", action="store_true",
                        help="Disable the English-language gate (see cli.py --no-language-filter).")
    parser.add_argument("--max-window-chars", type=int, default=200_000,
                        help="How much recent conversation each session keeps as context (default 200000).")
    parser.add_argument("--max-sessions", type=int, default=100,
                        help="Conversations kept in memory; least recently used is dropped first.")
    parser.add_argument("--verbose", action="store_true", help="Log every pipeline stage (INFO level).")
    args = parser.parse_args()

    configure_logging(BASE_DIR / "logs")
    if not args.verbose:
        # Every request logs each pipeline stage at INFO - fine in
        # logs/pipeline.log, unreadable on a busy console. Quiet only the
        # console handler; the file keeps the full record. (Setting the
        # level via configure_logging() isn't enough: the first
        # get_logger() at import time already configured logging.)
        for handler in logging.getLogger("person_extractor_v4").handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.WARNING)

    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.no_feedback_log:
        config["feedback"]["enabled"] = False
    if args.no_language_filter:
        config["language_filter"]["enabled"] = False

    print("Loading models and knowledge base (one time only)...", flush=True)
    service = RealtimeService(
        Pipeline(config=config, base_dir=BASE_DIR),
        max_window_chars=args.max_window_chars,
        max_sessions=args.max_sessions,
    )
    server = make_server(service, args.host, args.port)
    print(f"Ready: http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
