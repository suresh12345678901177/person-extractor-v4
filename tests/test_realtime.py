"""Tests for real-time mode: RealtimeService (one-shot + conversation
sessions) and the server.py HTTP layer."""
import copy
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from config import BASE_DIR, DEFAULT_CONFIG
from src.pipeline.orchestrator import Pipeline
from src.realtime.service import RealtimeService


@pytest.fixture(scope="module")
def pipeline():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["feedback"]["enabled"] = False  # never write to the real feedback log from tests
    return Pipeline(config=config, base_dir=BASE_DIR)


@pytest.fixture
def service(pipeline):
    return RealtimeService(pipeline)


def test_extract_matches_file_mode_and_reports_locations(service, pipeline, tmp_path):
    text = "Yesterday Dr. Suresh Kumar met Priya Raman at the office.\nPriya Raman agreed."
    result = service.extract(text)

    path = tmp_path / "doc.txt"
    path.write_text(text, encoding="utf-8")
    file_names = {p.display_text for p in pipeline.run(path).persons}
    assert {p["name"] for p in result["accepted"]} == file_names

    for person in result["accepted"]:
        for span in person["spans"]:
            assert text[span["start"]:span["end"]] == span["text"]
            assert span["line"] >= 1 and span["column"] >= 1


def test_session_accepts_name_once_repetition_builds_up(service):
    """'Femi Zqarlowski': a known first name + a surname no dictionary has.
    Alone it is only REVIEW; once repeated across enough separate messages
    the session's shared window supplies the repetition evidence that
    promotes it - exactly what it would get inside one chat-export file."""
    messages = [
        "Yesterday I spoke with Femi Zqarlowski about the shipment.",
        "Femi Zqarlowski said the paperwork would arrive Monday.",
        "Did Femi Zqarlowski send the invoices yet?",
        "I will call Femi Zqarlowski again tomorrow morning.",
    ]
    replies = [service.add_message("chat-1", m) for m in messages]

    first = replies[0]
    assert "Femi Zqarlowski" not in {p["name"] for p in first["accepted"]}

    promoted = [r for r in replies if "Femi Zqarlowski" in r["newly_accepted"]]
    assert promoted, [r["accepted"] + r["review"] for r in replies]

    roster = {e["name"]: e for e in service.get_session("chat-1")["names"]}
    assert roster["Femi Zqarlowski"]["decision"] == "accepted"
    assert roster["Femi Zqarlowski"]["first_message"] == 1


def test_session_spans_are_relative_to_each_message(service):
    service.add_message("chat-2", "Hello there, how are you doing today?")
    message = "Dr. Suresh Kumar will join the call at noon."
    reply = service.add_message("chat-2", message)
    spans = [s for p in reply["accepted"] + reply["review"] for s in p["spans"]]
    assert spans
    for span in spans:
        assert message[span["start"]:span["end"]] == span["text"]


def test_session_window_is_capped_and_sessions_are_isolated(pipeline):
    service = RealtimeService(pipeline, max_window_chars=120)
    for i in range(10):
        service.add_message("capped", f"Message number {i} with a little bit of filler text.")
    session = service.get_session("capped")
    assert session["messages"] == 10
    assert session["window_chars"] <= 120
    assert service.get_session("other") is None
    assert service.delete_session("capped") is True
    assert service.get_session("capped") is None


def test_least_recently_used_session_is_evicted(pipeline):
    service = RealtimeService(pipeline, max_sessions=2)
    service.add_message("a", "First chat.")
    service.add_message("b", "Second chat.")
    service.add_message("a", "Touch a so b is now the oldest.")
    service.add_message("c", "Third chat.")
    assert service.get_session("b") is None
    assert service.get_session("a") is not None and service.get_session("c") is not None


# --- HTTP layer ------------------------------------------------------------

@pytest.fixture
def base_url(service):
    from server import make_server

    server = make_server(service, "127.0.0.1", 0, max_body_bytes=2000)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _call(method, url, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_http_endpoints_round_trip(base_url):
    status, health = _call("GET", f"{base_url}/health")
    assert status == 200 and health["status"] == "ok"

    status, reply = _call("POST", f"{base_url}/sessions/web-1/messages",
                          {"text": "Dr. Suresh Kumar joined the call."})
    assert status == 200 and reply["message_index"] == 1

    status, session = _call("GET", f"{base_url}/sessions/web-1")
    assert status == 200 and session["messages"] == 1

    status, extracted = _call("POST", f"{base_url}/extract", {"text": "Dr. Suresh Kumar said hello."})
    assert status == 200 and "accepted" in extracted

    status, _ = _call("DELETE", f"{base_url}/sessions/web-1")
    assert status == 200
    status, _ = _call("GET", f"{base_url}/sessions/web-1")
    assert status == 404


def test_http_rejects_bad_requests(base_url):
    assert _call("POST", f"{base_url}/extract", raw=b"not json")[0] == 400
    assert _call("POST", f"{base_url}/extract", {"text": ""})[0] == 400
    assert _call("POST", f"{base_url}/extract", {"text": "x", "include_review": "yes"})[0] == 400
    assert _call("POST", f"{base_url}/extract", {"text": "x" * 5000})[0] == 413
    assert _call("POST", f"{base_url}/sessions/bad%21id/messages", {"text": "hi"})[0] == 400
    assert _call("GET", f"{base_url}/nope")[0] == 404


def test_session_first_name_propagation_never_duplicates_a_full_name_mention(service):
    """Propagating 'Kenji' must not create a second mention inside an
    earlier message's 'Kenji Ito' - that span is already a candidate even
    though a session call only re-judges the newest message (this used to
    count the same text as both 'Kenji Ito' and 'Kenji')."""
    for text in ("Dr. Kenji Ito signed the report on Monday.",
                 "Kenji Ito called again about the invoice.",
                 "Kenji confirmed the numbers this morning."):
        service.add_message("prop", text)

    session = service._sessions["prop"]
    positions = [p for entry in session.roster.values() for p in entry["_seen"]]
    assert len(positions) == len(set(positions)), session.roster_rows()

    rows = {r["name"]: r for r in service.get_session("prop")["names"]}
    assert rows["Kenji"]["decision"] == "accepted"
    assert rows["Kenji"]["mentions"] == 1
