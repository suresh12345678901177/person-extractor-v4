"""Tests for src.review.llm_reviewer - with a fake Ollama, never a real model."""
import copy
import sys
import urllib.error
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import src.review.llm_reviewer as llm
from config import BASE_DIR, DEFAULT_CONFIG
from src.core.models import Candidate, CandidateResult, CandidateState, Decision, Detection, DetectorName
from src.pipeline.orchestrator import Pipeline

SETTINGS = copy.deepcopy(DEFAULT_CONFIG["llm_reviewer"])
MODEL, DIGEST = SETTINGS["model"], SETTINGS["model_digest"]


def _fake_ollama(monkeypatch, *, running=True, models=None, vram=(4_000_000_000,)):
    """Installs a fake _request; `vram` is size_vram reported per load attempt."""
    models = {MODEL: DIGEST} if models is None else models
    loads = list(vram)

    def fake(url, payload=None, timeout=10.0):
        if not running:
            raise urllib.error.URLError("connection refused")
        if url.endswith("/api/version"):
            return {"version": "test"}
        if url.endswith("/api/tags"):
            return {"models": [{"name": n, "digest": d} for n, d in models.items()]}
        if url.endswith("/api/generate"):
            return {}
        if url.endswith("/api/ps"):
            return {"models": [{"name": MODEL, "size_vram": loads.pop(0) if loads else 0}]}
        raise AssertionError(url)
    monkeypatch.setattr(llm, "_request", fake)
    monkeypatch.setattr(llm, "_start_ollama", lambda url: False)
    monkeypatch.delenv(llm.ENV_SWITCH, raising=False)


def test_active_only_with_the_pinned_model_on_a_gpu(monkeypatch):
    _fake_ollama(monkeypatch)
    assert llm.resolve_status(SETTINGS).active


@pytest.mark.parametrize("kwargs, why", [
    ({"running": False}, "not installed or would not start"),
    ({"models": {}}, "is not installed"),
    ({"models": {MODEL: "0" * 64}}, "pinned"),
    ({"vram": (0, 0)}, "no usable GPU"),
])
def test_switches_itself_off_and_says_why(monkeypatch, kwargs, why):
    _fake_ollama(monkeypatch, **kwargs)
    status = llm.resolve_status(SETTINGS)
    assert not status.active and why in status.reason


def test_a_cpu_only_load_is_retried_once_on_the_gpu(monkeypatch):
    # A model left loaded CPU-only by an earlier request is unloaded and placed again.
    _fake_ollama(monkeypatch, vram=(0, 4_000_000_000))
    assert llm.resolve_status(SETTINGS).active


def test_off_by_config_or_environment(monkeypatch):
    _fake_ollama(monkeypatch)
    assert not llm.resolve_status({**SETTINGS, "enabled": False}).active
    monkeypatch.setenv(llm.ENV_SWITCH, "off")
    assert not llm.resolve_status(SETTINGS).active


def _cand(text: str, start: int, decision: Decision, reason: str | None = None) -> CandidateResult:
    det = Detection(text=text, start=start, end=start + len(text), page_index=0,
                    detector=DetectorName.REGEX, confidence=0.9, metadata={"pattern": "bare"})
    state = CandidateState()
    state.decision, state.rejection_reason = decision, reason
    return CandidateResult(candidate=Candidate.new(text, text, start, start + len(text), 0, (det,)), state=state)


def _reviewer(monkeypatch, scores: dict[str, float], **overrides) -> tuple[llm.LlmReviewer, list[str]]:
    asked: list[str] = []
    reviewer = llm.LlmReviewer({**SETTINGS, **overrides}, BASE_DIR / "assets")
    monkeypatch.setattr(reviewer, "p_person", lambda name, excerpts: (asked.append(name), scores[name])[1])
    return reviewer, asked


TEXT = "Invoice from Turk Telekom to Ramesh Kumar. Alexa Help Videos. Hopkins, 1996. Bree"


def test_demotes_doubted_accepted_names_and_records_every_verdict(monkeypatch):
    reviewer, _ = _reviewer(monkeypatch, {"Turk Telekom": 0.02, "Ramesh Kumar": 0.99, "Alexa": 0.1})
    org, person, alexa = (_cand("Turk Telekom", 13, Decision.ACCEPTED), _cand("Ramesh Kumar", 29, Decision.ACCEPTED),
                          _cand("Alexa", 43, Decision.ACCEPTED))
    reviewer.review([org, person, alexa], TEXT)
    assert org.state.decision == Decision.REVIEW and "LLM reviewer: P(person) 0.02" in org.state.rejection_reason
    assert person.state.decision == Decision.ACCEPTED
    assert any(e.label == "LLM reviewer P(person) 0.99" for e in person.state.evidence)
    # Owner-confirmed names (assets/owner_decisions/confirmed_names.txt) are never demoted.
    assert alexa.state.decision == Decision.ACCEPTED


def test_promotion_is_off_by_default(monkeypatch):
    reviewer, asked = _reviewer(monkeypatch, {"Hopkins": 0.98})
    cited = _cand("Hopkins", 63, Decision.REVIEW, "bare common-word surname")
    reviewer.review([cited], TEXT)
    assert cited.state.decision == Decision.REVIEW and asked == []


def test_promotion_when_switched_on_skips_lone_words_the_rules_held(monkeypatch):
    reviewer, asked = _reviewer(monkeypatch, {"Hopkins": 0.98}, promote_at_or_above=0.9)
    cited = _cand("Hopkins", 63, Decision.REVIEW, "bare common-word surname")
    lone = _cand("Bree", 81, Decision.REVIEW, "Single word in a very short document ...")
    reviewer.review([cited, lone], TEXT)
    assert cited.state.decision == Decision.ACCEPTED and cited.state.rejection_reason is None
    assert lone.state.decision == Decision.REVIEW and asked == ["Hopkins"]


def test_an_ollama_failure_leaves_decisions_alone(monkeypatch):
    reviewer = llm.LlmReviewer(SETTINGS, BASE_DIR / "assets")

    def broken(name, excerpts):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(reviewer, "p_person", broken)
    org = _cand("Turk Telekom", 13, Decision.ACCEPTED)
    reviewer.review([org], TEXT)
    assert org.state.decision == Decision.ACCEPTED and reviewer.failed


def test_files_are_reviewed_but_live_text_never_is(monkeypatch, tmp_path):
    asked: list[str] = []
    monkeypatch.setattr(llm.LlmReviewer, "p_person", lambda self, name, excerpts: (asked.append(name), 0.01)[1])
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["feedback"]["enabled"] = False
    cfg["llm_reviewer"]["resolved"] = {"active": True, "reason": "on (test)"}
    text = "Dr. Suresh Kumar met Mrs. Jennifer Wilson in the office yesterday afternoon.\n"
    live = Pipeline(config=cfg, base_dir=BASE_DIR).run_text(text)
    assert asked == [] and live.persons
    sample = tmp_path / "note.txt"
    sample.write_text(text, encoding="utf-8")
    result = Pipeline(config=cfg, base_dir=BASE_DIR).run(sample)
    assert asked and not result.persons and result.review_persons
    assert result.model_info["llm_reviewer"] == "on (test)"


def test_no_llm_review_option_switches_it_off_in_the_scan_and_the_cli():
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import cli
    from scan_directory import _build_config as scan_config
    flags = dict(no_feedback_log=False, loose_gate=False, no_language_filter=False)
    assert scan_config(argparse.Namespace(**flags, no_llm_review=False))["llm_reviewer"]["enabled"] == "auto"
    assert scan_config(argparse.Namespace(**flags, no_llm_review=True))["llm_reviewer"]["enabled"] is False
    assert cli._build_config(argparse.Namespace(no_llm_review=True))["llm_reviewer"]["enabled"] is False
