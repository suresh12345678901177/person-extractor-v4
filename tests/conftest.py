"""Test-wide settings."""
import pytest

from src.review.llm_reviewer import ENV_SWITCH


@pytest.fixture(autouse=True)
def _no_llm_reviewer(monkeypatch):
    """Tests never call a local LLM: results must not depend on whether
    Ollama happens to be running on the machine (see src/review/llm_reviewer.py).
    tests/test_llm_reviewer.py exercises the reviewer with a fake server."""
    monkeypatch.setenv(ENV_SWITCH, "off")
