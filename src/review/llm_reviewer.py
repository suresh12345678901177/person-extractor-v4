"""
src.review.llm_reviewer
=========================
Optional final check of each FILE's names by a local LLM, through Ollama
(http://127.0.0.1:11434 - nothing leaves the machine). The tool never
installs or downloads anything: scripts/setup_llm_reviewer.ps1 is the
one-time setup, run by hand with internet, and only on a machine with a GPU.

Measured before it was built (experiments/llm_reviewer_probe.py, 2026-09-28,
Llama 3.1 8B): of the ACCEPTED names in a real case scan, 13 of the 14 the
model gave P(person) < 0.3 were not people (companies, places, interface
labels, OCR scraps); on the benchmark the same move removed one false
positive and nothing real. With GPU+CPU a name takes 0.41 s, on CPU only
4.25 s.

Rules stay the gate. The reviewer only moves ACCEPTED -> REVIEW below
`demote_below` (it never rejects anything), and REVIEW -> ACCEPTED at or
above `promote_at_or_above` only when that is switched on (off by default -
an owner decision). Names in assets/owner_decisions/confirmed_names.txt are
never demoted. Every verdict goes into the evidence trail with the model,
its digest and the prompt version.

Active only when all of these hold - otherwise the pipeline runs exactly as
without it and the reason is recorded in the run's provenance:
  - config llm_reviewer.enabled is "auto" or True (and the environment
    variable PERSON_EXTRACTOR_LLM_REVIEWER isn't "off" - the test suite
    sets it),
  - Ollama answers (if it's installed but not running, it is started),
  - the pinned model digest is installed,
  - the model runs on a GPU (Ollama's /api/ps reports GPU memory) - on a
    machine without one it is skipped, as it would be 10x slower.
Live text (Pipeline.run_text, realtime sessions) is never reviewed - see
PersonExtractor.extract.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from src.core.models import CandidateResult, Decision
from src.utils.logger import get_logger

logger = get_logger("review.llm_reviewer")

PROMPT_VERSION = "v1"
SYSTEM_PROMPT = (
    "You check the output of a person-name extractor that runs on forensic evidence: chats, emails, "
    "documents, web pages, app data and OCR text. For the candidate, decide whether the excerpts use it "
    "as the name of a specific human being - a first name, a surname, a full name, or a name with a "
    "title or initials. Answer only 'yes' or 'no'. Answer 'no' for organizations, companies, brands, "
    "products, apps, software and code identifiers; places, streets, localities, countries, languages "
    "and nationalities; job titles; days and months; journal and publication titles; ordinary words; "
    "interface labels; and garbled text."
)
ENV_SWITCH = "PERSON_EXTRACTOR_LLM_REVIEWER"
MAX_EXCERPTS = 3
EXCERPT_WIDTH = 120
# Set by _cap_lone_single_tokens; such names aren't promoted - the model
# sees no more context there than the rules did.
_LONE_WORD_REASON_PREFIX = "Single word in"


def excerpt(text: str, start: int, end: int, width: int = EXCERPT_WIDTH) -> str:
    """The mention with up to `width` characters either side, within its line,
    marked [[like this]]."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end
    a, b = max(line_start, start - width), min(line_end, end + width)
    return f"{text[a:start]}[[{text[start:end]}]]{text[end:b]}".strip()


def _request(url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _start_ollama(url: str) -> bool:
    """Starts an installed-but-stopped local Ollama server. Never installs."""
    exe = shutil.which("ollama") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe")
    if not Path(exe).exists():
        return False
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    for _ in range(30):
        time.sleep(1)
        try:
            _request(f"{url}/api/version", timeout=2)
            return True
        except (urllib.error.URLError, OSError):
            continue
    return False


@dataclass(frozen=True)
class ReviewerStatus:
    active: bool
    reason: str


def resolve_status(settings: dict) -> ReviewerStatus:
    """Whether the reviewer can run here - see the module docstring."""
    if os.environ.get(ENV_SWITCH, "").lower() == "off":
        return ReviewerStatus(False, f"off ({ENV_SWITCH}=off)")
    enabled = settings.get("enabled", "auto")
    if enabled is False:
        return ReviewerStatus(False, "off (disabled in config)")
    url, model, digest = settings["url"], settings["model"], settings["model_digest"]
    try:
        _request(f"{url}/api/version", timeout=2)
    except (urllib.error.URLError, OSError):
        if not _start_ollama(url):
            return ReviewerStatus(False, "off (Ollama is not installed or would not start)")
    try:
        installed = {m["name"]: m["digest"] for m in _request(f"{url}/api/tags")["models"]}
        if model not in installed:
            return ReviewerStatus(False, f"off (model {model} is not installed)")
        if installed[model] != digest:
            return ReviewerStatus(False, f"off (model {model} is {installed[model][:12]}, pinned {digest[:12]})")
        for attempt in range(2):
            _request(f"{url}/api/generate", {"model": model, "prompt": "", "keep_alive": "30m"}, timeout=600)
            loaded = [m for m in _request(f"{url}/api/ps")["models"] if m["name"] == model]
            if loaded and loaded[0].get("size_vram", 0) > 0:
                return ReviewerStatus(True, f"on ({model} {digest[:12]}, prompt {PROMPT_VERSION}, GPU)")
            # Loaded CPU-only (no usable GPU, or an earlier CPU-only request) -
            # unload once and let Ollama place it again.
            _request(f"{url}/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
        return ReviewerStatus(False, "off (no usable GPU - the model would run on the CPU only)")
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        return ReviewerStatus(False, f"off (Ollama error: {exc})")


def _load_names(path: Path) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    return frozenset(
        line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )


class LlmReviewer:
    def __init__(self, settings: dict, assets_dir: Path) -> None:
        self.url = settings["url"]
        self.model = settings["model"]
        self.digest = settings["model_digest"]
        self.demote_below = settings.get("demote_below")
        self.promote_at_or_above = settings.get("promote_at_or_above")
        self.confirmed_names = _load_names(assets_dir / "owner_decisions" / "confirmed_names.txt")
        self.failed: str | None = None

    def p_person(self, name: str, excerpts: list[str]) -> float:
        user = (f"Candidate: {name}\nExcerpts:\n" + "\n".join(f"{i}. {e}" for i, e in enumerate(excerpts, 1))
                + f"\nIs [[{name}]] used as a person's name here?")
        reply = _request(f"{self.url}/api/chat", {
            "model": self.model, "stream": False, "logprobs": True, "top_logprobs": 10, "keep_alive": "30m",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": 0, "num_predict": 1},
        }, timeout=600)
        yes = no = 0.0
        for cand in reply["logprobs"][0]["top_logprobs"]:
            token = cand["token"].strip().lower()
            if token in ("yes", "y"):
                yes += math.exp(cand["logprob"])
            elif token in ("no", "n"):
                no += math.exp(cand["logprob"])
        return yes / (yes + no) if yes + no else 0.5

    def review(self, candidates: list[CandidateResult], text: str) -> None:
        """Judges each distinct ACCEPTED name (and REVIEW name, if promotion is
        on) once per file, from up to three of its mentions, and applies the
        thresholds to all its mentions. On an Ollama failure it stops
        reviewing for the rest of this process and leaves decisions as they are."""
        if self.failed:
            return
        groups: dict[tuple[Decision, str], list[CandidateResult]] = {}
        for c in candidates:
            if c.state.decision == Decision.ACCEPTED or (
                    c.state.decision == Decision.REVIEW and self.promote_at_or_above is not None
                    and not (c.state.rejection_reason or "").startswith(_LONE_WORD_REASON_PREFIX)):
                groups.setdefault((c.state.decision, c.candidate.normalized_text), []).append(c)
        tag = f"{self.model} {self.digest[:12]}, prompt {PROMPT_VERSION}"
        for (decision, name), group in groups.items():
            group.sort(key=lambda c: c.candidate.start)
            picks = sorted({0, len(group) // 2, len(group) - 1})[:MAX_EXCERPTS]
            try:
                p = self.p_person(name, [excerpt(text, group[k].candidate.start, group[k].candidate.end) for k in picks])
            except (urllib.error.URLError, OSError, KeyError, ValueError, IndexError) as exc:
                self.failed = str(exc)
                logger.warning("LLM reviewer stopped for this process (Ollama error: %s) - "
                               "remaining files keep their rule-based decisions", exc)
                return
            if decision == Decision.ACCEPTED and self.demote_below is not None and p < self.demote_below \
                    and name.lower() not in self.confirmed_names:
                for c in group:
                    c.state.decision = Decision.REVIEW
                    c.state.rejection_reason = (f"LLM reviewer: P(person) {p:.2f} is below {self.demote_below} "
                                                f"({tag}) - held for review")
                    c.state.add_evidence("llm_reviewer", f"LLM reviewer P(person) {p:.2f} - held for review", 0.0)
            elif decision == Decision.REVIEW and p >= self.promote_at_or_above:
                for c in group:
                    c.state.decision = Decision.ACCEPTED
                    c.state.rejection_reason = None
                    c.state.add_evidence("llm_reviewer", f"LLM reviewer P(person) {p:.2f} - promoted", 0.0)
            else:
                for c in group:
                    c.state.add_evidence("llm_reviewer", f"LLM reviewer P(person) {p:.2f}", 0.0)
