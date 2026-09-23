"""
src.utils.provenance
======================
Answers one question every exported result should be able to answer on
its own, without a reviewer reconstructing it from file timestamps: "what
code and model produced this file?" (independent-audit finding #3).

Before this, two scan CSVs from different days looked identical - nothing
tied a result to the git state or model version that produced it, so a
stale, pre-fix result could sit around looking as authoritative as a
current one (this actually happened: a 2026-09-22 scan of the real case
folder still contained false positives - "June", "Fair", "Handler" among
them - that a fix the very next day removed; nothing in that CSV said so).

Computed ONCE per process (a batch scan of thousands of files must not
shell out to git / re-hash the model file per document) and threaded
through `ExtractionResult.model_info` and every exporter from there - see
`PersonExtractor.__init__`.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger("utils.provenance")


def _git_state(repo_dir: Path) -> dict[str, Any]:
    """Never raises - a missing git binary or a non-repo checkout (e.g. a
    zipped release) must degrade to an honest 'not available' rather than
    break provenance stamping, let alone the whole scan."""
    try:
        # Explicit UTF-8, not text=True's locale default (cp1252 on
        # Windows) - a dirty file's path or a diff can legitimately
        # contain non-cp1252 bytes; decoding must not crash provenance
        # stamping (found the hard way in check_feedback_privacy.py,
        # which shells out to git the same way).
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, timeout=10, check=True,
            encoding="utf-8", errors="replace",
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir, capture_output=True, timeout=10, check=True,
            encoding="utf-8", errors="replace",
        ).stdout
        dirty_files = [line for line in status.splitlines() if line.strip()]
        return {
            "git_commit": commit,
            "git_dirty": bool(dirty_files),
            "git_dirty_file_count": len(dirty_files),
        }
    except Exception as exc:  # noqa: BLE001 - provenance must never break a scan
        logger.warning("Could not determine git state for provenance stamping: %s", exc)
        return {
            "git_commit": "unavailable",
            "git_dirty": None,
            "git_dirty_file_count": None,
        }


def _model_state(model_path: Path | None) -> dict[str, Any]:
    if model_path is None:
        return {"model_file": None, "model_mtime_utc": None, "model_sha256_short": None}
    try:
        stat = model_path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
        return {
            "model_file": model_path.name,
            "model_mtime_utc": mtime,
            "model_sha256_short": digest,
        }
    except Exception as exc:  # noqa: BLE001 - same "never break a scan" contract as above
        logger.warning("Could not stat/hash model file %s for provenance stamping: %s", model_path, exc)
        return {"model_file": model_path.name, "model_mtime_utc": "unavailable", "model_sha256_short": "unavailable"}


def build_run_provenance(base_dir: Path, model_path: Path | None) -> dict[str, Any]:
    """One dict, computed once per process/worker, describing exactly what
    produced a run: the git commit (or 'uncommitted, N files dirty' via
    git_dirty/git_dirty_file_count), and the model file's identity
    (name + mtime + a short content hash, so an in-place-overwritten
    model file is still distinguishable from the one before it - a
    filename alone is not, since every retrain writes to the same path)."""
    provenance = {"scan_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    provenance.update(_git_state(base_dir))
    provenance.update(_model_state(model_path))
    return provenance


def provenance_summary_line(provenance: dict[str, Any]) -> str:
    """Single human-readable line for report.txt / console output."""
    if provenance.get("git_commit") == "unavailable":
        git_part = "git: unavailable"
    elif provenance.get("git_dirty"):
        git_part = f"git {provenance['git_commit'][:12]} (UNCOMMITTED, {provenance['git_dirty_file_count']} file(s) dirty)"
    else:
        git_part = f"git {provenance['git_commit'][:12]} (clean)"

    model_hash = provenance.get("model_sha256_short") or "n/a"
    model_part = f"model {provenance.get('model_file') or 'n/a'} @ {model_hash} ({provenance.get('model_mtime_utc') or 'n/a'})"
    return f"{git_part} | {model_part} | scanned {provenance.get('scan_timestamp_utc')}"
