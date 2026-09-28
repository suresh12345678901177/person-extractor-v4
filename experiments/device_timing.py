"""
experiments/device_timing.py
===============================
Time taken with the GPU and without it, for each part of the tool that can
use one, on this machine. Numbers only go to experiments/probes/device_timing.json.

  1. LLM reviewer, per name - Ollama with GPU+CPU (its default: as much of
     the model on the GPU as fits, the rest on the CPU) vs CPU only
     (num_gpu 0). Same 30 benchmark names, model load excluded.
  2. Whole pipeline on the benchmark (7 files) - rules only (CPU; nothing
     in it can use this GPU, see README "larger NER models and the GPU"),
     with the LLM reviewer on GPU+CPU, and with it on CPU only.

Name-detector models on CPU vs GPU come from experiments/gpu_detector_probe.py
(experiments/probes/gpu_probe.json); full real-case scans from the ledger.

    python experiments/device_timing.py
"""
from __future__ import annotations

import copy
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.review.llm_reviewer as llm  # noqa: E402
from config import BASE_DIR, DEFAULT_CONFIG  # noqa: E402
from src.core.models import Decision  # noqa: E402
from src.pipeline.orchestrator import Pipeline  # noqa: E402

OUT = BASE_DIR / "experiments" / "probes" / "device_timing.json"
BENCHMARK = sorted((BASE_DIR / "datasets" / "benchmark").glob("*.txt"))
_real_request = llm._request


def _cpu_only(on: bool) -> None:
    """Routes every Ollama call through num_gpu 0 (CPU only) or restores the default."""
    def forced(url, payload=None, timeout=10.0):
        if payload is not None and url.endswith(("/api/chat", "/api/generate")):
            payload = {**payload, "options": {**payload.get("options", {}), "num_gpu": 0}}
        return _real_request(url, payload, timeout)
    llm._request = forced if on else _real_request


def _unload(settings: dict) -> None:
    _real_request(f"{settings['url']}/api/generate", {"model": settings["model"], "keep_alive": 0}, timeout=60)


def _load(settings: dict) -> float:
    """Loads the model in the current mode; returns GPU memory used (GB)."""
    llm._request(f"{settings['url']}/api/generate", {"model": settings["model"], "prompt": "", "keep_alive": "30m"}, timeout=900)
    ps = [m for m in _real_request(f"{settings['url']}/api/ps")["models"] if m["name"] == settings["model"]]
    return round(ps[0].get("size_vram", 0) / 2**30, 2) if ps else 0.0


def _config(llm_on: bool) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["feedback"]["enabled"] = False
    cfg["llm_reviewer"]["resolved"] = {"active": llm_on, "reason": "on (timing)" if llm_on else "off (timing)"}
    return cfg


def _pipeline_seconds(llm_on: bool) -> float:
    pipeline = Pipeline(config=_config(llm_on), base_dir=BASE_DIR)
    pipeline.run(BENCHMARK[0])  # warm-up (spaCy, model files)
    start = time.perf_counter()
    for path in BENCHMARK:
        pipeline.run(path)
    return round(time.perf_counter() - start, 2)


def _hardware() -> dict:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    import os
    return {"cpu": platform.processor(), "logical_cpus": os.cpu_count(), "gpu": gpu or "none"}


def main() -> int:
    logging.getLogger("person_extractor_v4").setLevel(logging.ERROR)
    settings = DEFAULT_CONFIG["llm_reviewer"]
    status = llm.resolve_status(settings)
    if not status.active:
        print(f"LLM reviewer unavailable here: {status.reason}")
        return 1

    # 30 benchmark names with their excerpts, from the pipeline's own output.
    probe = Pipeline(config=_config(False), base_dir=BASE_DIR)
    names = []
    for path in BENCHMARK:
        text = path.read_text(encoding="utf-8")
        for person in probe.run(path).persons:
            m = person.mentions[0].candidate
            names.append((person.display_text, [llm.excerpt(text, m.start, m.end)]))
    names = names[:30]
    reviewer = llm.LlmReviewer(settings, BASE_DIR / "assets")

    results = {"hardware": _hardware(), "model": f"{settings['model']} {settings['model_digest'][:12]}",
               "llm_seconds_per_name": {}, "llm_gpu_memory_gb": {}, "pipeline_benchmark_seconds": {}}
    print("rules only ...", flush=True)
    results["pipeline_benchmark_seconds"]["rules_only_cpu"] = _pipeline_seconds(False)
    for label, cpu_only in (("gpu_plus_cpu", False), ("cpu_only", True)):
        print(f"LLM {label} ...", flush=True)
        _unload(settings)
        _cpu_only(cpu_only)
        results["llm_gpu_memory_gb"][label] = _load(settings)
        start = time.perf_counter()
        for name, excerpts in names:
            reviewer.p_person(name, excerpts)
        results["llm_seconds_per_name"][label] = round((time.perf_counter() - start) / len(names), 3)
        results["pipeline_benchmark_seconds"][f"with_llm_{label}"] = _pipeline_seconds(True)
    _cpu_only(False)
    _unload(settings)  # the next normal load goes back onto the GPU

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
