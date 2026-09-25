"""
experiments/gpu_detector_probe.py
====================================
Observer-only comparison of person-NER models against the detector the
pipeline uses today (spaCy en_core_web_sm). Changes no pipeline code and no
decision - it answers "would a bigger model, on the GPU, be worth adding as
an extra detector?" before any of that is built:

  1. Benchmark coverage - of the gold mentions the pipeline leaves in REVIEW
     or misses entirely, how many each model tags as a person.
  2. False-positive propensity - person tags overlapping no gold mention
     (datasets/benchmark/, N=7 files), on benchmark_structured_dump_005
     (no people by construction) and on the held-out benchmark_unseen.
  3. Real-case noise volume, COUNTS ONLY - on a hash-random sample of case
     .txt files, how many distinct texts each model tags that the pipeline
     accepts, reviews, rejects, or never saw.
  4. Speed (chars/s, model load excluded), device, GPU memory and
     utilization (nvidia-smi sampled during inference).

trf_gpu needs spaCy's GPU backend (CuPy). On the machine this was written
on, Windows Application Control blocks CuPy's DLL, so the transformer
spaCy model was measured on CPU (trf_cpu) and GLiNER (torch) on the GPU.

Each model runs in its own process (spacy.require_gpu() is process-wide):
    python experiments/gpu_detector_probe.py detect --model trf_gpu
    python experiments/gpu_detector_probe.py analyze
Needs the GPU environment (requirements-gpu.txt). Tag offsets are cached in
--cache-dir (outside git). Only counts go to experiments/results/; a
per-text review list for the owner goes to output/ (gitignored).
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

BENCHMARK_DIR = BASE_DIR / "datasets" / "benchmark"
UNSEEN_DIR = BASE_DIR / "datasets" / "benchmark_unseen"
DUMP_FILE = "benchmark_structured_dump_005.txt"
RESULT_PATH = BASE_DIR / "experiments" / "probes" / "gpu_probe.json"
REVIEW_PATH = BASE_DIR / "output" / "self_upgrade" / "gpu_probe_review.csv"

MODELS = ("sm_cpu", "lg_cpu", "trf_cpu", "trf_gpu", "gliner_small_gpu", "gliner_medium_gpu")
CASE_SAMPLE_MAX_CHARS = 6_000_000
CASE_FILE_CHARS = (1_000, 600_000)
TRF_CPU_CASE_CHARS = 600_000


# --------------------------------------------------------------------- texts

def _clean(path: Path) -> str:
    from src.core.document_factory import DocumentFactory
    from src.preprocessing.cleaner import clean_text
    result = DocumentFactory().create(str(path))
    return clean_text(result.document.full_text) if result.success else ""


def load_texts(case_dir: Path | None, include_case: bool) -> dict[str, str]:
    """Keys: 'bench/<name>', 'unseen/<name>', 'case/<n>'. Case files: every
    .txt in the case folder ordered by a hash of its relative path (a random
    sample nobody chose), kept while within size bounds, until
    CASE_SAMPLE_MAX_CHARS. Order and membership are deterministic."""
    texts = {f"bench/{p.name}": _clean(p) for p in sorted(BENCHMARK_DIR.glob("*.txt"))}
    texts.update({f"unseen/{p.name}": _clean(p) for p in sorted(UNSEEN_DIR.glob("*.txt"))})
    if include_case and case_dir is not None:
        files = sorted(
            (p for p in case_dir.rglob("*.txt") if p.is_file()),
            key=lambda p: hashlib.sha1(p.relative_to(case_dir).as_posix().encode()).hexdigest(),
        )
        total = 0
        for i, path in enumerate(files):
            if not (CASE_FILE_CHARS[0] <= path.stat().st_size <= CASE_FILE_CHARS[1] * 2):
                continue
            text = _clean(path)
            if not (CASE_FILE_CHARS[0] <= len(text) <= CASE_FILE_CHARS[1]):
                continue
            texts[f"case/{i}"] = text
            total += len(text)
            if total >= CASE_SAMPLE_MAX_CHARS:
                break
    return texts


def case_paths(case_dir: Path, keys: list[str]) -> dict[str, Path]:
    files = sorted(
        (p for p in case_dir.rglob("*.txt") if p.is_file()),
        key=lambda p: hashlib.sha1(p.relative_to(case_dir).as_posix().encode()).hexdigest(),
    )
    return {k: files[int(k.split("/")[1])] for k in keys if k.startswith("case/")}


# ----------------------------------------------------------------- detection

class _GpuSampler:
    """nvidia-smi utilization/memory every 0.5s while a model runs."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, encoding="utf-8", timeout=5,
                ).stdout.strip().split(",")
                self.samples.append((float(out[0]), float(out[1])))
            except Exception:  # noqa: BLE001 - sampling is best-effort
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        if not self.samples:
            return {}
        util = [s[0] for s in self.samples]
        mem = [s[1] for s in self.samples]
        return {"gpu_util_mean_pct": round(statistics.mean(util), 1), "gpu_util_max_pct": max(util),
                "gpu_mem_used_max_mib": max(mem), "samples": len(self.samples)}


def _line_chunks(text: str, max_chars: int) -> list[tuple[int, str]]:
    from src.detection.spacy_detector import iter_line_bounded_chunks
    return [(o, c) for o, c in iter_line_bounded_chunks(text, max_chars) if c.strip()]


def _spacy_spans(model: str, texts: dict[str, str], gpu: bool, chunk_chars: int, batch_size: int):
    import spacy
    from src.detection.spacy_detector import SpacyDetector
    if gpu:
        spacy.require_gpu()
    nlp = spacy.load(model, disable=["lemmatizer", "attribute_ruler", "tagger", "parser"])
    spans: dict[str, list[list[int]]] = {}
    started = time.perf_counter()
    for key, text in texts.items():
        chunks = _line_chunks(text, chunk_chars)
        found = []
        for (offset, _), doc in zip(chunks, nlp.pipe((c for _, c in chunks), batch_size=batch_size)):
            found += [[d.start, d.end] for d in SpacyDetector._entities_to_detections(doc, 0, offset)]
        spans[key] = found
    return spans, time.perf_counter() - started


def _gliner_spans(model_id: str, texts: dict[str, str], chunk_chars: int, batch_size: int, threshold: float):
    """model_id is a vendored local folder whose gliner_config.json points
    model_name at a local copy of the backbone's tokenizer/config -
    HF_HUB_OFFLINE is set in detect(), so nothing is fetched at run time."""
    import torch
    from gliner import GLiNER
    model = GLiNER.from_pretrained(model_id, local_files_only=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    spans: dict[str, list[list[int]]] = {}
    started = time.perf_counter()
    for key, text in texts.items():
        chunks = [(o, c) for o, c in _line_chunks(text, chunk_chars)]
        # GLiNER reads ~384 words: split any chunk still too long at whitespace.
        pieces: list[tuple[int, str]] = []
        for offset, chunk in chunks:
            while len(chunk) > chunk_chars:
                cut = chunk.rfind(" ", 0, chunk_chars)
                cut = cut if cut > 0 else chunk_chars
                pieces.append((offset, chunk[:cut]))
                offset, chunk = offset + cut, chunk[cut:]
            pieces.append((offset, chunk))
        found = []
        for i in range(0, len(pieces), batch_size):
            batch = pieces[i:i + batch_size]
            with torch.no_grad():
                results = model.batch_predict_entities([c for _, c in batch], ["person"], threshold=threshold)
            for (offset, chunk), ents in zip(batch, results):
                for ent in ents:
                    # one tag per line segment, like SpacyDetector
                    s, e = ent["start"], ent["end"]
                    cursor = s
                    for part in chunk[s:e].split("\n"):
                        if part.strip():
                            lead = len(part) - len(part.lstrip())
                            found.append([offset + cursor + lead, offset + cursor + lead + len(part.strip())])
                        cursor += len(part) + 1
        spans[key] = found
    return spans, time.perf_counter() - started


def detect(model: str, cache_dir: Path, case_dir: Path | None, include_case: bool, weights_dir: Path | None) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    texts = load_texts(case_dir, include_case)
    # The CPU transformer would take hours on the whole case sample: benchmark
    # files plus the first TRF_CPU_CASE_CHARS of the sample only.
    if model == "trf_cpu":
        kept, case_chars = {}, 0
        for k, v in texts.items():
            if k.startswith("case/"):
                if case_chars >= TRF_CPU_CASE_CHARS:
                    continue
                case_chars += len(v)
            kept[k] = v
        texts = kept
    chars = sum(len(t) for t in texts.values())
    info: dict = {"model": model, "texts": len(texts), "chars": chars}

    with _GpuSampler() as sampler:
        if model == "sm_cpu":
            spans, seconds = _spacy_spans("en_core_web_sm", texts, False, 400_000, 1)
        elif model == "lg_cpu":
            spans, seconds = _spacy_spans("en_core_web_lg", texts, False, 400_000, 1)
        elif model in ("trf_gpu", "trf_cpu"):
            spans, seconds = _spacy_spans("en_core_web_trf", texts, model == "trf_gpu", 5_000, 32)
        elif model.startswith("gliner_"):
            if weights_dir is None:
                raise SystemExit("GLiNER needs --weights-dir (vendored weights; no hub lookups)")
            name = {"gliner_small_gpu": "gliner_small-v2.1", "gliner_medium_gpu": "gliner_medium-v2.1"}[model]
            spans, seconds = _gliner_spans(str(weights_dir / name), texts, 1_000, 16, 0.5)
        else:
            raise SystemExit(f"unknown model {model}")
    info["seconds"] = round(seconds, 2)
    info["chars_per_second"] = round(chars / seconds) if seconds else None
    info["gpu"] = sampler.summary() if model.endswith("_gpu") else {}
    try:
        import torch
        if torch.cuda.is_available():
            info["torch_cuda_max_mem_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
            info["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{model}.json").write_text(json.dumps({"info": info, "spans": spans}), encoding="utf-8")
    print(json.dumps(info, indent=2))


# ------------------------------------------------------------------ analysis

def _overlaps(a: list[int] | tuple[int, int], spans) -> bool:
    return any(not (a[1] <= s or a[0] >= e) for s, e in spans)


def analyze(cache_dir: Path, case_dir: Path | None) -> None:
    import logging
    from config import DEFAULT_CONFIG
    from src.evaluation.evaluator import _gold_mention_shapes
    from src.pipeline.orchestrator import Pipeline

    for handler in logging.getLogger("person_extractor_v4").handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.WARNING)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["feedback"]["enabled"] = False
    pipeline = Pipeline(config=config, base_dir=BASE_DIR)
    kb = pipeline.extractor.knowledge_base

    runs = {m: json.loads((cache_dir / f"{m}.json").read_text(encoding="utf-8"))
            for m in MODELS if (cache_dir / f"{m}.json").exists()}
    if not runs:
        raise SystemExit(f"no detect results in {cache_dir}")
    keys = sorted({k for r in runs.values() for k in r["spans"]})

    # The pipeline's own verdict on every text the models saw.
    paths = {f"bench/{p.name}": p for p in BENCHMARK_DIR.glob("*.txt")}
    paths.update({f"unseen/{p.name}": p for p in UNSEEN_DIR.glob("*.txt")})
    if case_dir is not None:
        paths.update(case_paths(case_dir, [k for k in keys if k.startswith("case/")]))
    verdicts = {}
    for key in keys:
        result = pipeline.run(paths[key])
        verdicts[key] = {
            "accepted": [(m.candidate.start, m.candidate.end) for p in result.persons for m in p.mentions],
            "review": [(m.candidate.start, m.candidate.end) for p in result.review_persons for m in p.mentions],
            "rejected": [(c.candidate.start, c.candidate.end) for c in result.rejected],
            "skipped": bool(result.model_info.get("skipped_reason")),
        }
    gold = {}
    for key in keys:
        if key.startswith(("bench/", "unseen/")):
            gpath = paths[key].with_name(paths[key].stem + "_gold.json")
            gold[key] = json.loads(gpath.read_text(encoding="utf-8"))["mentions"] if gpath.exists() else []

    report: dict = {"note": "Counts only. Benchmark = datasets/benchmark (N=7 files); unseen = held out, "
                            "measured not tuned; case = hash-random sample of case .txt files, no gold labels.",
                    "models": {}}
    review_rows = []
    for model, run in runs.items():
        spans = run["spans"]
        out: dict = {"speed": run["info"]}
        for group in ("bench", "unseen"):
            g_keys = [k for k in keys if k.startswith(group + "/") and k in spans]
            tags = sum(len(spans[k]) for k in g_keys)
            tags_on_gold = sum(1 for k in g_keys for t in spans[k] if _overlaps(t, [(m["start"], m["end"]) for m in gold[k]]))
            buckets = {"accepted": [0, 0], "review_only": [0, 0], "missed": [0, 0], "unseen_name_review_or_missed": [0, 0]}
            for k in g_keys:
                v = verdicts[k]
                for m in gold[k]:
                    span = (m["start"], m["end"])
                    state = "accepted" if _overlaps(span, v["accepted"]) else \
                        "review_only" if _overlaps(span, v["review"]) else "missed"
                    tagged = _overlaps(span, spans[k])
                    buckets[state][0] += 1
                    buckets[state][1] += tagged
                    if state != "accepted" and _gold_mention_shapes(m["text"], kb)[1] == "unseen_name":
                        buckets["unseen_name_review_or_missed"][0] += 1
                        buckets["unseen_name_review_or_missed"][1] += tagged
            gold_total = sum(len(gold[k]) for k in g_keys)
            out[group] = {
                "person_tags": tags,
                "tags_on_a_gold_mention": tags_on_gold,
                "tags_on_no_gold_mention": tags - tags_on_gold,
                "tag_precision": round(tags_on_gold / tags, 4) if tags else None,
                "gold_mentions": gold_total,
                "gold_tagged": sum(b[1] for s, b in buckets.items() if s != "unseen_name_review_or_missed"),
                "by_pipeline_state (gold, tagged_by_model)": buckets,
            }
            if group == "bench":
                out[group]["dump005_person_tags"] = len(spans.get(f"bench/{DUMP_FILE}", []))

        case_keys = [k for k in keys if k.startswith("case/") and k in spans]
        if case_keys:
            texts = {k: None for k in case_keys}
            classes: dict[str, set] = {"pipeline_accepted": set(), "pipeline_review": set(),
                                       "pipeline_rejected": set(), "pipeline_never_saw": set(),
                                       "file_skipped_by_language_gate": set()}
            tag_count = 0
            for k in case_keys:
                if texts[k] is None:
                    texts[k] = _clean(paths[k])
                v = verdicts[k]
                for s, e in spans[k]:
                    tag_count += 1
                    surface = " ".join(texts[k][s:e].split())
                    if v["skipped"]:
                        cls = "file_skipped_by_language_gate"
                    elif _overlaps((s, e), v["accepted"]):
                        cls = "pipeline_accepted"
                    elif _overlaps((s, e), v["review"]):
                        cls = "pipeline_review"
                    elif _overlaps((s, e), v["rejected"]):
                        cls = "pipeline_rejected"
                    else:
                        cls = "pipeline_never_saw"
                    if surface.lower() not in classes[cls]:
                        review_rows.append([model, cls, surface, str(paths[k]), texts[k].count("\n", 0, s) + 1])
                    classes[cls].add(surface.lower())
            out["case_sample"] = {"files": len(case_keys), "person_tags": tag_count,
                                  "distinct_texts_by_pipeline_state": {c: len(v) for c, v in classes.items()}}
        report["models"][model] = out

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if review_rows:
        REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REVIEW_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["model", "pipeline_state", "tagged_text", "file", "line"])
            writer.writerows(review_rows)
    print(json.dumps(report, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--model", choices=MODELS, required=True)
    a = sub.add_parser("analyze")
    for p in (d, a):
        p.add_argument("--cache-dir", type=Path, required=True)
        p.add_argument("--case-dir", type=Path, default=None)
    d.add_argument("--no-case", action="store_true")
    d.add_argument("--weights-dir", type=Path, default=None,
                   help="Folder holding vendored GLiNER weights (e.g. gliner_small-v2.1/)")
    args = parser.parse_args()
    if args.cmd == "detect":
        detect(args.model, args.cache_dir, args.case_dir, not args.no_case, args.weights_dir)
    else:
        analyze(args.cache_dir, args.case_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
