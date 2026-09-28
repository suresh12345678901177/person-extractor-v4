"""
experiments/llm_reviewer_probe.py
====================================
Observer-only test of a local LLM as a final reviewer of the pipeline's
output. Changes no pipeline code and no decision - it answers "would asking
a local model 'is this a person's name here?' about each ACCEPTED/REVIEW
name improve the results?" before anything is built.

For every name the pipeline ACCEPTS or sends to REVIEW in the benchmark
files (datasets/benchmark/, N=7), the model sees the name and up to three
lines where it occurs, and its probability of answering "yes" is read from
the first token's log-probabilities. Scenarios are then scored with the
evaluator's own mention-level matching, pooled exactly like run_benchmark:

  baseline   ACCEPTED as the pipeline decides today
  cleanup@t  ACCEPTED names with P(yes) < t moved out of ACCEPTED
  promote@t  REVIEW names with P(yes) >= t moved into ACCEPTED
  both@t     cleanup@(1-t) and promote@t together

The model runs locally through Ollama (http://127.0.0.1:11434 - nothing
leaves the machine), temperature 0 and a fixed seed. Only counts go to
experiments/probes/; the per-name verdicts go to output/ (gitignored).

    python experiments/llm_reviewer_probe.py --model llama3.1:latest
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR, DEFAULT_CONFIG  # noqa: E402
from src.evaluation.evaluator import _FILE_SPAN_STRIDE, _find_benchmark_pairs  # noqa: E402
from src.evaluation.metrics import compute_span_metrics  # noqa: E402
from src.pipeline.orchestrator import Pipeline  # noqa: E402

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
THRESHOLDS = (0.5, 0.7, 0.9)


def _excerpt(text: str, start: int, end: int, width: int = 120) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end
    a, b = max(line_start, start - width), min(line_end, end + width)
    return f"{text[a:start]}[[{text[start:end]}]]{text[end:b]}".strip()


def _p_yes(model: str, name: str, excerpts: list[str], url: str) -> float:
    user = f"Candidate: {name}\nExcerpts:\n" + "\n".join(f"{i}. {e}" for i, e in enumerate(excerpts, 1)) + \
        f"\nIs [[{name}]] used as a person's name here?"
    body = {"model": model, "stream": False, "logprobs": True, "top_logprobs": 10,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": 0, "num_predict": 1}}
    req = urllib.request.Request(f"{url}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    reply = json.loads(urllib.request.urlopen(req, timeout=600).read())
    yes = no = 0.0
    for cand in reply["logprobs"][0]["top_logprobs"]:
        token = cand["token"].strip().lower()
        if token in ("yes", "y"):
            yes += math.exp(cand["logprob"])
        elif token in ("no", "n"):
            no += math.exp(cand["logprob"])
    return yes / (yes + no) if yes + no else 0.5


def case_mode(args) -> int:
    """Real case, from a finished scan: every ACCEPTED/REVIEW name with up to
    three excerpts found in its source files. COUNTS ONLY go to --out; the
    per-name verdicts (real names and context) go to --details under output/."""
    import csv
    import re
    from src.core.document_factory import DocumentFactory
    csv.field_size_limit(10**9)
    names_rows = list(csv.DictReader(open(args.scan_names, encoding="utf-8")))
    files_of: dict[str, list[str]] = {}
    for r in csv.DictReader(open(args.scan_files, encoding="utf-8")):
        path = Path(r["file_path"]) if Path(r["file_path"]).is_absolute() else args.case_dir / r["file_path"]
        for n in (r["names_found"] or "").split("; "):
            files_of.setdefault(n.strip(), []).append(str(path))
    # The per-file CSV lists ACCEPTED names only; REVIEW names are found
    # through the names CSV's source_files (base names) and a folder index.
    by_basename: dict[str, list[str]] = {}
    for p in args.case_dir.rglob("*"):
        if p.is_file():
            by_basename.setdefault(p.name, []).append(str(p))
    for r in names_rows:
        if r["name"] not in files_of:
            files_of[r["name"]] = [p for b in (r["source_files"] or "").split("; ") for p in by_basename.get(b.strip(), [])]
    factory, texts = DocumentFactory(), {}

    def text_of(path: str) -> str:
        if path not in texts:
            res = factory.create(path)
            texts[path] = res.document.full_text if res.success and res.document else ""
        return texts[path]

    judged, started = [], time.time()
    for r in names_rows:
        if r["decision"] not in ("accepted", "review"):
            continue
        name, excerpts = r["name"], []
        rx = re.compile(r"(?<![\w])" + re.escape(name) + r"(?![\w])")
        for path in files_of.get(name, [])[:6]:
            m = rx.search(text_of(path))
            if m:
                excerpts.append(_excerpt(text_of(path), m.start(), m.end()))
            if len(excerpts) == 3:
                break
        if not excerpts:
            continue
        judged.append({"name": name, "bucket": r["decision"], "occurrences": int(r["total_occurrences"]),
                       "excerpts": excerpts, "p_yes": _p_yes(args.model, name, excerpts, args.url)})
    seconds = time.time() - started

    def band(group, lo, hi):
        return sum(1 for n in group if lo <= n["p_yes"] < hi)
    summary = {"model": args.model, "prompt_version": PROMPT_VERSION, "names_judged": len(judged),
               "seconds": round(seconds, 1), "p_yes_bands": {}}
    for bucket in ("accepted", "review"):
        g = [n for n in judged if n["bucket"] == bucket]
        summary["p_yes_bands"][bucket] = {"total": len(g), "<0.1": band(g, 0, 0.1), "0.1-0.3": band(g, 0.1, 0.3),
                                          "0.3-0.7": band(g, 0.3, 0.7), "0.7-0.9": band(g, 0.7, 0.9), ">=0.9": band(g, 0.9, 2)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.details.write_text(json.dumps(sorted(judged, key=lambda n: n["p_yes"]), indent=1, ensure_ascii=False),
                            encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "experiments" / "probes" / "llm_reviewer_probe.json")
    parser.add_argument("--details", type=Path, default=BASE_DIR / "output" / "llm_reviewer_probe_details.json")
    parser.add_argument("--case-dir", type=Path, default=None, help="Real-case mode: the scanned folder")
    parser.add_argument("--scan-names", type=Path, default=None, help="Real-case mode: the scan's names CSV")
    parser.add_argument("--scan-files", type=Path, default=None, help="Real-case mode: the scan's per-file CSV")
    args = parser.parse_args()
    logging.getLogger("person_extractor_v4").setLevel(logging.ERROR)
    if args.case_dir:
        return case_mode(args)

    logging.getLogger("person_extractor_v4").setLevel(logging.ERROR)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["feedback"]["enabled"] = False
    pipeline = Pipeline(config=cfg, base_dir=BASE_DIR)

    names: list[dict] = []   # one per (file, name, bucket)
    gold_pooled: list[tuple[int, int]] = []
    started = time.time()
    for i, (txt_path, gold_path) in enumerate(_find_benchmark_pairs(BASE_DIR / "datasets" / "benchmark"), 1):
        text = txt_path.read_text(encoding="utf-8")
        gold = [(m["start"], m["end"]) for m in json.loads(gold_path.read_text(encoding="utf-8"))["mentions"]]
        base = i * _FILE_SPAN_STRIDE
        gold_pooled += [(s + base, e + base) for s, e in gold]
        result = pipeline.run(txt_path)
        for bucket, persons in (("accepted", result.persons), ("review", result.review_persons)):
            for person in persons:
                spans = [(m.candidate.start, m.candidate.end) for m in person.mentions]
                picks = sorted({0, len(spans) // 2, len(spans) - 1})
                excerpts = [_excerpt(text, *spans[k]) for k in picks]
                names.append({
                    "file": txt_path.name, "name": person.display_text, "bucket": bucket,
                    "spans": [(s + base, e + base) for s, e in spans],
                    "gold_hits": sum(any(not (e <= gs or s >= ge) for gs, ge in gold) for s, e in spans),
                    "p_yes": _p_yes(args.model, person.display_text, excerpts, args.url),
                })
        print(f"  {txt_path.name}: {sum(1 for n in names if n['file'] == txt_path.name)} names", flush=True)
    seconds = time.time() - started

    def score(accepted: list[dict]) -> dict:
        m = compute_span_metrics([sp for n in accepted for sp in n["spans"]], gold_pooled)
        return {"P": round(m.precision, 4), "R": round(m.recall, 4), "F1": round(m.f1, 4),
                "TP": m.true_positives, "FP": m.false_positives, "names": len(accepted)}

    acc = [n for n in names if n["bucket"] == "accepted"]
    rev = [n for n in names if n["bucket"] == "review"]
    scenarios = {"baseline": score(acc), "accepted_plus_review": score(acc + rev)}
    for t in THRESHOLDS:
        scenarios[f"cleanup@{t}"] = score([n for n in acc if n["p_yes"] >= 1 - t])
        scenarios[f"promote@{t}"] = score(acc + [n for n in rev if n["p_yes"] >= t])
        scenarios[f"both@{t}"] = score([n for n in acc if n["p_yes"] >= 1 - t] + [n for n in rev if n["p_yes"] >= t])

    def is_person(n: dict) -> bool:
        return n["gold_hits"] * 2 >= len(n["spans"])

    summary = {
        "model": args.model, "prompt_version": PROMPT_VERSION, "files": len({n["file"] for n in names}),
        "names_judged": len(names), "seconds": round(seconds, 1),
        "seconds_per_name": round(seconds / max(1, len(names)), 3),
        "name_level": {
            f"{bucket}_{'person' if want else 'not_person'}": {
                "count": len(group), "mean_p_yes": round(sum(n["p_yes"] for n in group) / max(1, len(group)), 3)}
            for bucket in ("accepted", "review") for want in (True, False)
            for group in [[n for n in names if n["bucket"] == bucket and is_person(n) == want]]
        },
        "scenarios_mention_level": scenarios,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.details.write_text(json.dumps(names, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
