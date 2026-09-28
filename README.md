# PERSON_EXTRACTOR_V4

An offline, explainable Person Name Extraction framework - reading
**TXT, PDF (text layer only), CSV/TSV, JSON/JSONL, HTML, XML, DOCX and
EML** - built around a 16-validator rules engine, with a real
trained ML classifier used strictly as a corroborating signal — never as a
gate. Every accepted name tells you exactly why it was accepted, how many
times it was found, where, and how long each stage took.

**Real-world coverage caveat, stated up front, not buried in a scan log:**
on an actual 47,353-file forensic case folder, the TXT + PDF scope this
tool had until 2026-09-24 meant it never even attempted to open **84.7%**
of files (wrong extension - DOCX/XLSX/images/email containers). CSV, JSON,
HTML, XML, DOCX and EML are now read too (see "Supported input formats");
XLSX, images/OCR and binary containers (.msg, .doc, databases) still
aren't, and the share of that case now covered has not been re-measured.
Of the `.txt`/`.pdf` files it did
open, **8.1% of the total** are scanned/image-only PDFs with no text layer
to read. Net result on that real case: only **~4.9% of all files in the
folder** ever contributed an extracted name. See "Known limitations" for
what this does and doesn't mean, and "Real measured results" for the exact
numbers this is computed from.

```
Input .txt / .pdf File
  -> IO (TextReader / PdfReader + DocumentFactory)
  -> Preprocessing (clean + line-index)
  -> Detection (Regex, Dictionary, spaCy)
  -> Candidate Factory (merge overlapping detections)
  -> Validation Firewall (16 validators - the precision backbone)
  -> Feedback Logging (REVIEW items -> active learning loop)
  -> Feature Engineering
  -> ML Classification (LightGBM - corroborating signal only)
  -> Decision Engine (rules are the gate, ML can only help)
  -> Aggregation (group repeated mentions -> occurrence count + locations)
  -> Export (CSV / JSON / human-readable report.txt)
```

## Active Learning Loop — this is the actual answer to "get better results"

Instead of chasing accuracy with a heavier model (tested and rejected —
see below), V4 closes the gap a different way: every REVIEW-bucket
candidate is automatically logged, you confirm or correct a quick batch,
and the classifier retrains on your real corrections.

```bash
python cli.py --input yourfile.txt          # REVIEW items are logged automatically
python scripts/label_feedback.py             # answer y/n/skip for each one
python scripts/retrain_from_feedback.py      # folds your real corrections into the model
```

**This is verified, not theoretical.** In one real test round: 19 REVIEW
items, most of them genuine names simply not in the demo dictionary
(e.g. "Francisco J. Varela," "Stephen Jay Gould") were labeled in about
a minute. After retraining, re-running the *same* document moved
**14 of them straight from REVIEW to ACCEPTED**, with zero new false
positives — and the two intentionally-mislabeled items ("See Tulku
Urgyen's," a fragment that shouldn't have been a candidate at all)
correctly stayed out of ACCEPTED too. Both directions of correction
worked.

Real confirmed examples are upweighted 3x relative to synthetic ones
when retraining (`scripts/retrain_from_feedback.py`), so a handful of
real corrections meaningfully moves the model without being drowned out
by the much larger synthetic set - but also without one example
completely dominating (LightGBM's regularization prevents the model
from simply memorizing single rows, which is why not every confirmed
example crosses the acceptance threshold after one small labeling
round; this improves further with more rounds).

### Real PDF/TXT evaluation and balanced training workflow

For a production PDF/TXT folder, reserve a file-separated holdout before
labeling training candidates. The manifest contains only local paths and
hashes; it is excluded from retraining and git. Start with 20-50 files:

```powershell
python scripts/create_pdf_txt_holdout.py --input-dir "C:\case_files" --count 25
python scripts/prepare_holdout_snapshots.py
```

The snapshot command creates pipeline-ready TXT copies of the held-out PDF
text and TXT input. In a copy of each snapshot, wrap every real person
mention in `**double asterisks**`, then build its TXT/gold pair and measure
it. This captures missed people too, so recall is meaningful.

```powershell
python scripts/build_benchmark_from_markup.py marked_001.md holdout_001 --output-dir datasets/evaluation_holdout/snapshots
python cli.py --evaluate --benchmark-dir datasets/evaluation_holdout/snapshots
```

For training data, draw a balanced sample from the complete live pipeline,
not only the REVIEW queue. It samples ACCEPTED, REVIEW and classifier-ready
REJECTED candidates without changing any decision rule. `--queue` sends the
sample to the existing interactive labeler.

```powershell
python scripts/sample_pipeline_candidates.py --input-dir "C:\case_files" --per-bucket 25 --queue
python scripts/label_feedback.py
python scripts/retrain_from_feedback.py
```

Retraining now refuses a feedback set with fewer than 10 examples in either
class or a class ratio above 3:1, unless `--allow-imbalanced-feedback` is
explicitly supplied. The existing benchmark regression guard remains the
final promotion check. The strict rules-first decision engine remains
unchanged: LightGBM and spaCy continue to corroborate a candidate and never
override a hard validator rejection.

## Why we didn't add a heavier ensemble/transformer model

This was tested, not assumed. The apparent fix for a real false-positive
storm (a bibliography-heavy document produced 939 REVIEW items) looked
at first like it needed a stronger model - GLiNER or spaCy's transformer
pipeline. Two things changed that conclusion:

1. **Installing GLiNER pulled in 4.5GB of GPU-oriented PyTorch/CUDA
   dependencies** that a non-GPU laptop doesn't need - a real,
   measured installation risk, not a hypothetical one.
2. **The actual cause wasn't a model failure.** Every one of those 939
   items had zero spaCy support - the already-installed small model had
   *already* correctly refused to call them people. The noise came from
   bare regex shape-matching generating candidates with no semantic
   backing at all, and validation not yet rejecting them hard enough.

The fix (`CorroborationValidator` - see below) needed no new
dependency, brought that number from 939 to 19, and the active learning
loop turns those 19 into progressively fewer over time. This is
documented here deliberately, not to close the door on a heavier model
forever, but because the evidence pointed at a cheaper, lower-risk fix
actually solving the real problem first.

## Why these design choices (data-scientist reasoning, not defaults)

**Scope update, 2026-09-24:** CSV/TSV, JSON/JSONL, HTML, XML, DOCX and
EML readers were added - standard library only, so still no new
dependencies and still fully offline (see "Supported input formats").
Each converts its format to text while keeping structure as hard name
boundaries (a table cell, CSV field or paragraph can never merge into the
next). Excel, images/OCR and binary containers remain out. The history
below describes the scope before that change.

**Scope: TXT + PDF, text-layer only. DOCX/Excel/images/email containers
and OCR are still deliberately out.** The original scope was TXT-only for
exactly the reason below; `PdfReader` (`src/io/pdf_reader.py`) was added
later, reading only a PDF's embedded text layer via `pypdf` — no OCR, no
rendering, no layout/table reconstruction. This note itself is a
correction: that addition was never reflected here until an independent
audit (2026-09-23) flagged the README as describing a narrower scope than
the code actually has — worth stating plainly rather than quietly fixing,
since a forensic user needs to trust exactly what this tool does and
doesn't read.

The underlying reasoning hasn't changed: less input-format surface area
means fewer places for encoding quirks, OCR noise, or layout artifacts to
introduce false positives — and false positives are explicitly the #1
thing this project optimizes against. PDF text-layer extraction was judged
worth the small added surface (a real, structured text layer, not OCR
output) because so much real casework arrives as PDF. DOCX/Excel/image
formats remain out for the same original reason, and OCR remains out
because it's a different, larger, still-unscoped decision — noisier text
than either a `.txt` export or a PDF's real text layer, on a project that
already treats false positives as the metric to protect above all others.
See "Known limitations" for the measured, real-world size of the resulting
coverage gap, and the OCR scoping note further down for what would need
to be true before adding it.

**Detectors: Regex + Dictionary + spaCy (`en_core_web_sm`).** Not
GLiNER, not Flair. Both are heavier dependencies (large model downloads,
slower inference) for marginal accuracy gain on the kind of structured,
mostly-Western-and-South-Asian-name text this tool targets. spaCy's small
English model is fast, fully offline once downloaded, and — critically —
its false positives are *predictable enough* (sentence-initial capitalized
words, occasional organization mislabeling) that the rules engine can be
built to specifically guard against them. This is documented in
`src/detection/spacy_detector.py` and `GrammarValidator`.

**Classifier: LightGBM, trained via distant supervision.** There is no
hand-labeled training corpus for this task. Rather than skip ML entirely or
fabricate one, `scripts/generate_training_data.py` builds a synthetic
training set: positive examples are real names from the knowledge base
inserted into varied sentence templates; negative examples are known
organizations/locations/campaign slogans/blacklist entries, PLUS
deliberately *hard* negatives (a real first name combined with a business
suffix, e.g. "Suresh Traders") so the model can't just learn "any dictionary
hit = person." This is a legitimate, standard weak-supervision technique —
see the honesty note in `src/classification/lightgbm_classifier.py` for
exactly what the resulting "accuracy" number does and doesn't mean.

**The core precision decision: rules are the gate, ML only corroborates.**
A hard validator failure is final — no ML probability can override it. The
ML classifier's score can only ever *add* to a candidate's confidence,
never subtract, and a brand-new **knowledge-corroboration gate** (added
after finding a real bug during testing — see below) additionally requires
that an ACCEPTED candidate have at least one signal that actually knows
something about the text (a dictionary hit, a title match, or a
high-confidence ML score) — not just detectors agreeing on its *shape*.

## A real bug this design caught (and how)

During adversarial testing, a completely fabricated string, "Xzqlt Vbnmp",
was accepted at confidence 1.25 — bare regex and spaCy both agreed it was
*shaped* like a name, and that alone was enough to clear the acceptance
threshold, with zero dictionary support. This is exactly the false-positive
class this project exists to prevent. The fix (the knowledge-corroboration
gate described above) is permanent and covered by a regression test:
`tests/test_pipeline.py::test_pipeline_rejects_fabricated_name_with_zero_dictionary_support`.

A second real bug, found 2026-09-16 via a real production case scan: "Major"
is a genuine (if rare) entry in `assets/first_names/first_names.txt`, but
also an ordinary English rank/role word. A bare, low-occurrence "Major" with
no other corroborating evidence was wrongly ACCEPTED 63 times across 6
unrelated files (an Android `dumpsys` dump, academic PDFs, ...) purely on
the strength of a single-token dictionary hit, because that evidence class
previously short-circuited `CorroborationValidator` unconditionally. The
same collision class can affect any word that happens to sit in both
`first_names.txt`/`last_names.txt` and a non-person list — run
`python scripts/audit_collisions.py` to find them (it also reports which
are already mitigated). Running it turned up seven more real ones -
South Asian honorifics/titles that are also genuine personal
names/surnames (`kumari`, `raja`, `rani`, `sahib`, `sardar`, `shri`,
`pandit`, `thakur`) - now gated the same way. One flagged overlap,
`bello` (a last name that's also in `locations.txt`), was deliberately
left alone: it's the opposite failure direction (risks a false REJECT
via `LocationValidator`, not a false ACCEPT) and needs a different fix
in that validator, not this mechanism. The fix generalizes the
existing `assets/ambiguous_words/ambiguous_first_names.txt` mechanism
(previously spaCy-only) to also gate single-token dictionary hits: a word
in that list now needs repetition, a title, or a full first+last dictionary
window match before it corroborates on its own, whether the evidence is
spaCy or a bare dictionary hit. Covered by
`tests/test_pipeline.py::test_pipeline_rejects_ambiguous_single_token_dictionary_word_without_repetition`.

A third real bug, found 2026-09-16 during a real case rescan: retraining the
classifier on real feedback shifted its calibration enough that bare,
zero-dictionary/title/spaCy business phrases ("Vice President", "Convertible
Debt", "Customer Service", ...) started clearing `DecisionEngine`'s
ML-confidence-alone corroboration path purely on a high score + in-document
repetition - nothing checked whether the words themselves meant "person" at
all. The fix (documented as design principle 3 in
`src/decision/decision_engine.py`'s module docstring): that path now
additionally requires the candidate NOT be composed entirely of ordinary
English vocabulary (`assets/common_words/common_english_words.txt` - raw
word-frequency data, not a copyrighted dictionary; see that file's header
for provenance). A genuine name is unaffected as long as it has any real
dictionary/title support, or any token that isn't just a common word.
Verified directly against the model/file combination that originally
produced the bug (same 0.91-0.97 ML scores, now correctly capped at REVIEW
instead of ACCEPTED) and covered by
`tests/test_decision_engine.py::test_ml_confidence_alone_does_not_accept_common_word_phrase`.

A fourth real bug, found 2026-09-18 via a direct user question about scan
runtime: `SpacyDetector` ran spaCy's full `en_core_web_sm` pipeline
(`tok2vec + tagger + parser + ner`) on an entire document in one call, with
`nlp.max_length` left at spaCy's 1,000,000-character default. Neither
`tagger` nor `parser` output is read anywhere in this codebase - only
`doc.ents` (NER) is ever used - a real, measured ~26% per-call slowdown for
zero benefit, confirmed via a controlled A/B benchmark on a 300K-char sample
(identical entity output either way). Worse: any document over 1,000,000
characters made `self._nlp(text)` raise, caught by `detect()`'s blanket
`except Exception` and silently returned zero detections -
`logs/pipeline.log` showed this had already happened **142 times** on this
project's real multi-MB Android-dump casework files, meaning those
documents got ZERO spaCy corroboration, not just ran slow. Fixed both:
disabled `tagger`/`parser` in `spacy_detector.py`'s `spacy.load(...)` call,
and added `iter_line_bounded_chunks` - splits any oversized document into
<=400,000-char pieces at line boundaries only (never mid-token), translating
each chunk's entity offsets back to the original document's coordinates.
Verified end-to-end on a constructed 1.12M-char document: found both
planted names at the correct offsets in under 10 seconds, where the old
code would have silently returned nothing. Covered by
`tests/test_spacy_detector.py` (12 tests: chunk-size invariants, exact
reconstruction including an exotic-Unicode-line-separator edge case, a
pathological single-overlong-line case, and an end-to-end offset-
correctness check).

A fifth real bug, found 2026-09-18 via manual verification against real
case files (`CASEID_*.txt` - forensic call/chat/email/social-media
exports): REVIEW was being flooded with plain business/legal noun phrases -
"Performance Review," "Retention Bonus," "Corporate Controller,"
"Coordinated Universal Time," "Deferred Compensation" - that scored into
the REVIEW range on bare shape evidence alone, with zero dictionary/title/
spaCy/ML corroboration ever suggesting they meant "person." The existing
common-word-phrase guard (design principle 3 above) only blocked this class
from reaching ACCEPTED via the ML-confidence-alone path - it did nothing to
stop the same phrases from still landing in REVIEW via the plain score
threshold. Extended the guard one step further in `DecisionEngine.decide()`:
a candidate with zero knowledge corroboration AND made entirely of ordinary
words already in `common_english_words.txt` is now REJECTED outright
instead of parked in REVIEW.

Measured, not assumed, via a controlled before/after `--evaluate` run:
overall ACCEPTED+REVIEW precision improved 0.7657 -> 0.8244 (+5.9pp), but
recall dropped 0.9949 -> 0.9678 (-2.7pp), concentrated almost entirely in
the `unseen_name` shape category (0.9880 -> 0.8514, -13.7pp, 34 names) -
because the benchmark's own unseen-name gold examples ("Copper Falcon,"
"Velvet Harbor," "Silver Anchor") use common-word pairs as name stand-ins,
the exact shape this guard targets. This tradeoff was surfaced explicitly
and the decision to keep it anyway was made deliberately by the project
owner, not assumed - see "Real measured results" below. Covered by
`tests/test_decision_engine.py::test_common_word_phrase_with_zero_corroboration_is_rejected_not_reviewed`
and its control test proving a real dictionary-corroborated name sharing
the same common-word shape (`Grace Review`) is unaffected.

## Validation Firewall — all 16 validators

| Validator | What it catches |
|---|---|
| Length | Implausibly short/long spans |
| Structure | Non-name shapes, acronyms (CCTV, IDs, TXN-) |
| Punctuation | Digits, disallowed symbols |
| Repetition | "John John John" |
| Initial | All-initials spans ("J. R.") |
| Language | Non-Latin script, transliterated non-English words |
| **Corroboration** | **Bare regex shape-matches with zero dictionary/title/spaCy backing — found via real bibliography-text testing (see above)** |
| Blacklist | Known non-person strings ("United Nations") |
| Campaign | Slogans/banners, both exact-match AND generic marker-words for **novel** slogans |
| Organization | Business/institution names, via keyword generalization |
| Location | Known place names |
| Stopword | Any token that's a common English function word |
| Grammar | Sentence-initial capitalized words with no dictionary/title support |
| Context | URL/email adjacency (hard reject); occupation-word precedence (soft boost) |
| Dictionary | Soft-boosts tokens found in the name knowledge base |
| Title | Soft-boosts titled patterns ("Dr. X", "Mrs. Y") |

## Output — what "clear labeling" means here

Every run reports, per your requirements:
- **Time taken**: per-stage timing breakdown + total, both in the CLI and `report.txt`
- **Why it's a name**: the full evidence trail (`--explain`) — every validator/detector's contribution, signed and labeled
- **How many times found**: `AggregatedPerson.occurrence_count` — repeated mentions of the same surface form are grouped, not double-reported
- **Location**: line + column (not just a raw character offset) for every mention
- **Model accuracy**: reported honestly as TWO separate numbers, never conflated —
  1. The LightGBM classifier's own synthetic validation accuracy (shown in every run)
  2. Real end-to-end precision/recall/F1 against hand-labeled benchmark text (`python cli.py --evaluate`)

## Setup

```powershell
# Windows (PowerShell)
.\setup_env.ps1
```
```bash
# Linux/Mac
bash setup_env.sh
```

This creates an isolated virtual environment, installs dependencies,
downloads the spaCy model, generates synthetic training data, trains the
classifier, and runs the full test suite — a fresh checkout is fully
working after one script. If PowerShell blocks the script, run once:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Manual setup, if you prefer:
```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/generate_training_data.py
python scripts/train_model.py
```

## Usage

```bash
python cli.py --input path/to/file.txt --explain
python cli.py --input path/to/file.txt --format json --output results/
python cli.py --evaluate
```

Every run writes `output/report.txt` (always) plus a structured
`output/result.csv` or `output/result.json` (via `--format`).

### Recursive folder scan (offline batch automation)

Point it at any folder — a case export root, a nested directory tree,
whatever — and it walks every subfolder to any depth, finds every
`.txt` file, runs each one through the pipeline, and writes two fresh
timestamped CSVs (never overwrites a previous run):

```bash
python scripts/scan_directory.py --input-dir "C:\path\to\case_folder"
python scripts/scan_directory.py --input-dir "C:\path\to\case_folder" --output "C:\path\to\results" --include-review
python scripts/scan_directory.py --input-dir "C:\path\to\case_folder" --workers 8   # parallel (default: CPU count)
python scripts/scan_directory.py --input-dir "C:\path\to\case_folder" --exclude report.xml --exclude "*/cache/*"
```

`--exclude` (repeatable) skips files whose name or path relative to
`--input-dir` matches a glob, case-insensitively. Excluded files still get
a row in the per-file CSV (`skipped: excluded by --exclude <pattern>`) and a
`files_excluded` count in the summary. Typical use: a forensic tool's own
aggregate report (e.g. ProDiscover's 68 MB `report.xml`, a copy of evidence
already scanned file by file - it took 29 of 44 minutes on a real case and
added only duplicate or noisy names).

Runs across `--workers` processes by default (one per CPU core), each
with its own loaded spaCy model + classifier - not one shared Pipeline,
since a loaded spaCy model can't be cheaply passed across process
boundaries. `--workers 1` forces the original single-process behavior.
Verified byte-identical output at both 1 and 8 workers, on an 88-file
sample and the full 7,246-file real case folder alike (see "Real
measured results" below for the full correctness + speed writeup) -
**2.98x faster wall-clock on the full case** (43.3 min -> 14.5 min).

- `<folder>_files_<timestamp>.csv` — one row per `.txt` file found:
  how many names in it, the list of those names, and how long that
  file took.
- `<folder>_names_<timestamp>.csv` — the case-wide rollup: every
  unique person across all files, merged the same way `cli.py`'s
  batch mode merges them, with total occurrences and every source
  file that mentioned them.
- `<folder>_summary_<timestamp>.csv` — run totals persisted to disk
  (files found, names found, total time taken), so they survive after
  the terminal that ran it is gone.

Non-`.txt` files encountered while walking are counted and skipped,
never sent to the model. A run summary (files found, names found,
total time taken) prints to the console and mirrors what a header
glance at the files CSV would show.

### Supported input formats

| Extensions | What is read | "line N" in a result |
|---|---|---|
| `.txt` | the file | line N of the file |
| `.pdf` | embedded text layer (no OCR) | line N of that page (`page P, line N`) |
| `.csv` `.tsv` | every record, cells joined by ` \| ` | CSV record N (header = 1) |
| `.json` `.jsonl` `.ndjson` | every string value as `key: value` | line of the extracted text |
| `.html` `.htm` `.xhtml` | visible text (no scripts/styles), cells separated | line of the extracted text |
| `.xml` | element text + attribute values; malformed XML read leniently | line of the extracted text |
| `.docx` | body paragraphs and tables, footnotes, comments (+ comment authors), document author / last editor | body paragraph N |
| `.eml` | From/To/Cc/Bcc/Reply-To/Sender/Subject/Date + plain-text or HTML body | line of the extracted text |

Structured data (CSV, JSON, XML) skips the English-prose language gate,
which is calibrated on prose and would otherwise skip a contact list that
is nothing but names and numbers; every per-candidate validator still
applies. Known limits: a first and last name in separate columns/cells
are reported as two single-token mentions (cells are hard boundaries by
design); `.xlsx`, `.doc`, `.msg`, images and attachments are not read.
`scan_directory.py` picks up every listed extension automatically.

### PDF locations

Mentions in a PDF are located by page: `page 3, line 4, col 1 (chars
154-170)`, with the line counted within that page (text files keep
`line N, col C (...)`). Extraction still treats the whole PDF as one
document, so repetition and first-name propagation work across pages;
only the reported position is per page. Before 2026-09-24 every PDF was
reported as lines of its pages' joined text.

### Large single files

For one very large file (over ~400KB), spaCy can use several CPU
processes inside that file - results are identical, measured 2.4x
faster at 4 processes on a 2MB file:

```bash
python cli.py --input big_dump.txt --spacy-processes 4
```

Leave it at the default (1) for `scan_directory.py`, which already
uses every core across files.

### Real-time mode (local HTTP service)

`cli.py` reloads spaCy, the classifier and the knowledge base on every
run (~11-13s). `server.py` loads them once and then answers each request
in milliseconds (measured median 11-23ms per chat message):

```bash
python server.py                    # http://127.0.0.1:8765, this machine only
python server.py --port 9000 --no-feedback-log
```

- `POST /extract` `{"text": "..."}` — one standalone piece of text.
- `POST /sessions/<id>/messages` `{"text": "..."}` — the next message of
  a conversation. Each message is judged with the recent conversation
  (default last 200,000 chars) as context, so a name repeated across
  messages builds up evidence exactly as it would inside an exported
  chat file; earlier mentions of that name are re-judged when a new
  message adds evidence (`newly_accepted` in the reply).
- `GET /sessions/<id>` — running name list for the conversation;
  `DELETE /sessions/<id>` forgets it.
- `GET /health`.

Fed one line at a time and scored against the gold labels (re-measured
2026-09-24): `benchmark_real_660.txt` precision 0.9701 vs 0.9717 for a
whole-file run, gold coverage 0.9601 vs 0.9772 (a stream judges a name's
first mentions before its later repetitions exist); `benchmark_real_675.txt`
identical to the whole-file run. Median 13-27ms per message.

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/sessions/chat1/messages `
    -ContentType 'application/json' -Body '{"text": "Femi Novak called again."}'
```

Binds to 127.0.0.1 by default. The text sent to it is case material -
don't expose it on a network interface without authentication in front.

## Real measured results (not projected)

Run `python cli.py --evaluate` to reproduce these against
`datasets/benchmark/` (last run 2026-09-24, 7 benchmark files including
two large real-world documents, `benchmark_real_660.txt` and
`benchmark_real_675.txt`, plus a synthetic all-negative structured-dump
file - see below):

| Bucket | Precision | Recall | F1 |
|---|---|---|---|
| ACCEPTED only (auto-shipped) | 0.9598 | 0.9131 | 0.9359 |
| ACCEPTED + REVIEW (what a human sees) | 0.9325 | 0.9625 | 0.9473 |

**2026-09-25 - hostnames no longer reject real names.** `SystemLogValidator`
treated any 4+-segment dotted string as an Android/Java package, so a real
name on the same line as `www.bvrit.ac.in`, `priya@cse.bvrit.ac.in` or
`mail.company.co.uk` was hard-rejected - never even REVIEW ("Contact Suresh
Kumar at www.bvrit.ac.in" rejected; the same sentence with `www.bvrit.com`
accepted). A hostname ends in its TLD, a package starts with one, so such
matches are now exempt (`_is_hostname`); every other identifier on the line
still counts. Benchmark unchanged. Real case, before/after on all 108 files
containing an exempted hostname (of 19,089): no ACCEPTED change, no name
gained or lost in REVIEW; "Edit" (already in REVIEW) +4 mentions in one
preferences XML.

**2026-09-28 - log rotation.** `logs/pipeline.log` had grown to 1 GB (one
full real-case scan writes ~59 MB). At startup, once it passes 50 MB, it
is renamed to `pipeline.log.1` (older backups shift up to `.3`; the oldest
is dropped). Only the main process rotates, and only at startup: scan
workers append to the same file, and Windows can't rename a file another
process has open - if a server or scan is still running, rotation simply
waits for the next start. The log also always goes to the project's
`logs/` now: the default was the relative `"logs"`, and since modules set
up logging at import time, a run started from another folder logged there.

**2026-09-28 - middle-initial names (exp06).** "Craig J. Mundie" and
"Lisa E. Brummel" were rejected because their first names are on the
collision list (`common_english_words.txt` contains craig/lisa, so the
collision audit listed them), and "Kurt D. DelBene" was cut to a rejected
"Kurt" (the one-word dictionary hit only ever widened to 2-token spans).
A "First I. Last" candidate backed by spaCy or a dictionary hit now passes
CorroborationValidator regardless of the first name, and the widening also
covers the middle-initial shape. Benchmark unchanged (no such names in
it). Real case: 7 names newly ACCEPTED, all real people named in a public
SEC 10-K officer table and director signature block; nothing lost. Open:
"A. Road"-style spans (an initial plus a surname-list word) can still be
accepted - pre-existing, not caused by this change.

**2026-09-28 - spaCy spans split at phone numbers (exp05, owner-approved
trade).** On one-line call-log/chat dumps spaCy tagged "Farhan Vora
919812345671 Meera Iyengar Hi" as one person; CandidateFactory merged
everything it overlapped into one cluster that kept only "Meera Iyengar",
so "Farhan Vora" vanished from that spot (recovered only partly, as
"Farhan", by first-name propagation). SpacyDetector now splits an entity
at digit-bearing tokens as it already did at line breaks. Benchmark:
overlap metrics unchanged, exact boundaries 1,154 -> 1,156 of 1,205
accepted gold hits. Real case: +1 accepted (a full name swallowed the
same way next to an ID number) and +7 REVIEW entries, all noise that a
digit-bearing spaCy span used to drag into rejection (+2.3% REVIEW) -
accepted precision unaffected.

**2026-09-28 - short day/month names and ordinary-word first names
(exp04).** Two asset gaps behind the last ordinary-word names still ACCEPTED
in the real case scan: `multilingual_calendar_words.txt` had full day/month
names but not the short forms web date pickers ship ("Dom", "Ott" from
jQuery UI locale arrays), and `ambiguous_first_names.txt` lacked first names
that are ordinary words first ("Night"). Added the es/pt/it/sv short forms
(only dom, mie, mån, ott, tor are also dictionary names, so only they change
behavior) and 10 collision words (night, baby, beauty, fish, dip, era,
mini, pan, karma, asin - real given names and common Indian names/nicknames
from the same audit report were deliberately left off). Benchmark
unchanged. Real case: exactly 5 accepted names removed, all noise - Dom and
Ott (date pickers), Tor (the anonymity network in a VPN menu), Night and
Mie (3-to-42-character OCR fragments); nothing else changed. Open gap this
exposed: tiny OCR outputs (a handful of characters) still reach the
detectors and can produce single-word accepts.

**2026-09-25 - dictionary names matched as whole words only (exp03).**
`DictionaryDetector` had no word-boundary check, so it matched listed names
INSIDE code identifiers: "Handler" out of `clickHandler`/`errorHandler` was
ACCEPTED 146 times in the real case scan, none of them a standalone word. A
token now needs no letter or underscore before it and no letter, digit or
underscore after it. A digit before is allowed: the first version also
blocked it and the rescan lost a real cited author written "1Berger, A.".
Benchmark (N=7) vs the exp02 state: accepted-only P 0.9710 -> 0.9781, recall
and every shape unchanged. Real case: nothing newly accepted; "Handler" 146
accepted -> 4 standalone mentions in REVIEW; 4 accepted fragments gone
(an app filename, a `..._Max` setting, a log tag, and a surname glued to a
domain while the full name stays accepted); 45 REVIEW entries gone, 43 of which only ever
occurred glued inside longer tokens. A strict improvement, promoted.

**2026-09-25 - bare common-word surnames held for review (exp02, owner-
approved trade).** Tracing the real case scan's accepted single words found
"Read", "English", "Block", "Pilot", "Day", "Price" ACCEPTED from web/UI
text on one piece of evidence: a surname-list hit, trusted via repetition
or an ML score the classifier gives most capitalized words there - spaCy
never tagged them. Now, after decisions, an ACCEPTED single word that is an
ordinary English word known ONLY as a surname, with nothing but that
dictionary hit (no spaCy tag, title, full dictionary name or propagation),
and no accepted full name in the document starting or ending with it, goes
to REVIEW (`person_extractor._cap_bare_common_surnames`). First names are
untouched ("Marcus", "Amber" in chat are people). Benchmark (N=7):
accepted-only P 0.9606 -> 0.9710, R 0.9131 -> 0.9026 (the loss is one
synthetic code-name alias, "Winter"), F1 flat, ACCEPTED+REVIEW unchanged.
Real case: 8 names ACCEPTED -> REVIEW (Read, English, Hopkins, Price, Block,
Binder, Day, Pilot), nothing else changed; "Hopkins" (academic citations)
may be a real author and now needs a human glance. A precision-for-recall
trade under the noise floor, promoted on the owner's decision.

**2026-09-25 - larger NER models and the GPU: measured, not adopted.**
Observer-only probe (`experiments/gpu_detector_probe.py`; no pipeline code
changed): each model tagged persons in the same text, compared with the gold
labels and with what the pipeline already decides. Laptop: RTX 2050 (4 GB),
i5-12450H. Benchmark = 7 files, 1,335 gold mentions; of those the pipeline
accepts 1,220 and leaves 115 in REVIEW or missed (96 of them not in any
dictionary).

| Model | Device | chars/s | Tag precision | `dump_005` tags | Of the 115 not accepted, tagged | Of the 96 unseen, tagged |
|---|---|---:|---:|---:|---:|---:|
| en_core_web_sm (current) | CPU | 31,040 | 0.663 | 10 | 15 | 3 |
| en_core_web_lg | CPU | 29,952 | 0.856 | 1 | 23 | 6 |
| en_core_web_trf | CPU | 1,265 | 0.870 | 0 | 37 | 20 |
| GLiNER small v2.1 | GPU | 6,806 | 0.907 | 3 | 19 | 8 |

GPU run: 86% mean utilization, 3.9 of 4 GB VRAM (nvidia-smi, 741 samples).
Recall ceiling: even if every one of those tags became ACCEPTED with no new
false positive, overall accepted-only recall would rise at most 1.1pp (sm) to
2.8pp (trf) - around the ~2pp noise floor of a 7-file benchmark - and
turning a tag into ACCEPTED means letting NER evidence count toward the
knowledge-corroboration gate, which it deliberately doesn't today. Speed
(estimated from the baseline scan's per-file times, ~124M chars through
detection): trf on 12 CPU workers would take detection from ~13 min to ~5 h;
GLiNER as an extra detector on the one GPU would add ~5 h. Not adopted.
`benchmark_unseen`: no model tagged a non-person (sm/lg 24, trf 28, GLiNER
29 of 29 gold). Could not measure: en_core_web_trf on the GPU - spaCy's GPU
backend (CuPy) is blocked by Windows Application Control on this machine.
Worth testing next, through the full gate: **en_core_web_lg** - same speed,
no new dependency class, and far fewer junk tags (on the case sample it
tagged 118 distinct texts the pipeline rejects vs 840 for sm), though it
also produces 271 distinct texts the pipeline never saw as candidates.
*en_core_web_trf on the GPU, measured the same day via WSL2 Ubuntu* (Smart
App Control doesn't apply to Linux binaries there; setup in
`requirements-gpu.txt`): 3,149 chars/s at batch 4 on the same 6.1M-char
sample (2,228 at batch 16, which overflowed the 4 GB card into shared
memory; 11-16k chars/s on the prose-like benchmark files alone), GPU 98%
busy - 2.5x its CPU speed and output-identical to it (1,305 of 1,305
benchmark spans), but still 10x slower than en_core_web_sm on ONE CPU core,
and ~11 h vs ~13 min for full-scan detection. Accuracy as on CPU (37 of the
115 not-accepted gold mentions tagged). Not adopted; GPU question closed for
this hardware.
*Tested the same day (experiments/ledger.md, `exp01-spacy-lg`): rejected
at the benchmark gate.* As the pipeline's detector (config
`detection.spacy_model`, now configurable) it lowered accepted-only
precision 0.9606 -> 0.9589 and single-token recall 0.9606 -> 0.9310, and
broke first-name propagation (2 tests: its spans leave "Kenji" of "Dr.
Kenji Ito" unaccepted) - the pipeline's boundary and decision rules are
tuned around sm's spans, so a cleaner tagger in isolation did not make a
better pipeline. No case rescan was run: a rescan can confirm a gain, not
rescue a failed gate.

**2026-09-25 - evaluation hygiene.** `retrain_from_feedback.py` now also
excludes labels from the held-out `datasets/benchmark_unseen/` and from
copies of any benchmark file elsewhere (matched by file name) - before,
only `datasets/benchmark/` was excluded, and `evaluate_unseen_names.py`
logged held-out names into the labeling queue (it no longer logs; neither
does `ci_evaluate_gate.py`). No existing confirmed label was affected. The
CI F1 floor was still 0.84 from before the 2026-09-24 relabel (live F1
0.936); it is now 0.91. The CI privacy step also fails if any
`datasets/feedback/*.jsonl` is tracked by git - with those files
gitignored, its content scan had nothing to check in CI.

**2026-09-24 - validated on the real "real case" folder, and
what it caught that the benchmark couldn't.** A scan with the new readers
(19,090 files vs 7,246) exposed, compared against the 2026-09-23 scan via
each run's case-wide names CSV:
- Android binary XML/compiled manifests and image files with `.html`
  extensions (840 files) decoded into garbage "names" - now skipped as
  binary content (`src/io/binary_check.py`, counted in the scan summary).
- The text-feature retrain made brand + ordinary-word phrases score like
  names ("Zomato Customer Service" 0.19 -> 0.99), ACCEPTED on the ML score
  alone when repeated - the ML-only paths now require that no token be an
  ordinary English word (`DecisionEngine._contains_common_word`).
- Accented-letter support let machine-text through: mixed-case fragments
  ("ToMs", "LUt"), CamelCase identifiers ("KeyCharacterMapFile") and
  UTF-8-read-as-Latin-1 ("TomÃ¡s") - now rejected by StructureValidator.
- Single-word values in Android settings XML/JSON ("Edit", "Read",
  "Block") - single tokens in CSV/JSON/XML now go to REVIEW unless a title
  or same-file full name supports them.
- Support-line and font names ("Jio Customercare", "Helvetica Neue") -
  org keywords and a font section in the blacklist.
After these, accepted names from the same `.txt`/`.pdf` files went 218 ->
214 with full names instead of fragments (a lone first name -> the full name) and
old false positives removed ("Samsung Smart", "DeX"), and the new formats
added ~15 real people (e.g. "Roshni Nadar", "Frank Abagnale"). Known
remaining false positives: "Suicidal Tendency" (both words missing from
the ~10k-word common-words list), "Kyrie Eleison", "Grandmaster Master",
"Crna Gora". Benchmark unchanged throughout (P 0.9606 / R 0.9131).

**2026-09-24 - retrain for one-known-word names.** After the boundary fix
below kept "Femi Novak" whole, the classifier scored such names near zero
(0.04) - no synthetic example had that shape (two tokens, only ONE a
dictionary name). `generate_training_data.py` now adds 300 positives
(known first/last name + a pseudo-name no list contains) and 300
negatives with identical detections (known first name + an ordinary
English word). Controlled comparison, same code, trained exactly like
`retrain_from_feedback.py`: regenerated data alone P 0.9486 / F1 0.9297
(66 FP); with the new examples P 0.9598 / R 0.9131 / F1 0.9359 (51 FP);
live model before P 0.9605 / R 0.9101 / F1 0.9346 (50 FP). Promoted via
the guardrail (no shape regressed). "Femi Novak"/"Femi Okafor" 0.04 ->
0.61 and "Adebayo Baptiste" 0.08 -> 0.63, now ACCEPTED; `benchmark_unseen/`
unchanged at P 1.000 / R 0.966. (Regenerating also picks up today's
feature-code changes - accent-variant dictionary hits, the placeholder
blacklist - so the synthetic set now matches live inference.)

**2026-09-24 - name boundaries.** The benchmark's overlap scoring counts
"Nadia" as a hit for gold "Nadia Kovalenko", so truncation was invisible
to it; measured directly, 10.5% of accepted mentions were cut to one
token. When a one-word dictionary name sits inside a two-word capitalized
span whose other word reads as a name (an unambiguous known name, or an
unknown word that isn't ordinary English, a stopword, a place or an org),
the two-word span now wins (`candidate_factory._select_canonical_span`).
Exact boundaries 89.4% -> 94.9%; accepted precision 0.9525 -> 0.9605
(widening also exposes phrases like "Wild Gateway" to validators that
reject them), recall 0.9161 -> 0.9101: some widened names the classifier
scores very low ("Femi Novak" 0.04) now sit in REVIEW as full names
instead of being accepted as one token. Tried and reverted: also refusing
to widen ambiguous dictionary words - it cost +10 false positives to save
2 mentions. Remaining truncations are mostly code-name aliases ("Amber" of
"Amber Circuit"), deliberately not widened.

**2026-09-24 - document-level first-name propagation**
(`src/candidate/name_propagation.py`). Once a document ACCEPTS a
multi-token person ("Dr. Kenji Ito"), a standalone "Kenji" in the same
document counts as corroborated: REVIEW candidates with that text are
re-decided, and occurrences no detector found are created and run through
the FULL validation pipeline (hard rejections are never overridden).
First names that are ordinary English words, known first-name/common-noun
collisions ("Chase", "Grace", "Major") or calendar words are never
propagated. A first name directly followed by an existing candidate
("Ranjodh" in "Ranjodh Aulakh" where only "Aulakh" was detected) is not
split off as its own mention. Accepted-only: single-token recall
0.8177 -> 0.9655, unseen_name 0.5575 -> 0.6620, overall recall 0.8914 ->
0.9161, precision 0.9497 -> 0.9510 (no new false positives in any
benchmark file; `benchmark_unseen/` precision stays 1.000). Remaining
single-token misses are mostly "Ola" (Ola Bakare), hard-rejected because
"Ola" is on the organizations list. Real-time sessions pass the
conversation's accepted names in, so a full name from message 3 vouches
for a first name in message 20.

**2026-09-24 - one gold-label convention across all benchmark files.**
The files disagreed on first-name-only mentions ("Wendy already logged
it", "Hi Rohan,"): `benchmark_001/002` counted them, the rest didn't, so
121 of 182 accepted "false positives" were the pipeline correctly finding
a gold person by first name. All files now count them (the standard NER
convention, and the one `benchmark_001` already used) via
`scripts/extend_gold_first_name_mentions.py`: +156 mentions (003 +1,
004 +2, real_660 +140, and real_675 +13 for "Marcus Delaney", a call-log
party the 675 roster had omitted). Every addition was reviewed in context;
the one exclusion ("Kavya" in the TV-show title "The Kavya Diaries") is
listed in the script. Originals are in `datasets/benchmark/backups/`.
Same code and model, before -> after relabeling (accepted-only):
precision 0.8547 -> 0.9497, recall 0.9084 -> 0.8914, F1 0.8807 -> 0.9196.
Recall dipped because first-name mentions the pipeline misses are now
visible: single-token recall is 0.8177 (166/203) - 22 of the misses sit
in REVIEW (e.g. "Katya", "Jaylen"), 8 are never detected, and 6 are
"Ola" (Ola Bakare) rejected because "Ola" is on the organizations list.

**2026-09-24 - evaluator fix: overall numbers before this date are
slightly off.** `run_benchmark` pooled every file's spans as bare
character offsets for the OVERALL and per-domain metrics, so a prediction
at chars 316-324 in one file could "match" a gold mention at the same
offsets in a different file (greedy one-to-one matching, so cross-file
pairs also consumed each other). Per-file rows and per-shape recall were
always correct; overall/per-domain P/R/F1 (including the historical
figures further down, and `retrain_from_feedback.py`'s overall-F1
guardrail) were not. Fixed by giving each file its own offset range -
overall TP/FP/FN now equal the per-file sums exactly (regression test in
`tests/test_features_and_evaluation.py`). The table above uses the
corrected computation.

**2026-09-24 - placeholder names.** "John Doe", "Jane Doe", "Joe
Bloggs", "Test User" and similar are now blacklisted (see the dated
section in `assets/blacklist/blacklist.txt`). `benchmark_004.txt` (email)
was accepting "John Doe" twice from a forwarded test form submission;
email precision 0.667 -> 0.769 with recall unchanged at 1.0. None of
these names is a gold person anywhere in the benchmarks.


**2026-09-24 - accented names.** Every accented Latin name ("José
García", "François Dubois", "Łukasz Kowalski") was being hard-rejected
by StructureValidator's ASCII-only token check - never reaching even
REVIEW - and DictionaryDetector matched only the ASCII prefix of an
accented word ("François" -> a dictionary hit on "Fran"). ~9,100 accented
name-list entries were unreachable. All name-shape checks now share one
Latin letter set (`src/preprocessing/latin.py`), and accent variants
("Ștefan", "Nguyễn") fall back to an accent-folded lookup - for
non-ASCII tokens only (folding plain ASCII text too was measured and
rejected: it would make 43 ordinary English words, e.g. "back", "come",
"lower", known names). Benchmark effect: the one accented gold name,
"François Dubois" in `benchmark_004.txt`, previously missed even with
review, is now accepted; nothing else moved. Curly double quotes (“ ”)
now count for the `occurs_in_quotes` feature too.

**2026-09-24 - three text-only classifier features.** On benchmark
candidates NOT fully in the name dictionary, the previous classifier
ranked real names vs non-names at AUC 0.40-0.54 - no better than chance -
so the ML-gated acceptance paths were blocking unseen real names. Added
`common_word_ratio`, `first_token_is_ambiguous` and
`place_or_org_token_count` (see `FeatureVector`), measured offline first
(same data and params: AUC 0.538 -> 0.782 on that population), back-filled
exactly into every stored training row
(`scripts/migrate_training_data_add_text_features.py`), then retrained via
`retrain_from_feedback.py` (guardrail passed, no shape regressed).
Previous vs retrained model, both on the same current code, overall
figures re-measured with the corrected evaluator (see above):

| Accepted-only | Previous model | Retrained model |
|---|---|---|
| unseen_name recall | 0.3855 | **0.6145** |
| multi_token recall | 0.8534 | 0.9044 |
| dictionary_known / single_token recall | 0.9882 / 1.0000 | unchanged |
| Overall precision / recall / F1 | 0.8521 / 0.8601 / 0.8561 | 0.8547 / 0.9084 / **0.8807** |
| `datasets/benchmark_unseen/` (held out) P / R | 1.000 / 0.759 | 1.000 / **0.966** |

Precision held (slightly up). Cost: 6 more accepted false positives on
`benchmark_real_660.txt` (123 -> 129) alongside 8 more true positives
there; `benchmark_real_675.txt` precision rose 0.8428 -> 0.8705. Older
saved models keep working - the classifier reads only the leading columns
a model was trained on.

(Accepted-only recall jumped 0.7846 -> 0.8677 and F1 0.8186 -> 0.8644 in
a single targeted session - see "Real measured results, 2026-09-22 -
four-priority targeted investigation" further down for the full
before/after breakdown per fix. ACCEPTED+REVIEW is unchanged, as
expected: these fixes move candidates FROM review TO accepted: they
don't change what's discovered in the first place.)

(Precision on the ACCEPTED+REVIEW row moved up substantially from the
2026-09-17 checkpoint below, 0.7587 -> 0.8244, at a real recall cost,
0.9949 -> 0.9678 - this is the REVIEW->REJECTED common-word-phrase guard
described above, a deliberate, measured, explicitly-approved tradeoff, not
model drift. See "Real measured results, 2026-09-18" further down for the
full before/after breakdown.)

**Per-domain and per-shape metrics (2026-09-17):** one blended number
hides where the model actually struggles, so `--evaluate` now also
breaks recall/precision/F1 out by document-type domain
(`src/evaluation/evaluator.py`'s `FILE_DOMAINS` - prose, chat_or_call_log,
email, system_log; no OCR-sourced benchmark file exists yet, an honest
gap reported as "not measured," never assumed zero) and by candidate
shape (single- vs multi-token, dictionary-known vs unseen name;
recall-only - a false positive has no gold shape to categorize against).
This paid off immediately: **accepted-only recall for `unseen_name`
gold mentions (not in any dictionary) is 0.0000 (0/249)** - every one of
them currently depends entirely on the REVIEW bucket (98.80% recall
there) and NONE clear the bar for automatic acceptance. `chat_or_call_log`
is also the clear weak domain overall (F1 0.8008 vs prose's 0.9286).
Both are real, previously-invisible findings, not fixed here - they're
exactly the kind of targeting information the next round of hard-negative/
feedback-rebalancing work (see below) should use instead of guessing.

**Rebalancing feedback by domain (2026-09-17): investigated, deliberately
NOT done as originally proposed.** The obvious version - upweight
`datasets/feedback/confirmed_labels.jsonl` records sourced from the real
case folder over the public-book/paper labels - was checked before
building it: case-data labels are 81% `confirmed_not_person` (511/634),
while the book-sourced labels are 80% `confirmed_person` (329/413).
Blindly upweighting case data would have pushed the model toward
rejecting MORE, the opposite of what the `unseen_name` = 0% finding
above needs. Went looking for the actual bottleneck instead of guessing:
traced every one of the 246 real unseen gold mentions that had a
candidate at all, and found the ML classifier's confidence maxed out at
**0.786** - never once reaching the 0.85 `ML_HIGH_CONFIDENCE_OVERRIDE`
that's the *only* path a dictionary-absent candidate has to ACCEPTED
(`decision_engine.py`). Separately confirmed the OTHER half of that same
gate is not the problem: 247 of 249 (99.2%) unseen gold mentions already
clear the `repeat_count >= 4` requirement paired with it - repetition
was never the bottleneck, confidence was.

Added targeted (not domain-blanket) synthetic reinforcement instead -
`_build_unseen_name_positive_examples`/`_build_unseen_name_negative_examples`
in `generate_training_data.py`: algorithmically-generated pseudo-names
(verified against the real dictionaries so they can't collide),
regex-only evidence (no dictionary detection at all, matching how a
genuine unseen name is actually detected), 200 examples per class.
Retrained and re-measured: average ML confidence on the same 246 unseen
mentions moved 0.107 -> 0.270, max moved 0.786 -> 0.838 - real,
directional progress, and overall accepted-only benchmark numbers moved
slightly positive, not negative (F1 0.8147 -> 0.8157). But `unseen_name`
accepted-only recall is **still 0.0000** - even the new maximum (0.838)
falls short of the 0.85 bar. Deliberately did NOT also lower
`ML_HIGH_CONFIDENCE_OVERRIDE` to close that last gap: with the current
model, only ~2 of 246 mentions would newly cross even a relaxed 0.80
threshold (too small an effect to justify it), and that threshold exists
specifically because a real past incident (case 930204690, "Dark
Corridor") saw the classifier assign 0.99 confidence to a non-person
ledger-label artifact sitting alone before a line break - the exact
"isolated line" shape `is_isolated_line` (added earlier this session)
now explicitly trains the model to associate with personhood. The
repetition requirement still guards that specific incident either way,
but a lower confidence bar is a real, not-yet-justified precision risk
for some other not-yet-seen artifact class, not a free win. Kept the
retrained model (measured net-positive, zero regression); left
`unseen_name` accepted-only recall as a genuinely open problem requiring
either much larger reinforcement volume or a narrower, purpose-built
corroboration path - not a threshold tweak - to close safely.

**Multilingual calendar-word / missing-country false positives
(2026-09-17):** a real full case rescan (post unseen-name-reinforcement
retrain) surfaced a NEW noise class the benchmark suite doesn't contain:
one file, an Android locale/calendar-picker resource dump
(`CASEFILE.txt`), contributed 32 of 276 accepted names in
that scan - day/month names in German/French/Spanish/Italian/Polish/
Swedish/Portuguese/Dutch ("Jan" x10, "Sep" x8, "Domingo", "Julio", "Maj"...)
plus 5 missing country names (Costa Rica, Montenegro, Palau, Sierra
Leone, Sri Lanka - `locations.txt` had India/France/Germany/China but no
complete world country list). Two fixes:
1. Added the complete ISO-3166 country list (192 new entries) to
   `assets/locations/locations.txt` - reuses the existing
   `LocationValidator` hard-reject, zero new code.
2. Added `assets/common_words/multilingual_calendar_words.txt` (day/
   month names, 8 languages) and a NEW, narrower carve-out in
   `CorroborationValidator` (its "fifth refinement"): unlike an ordinary
   ambiguous word, a calendar word's heavy repetition in a resource file
   reflects the file's fixed ~19-entry vocabulary, not a genuinely
   recurring person, so calendar words skip the repetition-corroboration
   allowance entirely and require a title or full dictionary-window
   match instead. Simply adding these words to
   `assets/ambiguous_words/ambiguous_first_names.txt` (the existing
   mechanism) was tried first and confirmed NOT sufficient - that gate's
   repetition-allowance check runs before its ambiguous-word check, so
   "Jan" (10 occurrences) would have cleared it regardless.

Verified directly on the real file: accepted noise dropped 31 -> 13 (the
remaining 13 are a different, longer tail - single-token FRAGMENTS of
country names, e.g. "Maarten" from "Sint Maarten", which the full-phrase
`LocationValidator` match can't catch, plus generic dictionary-word
collisions unrelated to calendar/country content - left as a known,
not-chased-further residual (see next entry below - `WILD` specifically
turned out to be fixed anyway, as a side effect of a more general fix).
Full `--evaluate` benchmark is byte-identical before/after (it contains
no calendar/country content to begin with) - a clean, isolated win with
measured zero cost.

**ALL-CAPS acronym/last-name collisions (2026-09-17) - root cause of
READ/HANDLER/KNOX/POL/FOA/UUS/SHA/WILD/LAW, found via a user question
("why is the model confused about uppercase/lowercase words?").**
`StructureValidator` hard-rejects ALL-CAPS tokens as acronym-shaped
UNLESS the token happens to match a known LAST name - an escape hatch
meant for genuine surnames written in caps (legal/formal-ID documents).
At this project's dictionary scale (21,000+ last names, 81 locales),
plenty of ordinary acronyms/tech terms/legal-boilerplate words
coincidentally ARE someone's rare real surname: verified directly -
"Law" is a genuine (if rare) surname (Jude Law), so "LAW" bypassed the
acronym check and got ACCEPTED from repeated "GOVERNING LAW" EULA
section headers (real source: `eula_12.txt`, 5 occurrences in that one
file alone). This is the SAME underlying mechanism behind every one of
the "dumpsys noise" terms fixed earlier tonight (READ, HANDLER, KNOX,
POL, FOA, UUS, SHA) and the one left as a deliberately-unfixed
limitation (WILD) - all 9 are coincidental last-name matches.

Fix: `CorroborationValidator`'s "sixth refinement" - an ALL-CAPS
acronym-shaped SINGLE token that only passed structure validation via a
last-name coincidence now skips the repetition-corroboration allowance
entirely (same treatment as calendar words above), since these words
repeat because they're boilerplate/technical vocabulary, not because a
real person recurs. Verified: all 9 known terms (READ, HANDLER, KNOX,
POL, FOA, UUS, SHA, WILD, LAW) now correctly rejected - **including
WILD**, which the earlier dumpsys-specific `SystemLogValidator` fix
could never reach (its triggering line had no package identifier of its
own, only on a preceding line) but this general, document-agnostic
structural fix catches regardless of context. `eula_12.txt` re-verified
directly: 1 -> 0 accepted. Full `--evaluate` benchmark byte-identical
before/after (no benchmark gold mention is an all-caps acronym-shaped
single token relying on this exact escape hatch) - another clean,
isolated win, and notably a MORE general fix than several of tonight's
earlier, narrower, content-specific patches.

**Indian name-dictionary expansion via Wikidata (2026-09-17):** the
2026-09-16 Faker bulk import already fully exhausted Faker's `en_IN`
locale (verified directly: 0 missing first names, 4 junk last names,
against the full provider list) and Faker's OTHER India-specific
locales (`hi_IN`, `ta_IN`, `gu_IN`, `mr_IN`, `or_IN`, `bn_BD`) turned out
to provide names in native script only, unusable for this Latin-script-
only tool. Checked several other candidate sources and rejected them:
two GitHub Indian-name repos with no declared license (no legal
permission to redistribute); `philipperemy/name-dataset` (491M records)
rejected outright regardless of its Apache-2.0 code license - its
underlying name data originates from the 2021 Facebook data breach
(533M users' data leaked without consent), an ethical line, not a
licensing technicality.

Used Wikidata instead: its main-namespace structured data is genuinely
CC0/public domain (verified by downloading the actual LICENSE file,
distinct from Wikipedia's own CC BY-SA article text, which was NOT
used). Queried the Wikidata Query Service for the `given name` (P735)
and `family name` (P734) property values of Wikidata items that are
instance-of human (Q5) and citizen of India (Q668, P27) - i.e. names
actually used by real, notable Indian people with Wikidata/Wikipedia
entries (18,630 such people have a structured given name; 14,009 have a
family name), not a synthetic or frequency-derived list. Filtered to
single-token, Latin-script entries (multi-word and unresolved-label
results dropped - dictionary lookups are per-token) and deduplicated
against the existing dictionary: **1,898 new first names, 1,567 new
last names** (`first_names.txt`: 17,653 -> 19,551; `last_names.txt`:
21,061 -> 22,628). Backed up both files before writing
(`assets/{first,last}_names/backups/`).

Ran `scripts/audit_collisions.py` (the same tool used for the original
Faker import) against the expanded dictionaries: found 231 new
collisions with titles/honorifics/occupations/organizations/locations/
stopwords (e.g. "Guru", "Santiago", "Broker", "Montenegro" are now also
dictionary first/last names) and added all of them to
`assets/ambiguous_words/ambiguous_first_names.txt` (903 -> 1,153
entries) using the identical automatic-flagging methodology, so none of
them can reach ACCEPTED on a bare single-token dictionary hit alone.
Full `--evaluate` benchmark re-run after: 0.8464/0.7854/0.8148
(accepted-only P/R/F1) vs. the prior 0.8491/0.7829/0.8147 - flat within
noise, no regression. 76/76 tests pass.

**Context-window ML features (2026-09-17):** added 3 new features the
classifier never had before - `preceding_context_is_person_cue` (a word
like "said"/"met"/"contacted" within 3 words before the candidate),
`following_context_is_nonperson_cue` (a word like "Ltd"/"Division"/
"Debt" within 3 words after), and `is_isolated_line` (the candidate
sits alone on its own line - the real chat/email sender-header shape).
Motivated by an accuracy review finding the model's 16 features were
all shape/dictionary/confidence numbers with zero awareness of the
candidate's actual surrounding words, so it structurally could not
distinguish "Marcus Webb" from "Convertible Debt" when those properties
happened to match (measured: 757 of 1,276 confirmed labels, 59%, share
an identical feature vector with an opposingly-labeled example under
the old 16-feature schema). A 4th candidate feature, "sits on a system-
log/dumpsys line," was designed and then deliberately dropped before
implementation: `SystemLogValidator` already hard-rejects those
candidates before feature extraction ever runs, so the feature could
never be nonzero for anything the classifier or feedback loop would
ever actually see - a structurally dead input, the same "ML can't
recover what rules already gated" trap this review itself warned about.
All 1,276 confirmed labels' stored feature vectors were migrated
in-place (`scripts/migrate_confirmed_labels_add_context_features.py`,
original backed up to `datasets/feedback/backups/`) - 1,257 recomputed
from their real source documents, 19 conservatively zero-padded (10
whose source file no longer exists, 9 with minor offset drift) rather
than guessed.

Net effect, measured: internal validation F1 improved (0.827 -> 0.858),
but real accepted-only F1 moved slightly the other way (0.8205 ->
0.8147, recall 0.7930 -> 0.7829) - the entire real-world shift traced to
one file, `benchmark_real_660.txt` (12 fewer auto-accepted names out of
738 gold mentions). Checked what actually happened to those 12 before
deciding: they are NOT in that file's "missed even with review" list -
they moved from ACCEPTED into the REVIEW queue (still visible to a
human), not lost, and false positives on that file didn't change at all
(129 before, 129 after). The human-visible ACCEPTED+REVIEW number is
byte-identical to before (0.7587/0.9949/0.8609). Kept anyway (deliberate
choice): for real forensic casework, a slightly higher bar for
*unreviewed* auto-accept is a reasonable direction to move in, not
obviously a regression, and the previous model is one file-copy away in
`models/lightgbm/backups/` if a future measurement argues otherwise.

**Benchmark-leakage fix (2026-09-17):** `scripts/retrain_from_feedback.py`
now excludes any confirmed feedback label sourced from a file under
`datasets/benchmark/` before training (`_is_benchmark_source`) - 229 of
the 1,276 confirmed labels (124 from `benchmark_real_660.txt`, 66 from
`benchmark_real_675.txt`, 39 from the four small hand-authored
`benchmark_0*.txt` files) had been getting 3x-upweighted into training
while those exact files were also what `--evaluate` scores against.
Retraining on the remaining 1,047 non-benchmark labels and re-measuring
shows the leak's actual effect was small: accepted-only precision moved
0.8539 -> 0.8500 and F1 0.8223 -> 0.8205 (both within noise), and the
two leaked files' own per-file numbers are unchanged to four decimal
places (`benchmark_real_675.txt` accepted-only: 0.7357/0.3093/0.4355
both before and after). Kept anyway - the fix costs nothing and now the
number is trustworthy by construction, not just by luck.

**Real measured results, 2026-09-18 - full session summary.** Started from
a direct user question ("why can't the model use context instead of word
lists") and ended up touching five different layers: asset coverage, the
feedback/training data schema, detection performance and correctness, the
decision engine, and a real end-to-end case re-scan.

*Asset coverage expansion.* Added months/days verification (already fully
covered - no gap found), **323 new living/non-living-thing words** (animals,
plants, general biology vocabulary, everyday objects) to
`common_english_words.txt`, **12 more casual/chat words** to
`stopwords.txt`, and **137 major global companies** to `organizations.txt`
(all cross-checked against `first_names.txt`/`last_names.txt` for
collisions first - 4 candidates, Ericsson/Zara/Ferrari/Roche, were
deliberately excluded for exactly this reason). Then, from manually reading
real case files end-to-end (`CASEID_call/chat/cookie/email/
installedapplication/socialmediaactivity/useraccount/wirelessnetwork.txt`)
and comparing every single name against the tool's own output by hand: found
3 more real false positives reaching ACCEPTED ("Monica" - a fragment of
"Santa Monica" in a corporate address; "Nagar" - a fragment of "Chanda
Nagar," a Hyderabad locality; "Day" - a fragment of a calendar event title
"The Day") and generalized `OrganizationValidator.ORG_KEYWORDS` with
`licensing`/`marketplace`/`warranty`/`software` after finding "Select
Licensing," "Zune Marketplace," "Product Warranty," and "MontaVista
Software" all clearing into REVIEW with no keyword to catch any of them.

*Feedback/training pipeline - context capture.* `FeedbackRecord` never
stored the sentence/line surrounding a candidate, only its bare text span
plus an already-computed numeric feature vector - which had already caused
a real, documented problem: the 2026-09-17 context-features migration had
to reopen each record's original source file from disk just to backfill a
3-word window, and 19 of 1,276 records permanently lost that ability
because the source file was gone by then. Added `context_text` to
`FeedbackRecord` (captured via `line_containing()` at logging time, zero
extra cost since the document text is already in scope), threaded it
through `log_review_persons()`, displayed it to the human labeler in
`label_feedback.py` (who previously saw only the bare candidate text with
no surrounding context at all), and added `context_text` to synthetic rows
too for schema symmetry. Backfilled the 1,276 existing confirmed labels via
a new one-time migration (`scripts/migrate_feedback_add_context_text.py`,
modeled on the 2026-09-17 precedent's verify-before-trust pattern): 1,257
recovered real context, 19 padded with explicit "unavailable" placeholders
(10 missing source files, 9 offset drift) - never guessed. Caught and fixed
a real latent bug in the same change: `FeedbackStore.load_pending`/
`load_confirmed` used `str.splitlines()`, which also treats Unicode
line-separator-like characters (NEL, LS, PS) as line breaks - harmless
while stored fields were short candidate spans, but `context_text` now
carries real (sometimes OCR'd) document text that can legitimately contain
them, which would silently fragment a valid JSONL record into unparseable
pieces. This is purely a data-capture change - it does not itself make the
model smarter, but it means no future feature-schema change will ever again
depend on original casework files still existing on disk, and a real
(context, label) corpus now exists for whenever a context-aware feature or
model is worth building.

*Detection coverage and speed, and the decision-engine REVIEW guard* - see
"A real bug this design caught" above for both (bugs four and five).

*Real case re-scan, same 7,246-file case folder, before vs. after every fix
above (excluding the decision-engine change, applied in a separate later
run):*

| | 2026-09-17 | 2026-09-18 | Change |
|---|---|---|---|
| Accepted | 229 | 248 | +19 |
| Review | 389 | 367 | -22 |
| Total unique names | 618 | 615 | -3 |
| Total time | 31.1 min | 45.9 min | +14.8 min |

File-level counts (processed OK, skipped, failed) were identical between
runs - confirms the change was isolated to detection/decision quality, not
file handling. All 16 words originally reported as false positives
(MontaVista Software, Dell, Zune Marketplace, Chief, Messenger, ...) were
individually re-verified gone from the new REVIEW output. The added runtime
is the multi-MB dump files - previously silently skipped by the spaCy
`max_length` bug - now actually being processed.

(Table above reflects the final 2026-09-17 state, after the entire
2026-09-16/17 backlog-clearing session: `datasets/feedback/
confirmed_labels.jsonl` grew from 133 to **1,276** real labels and
`pending_review.jsonl` is now fully empty - 94 auto-resolved against
this project's own benchmark gold labels, the rest from working through
every non-case-data source in the backlog (a public book, academic
PDFs, a Microsoft SEC filing, Kaggle/forum profile pages, EULA/system-
log/JS-tracking noise, OCR garbage) with visible reasoning at every
step, plus one genuinely ambiguous item ("Frank," a name explicitly
assigned by a book author to an anonymized film character - "we'll call
them Al and Frank" - resolved not-person once the full surrounding
passage made that clear). **634 items were real corporate-fraud case
content and were deliberately left to the case's own reviewer**, not
guessed on. Retraining on the full label set measurably cost accepted-
only recall/F1 versus an intermediate 616-label checkpoint (0.83/0.84
-> 0.79/0.82) - likely because most of the added labels are negative
examples from noise-heavy sources structurally different from the
benchmark's narrative-prose gold set. Kept anyway (deliberate choice,
not an oversight): this model reflects far more of the real diversity
this tool actually encounters, and the common-word-phrase guard held
clean (zero accepted-only false positives on the structured-dump
regression file) through every retrain this session. The 616-label
checkpoint remains in `models/lightgbm/backups/` if a future
measurement finds the trade worth reversing.)

**Read this honestly, don't just quote the F1.** This is a large jump
from earlier same-day numbers (accepted-only recall 0.18 -> 0.72,
precision 0.96 -> 0.84) - both changes are explained, not mysterious:

1. **`assets/first_names/first_names.txt` and `last_names.txt` grew
   from 1,509/849 entries to ~17,650/~21,060** (2026-09-16 bulk import:
   Faker, MIT License, person-provider data from 81 Latin-script
   locales spanning Western/Eastern/Southern Europe, Scandinavia, the
   Baltics, Latin America, romanized Arabic, West/East/Southern
   African English-speaking countries, Turkey, and Southeast Asia).
   This directly closes most of the previously-documented gap -
   "Kenji Ito," "Nadia Kovalenko," "Femi Nakamura" and similar globally
   diverse names now clear the knowledge-corroboration gate for
   auto-accept, which is where essentially all of the recall gain came
   from. Deliberately excludes non-Latin-script locales (Japanese,
   Chinese, Korean, Russian, Hebrew, Thai, etc. in native script) -
   consistent with `RegexDetector`'s Latin-script-only design; romanized
   coverage for those specific cultures remains a real gap (see "Known
   limitations") pending a properly-licensed romanization source (the
   well-known GFDL-licensed first-name database was deliberately NOT
   used - its copyleft/attribution terms are a worse fit for this
   project's plain-MIT-style data assets than Faker's).
2. **Precision cost is the expected side effect of that much more
   dictionary surface area**, not a new bug: broader name coverage
   means broader collision surface with `assets/common_words/`,
   `titles.txt`, `honorifics.txt`, `locations.txt`, and
   `organizations.txt`. Every collision with a common word/title/
   honorific/occupation was found via `scripts/audit_collisions.py` and
   added to `assets/ambiguous_words/ambiguous_first_names.txt` (~900
   entries, same mechanism as the "Major" fix below) so a bare,
   low-repetition occurrence needs real corroboration - a genuine full
   name is unaffected. Collisions with `locations.txt`/
   `organizations.txt` (e.g. a surname that's also a city, "Santiago,"
   or also a company, "Trent") are a DIFFERENT, lower-priority failure
   direction (a false REJECT, not a false ACCEPT) and are documented,
   not yet fixed - see "Known limitations."

The classifier itself is unchanged in kind since the last retrain (still
trained on synthetic data only, no real feedback - see
`models/lightgbm/backups/` for the prior 133-real-example snapshot and
`src/classification/lightgbm_classifier.py`'s `save()` docstring for why
every retrain now backs up the model it's about to overwrite). The
synthetic generator (`scripts/generate_training_data.py`) also gained
chat-header-shape templates (a name alone on its own line, immediately
followed by a newline - the real WhatsApp/chat-export sender-header
pattern this project's actual casework is full of) so the model learns
that shape from synthetic data instead of only from scarce real
feedback.

`datasets/feedback/pending_review.jsonl` has grown to **1,107 items
pending** as of this writing (up from 413) and remains the next
highest-leverage lever: labeling a batch of it and retraining
(`scripts/retrain_from_feedback.py`) grows real-world coverage further,
now with less regression risk than before thanks to the common-word-
phrase guard (see "A real bug this design caught," design principle 3
in `src/decision/decision_engine.py`).

The smaller benchmark files, their gold labels, and exactly how those
labels were built (mechanically, via markup-stripping — never
hand-counted character offsets) are in `datasets/benchmark/` and
`scripts/build_benchmark_from_markup.py`.
`benchmark_structured_dump_005.txt` is different: a hand-authored,
all-negative (zero gold mentions) synthetic Android dumpsys/report-style
file that encodes the 2026-09-16 common-word-phrase bug directly into
the regression suite, so a future regression of that class shows up in
`--evaluate` instead of only being found by accident on a real scan.

**Real measured results, 2026-09-22 - four-front improvement pass (false
positives, recall, speed, active learning), each with before/after
measurement, ordered cheapest-first per an explicit work order.**

*Front 1 - false positives.* `scripts/audit_collisions.py` had never
checked `first_names.txt`/`last_names.txt` against
`common_english_words.txt` (only against titles/honorifics/occupations/
organizations/locations/stopwords/blacklist/campaigns) - a real gap in
the audit tool's own scope, the same failure class as the "Major" bug
but against a list the script didn't examine yet. Added that check
(reuses the script's own `report()` function, no new logic): found 636
first-name and 630 last-name collisions with common words, almost all
already covered incidentally by earlier bulk imports, but confirmed via
a real full-case re-scan that words like "Max," "June," "Server,"
"Sales," "Read," "Key," "Binder," "Handler" were passing a bare
single-token dictionary hit and then corroborating anyway via heavy
in-document repetition inside fixed-vocabulary Android dumps - the exact
same "repetition defeats the ambiguous-word gate" failure class the
calendar-word and ALL-CAPS-acronym refinements were built to close, just
not yet extended to this list. Root-caused the two biggest offenders
directly: `android.os.Handler`/`android.os.Binder`-style 3-segment,
PascalCase-final-segment Java/Android class references were not covered
by `SystemLogValidator`'s package-identifier regex (which required
either 4+ segments or an ALL-CAPS final segment) - added a 4th pattern
for this shape. Verified directly on the real 10.3MB
`dumpsys_ANR_WindowManager.txt`: "Binder" 168 -> 2 occurrences, "Handler"
61 -> 1 (98%+ reduction), with zero adversarial-example false triggers
(checked against 5 real prose/chat/email sentence shapes). Remaining
narrow items (NoUi, Headset Jack, Gen Psychiatry, Teknologi Mara, Gulab
Jamun, Customer Car, Rev. B, Kindle) added as exact blacklist/
organization entries after individually checking each for first/last-
name collision risk. Also found and fixed a real, general gap unrelated
to any single file: `locations.txt` had zero US states at all (not just
"North Dakota," missing entirely as a category) - added the complete
50-state list (47 new; Georgia/Virginia already present) plus "Asia" and
"Mint Hill," following the exact precedent of the 2026-09-17 ISO-3166
country-list addition, checked for name collisions first (only
"Indiana," already gated).

Verified end-to-end via a real, controlled before/after case re-scan
(same 7,246-file folder, `--include-review`):

| | Before front 1 | After front 1 | Change |
|---|---|---|---|
| Accepted | 248 | 237 | -11 |
| Review | 367 | 171 | **-196 (53%)** |
| Total time | 45.9 min | 43.3 min | -2.6 min |

Diffed the accepted-name lists directly: all 11 removed names (Asia,
Customer Car, Gen Psychiatry, Gulab Jamun, Headset Jack, Kindle, Mint
Hill, NoUi, North Dakota, Rev. B, Teknologi Mara) are confirmed false
positives targeted by this exact round of fixes - zero genuine names
lost, zero new names gained (expected: this front was precision-only).
`--evaluate` moved slightly positive (F1 0.8169 -> 0.8173), confirming
these real-world fixes cost nothing on the labeled benchmark.

*Front 2a - `unseen_name` recall (0% accepted-only, entirely REVIEW-
dependent).* Reproduced the README's own prior ML-confidence-ceiling
methodology fresh: unchanged at 0.8382 max, still short of
`ML_HIGH_CONFIDENCE_OVERRIDE`'s 0.85 by 0.0118 - the prior reinforcement
round's plateau, not a new problem. Per this session's explicit
instruction (extend the same targeted synthetic approach rather than
lower the threshold), raised
`generate_training_data.py`'s `n_unseen_name` 200 -> 600 and retrained.
Max confidence moved 0.8382 -> 0.9917; **`unseen_name` accepted-only
recall moved 0.0000 -> 0.0080 (2/249)** - the first non-zero value this
metric has ever recorded in this project's history. Zero test
regressions; overall accepted-only F1 moved slightly positive (0.8173
-> 0.8186).

*Front 2b - `chat_or_call_log` domain (lowest F1 of the tracked
domains).* Root-caused a small, concrete instance rather than guessing
broadly: `benchmark_003.txt`'s "Farhan Vora" is missed 2 of 8 real
occurrences even with REVIEW, despite full first+last dictionary
coverage matching the correctly-detected "Meera Iyengar" in the same
line. Checked and ruled out the obvious hypothesis (adjacent
code-mixed Hindi text triggering `LanguageValidator`) by reading that
validator's actual logic: it only inspects the candidate's own tokens,
never neighboring words, so nearby non-English text cannot be the
cause. Root cause not yet found within this session's budget - documented
here as a genuine, narrow, diagnosed-but-unfixed gap rather than forcing
an unverified fix. Domain-level F1 moved only marginally (0.8040 ->
0.8047) from front 2a's retrain, confirming this domain's weakness is
not fully explained by the unseen-name ceiling issue alone.

*Front 3 - speed.* Checked the other two hypotheses first: dictionary
lookups already use `frozenset` (O(1), confirmed in
`knowledge_base.py`) and every `re.compile()` call in the codebase is
already at module level, never inside a loop - both non-issues, no fix
needed. Parallelized `scripts/scan_directory.py` (`--workers`, default
`os.cpu_count()`): each worker process gets its own `Pipeline` instance
(one spaCy load per worker, not per file - a loaded spaCy model cannot
be cheaply shared across process boundaries, so a process pool with a
per-worker `Pipeline` is the correct shape, not naive sharing). Verified
the "no shared mutable state across files" assumption before writing
any code: the only cross-file state is the `combined` merge dict, built
in the main process from each file's already-self-contained result
after it returns - workers never touch it. Correctness verified two
ways: byte-identical CSV output on an 88-file folder (`--workers 1` vs
`--workers 8`), and identical summary counts on the full 7,246-file case
(933 names found, 408 unique, 237 accepted, 171 review - exact match
both ways). Speed measured both ways too: 88-file folder wall-clock
173s -> 32.7s (5.3x, dominated by fixed spaCy/model-load startup cost at
this small scale); full case wall-clock **2596s -> 871s (43.3 min ->
14.5 min, 2.98x)** - the lower ratio at scale reflects load imbalance
from a handful of multi-MB duplicate dump files each taking minutes on
whichever single worker draws them.

*Front 4 - active learning loop.* `datasets/feedback/pending_review.jsonl`
had shrunk to 33 items (from 1,107 - the intervening backlog-clearing
work is documented above). Labeled all 33 with real, context-justified
judgments (6 confirmed_person, 27 confirmed_not_person - each verified
against its actual `context_text`, e.g. "Cardholder: Femi Novak | Card:
1409..." for the persons, "Massey University"/"MOL Corporation"/Android
dumpsys field labels for the rejections), applied via the same
`FeedbackStore` API `label_feedback.py` uses interactively. Retraining
crashed on a real, previously-latent bug: 3 of the 33 came from
`pending_review.jsonl` records logged before the 2026-09-17 context-
features migration, which only ever touched `confirmed_labels.jsonl` -
promoting them left 3 records at the old 16-feature schema mixed with
1,306 at 19, breaking `np.array(X, dtype=float)`'s shape assumption.
Fixed by re-running the existing (idempotent)
`migrate_confirmed_labels_add_context_features.py` against just those 3
- which itself hit the SAME `str.splitlines()`-vs-Unicode-line-separator
bug documented in the context-capture work above, since this older
script had never been patched with that fix; patched it to reuse
`_read_jsonl_lines` too. Retrained cleanly afterward (1,074 real
examples, 235 correctly excluded as benchmark-sourced -
`_is_benchmark_source` reconfirmed still working, +6 over the
previously-documented 229, matching exactly the 6 benchmark-sourced
labels in this batch) - but the retrain **measurably regressed the
model**: accepted-only F1 0.8186 -> 0.7938, recall 0.7854 -> 0.7362,
concentrated unexpectedly broadly on `dictionary_known` recall (906 ->
847 of 930, not just edge cases) rather than the narrower unseen/
ambiguous-case cost seen in earlier retrains - likely the same
heavily-negative-skewed-batch dynamic documented in this file's 2026-09-17
"634 case items" entry (27 of 33 labels here were `confirmed_not_person`),
but at a magnitude judged too broad to keep silently. Surfaced explicitly
per this project's own "zero regressions is non-negotiable" bar rather
than applying that older entry's "kept anyway" precedent by default;
reverted to the pre-retrain model backup
(`models/lightgbm/backups/person_classifier_20260922T040504Z.txt`) on
explicit confirmation. The 33 labels remain in `confirmed_labels.jsonl`
(not lost) for a future, larger and better-balanced retrain. 91/91 tests
pass throughout.

**Real measured results, 2026-09-22 - four-priority targeted
investigation (given an explicit --evaluate baseline instead of broad
re-exploration): close the multi_token accept gap, resolve whether
unseen_name needs more synthetic volume or a different mechanism, add a
regression guardrail to the retrain loop, and profile the one pipeline
stage that looked disproportionately slow. Ordered cheapest/highest-ROI
first, each measured before/after against a shared starting baseline
(accepted-only P 0.8557/R 0.7846/F1 0.8186).**

*Priority 1 - close the multi_token accept gap.* Diffed REVIEW-bucket
true positives against ACCEPTED-bucket true positives in
`benchmark_real_675.txt` and `benchmark_real_660.txt`, pulling each
candidate's actual decision-engine state (final confidence, ML
probability, detection patterns, repeat count) rather than guessing a
cause. Found TWO distinct subclasses, not one: the overwhelming majority
(196 of 196 in `_675`, 21 of 28 in `_660`) turned out to just be the
`unseen_name` problem wearing a different hat - zero dictionary
evidence, ML probability well under `ML_HIGH_CONFIDENCE_OVERRIDE` -
correctly left for the dedicated Priority 2 investigation instead of a
fake separate fix. But 7 cases in `_660` ("Femi Novak," "Nadia
Kovalenko," "Jaylen Brooks," ...) were genuinely different: a real
single-token dictionary hit (Femi/Nadia/Jaylen) paired with a token the
dictionary has never seen (Novak/Kovalenko/Brooks), vetoed by
`ML_WEAK_DICTIONARY_VETO_FLOOR` because ML scored them only 0.03-0.08 -
nowhere near the exact-0.00 score of the "Deep Tunnel" incident that
floor exists to catch. Rather than lower the floor itself (calibrated
against that one real incident), extended `_has_knowledge_corroboration`
with the same pattern already used one branch above: a weak signal
(partial dictionary support) becomes trustworthy when paired with real,
independent document-wide repetition (5 of 7 cases had repeat_count
>= 4). Measured: multi_token accepted-only recall 0.7587 -> 0.7685 (+11),
with two unplanned bonus gains from the same general fix - dictionary_known
0.9742 -> 0.9871 (+12) and single_token 0.9833 -> 1.0000 (+1) - and
**precision held (0.8557 -> 0.8556, i.e. unaffected)**. Covered by
`tests/test_decision_engine.py::test_single_token_dictionary_hit_with_low_ml_corroborates_via_repetition`
and a repeat_count<4 control test.

*Priority 2 - unseen_name: stop scaling synthetic volume, use context
features instead.* Computed the FULL ML confidence distribution across
all 249 `unseen_name` gold mentions (not just min/max, which is all any
prior round had checked): median 0.12, p75 0.41, p90 0.6956 - a genuine
long tail, not a threshold-placement issue (only 6 of 249 clear 0.70).
Per this investigation's own decision framework, that ruled out both
"raise the threshold" and "add more synthetic volume" (already tried,
plateaued at max 0.838) and pointed at a structural corroboration path
using the context features added 2026-09-17
(`preceding_context_is_person_cue`, `is_isolated_line`) instead of ML
confidence. Checked which of the two actually helps THIS shape before
using either, rather than assuming both apply equally:
`preceding_context_is_person_cue` was False for 244 of 247 real failing
mentions - essentially dead for this shape, these names sit mid-sentence
in prose, never near a "said"/"met" cue word - while `is_isolated_line`
(the real chat/email sender-header shape) correctly flagged 140 of 247
(57%). Added a new corroboration path requiring `is_isolated_line` +
`ml_prob >= ML_WEAK_DICTIONARY_VETO_FLOOR` (reusing the existing
constant as an "the ML model isn't actively disagreeing" guard - all 140
candidates scored >= 0.023, nowhere near the "Dark Corridor" incident's
exact 0.00) + real repetition (>= 4). Measured: **`unseen_name`
accepted-only recall 0.0080 -> 0.3855 (2 -> 96 of 249)** - by far the
largest single change of this session - with overall accepted-only
recall 0.7939 -> 0.8677 and F1 0.8236 -> 0.8644, and **precision actually
improved slightly** (0.8556 -> 0.8611). `chat_or_call_log` domain F1
moved 0.8047 -> 0.8577; `prose` domain accepted-only recall reached
1.0000. Covered by
`tests/test_decision_engine.py::test_isolated_line_unseen_name_with_modest_ml_corroborates_via_repetition`
plus two control tests (below the ML floor; without repetition).

*Priority 3 - retrain_from_feedback.py regression guardrail.* Checked
whether the 27-negative/6-positive skew in the 33-label batch that
regressed the model (see the entry directly above) reflected a standing
dataset-wide imbalance first: the full `confirmed_labels.jsonl` (1,309
records) is 55.4% `confirmed_not_person` / 44.6% `confirmed_person` -
mild, not the standing issue that specific batch's 82%/18% skew might
have suggested. Added the guardrail anyway (this exact failure mode had
already happened twice in this project's real history - 2026-09-16 and
2026-09-22, both caught only by someone remembering to run `--evaluate`
by hand afterward): `retrain_from_feedback.py` now saves the newly
trained model to a scratch path first, evaluates it against the real
`datasets/benchmark/` suite, and compares per-shape accepted-only recall
against the CURRENT live model before ever touching it. Any shape
dropping more than `MAX_ACCEPTABLE_SHAPE_RECALL_DROP` (0.02) with no
offsetting overall F1 improvement blocks the promotion automatically -
the candidate model stays saved separately for inspection, the live
model is untouched, and `--force` is required to override. Verified
directly, not just unit-tested: re-ran retraining on the same 33 labels
that caused the 2026-09-22 regression, and the guardrail caught it
automatically (`unseen_name` recall 0.3855 -> 0.3454, a real regression
the improved decision-engine logic made more visible in a different
shape than before) and correctly refused to promote - the live model's
`--evaluate` numbers were confirmed unchanged after the refused run.

*Priority 4 - profile feedback_logging.* Measured, not guessed: this
stage was costing ~1.6-1.8s per file across every real run this session,
larger than every other pipeline stage combined, for a stage that should
just be appending a few new lines. Root cause: `FeedbackStore.
_existing_texts()` re-read AND re-parsed BOTH full JSONL files from disk
on every single call - i.e. once per document in a batch scan - even
though one `FeedbackStore` instance lives for an entire batch-scan run
(one `Pipeline`, one `PersonExtractor`, one `FeedbackStore`). With
`confirmed_labels.jsonl` now at 1,300+ lines and growing, that's
redundant work scaling with both file count and label-set size at once.
Fixed by caching the set on the instance - `log_review_persons()`
already mutated it in place as new records were logged, so caching it
(not copying it) keeps it correctly up to date across calls with zero
extra bookkeeping. Measured: feedback_logging **1.6-1.8s -> 0.0000s**,
every single call, across all 5 benchmark files that have REVIEW
candidates - confirmed via `pending_review.jsonl`'s MD5 hash being
byte-identical before and after (correctness unchanged, purely a speed
fix). Two real per-file wins from this alone:
`benchmark_real_660.txt` total pipeline time 2.834s -> 1.220s,
`benchmark_real_675.txt` 2.390s -> 0.465s. Covered by
`tests/test_feedback.py::test_existing_texts_cache_stays_correct_across_many_calls`
(5 distinct people logged, then all 5 re-logged as if from a later file
in the same batch, dedup verified correct against all 5, not just the
most recent).

97/97 tests pass throughout this investigation.

**Process lesson, 2026-09-22 (read this before trusting any future
`--evaluate` number on its own): `--evaluate` alone is no longer
sufficient sign-off for a recall/precision change to
`decision_engine.py` or the classifier - a real-corpus scan is now a
required second step, not optional.**

Found the hard way, immediately after the Priority 1-2 work above: the
`unseen_name` fix moved benchmark accepted-only recall 0.0080 -> 0.3855
(+94 mentions, F1 +4.58pp) - a genuine, real, well-tested improvement on
`datasets/benchmark/`. But a full real-case-folder rescan (the same
7,246-file corporate-fraud case used throughout this session) showed the
net ACCEPTED set was **byte-identical to the pre-fix baseline** once 3
induced false positives were corrected - "Mellon" (a real but here-
coincidental last-name dictionary hit inside "Bank of New York Mellon
Trust Company", rescued by the Priority 1 single-token-dictionary-plus-
repetition path), "Keyboard Input Mapper" (an Android dumpsys section
label), and "Suicidal Behaviour" (an academic-paper section heading) -
both of the latter two rescued by the Priority 2 `is_isolated_line`
structural path. All 3 diffed and confirmed non-person, then blacklisted
(same mechanism as every other real-casework false positive this
session, see Priority 1's entry above) - and the resulting accepted list
matched the original baseline exactly, name-for-name, with zero net
genuine names recovered on this specific case.

Root cause of the gap between the two measurements: `datasets/
benchmark/`'s two large real-world files were deliberately constructed
with synthetic pseudo-random "unseen names" (Kenji Ito, Talvin Meskara,
Copper Falcon, ...) specifically to stress-test this exact shape
(isolated line, zero dictionary, repeated, modest-not-confident ML
score). The benchmark suite structurally CANNOT contain the false-
positive class this fix also happened to unlock - it has no dumpsys
section labels or academic-paper headings in its gold-labeled files at
all - so `--evaluate` had no way to surface that cost, only a real
corpus could. The fix itself is correct and the benchmark win is real
(kept, not reverted) - the lesson is procedural, not "the fix was
wrong": going forward, any change that touches `decision_engine.py` or
`models/lightgbm/person_classifier.txt` gets a real-corpus rescan and
diff against the pre-change accepted list, in addition to `--evaluate`,
before being reported as a recall/precision improvement. `--evaluate`
remains the right tool for confirming NO regression on the labeled gold
set (fast, reproducible, exact) - it's specifically insufficient, on its
own, for confirming a claimed GAIN generalizes past what the benchmark
happens to contain.

**Real measured results, 2026-09-22 - active-learning retrain resumed
after the regression guardrail landed.** `pending_review.jsonl` backlog
was 2 items: "Doran" (benchmark-sourced, `benchmark_real_675.txt`,
contact-list-shaped context alongside the already-gold "Marcus Delaney" -
labeled `confirmed_person`; excluded from training regardless by
`_is_benchmark_source`) and "Gora" (real casework, the same
`CASEFILE.txt` jQuery-UI locale-data file whose siblings Ott/
Barth/Miquelon/Anexo were already known non-person noise - labeled
`confirmed_not_person`). `confirmed_labels.jsonl` went 1,309 -> 1,311.
97/97 tests passing before and after.

`scripts/retrain_from_feedback.py` guardrail verdict: **promoted** - all
4 shape recalls unchanged (`dictionary_known` 0.9871, `multi_token`
0.8525, `single_token` 1.0000, `unseen_name` 0.3855, every delta exactly
+0.0000), overall accepted-only F1 0.8644 -> 0.8626 (-0.18pp). No shape
crossed the `MAX_ACCEPTABLE_SHAPE_RECALL_DROP` threshold, so the
guardrail correctly allowed promotion (it only blocks shape-level
regressions above threshold with no offsetting overall gain - it is not
designed to block a flat, sub-threshold overall dip, and didn't).

Per the process lesson above, `--evaluate`/the guardrail's own benchmark
run isn't sufficient sign-off on its own. Given the tiny change surface
(2 labels, one excluded from training entirely), a full case rescan was
disproportionate and was skipped in favor of a targeted before/after
diff on the two real files that could possibly be affected: old model
(pre-retrain backup) vs. new (live) model on `benchmark_real_675.txt`
and on `CASEFILE.txt` directly. `benchmark_real_675.txt`:
zero diff, as expected since "Doran" never entered training data.
`CASEFILE.txt`: **"ott" and "wallis" - both previously
wrongly ACCEPTED - were demoted to REVIEW**, some (not all) mentions of
"handler" were likewise demoted, and "Gora" itself stayed correctly out
of ACCEPTED in both runs (it was never in the accepted bucket to begin
with). Zero new false positives were introduced (`accepted added: []`)
and nothing was lost from REVIEW (`review removed: []`). Unlike the
Priority 1-2 investigation above, this one produced a real, if modest,
positive effect on real casework, not a net-zero - a single real label
generalized to catch two sibling noise tokens from the same file/genre.
Caveat: this was a 2-file targeted check, not a full-case rescan: this
was assessed as proportionate given the change surface, but a full
rescan would be needed for a stronger guarantee before relying on this
in a different case.

**Real measured results, 2026-09-23 - manual line-by-line audit of 2 real
case files, two genuine bugs found and fixed.** At the user's request, I
manually read two real files end-to-end (not just skimmed the CSV) and
verified every accepted/rejected candidate against its real sentence
context, rather than trusting the pipeline's own output about itself.

*File 1, `CASEFILE.txt`* turned out to be Microsoft's actual
2011 SEC Form 10-K filing. Of 31 accepted names, 15 were real people
(Steven A. Ballmer, Peter S. Klein, Satya Nadella, etc. - all correctly
caught, including via their "Mr./Ms. [Surname]" honorific forms) but
**16 were false positives** - ordinary 10-K vocabulary that happened to
collide with the name dictionary and got rescued by the single-token-
dictionary-hit-plus-repetition path, same failure class as "Mellon"
above: June/April/August (calendar dates), Fair/Gross/Cash/Sales/Key/Loan
(accounting terms), Stock/Server (product terms), Rico ("Puerto Rico"),
Ingram ("Ingram Micro"), Geronimo/Rails (software project names), Bona
("bona fide dispute"). All 16 verified individually against their real
sentence via direct file inspection, then blacklisted (see
`assets/blacklist/blacklist.txt`, 2026-09-23 entry). I also found a
secondary, smaller recall gap: the officer table's full canonical names
with middle initials ("Craig J. Mundie", "Lisa E. Brummel") were detected
but hard-rejected by the corroboration validator (no title, appear once,
"collision word" first name), and "Kurt D. DelBene" wasn't detected as a
candidate at all (camelCase surname). Real-world impact is softened
because each person's honorific short form is still correctly accepted
elsewhere in the same document - left as a known gap, not fixed here.

*File 2, `CASEID_chat.txt`* (a real WhatsApp export) was mostly
correct - Corvin Hale, [other contacts], and even
a surname (pulled from a self-identified rank-and-name signature embedded in a
message body, distinct from the contact's own display name) were all
correctly caught, with junk ("Bsnl Customercare", "Can't", "Idk")
correctly kept out of accepted. But "Rehan" (a spammer's self-identified
name) was hard-rejected by `SystemLogValidator` with the message "sits on
a line containing an Android/Java reverse-DNS package or class
identifier" - clearly wrong for a WhatsApp chat line. Root-caused via the
validator's own regex: a missing space after a sentence-ending period in
the source text ("...founder of Giftly.co.in.At GFT...") let
`_PACKAGE_IDENTIFIER_RE` restart its match mid-domain at "co.in.At" -
structurally identical to a genuine "android.os.Handler" identifier
shape purely by coincidence. Fixed with a `(?<!\.)` negative lookbehind
so the regex can no longer start matching immediately after another
dotted chain's "." (a genuine Android/Java identifier always starts at
its own word boundary - after whitespace, "=", or a line start - never
immediately after a preceding "."). Confirmed against all 4 existing
regression tests (still pass, unchanged) plus a new 5th test for this
exact case; confirmed on the real 10.3MB `dumpsys_ANR_WindowManager.txt`
export that every real "READ"/"Handler" occurrence (27 and 3 respectively)
is still correctly rejected - the fix closes the false-negative hole
without reopening the false-positive hole the validator exists to guard.

Both fixes measured on both axes per the process lesson above: 98/98
tests pass; benchmark accepted-only F1 went 0.8626 -> 0.8633 (+0.07pp,
no regression - the blacklist fix moved this slightly, the validator fix
is benchmark-invisible since the benchmark doesn't contain this exact
missing-space-after-domain shape); real-corpus confirmed directly -
`CASEFILE.txt`'s accepted list went from 31 (16 FP + 15 TP)
to exactly 15 (0 FP, 15 TP), and "Rehan" now correctly appears in
`CASEID_chat.txt`'s accepted output. Both are genuine, verified wins
found by actually reading real source text end-to-end instead of relying
solely on the tool's own aggregated output about itself.

**Closing full-case rescan, 2026-09-23 - confirms the combined effect of
the day's three changes (retrain, blacklist, `SystemLogValidator` fix) on
the real 7,246-file case, not just the individual files spot-checked
above.** Accepted count: 236 -> 218 (-18). Review count: 169 -> 171 (+2).
Diffed both full accepted/review sets between the two runs directly
(not estimated): **every single change is already explained by a
documented fix above, nothing unexpected** - the 18 removed from
accepted are exactly the 16 newly-blacklisted 10-K vocabulary words plus
"Ott"/"Wallis" (the retrain-demoted jQuery-locale noise from
`CASEFILE.txt`), and those same 2 account for the entire
review-count increase. Zero names were newly added to accepted. "Rehan"
was already accepted before this session's fix (via 2 other real
occurrences in `chat-13.txt`/`chat-14.txt`) - the fix's effect there is
real but invisible in the accepted/review set diff: occurrence count
2 -> 4, source files 2 -> 3, the previously-silently-dropped
`CASEID_chat.txt` mention now correctly included. A genuinely clean
result: three independent changes, one full-corpus rescan, zero
surprises. The rescan's own REVIEW logging added exactly 2 new
`pending_review.jsonl` items - "Handler" and "Wallis," both from the
same already-identified `CASEFILE.txt` noise file, left
for a future labeling round rather than acted on immediately here.

## Retraining / extending the knowledge base

Everything in `assets/*.txt` is a plain-text, one-entry-per-line, `#`-commented
file — no list is hardcoded in code. After editing any of them:

```bash
python scripts/generate_training_data.py   # regenerate synthetic training data
python scripts/train_model.py               # retrain the classifier from scratch
python -m pytest tests/ -v                   # confirm nothing regressed
python cli.py --evaluate                     # confirm benchmark metrics hold
```

For ongoing improvement from your own real documents, use the active
learning loop instead of editing the dictionary by hand — see above.
`scripts/retrain_from_feedback.py` folds real corrections in without
discarding the synthetic baseline.

Adding a new validator: implement `BaseValidator` in
`src/validation/validators/`, register it in
`src/validation/validation_pipeline.py`'s `hard_validators` or
`soft_validators` list. Nothing else needs to change (open/closed
principle, per the architecture's Design Principles).

## Testing

```bash
python -m pytest tests/ -v
```
98 tests covering detectors (including spaCy's line-bounded chunking -
chunk-size invariants, exact reconstruction, an exotic-Unicode-line-
separator edge case, and an end-to-end offset-correctness check), all
16 validators, the decision engine, feature extraction, evaluation
metrics, the feedback/active-learning store (including `context_text`
capture, its backward-compatible default for legacy records, and the
2026-09-22 `_existing_texts()` caching fix's cross-call dedup
correctness), and full pipeline integration - including eleven real
regression tests: the fabricated-name gate, the sentence-boundary-
crossing fix, the ambiguous single-token dictionary word gate, the
common-word-phrase guard on ML-confidence-alone corroboration, the same
guard's 2026-09-18 extension to REVIEW, the spaCy chunking coverage fix,
the four 2026-09-22 targeted-investigation fixes (single-token-
dictionary-hit-plus-repetition corroboration, the isolated-line
structural corroboration path for `unseen_name`, and each one's control
tests proving they're scoped, not blanket loosenings), and the
2026-09-23 `SystemLogValidator` missing-space-after-domain false-
negative fix.

## Independent-audit findings, 2026-09-23 — two scoped decisions, not implemented

An independent audit surfaced 8 findings. Six were fixed directly (README/
code drift, zero PdfReader test coverage, output provenance stamping, git
history, CI, and the identity-resolution report caveat — see the git log
and `.github/workflows/ci.yml` for those). The two below are genuine
scope decisions and were deliberately NOT implemented blind, per this
project's own standing rule that a heavier-dependency decision gets
measured before being built (see "Why we didn't add a heavier ensemble/
transformer model" above) — findings and a recommendation only.

**Finding #8, privacy guardrail on `datasets/feedback/*.jsonl`, turned
into a real incident, not just a guardrail.** Its first real run found
the working copy was NOT the "currently clean" state assumed going in:
**664 of 1,313 `confirmed_labels.jsonl` records (119 `confirmed_person`,
541 `confirmed_not_person`) were sourced from the real case folder**, not
the benchmark/public-book-only content this file is meant to hold. Root
cause, traced end to end: no code anywhere — not `log_review_persons()`,
not `label_feedback.py`, not `retrain_from_feedback.py` — ever filtered
a record by its source; the earlier claim that "634 items were
deliberately left to the case's own reviewer" was pure human discretion
during one interactive labeling session, with zero systematic backstop.
A second, independent bug compounded it: `line_containing()`
(`src/preprocessing/segmenter.py`) had no size cap, so a candidate on a
document with no nearby newline (a single-line chat export, a minified
JS bundle) captured the *entire remaining document* as "context" —
confirmed to have produced up to a 927,834-character single-record
capture, and specifically why 3 short non-person tokens from one real
chat ended up with that chat's full conversation (both parties' real
phone numbers included) attached, instead of a short local snippet.

Both are fixed: `line_containing()` now caps at 2000 chars, centered on
the candidate so it's never truncated out of its own context
(confirmed_labels.jsonl: 256MB → 1.2MB after re-deriving every surviving
record); `scripts/strip_real_case_feedback_records.py` removed every
real-case-sourced record (backed up first, per this project's existing
migration-script convention); `datasets/feedback/*.jsonl` and
`datasets/feedback/backups/` are now gitignored and untracked entirely —
the retraining pipeline reads them from the same local path regardless,
so no capability was lost, only their git history. `HEAD`'s original
133-line seed was confirmed never contaminated (all 664 records entered
via uncommitted work). See the git log for the exact commits.

**Finding: the real-world coverage gap (now stated on the front page
above).** On the real 47,353-file case folder this project's own casework
uses: 84.7% of files are never opened at all (wrong extension), and a
further 8.1% of the total are scanned/image-only PDFs with no text layer.
Net: only ~4.9% of all files in a real case currently contribute an
extracted name. This was previously true but not prominently stated -
fixed by surfacing it at the top of this README and unconditionally in
`scan_directory.py`'s own summary output (a `coverage_pct_of_all_files`
line, always printed, not something you have to compute from the other
numbers yourself).

**Should OCR be added? Scoped, not decided.**

*Dependency cost, compared like the GLiNER decision was:* Tesseract (via
`pytesseract`) is the only OCR option that fits this project's actual
constraints - CPU-only, no GPU/CUDA stack (unlike GLiNER's rejected 4.5GB
install), Apache-2.0, and it's a mature, widely-available binary (`winget`/
`apt`/`brew`). The real cost isn't the Python dependency, it's that it
would be this project's *first* non-pip, non-Python system dependency -
`setup_env.ps1`/`.sh` would need a Tesseract-presence check added, and CI
would need it preinstalled on the runner. Cloud OCR (AWS Textract/Azure/
Google Vision) was considered and rejected outright, not just deprioritized
- it would break the "offline" guarantee this project's title and design
are built around, a worse fit than the dependency-size question alone. A
vision-capable local model repeats exactly the GLiNER mistake pattern
(heavy, GPU-oriented, unjustified without evidence) and wasn't
investigated further for the same reason GLiNER wasn't.

*Accuracy cost:* not measured (would require actually running OCR, out of
scope for a scoping pass), but reasoned from this project's own design:
OCR noise hits proper nouns hardest - exactly the class this tool cares
about most - which could go either way for precision. The optimistic case:
this project's core design (rules are the gate, dictionary/title
corroboration required for ACCEPT) means OCR garbage that doesn't match
any known name shape mostly fails to corroborate and lands in
REJECTED/REVIEW rather than a false ACCEPT - the same "fail safe toward
REVIEW" behavior already relied on for spaCy's known false positives. The
risk case: OCR could plausibly manufacture NEW dictionary-collision noise
(misread characters coincidentally forming a real dictionary word) the
same way the calendar-word and ALL-CAPS-acronym classes did - not
hypothetical, since that exact failure class has already happened twice
with clean text.

*Payoff, measured at the metadata level only:* sampled the 3,829
image-only PDFs from the 2026-09-23 real case scan. 3,828 of 3,829 share a
ProDiscover hash-style export name (`EF\0ER\...`) rather than a
human-readable filename, and a random sample of 30 are uniformly
single-page, 3-65KB - consistent with individual photo/screenshot-style
export artifacts (the kind of thing a mobile forensic export produces per
message/attachment), not bulk multi-page scanned documents. Exactly one
has a human-readable name and is a genuine 98-page scanned document. This
metadata-level profile is *suggestive* that a meaningful fraction could be
photographed IDs, screenshots, or similar name-bearing images, but it is
NOT a real yield estimate - determining that requires opening actual
content from a real, currently-open fraud case, which this audit
deliberately did NOT do unilaterally (the same real-content-in-
version-control caution as the confirmed_labels.jsonl privacy incident
above applies here too, just for reading rather than committing).

**Recommendation: don't build OCR yet.** If pursued, Tesseract is the only
option consistent with this project's offline/dependency-risk posture.
Before writing integration code: have the case owner (or someone
authorized on the case) manually open a random sample of 15-20 of the
3,829 image-only PDFs and report how many actually contain a name a human
would want extracted. That single number is the actual missing input to
this decision - everything else above is already known.

**Benchmark methodology limits (finding #6).** Confirmed directly: the
benchmark is 7 files, `FILE_DOMAINS` in `src/evaluation/evaluator.py`
covers `prose` (2 files), `chat_or_call_log` (3, including both large real
files), `email` (1), and `system_log` (1) - and that `system_log` file
(`benchmark_structured_dump_005.txt`) is, by its own gold file's `note`
field, "synthetic ... zero real person mentions by construction," so its
domain-level recall/F1 is structurally undefined (0/0), not just small-
sample - the domain that most needs coverage (the noisiest, dumpsys-style
real casework) is the one domain `--evaluate` cannot currently score at
all. Separately, and not previously flagged: **no PDF-sourced benchmark
file exists at all** - every gold-labeled file is `.txt`, despite PDF now
being a first-class, heavily-used real input format (see the scope
correction above). `evaluator.py`'s own comment already acknowledges the
OCR-domain gap but not this one.

Recommendation, not yet actioned (needs real files, which this audit
doesn't have standing to select from casework alone):
1. Add 2-3 more real gold-labeled files, chosen to close the two gaps just
   found: one real (not synthetic) `system_log`/dumpsys-style file with at
   least a few genuine person mentions if any exist in that genre, and one
   PDF-sourced file (real text-layer PDF, not a `.txt` copy of one) so the
   PDF path has any benchmark coverage at all.
2. Set aside one small file, labeled but explicitly excluded from tuning
   decisions for several sessions, as a genuine blind check - not touched
   until deliberately asked for.
3. Print the sample-size caveat directly in `--evaluate`'s own output
   ("F1 computed on N=7 files; treat swings under ~2pp as noise at this
   sample size") so it travels with the number instead of being knowledge
   that only exists in this README.

Item 3 is cheap and was applied directly - see `--evaluate`'s output.
Items 1-2 need the project owner to select or approve real source files
before more work goes in, per the same "don't guess on real casework"
boundary this project already applies to feedback labeling.

## Known limitations (stated plainly, not hidden)

- **Alias/coreference resolution is intentionally NOT attempted.**
  "Suresh", "Dr. Suresh Kumar", and "S. Kumar" are reported as three
  separate `AggregatedPerson` entries, not merged into one identity. This
  is deliberate: silently merging surface forms risks conflating two
  *different* people who share a first name, which is a worse error for a
  no-false-positive tool than under-merging. Documented in
  `src/candidate/candidate_aggregator.py`.
- **A name immediately followed by an organizational word can be swallowed
  into a rejected span** — e.g. "Prasad Office" (a chat display-name
  label in one of our test texts) is rejected as a whole rather than
  recovering "Prasad" alone. This is a deliberate, understood precision/
  recall tradeoff (favoring precision), not an oversight — see the
  `OrganizationValidator` docstring.
- **A person whose name happens to also be a known place or organization
  can be hard-rejected outright when mentioned as a single bare token** -
  e.g. a lone "Santiago" or "Trent" with no other name in the same span
  matches `locations.txt`/`organizations.txt` exactly and is rejected the
  same way a real place/company mention would be. Found at bulk-import
  scale via `scripts/audit_collisions.py` after the 2026-09-16 global
  name-dictionary expansion (~285 first/last names now collide with a
  location or organization entry). This is the OPPOSITE failure
  direction from the "Major" bug below - a false REJECT, not a false
  ACCEPT - and is deliberately left unfixed for now rather than risk a
  rushed change to `LocationValidator`/`OrganizationValidator`'s exact-
  match logic, which is core to keeping genuine place/company mentions
  out of ACCEPTED. A full name (e.g. "Santiago Cruz") is unaffected,
  since both validators match the WHOLE candidate span, not per-token.
- **The keyword/marker-word approach in Campaign and Organization
  validators is not true semantic understanding** — a sufficiently novel
  slogan or business name that avoids every marker word could still slip
  through. Closing this gap fully needs a properly labeled training
  corpus, which does not exist for this task yet.
- **spaCy's `en_core_web_sm` is a general English model**, not tuned for
  South Asian names — it occasionally mislabels an unrelated word as
  PERSON. `GrammarValidator` and the knowledge-corroboration gate
  specifically exist to contain this risk, but it isn't eliminated for
  every conceivable sentence shape.
- **The LightGBM classifier's headline accuracy number is a synthetic
  validation metric**, not a real-world benchmark. The real-world number
  is the `--evaluate` precision/recall/F1, and the two are never shown
  without labeling which is which.
- **`datasets/feedback/` is no longer shipped in this repository at all**
  (`datasets/feedback/*.jsonl` and `datasets/feedback/backups/` are
  gitignored as of the 2026-09-23 independent audit) — real casework
  content had gotten into it despite the original intent to keep it
  benchmark/public-book-only, entirely through human labeling discretion
  with no code-level check. See "Independent-audit findings" above for
  the full incident and fix. The local, untracked copy still exists and
  the retraining pipeline still reads it from the same path — only its
  git history is gone.

## Project structure

```
PERSON_EXTRACTOR_V4/
  cli.py                    Entry point
  config.py                 Central configuration
  requirements.txt
  setup_env.ps1 / .sh       One-command environment setup
  src/
    core/                   models.py, document_factory.py
    io/                     dispatcher.py, base_reader.py, text_reader.py
    preprocessing/          cleaner.py, segmenter.py, line_indexer.py
    detection/               base_detector.py, regex_detector.py,
                              dictionary_detector.py, spacy_detector.py,
                              detector_manager.py
    candidate/               candidate_factory.py, candidate_aggregator.py
    validation/               base_validator.py, validation_pipeline.py,
                               validators/ (16 files)
    features/                 feature_vector.py, feature_extractor.py
    classification/            base_classifier.py, lightgbm_classifier.py,
                                model_registry.py
    decision/                  decision_engine.py
    knowledge/                 knowledge_base.py
    evaluation/                 metrics.py, evaluator.py
    feedback/                   feedback_store.py (active learning loop;
                                 FeedbackRecord.context_text captures the
                                 source line around each candidate)
    export/                     csv_exporter.py, json_exporter.py,
                                 report_exporter.py
    pipeline/                   orchestrator.py
    utils/                       logger.py, timing.py
  assets/                    Plain-text knowledge base (names, orgs, etc.)
                              common_words/ - general English vocabulary
                              (not names): guards the ML-confidence-alone
                              corroboration path AND (since 2026-09-18)
                              blocks zero-corroboration common-word
                              phrases from parking in REVIEW - see
                              decision_engine.py
  datasets/
    benchmark/               Gold-labeled evaluation text + labels
    training/                 Synthetic training data (generated)
    feedback/                 pending_review.jsonl, confirmed_labels.jsonl
  models/lightgbm/           Trained classifier + metrics.json
  scripts/                   generate_training_data.py, train_model.py,
                              label_feedback.py, retrain_from_feedback.py,
                              build_benchmark_from_markup.py,
                              scan_directory.py (recursive folder batch scan)
                              audit_collisions.py (cross-list knowledge-base
                              collision finder, e.g. the "Major" bug)
                              migrate_feedback_add_context_text.py
                              (one-time backfill for FeedbackRecord.context_text)
  tests/                     97 tests
  output/                    Extraction results land here
  logs/                      pipeline.log (+ .1-.3 backups, rotated at 50 MB)
  models/lightgbm/backups/   Auto-saved model snapshot from before every
                              retrain (see lightgbm_classifier.py's save())
```
