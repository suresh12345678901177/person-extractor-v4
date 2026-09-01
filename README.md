# PERSON_EXTRACTOR_V4

An offline, explainable, **TXT-only** Person Name Extraction framework built
around a 16-validator rules engine, with a real trained ML classifier used
strictly as a corroborating signal — never as a gate. Every accepted name
tells you exactly why it was accepted, how many times it was found, where,
and how long each stage took.

```
Input .txt File
  -> IO (TextReader + DocumentFactory)
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

**Scope: TXT only.** Every reader for PDF/DOCX/Excel/images was deliberately
left out. Less input-format surface area means fewer places for encoding
quirks, OCR noise, or layout artifacts to introduce false positives — and
false positives are explicitly the #1 thing this project optimizes against.

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

## Real measured results (not projected)

Run `python cli.py --evaluate` to reproduce these against
`datasets/benchmark/` (last run 2026-09-01, 6 benchmark files including
two large real-world documents, `benchmark_real_660.txt` and
`benchmark_real_675.txt`):

| Bucket | Precision | Recall | F1 |
|---|---|---|---|
| ACCEPTED only (auto-shipped) | 0.9627 | 0.3062 | 0.4646 |
| ACCEPTED + REVIEW (what a human sees) | 0.7920 | 0.9686 | 0.8714 |

**Read this honestly, don't just quote the F1.** Precision on
auto-accepted output stays high (as designed), but accepted-only recall
has dropped to ~30% on the two large real-world files — driven by
globally diverse names (e.g. "Kenji Ito," "Nadia Kovalenko," "Femi
Nakamura") that aren't in `assets/first_names.txt` / `last_names.txt`
and so can't clear the knowledge-corroboration gate for auto-accept;
they correctly land in REVIEW instead, which is why the
ACCEPTED+REVIEW recall is still 0.97. This is the intended
precision-favoring tradeoff, not a bug — but it means the honest
framing today is "very high-precision review tool," not "94% F1
auto-extractor." The fix is exactly the active learning loop above:
label a batch of `datasets/feedback/pending_review.jsonl` (413 items
pending as of this writing) and retrain, which directly grows
real-world name coverage. On the two small synthetic-style benchmark
files (001/002) accept-only recall is still 1.00.

The smaller benchmark files, their gold labels, and exactly how those
labels were built (mechanically, via markup-stripping — never
hand-counted character offsets) are in `datasets/benchmark/` and
`scripts/build_benchmark_from_markup.py`.

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
59 tests covering detectors, all 16 validators, feature extraction,
evaluation metrics, the feedback/active-learning store, and full
pipeline integration (including two real regression tests: the
fabricated-name gate, and the sentence-boundary-crossing fix).

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
- **`datasets/feedback/` ships with only benchmark- and public-book-sourced
  entries.** Real third-party documents processed during development were
  intentionally excluded from this repository.

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
    feedback/                   feedback_store.py (active learning loop)
    export/                     csv_exporter.py, json_exporter.py,
                                 report_exporter.py
    pipeline/                   orchestrator.py
    utils/                       logger.py, timing.py
  assets/                    Plain-text knowledge base (names, orgs, etc.)
  datasets/
    benchmark/               Gold-labeled evaluation text + labels
    training/                 Synthetic training data (generated)
    feedback/                 pending_review.jsonl, confirmed_labels.jsonl
  models/lightgbm/           Trained classifier + metrics.json
  scripts/                   generate_training_data.py, train_model.py,
                              label_feedback.py, retrain_from_feedback.py,
                              build_benchmark_from_markup.py
  tests/                     59 tests
  output/                    Extraction results land here
  logs/                      pipeline.log
```
