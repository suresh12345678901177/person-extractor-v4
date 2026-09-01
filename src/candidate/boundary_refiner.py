"""
src.candidate.boundary_refiner
=================================
Trims a leading/trailing token off a multi-token candidate when that
token is boundary noise rather than part of the name - detected purely
from within-document repetition, not any word list.

Why this exists: on chat/email exports that concatenate a sender label
directly against message text with no delimiter ("Testing Severus
Morris", "Kk Severus Morris", "Severus Morris Kakao"), the same real
name recurs dozens of times cleanly, but occasionally picks up one
extra glued-on word. Enumerating every possible glue-word in
stopwords.txt is a losing, data-specific game - it only ever covers
words already seen. This instead asks a document-relative question that
works on ANY real-world text, known vocabulary or not: does trimming
one edge token reveal a form that is overwhelmingly more common
elsewhere in this same document? If "Severus Morris" independently
recurs 80+ times and "Testing Severus Morris" appears once, the
extra word is almost certainly noise, regardless of whether "Testing"
is a word this tool has ever heard of.

Deliberately NOT the same thing as alias/coreference merging (which
candidate_aggregator.py explicitly avoids, for good reason - conflating
two different people is a worse error than under-merging). This never
decides two DIFFERENT surface forms refer to the same person; it only
corrects one candidate's own span boundary using how often that exact
shorter text already recurs as its own independent candidate in this
same document.

A second function, refine_combined_counts(), applies the identical
dominance rule at BATCH granularity instead of single-document
granularity - across every file processed together for one case (see
cli.py's run_batch()). This is a deliberate, narrower relaxation of the
"never mix signal across documents" boundary: safe when every file in
the batch is known to be about the same subject (one case folder), but
NOT something to reach for if a folder ever mixes files from multiple
unrelated people, since it would then risk conflating them. It operates
on already-finalized per-file combined name/occurrence counts, not raw
candidates, so it never changes any individual file's own report/
decision - only the cross-file combined names view.

A third function, trim_stopword_edges(), fixes a gap the frequency-based
trim can't cover: a person mentioned only once or twice in one document
(so the "clean" form never reaches MIN_CORE_OCCURRENCES on its own)
whose corrupted span ends in a stopword ("Devraj Bhatt We", "Devraj
Bhatt Will") gets HARD-REJECTED WHOLESALE by StopwordValidator later in
the pipeline - destroying the real name along with the noise, because
that validator has no way to know only the edge token is the problem.
This needs no repetition evidence at all: a stopword can never
legitimately be part of a real name by this tool's own definition of
the word list, so trimming one sitting at a candidate's very edge is
safe unconditionally. Must run BEFORE validation, same as the other two
passes, so the trimmed form is what StopwordValidator actually sees.

A fourth function, trim_dictionary_confirmed_organization_edge(), fixes
a narrower relative of the same problem: "Prasad Office" (a chat display-
name label) gets hard-rejected WHOLESALE by OrganizationValidator for
containing "Office", destroying the real name "Prasad" along with it.
Unlike the stopword case, an organization keyword is NOT unconditionally
safe to strip - "Sri Venkatesh Traders" and "Rajupalem Municipal Office"
are genuine multi-word organization names, not people with a glued-on
extra word, and a blanket/iterated org-keyword strip was tried and
reverted twice for turning both into fake person names (see project
history). The two failure cases are both 3 tokens; this function
deliberately only ever fires on an EXACTLY two-token span (OrgKeyword +
Name, or Name + OrgKeyword) where the REMAINING token is independently
a KNOWN first or last name in the dictionary - "Prasad" qualifies
("Prasad Office"), "Rajupalem"/"Sri"/"Venkatesh" combinations never
reach this function at all (3 tokens), so this is a strictly narrower,
dictionary-gated rescue, not a repeat of the earlier broader attempts.
"""

from __future__ import annotations

import re
from collections import Counter

from src.core.models import Candidate, CandidateResult
from src.detection.dictionary_detector import _strip_possessive
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.line_indexer import LineIndex
from src.validation.validators.organization_validator import ORG_KEYWORDS

_TOKEN_RE = re.compile(r"\S+")

# A frequency-based trim only fires when the shorter form is both
# well-established on its own (not itself a rare coincidence) AND
# overwhelmingly more common than the untrimmed form - conservative on
# both counts, since this is a hard structural rewrite of the candidate,
# not a soft evidence nudge.
MIN_CORE_OCCURRENCES = 3
MIN_DOMINANCE_RATIO = 3


def _rewrite_span(cr: CandidateResult, new_core: str, line_index: LineIndex) -> CandidateResult | None:
    """Rebuilds cr with its span narrowed to new_core, which must be
    either tokens[:-1] or tokens[1:] of the candidate's own normalized
    text - locates the real character boundary via the RAW text's own
    whitespace-delimited tokens (not the normalized text), so this stays
    correct even when the original text has irregular spacing (double
    spaces, tabs) that normalize_name() already collapsed away."""
    tokens = cr.candidate.normalized_text.split()
    spans = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(cr.candidate.text)]
    if len(spans) != len(tokens):
        return None  # raw/normalized token count mismatch - bail out rather than guess

    if new_core == " ".join(tokens[:-1]):
        new_start_rel, new_end_rel = spans[0][0], spans[-2][1]
    elif new_core == " ".join(tokens[1:]):
        new_start_rel, new_end_rel = spans[1][0], spans[-1][1]
    else:
        return None

    new_start = cr.candidate.start + new_start_rel
    new_end = cr.candidate.start + new_end_rel
    new_text = cr.candidate.text[new_start_rel:new_end_rel]

    location = line_index.locate(new_start, new_end)
    new_candidate = Candidate.new(
        text=new_text,
        normalized_text=new_core,
        start=new_start,
        end=new_end,
        page_index=cr.candidate.page_index,
        source_detections=cr.candidate.source_detections,
        location=location,
    )
    return CandidateResult(candidate=new_candidate, state=cr.state)


def _slice_candidate(cr: CandidateResult, spans: list[tuple[int, int]], tokens: list[str], lo: int, hi: int, line_index: LineIndex) -> CandidateResult:
    """Builds a fresh CandidateResult covering tokens[lo:hi] (raw text,
    via precomputed per-token spans). Always gets its own new
    CandidateState (not cr.state) since this is used to split one
    original candidate into multiple independent ones - sharing the
    same mutable state object across them would let validation of one
    half silently corrupt the other's evidence/decision."""
    new_start_rel, new_end_rel = spans[lo][0], spans[hi - 1][1]
    new_start = cr.candidate.start + new_start_rel
    new_end = cr.candidate.start + new_end_rel
    new_text = cr.candidate.text[new_start_rel:new_end_rel]
    new_core = " ".join(tokens[lo:hi])

    location = line_index.locate(new_start, new_end)
    new_candidate = Candidate.new(
        text=new_text,
        normalized_text=new_core,
        start=new_start,
        end=new_end,
        page_index=cr.candidate.page_index,
        source_detections=cr.candidate.source_detections,
        location=location,
    )
    return CandidateResult(candidate=new_candidate)


def _pick_dominant_core(key: str, own_count: int, counts: dict[str, int]) -> str | None:
    """Shared decision rule for both refine_boundaries (single-document)
    and refine_combined_counts (batch): does trimming a leading/trailing
    token reveal a form that is both well-established and overwhelmingly
    more common than the untrimmed form? Returns that form, or None.

    Deliberately only ever returns a core with 2+ tokens remaining - a
    lone bare first name is NEVER an eligible trim target, no matter how
    dominant. Found via two independent real-case confirmations
    ("Katya Sokolova" -> "Katya", "Devon Marsh" -> "Devon"): a person
    who is usually referred to by first name alone and only occasionally
    by their full name is a completely normal, common chat-log pattern -
    the high standalone frequency of the bare first name is NOT evidence
    that the surname in the full-name mentions is glued-on noise, unlike
    "Testing Severus Morris" -> "Severus Morris" (still 2 tokens after
    trimming - a fully plausible name on its own), which this heuristic
    was actually built for.
    """
    tokens = key.split()
    if len(tokens) < 2:
        return None

    best_core, best_count = None, own_count
    for core in (" ".join(tokens[:-1]), " ".join(tokens[1:])):
        if not core or len(core.split()) < 2:
            continue
        core_count = counts.get(core, 0)
        if core_count >= MIN_CORE_OCCURRENCES and core_count >= own_count * MIN_DOMINANCE_RATIO:
            if core_count > best_count:
                best_core, best_count = core, core_count
    return best_core


def refine_boundaries(candidates: list[CandidateResult], line_index: LineIndex) -> list[CandidateResult]:
    text_counts = Counter(c.candidate.normalized_text for c in candidates)

    refined: list[CandidateResult] = []
    for cr in candidates:
        key = cr.candidate.normalized_text
        best_core = _pick_dominant_core(key, text_counts[key], text_counts)
        trimmed = _rewrite_span(cr, best_core, line_index) if best_core else None
        refined.append(trimmed if trimmed is not None else cr)
    return refined


def trim_stopword_edges(candidates: list[CandidateResult], knowledge_base: KnowledgeBase, line_index: LineIndex) -> list[CandidateResult]:
    refined: list[CandidateResult] = []
    for cr in candidates:
        current = cr
        for _ in range(4):  # a candidate has at most 4 tokens (regex detectors cap there)
            tokens = current.candidate.normalized_text.split()
            if len(tokens) < 2:
                break
            if knowledge_base.is_stopword(tokens[0].rstrip(".")):
                new_core = " ".join(tokens[1:])
            elif knowledge_base.is_stopword(tokens[-1].rstrip(".")):
                new_core = " ".join(tokens[:-1])
            else:
                break
            next_cr = _rewrite_span(current, new_core, line_index)
            if next_cr is None:
                break
            current = next_cr
        refined.append(current)
    return refined


def trim_dictionary_confirmed_organization_edge(
    candidates: list[CandidateResult], knowledge_base: KnowledgeBase, line_index: LineIndex,
) -> list[CandidateResult]:
    """Rescues a real person's name from being swallowed whole by a
    single glued-on organization keyword ("Prasad Office" -> "Prasad").
    See this module's docstring for why this is deliberately much
    narrower than trim_stopword_edges: an organization keyword is NOT
    unconditionally safe to strip the way a stopword is (a genuine
    multi-word organization name looks structurally identical), so this
    only ever fires when BOTH hold:
      1. the candidate is EXACTLY two tokens - never longer, and
      2. the token that ISN'T the organization keyword is independently
         a KNOWN first or last name in the dictionary.
    A 3+-token organization-keyword-adjacent span never reaches this
    function at all and is left for OrganizationValidator to hard-reject
    wholesale, exactly as before this function existed.
    """
    refined: list[CandidateResult] = []
    for cr in candidates:
        tokens = cr.candidate.normalized_text.split()
        if len(tokens) != 2:
            refined.append(cr)
            continue

        first_is_org = tokens[0].rstrip(".,").lower() in ORG_KEYWORDS
        last_is_org = tokens[1].rstrip(".,").lower() in ORG_KEYWORDS
        if first_is_org == last_is_org:  # neither, or both - not this shape
            refined.append(cr)
            continue

        name_token = tokens[1] if first_is_org else tokens[0]
        stem = name_token.rstrip(".")
        if not (knowledge_base.is_known_first_name(stem) or knowledge_base.is_known_last_name(stem)):
            refined.append(cr)
            continue

        next_cr = _rewrite_span(cr, name_token, line_index)
        refined.append(next_cr if next_cr is not None else cr)
    return refined


def _is_dictionary_full_name_pair(a: str, b: str, knowledge_base: KnowledgeBase) -> bool:
    """Order-agnostic "is this pair a known first+last name" check, same
    rule dictionary_detector._extend_window uses to build a 2-token
    window - reused here so the split decision stays consistent with
    what the dictionary detector itself would call a real name."""
    a_stem = _strip_possessive(a)[0].rstrip(".")
    b_stem = _strip_possessive(b)[0].rstrip(".")
    has_first = knowledge_base.is_known_first_name(a_stem) or knowledge_base.is_known_first_name(b_stem)
    has_last = knowledge_base.is_known_last_name(b_stem) or knowledge_base.is_known_last_name(a_stem)
    return has_first and has_last


def split_double_name_candidates(candidates: list[CandidateResult], knowledge_base: KnowledgeBase, line_index: LineIndex) -> list[CandidateResult]:
    """Splits a single 4-token candidate into two 2-token candidates when
    both halves independently look like a real, standalone name.

    Why this exists: chat/call-log exports that concatenate two adjacent
    speaker-tag names with no delimiter ("Victor Cruz Michael Torres",
    two different people's names run together on one line) satisfy the
    regex/spaCy "2-4 capitalized tokens" pattern as a single span, and
    neither detector has any way to know it's actually two people, not
    one four-word name.

    A half is trusted two ways:
      1. Dictionary-known: both its tokens independently pass the same
         known-first+last-name check the dictionary detector itself uses.
      2. Document-frequent: that exact 2-token form already recurs
         MIN_CORE_OCCURRENCES+ times elsewhere in this document as its
         own candidate - same document-relative, no-word-list-required
         principle _pick_dominant_core already uses above. This matters
         because real names (foreign/uncommon surnames especially) often
         aren't in assets/first_names.txt or last_names.txt at all, so a
         dictionary-only check would miss exactly the real-world case
         this bug was found on.
    Splits when EITHER half is independently DICTIONARY-confirmed on its
    own, or when BOTH halves are at least document-frequent. A dictionary
    hit is much stronger, more specific evidence than mere frequency, so
    it doesn't need corroboration from the other half - found via real
    testing: "Priyanka Deol Felix Amaro" (two different real people's
    names run together with no delimiter) never split under the
    original both-halves-required rule, because "Felix Amaro" had no
    dictionary hit AND recurred zero times elsewhere in the document -
    even though "Priyanka Deol" alone was already a fully-confirmed
    dictionary name pair, strong evidence on its own that whatever
    follows is a separate span, not a genuine four-word single name.
    Splitting is only ever a boundary decision, not a validation
    decision - the resulting "Felix Amaro" half still goes through
    every validator independently afterward on its own merits, same as
    any other candidate, so trusting one confirmed half here doesn't
    smuggle a weak half past anything. Titled candidates ("Dr. John
    Michael Smith") are excluded since the title already anchors the
    whole span as one person.
    """
    text_counts = Counter(c.candidate.normalized_text for c in candidates)

    result: list[CandidateResult] = []
    for cr in candidates:
        tokens = cr.candidate.normalized_text.split()
        is_titled = any(d.metadata.get("pattern") == "titled" for d in cr.candidate.source_detections)
        if len(tokens) != 4 or is_titled:
            result.append(cr)
            continue

        left_key = " ".join(tokens[:2])
        right_key = " ".join(tokens[2:])
        left_dict = _is_dictionary_full_name_pair(tokens[0], tokens[1], knowledge_base)
        right_dict = _is_dictionary_full_name_pair(tokens[2], tokens[3], knowledge_base)
        left_freq = text_counts.get(left_key, 0) >= MIN_CORE_OCCURRENCES
        right_freq = text_counts.get(right_key, 0) >= MIN_CORE_OCCURRENCES
        if not (left_dict or right_dict or (left_freq and right_freq)):
            result.append(cr)
            continue

        spans = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(cr.candidate.text)]
        if len(spans) != 4:
            result.append(cr)  # raw/normalized token count mismatch - bail out rather than guess
            continue

        result.append(_slice_candidate(cr, spans, tokens, 0, 2, line_index))
        result.append(_slice_candidate(cr, spans, tokens, 2, 4, line_index))
    return result


def refine_combined_counts(combined: dict[str, dict]) -> dict[str, dict]:
    """Batch-level pass: combined is keyed by normalized_text ->
    {"display": str, "occurrences": int, "files": list[str], "decision":
    str (optional, defaults to "review")}, built from every file in one
    case (see cli.py's run_batch()). Merges any entry that is boundary
    noise around a form that dominates across the WHOLE batch, even if
    it didn't dominate within any single file alone.

    "decision" is carried through and merged "accepted wins" - same rule
    AggregatedPerson.decision already uses within a single file (a
    person is accepted case-wide if accepted in ANY file, even if other
    mentions elsewhere were only reviewed) - so an accepted+review split
    across files collapses into one accepted entry, not two, or a
    silently-downgraded review-only one. Missing "decision" (older
    callers/tests that never set it) defaults to "review", the more
    conservative reading.
    """
    counts = {k: v["occurrences"] for k, v in combined.items()}

    merged: dict[str, dict] = {}
    for key, entry in combined.items():
        target = _pick_dominant_core(key, entry["occurrences"], counts) or key
        target_display = combined[target]["display"] if target in combined else entry["display"]

        slot = merged.setdefault(target, {"display": target_display, "occurrences": 0, "files": [], "decision": "review"})
        slot["occurrences"] += entry["occurrences"]
        for f in entry["files"]:
            if f not in slot["files"]:
                slot["files"].append(f)
        if entry.get("decision", "review") == "accepted":
            slot["decision"] = "accepted"

    return merged
