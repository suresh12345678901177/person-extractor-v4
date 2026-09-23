"""
src.decision.decision_engine
===============================
Combines rule-based evidence (detector confidence + validator scores)
and the ML classifier's probability into one final confidence and
Accepted/Rejected/Review decision.

CORE NO-FALSE-POSITIVE DESIGN PRINCIPLES:

1. A hard validator failure is final. No ML probability, however high,
   can override it. This makes the ML classifier a *corroborating*
   signal rather than a gate.

2. KNOWLEDGE-CORROBORATION GATE (added after adversarial testing found
   a real bug): passing every hard validator is necessary but NOT
   sufficient for ACCEPTED status. A candidate supported ONLY by
   shape-based detectors (bare regex, spaCy) - with ZERO dictionary hit
   and ZERO title match - can still accumulate enough soft-validator
   credit to clear ACCEPT_THRESHOLD on pure pattern-matching alone, with
   no actual knowledge-base evidence that the text is a real name at
   all. This was caught directly: a fabricated two-word capitalized
   string ("Xzqlt Vbnmp") scored 1.25 and was wrongly accepted, because
   regex-bare (0.55) + spaCy (0.50) + small structural credits summed
   past 0.75, and the ML classifier's correctly-low probability
   couldn't pull the score back down (it is deliberately additive-only -
   see below). The fix: ACCEPTED now additionally requires at least one
   of (a) a dictionary hit, (b) a titled-pattern match, or (c) high ML
   confidence (>=0.85) - i.e. at least one signal that has ever looked
   at whether this specific text means something, not just whether it
   is shaped like a name. Candidates that clear the score threshold
   without any such corroboration are capped at REVIEW instead.

3. COMMON-WORD-PHRASE GUARD ON THE ML-CONFIDENCE PATH (added 2026-09-16
   after real casework found a second variant of the same bug class):
   the "(c) high ML confidence" corroboration path above still had no
   check for whether the candidate's words meant "person" at all -
   only whether the classifier scored it highly and it repeated often.
   A retrain on real feedback shifted the classifier's calibration
   enough that bare, zero-dictionary business/tech phrases ("Vice
   President", "Convertible Debt", "Customer Service", ...) started
   clearing this path purely on repetition + a high score, with no
   dictionary/title/spaCy evidence at all - confirmed directly on a
   real case export. The fix: ML-confidence-alone can no longer
   corroborate a candidate where EVERY token is ordinary English
   vocabulary (see `_is_common_word_phrase` and
   assets/common_words/common_english_words.txt). A genuine name is
   unaffected as long as it has any real dictionary/title support
   (which already short-circuits this check earlier) or contains any
   token that isn't just a common word.
"""

from __future__ import annotations

from src.core.models import CandidateResult, Decision, DetectorName
from src.knowledge.knowledge_base import KnowledgeBase
from src.preprocessing.segmenter import count_occurrences
from src.utils.logger import get_logger

logger = get_logger("decision.decision_engine")

ACCEPT_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.45
ML_WEIGHT = 0.35  # secondary signal - rules evidence still dominates
ML_SCORE_SCALE = 3.0  # rescales ML's [0,1] probability into the same rough
# magnitude as accumulated rule evidence (typically 0.5-3.2 for a strong
# candidate), so ML can meaningfully nudge borderline cases without
# dominating candidates the rules engine already scores strongly.
ML_HIGH_CONFIDENCE_OVERRIDE = 0.85  # ML alone can satisfy the knowledge
# gate only if it is this confident - a weak/ambiguous ML score should
# not, by itself, be treated as "the model recognized this as a name".
MIN_REPETITION_FOR_ML_ONLY_CORROBORATION = 4  # same bar as
# CorroborationValidator.MIN_REPETITION_FOR_CORROBORATION, deliberately
# kept in sync (see count_occurrences' shared-helper docstring) - found
# necessary via real testing (case 930204690): a single-occurrence,
# zero-dictionary/title, non-person token ("Dark Corridor" - a "Ref: X |
# Code: ..." ledger label, not a person) scored ML probability 0.99 and
# was wrongly ACCEPTED on that score alone, because the classifier has
# learned to associate a structural quirk (the candidate sitting alone
# right before a line break or delimiter) with personhood - the same
# quirk real standalone chat sender-header names also happen to share,
# which is why this can't be fixed by distrusting high ML confidence in
# general (measured: doing that collapsed real accepted-only recall from
# 31.55% to 8.65% across the official benchmark suite - see
# segmenter.py's following_char() docstring for that reverted attempt).
# Requiring real, document-wide repetition specifically for the
# ML-confidence-ALONE path (dictionary/title evidence is unaffected)
# targets the actual exploitable gap - a single occurrence with no other
# evidence - without touching the genuinely repeated real names or
# genuinely repeated real sender headers this project's real documents
# depend on the same signal for.
ML_WEAK_DICTIONARY_VETO_FLOOR = 0.10  # see _has_knowledge_corroboration:
# a single-token dictionary hit merged into a wider multi-token candidate
# only vouches for the ONE token it covers, not the whole span. Found via
# real testing (case 930204675): "Deep Tunnel" got ACCEPTED because "Deep"
# is a real (if unusual) first name, even though "Tunnel" has zero support
# and the ML classifier scored it 0.00 - strong, independent disagreement
# that was being ignored. This floor lets that disagreement veto ONLY the
# single-token-dictionary-alone case; it never touches candidates that also
# have spaCy support or a full first+last dictionary window match (that
# covers the overwhelming majority of real single-token-corroborated
# acceptances, e.g. "Michael Torres" - "Torres" isn't in the dictionary
# either, but spaCy support keeps its ML score well above this floor).


class DecisionEngine:
    def __init__(self, allow_spacy_only_corroboration: bool = False) -> None:
        # Comparison/testing knob, OFF by default (the strict, documented
        # behavior above). When True, a candidate with ANY spaCy PERSON
        # detection satisfies the knowledge-corroboration gate on its
        # own, even with zero dictionary/title support - i.e. trust
        # spaCy's contextual/grammatical judgment as sufficient evidence
        # by itself, not just as a tiebreaker. This reopens exactly the
        # precision risk the gate was built to close, so it must be
        # explicitly requested (see cli.py's --loose-gate), never the
        # silent default.
        self.allow_spacy_only_corroboration = allow_spacy_only_corroboration

    def decide(
        self, candidate: CandidateResult, document_text: str, knowledge_base: KnowledgeBase | None = None
    ) -> CandidateResult:
        state = candidate.state

        if state.is_hard_rejected:
            state.final_confidence = 0.0
            state.decision = Decision.REJECTED
            if not state.rejection_reason:
                state.rejection_reason = "Hard validator rejection"
            return candidate

        rule_based_score = state.total_evidence_score
        ml_prob = 0.0

        if state.classifier_result is not None and state.classifier_result.ok:
            ml_prob = state.classifier_result.probability
            # Deliberately additive-only (never subtractive): the ML
            # signal can corroborate and strengthen a candidate the rules
            # engine already likes, but a low ML probability can never by
            # itself drag an otherwise rule-confirmed candidate below
            # acceptance. Precision is instead protected by the knowledge-
            # corroboration gate below, not by letting ML subtract score.
            ml_contribution = ML_WEIGHT * (ml_prob * ML_SCORE_SCALE)
            final_score = rule_based_score + ml_contribution
            state.add_evidence(
                "ml_classifier",
                f"LightGBM classifier probability {ml_prob:.2f}",
                ml_contribution,
            )
        else:
            final_score = rule_based_score

        state.final_confidence = round(final_score, 4)

        has_knowledge_corroboration = self._has_knowledge_corroboration(
            candidate, ml_prob, document_text, knowledge_base
        )

        if state.final_confidence >= ACCEPT_THRESHOLD and has_knowledge_corroboration:
            state.decision = Decision.ACCEPTED
        elif state.final_confidence >= REVIEW_THRESHOLD:
            if not has_knowledge_corroboration and self._is_common_word_phrase(candidate, knowledge_base):
                # Every token is ordinary English vocabulary (see
                # _is_common_word_phrase below) AND nothing ever
                # positively recognized this span as a name - no
                # dictionary/title hit, no spaCy/high-ML-confidence
                # corroboration either. Real casework (2026-09-18) found
                # REVIEW being flooded with exactly this class of
                # candidate ("Performance Review", "Retention Bonus",
                # "Corporate Controller", "Coordinated Universal Time",
                # "Deferred Compensation") - plain business/technical
                # noun phrases with zero identity evidence, not
                # genuinely uncertain names. This extends the ML-
                # confidence-alone guard (principle 3 above) one step
                # further: a common-word phrase was already blocked from
                # reaching ACCEPTED on ML alone; now it's also blocked
                # from parking in REVIEW when it has no other
                # corroboration, since REVIEW's purpose (surfacing
                # plausible-but-unconfirmed names for human triage) is
                # defeated by noise this unambiguous. A genuine unseen
                # name is unaffected as long as it has ANY dictionary/
                # title support (already satisfies
                # has_knowledge_corroboration above) or contains even
                # one uncommon/foreign token (already fails
                # _is_common_word_phrase's ALL-tokens check) - the
                # remaining risk is a real name with zero dictionary
                # presence that ALSO happens to be made entirely of
                # ordinary English words, which no longer reaches a
                # human via REVIEW. Judged an acceptable trade given how
                # dominated REVIEW had become by this exact noise class.
                state.decision = Decision.REJECTED
                state.rejection_reason = (
                    "Every token is ordinary English vocabulary with zero "
                    "dictionary/title/spaCy/ML corroboration - reads as a plain "
                    "business/technical phrase, not a plausible name"
                )
            else:
                state.decision = Decision.REVIEW
                if state.final_confidence >= ACCEPT_THRESHOLD and not has_knowledge_corroboration:
                    state.rejection_reason = (
                        "Score cleared the acceptance threshold on shape-based evidence "
                        "only (no dictionary/title/high-confidence ML support) - capped "
                        "at review rather than accepted"
                    )
        else:
            state.decision = Decision.REJECTED
            state.rejection_reason = state.rejection_reason or "Final confidence below threshold"

        return candidate

    def _has_knowledge_corroboration(
        self,
        candidate: CandidateResult,
        ml_prob: float,
        document_text: str,
        knowledge_base: KnowledgeBase | None,
    ) -> bool:
        detections = candidate.candidate.source_detections
        has_window_dictionary_evidence = any(d.metadata.get("pattern") == "dictionary" for d in detections)
        has_single_token_dictionary_evidence = any(
            d.metadata.get("pattern") == "dictionary_single_token" for d in detections
        )
        has_title_evidence = any(d.metadata.get("pattern") == "titled" for d in detections)
        has_confident_ml = ml_prob >= ML_HIGH_CONFIDENCE_OVERRIDE

        if has_window_dictionary_evidence or has_title_evidence:
            return True

        if has_confident_ml and not self._is_common_word_phrase(candidate, knowledge_base):
            # High ML confidence alone (zero dictionary/title evidence)
            # additionally requires real document-wide repetition - see
            # MIN_REPETITION_FOR_ML_ONLY_CORROBORATION's docstring above
            # for the real exploit this closes and the real regression
            # a broader fix already caused when measured. It ALSO now
            # requires the candidate not be a plain common-English-word
            # phrase (see _is_common_word_phrase) - found via real
            # casework (2026-09-16): bare, zero-dictionary phrases like
            # "Vice President" and "Convertible Debt" cleared this exact
            # path on ML score + repetition alone, with nothing checking
            # whether the words themselves meant "person" at all.
            repeat_count = count_occurrences(candidate.candidate.normalized_text, document_text)
            if repeat_count >= MIN_REPETITION_FOR_ML_ONLY_CORROBORATION:
                return True

        if has_single_token_dictionary_evidence and ml_prob >= ML_WEAK_DICTIONARY_VETO_FLOOR:
            return True

        # Added 2026-09-22 after a real diagnosed gap (Priority 1 of a
        # targeted multi_token-recall investigation): real names like
        # "Femi Novak"/"Nadia Kovalenko"/"Jaylen Brooks" - a genuine
        # single-token dictionary hit on ONE name (Femi/Nadia/Jaylen)
        # paired with a token the dictionary has never seen (Novak/
        # Kovalenko/Brooks) - were being vetoed by the floor above even
        # though ML scored them only slightly low (0.03-0.08), nowhere
        # near the 0.00 score of the "Deep Tunnel" incident the floor
        # exists to catch. Rather than lower the floor itself (calibrated
        # against that one real incident; a future genuinely-bad case
        # might not score exactly 0.00 either), this mirrors the
        # ML-confidence-alone path's own pattern immediately above: a
        # weak signal (partial dictionary support OR a middling-but-not-
        # confident ML score) becomes trustworthy when paired with real,
        # independent document-wide repetition. Confirmed via the same
        # diagnostic: 5 of 7 real-name cases found in this investigation
        # had repeat_count >= 4; the 2 that didn't (repeat_count 2) are
        # correctly left requiring a higher bar, unaffected by this path.
        if has_single_token_dictionary_evidence:
            repeat_count = count_occurrences(candidate.candidate.normalized_text, document_text)
            if repeat_count >= MIN_REPETITION_FOR_ML_ONLY_CORROBORATION:
                return True

        # Structural corroboration path (2026-09-22, Priority 2 of a
        # targeted unseen_name investigation): computed the FULL ML
        # confidence distribution across all 249 unseen_name gold
        # mentions (not just min/max) - median 0.12, only ~2% clear
        # 0.70, p90 still under 0.70. A genuine long tail, not a
        # threshold-placement issue - raising ML_HIGH_CONFIDENCE_OVERRIDE
        # further, or adding more synthetic reinforcement (already tried
        # this session, plateaued at 0.838 max before this), would not
        # help. Checked which of feature_extractor.py's two context
        # features actually help THIS shape before using either:
        # preceding_context_is_person_cue was False for 244 of 247 real
        # failing mentions (essentially dead for this shape - these
        # names sit mid-sentence, not near a "said"/"met" cue word) and
        # is_isolated_line was True for 140 of 247 (57%) - the real
        # chat/email sender-header shape this project's actual casework
        # depends on this exact signal for elsewhere. Requiring
        # ml_prob >= ML_WEAK_DICTIONARY_VETO_FLOOR (reusing the existing
        # constant, not a new magic number) as an additional guard - "the
        # ML model isn't actively disagreeing" - keeps 94 of 140 (67%) of
        # the real fixable cases (all scored >=0.023, nowhere near the
        # exact-0.00 score of the "Dark Corridor" incident this guards
        # against) while protecting against a non-person artifact that
        # happens to sit alone on its own line and repeat often but that
        # the classifier correctly scores near zero.
        if (
            candidate.state.feature_vector is not None
            and candidate.state.feature_vector.is_isolated_line
            and ml_prob >= ML_WEAK_DICTIONARY_VETO_FLOOR
            and not self._is_common_word_phrase(candidate, knowledge_base)
        ):
            repeat_count = count_occurrences(candidate.candidate.normalized_text, document_text)
            if repeat_count >= MIN_REPETITION_FOR_ML_ONLY_CORROBORATION:
                return True

        if self.allow_spacy_only_corroboration:
            return any(d.detector == DetectorName.SPACY for d in detections)
        return False

    @staticmethod
    def _is_common_word_phrase(candidate: CandidateResult, knowledge_base: KnowledgeBase | None) -> bool:
        """True only if EVERY token in the candidate's text is ordinary
        English vocabulary (assets/common_words/common_english_words.txt)
        - i.e. the whole span reads as plain English, not a name-shaped
        span. Deliberately ALL-tokens, not ANY: a real name is safe as
        long as even one token is unusual/foreign/proper-noun-shaped (or
        has any real dictionary/title support, which already short-
        circuits this check higher up in _has_knowledge_corroboration).
        knowledge_base is optional only for defensive callers that don't
        have one on hand - treated as "not a common-word phrase" (i.e.
        this check is skipped, not fail-closed) so a missing knowledge
        base degrades to the pre-2026-09-16 behavior rather than
        blocking every ML-confidence corroboration outright."""
        if knowledge_base is None:
            return False
        tokens = candidate.candidate.normalized_text.split()
        if not tokens:
            return False
        return all(knowledge_base.is_common_word(t.rstrip(".'’")) for t in tokens)
