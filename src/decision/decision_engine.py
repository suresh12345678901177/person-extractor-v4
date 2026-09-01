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
"""

from __future__ import annotations

from src.core.models import CandidateResult, Decision, DetectorName
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

    def decide(self, candidate: CandidateResult, document_text: str) -> CandidateResult:
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

        has_knowledge_corroboration = self._has_knowledge_corroboration(candidate, ml_prob, document_text)

        if state.final_confidence >= ACCEPT_THRESHOLD and has_knowledge_corroboration:
            state.decision = Decision.ACCEPTED
        elif state.final_confidence >= REVIEW_THRESHOLD:
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

    def _has_knowledge_corroboration(self, candidate: CandidateResult, ml_prob: float, document_text: str) -> bool:
        detections = candidate.candidate.source_detections
        has_window_dictionary_evidence = any(d.metadata.get("pattern") == "dictionary" for d in detections)
        has_single_token_dictionary_evidence = any(
            d.metadata.get("pattern") == "dictionary_single_token" for d in detections
        )
        has_title_evidence = any(d.metadata.get("pattern") == "titled" for d in detections)
        has_confident_ml = ml_prob >= ML_HIGH_CONFIDENCE_OVERRIDE

        if has_window_dictionary_evidence or has_title_evidence:
            return True

        if has_confident_ml:
            # High ML confidence alone (zero dictionary/title evidence)
            # additionally requires real document-wide repetition - see
            # MIN_REPETITION_FOR_ML_ONLY_CORROBORATION's docstring above
            # for the real exploit this closes and the real regression
            # a broader fix already caused when measured.
            repeat_count = count_occurrences(candidate.candidate.normalized_text, document_text)
            if repeat_count >= MIN_REPETITION_FOR_ML_ONLY_CORROBORATION:
                return True

        if has_single_token_dictionary_evidence and ml_prob >= ML_WEAK_DICTIONARY_VETO_FLOOR:
            return True

        if self.allow_spacy_only_corroboration:
            return any(d.detector == DetectorName.SPACY for d in detections)
        return False
