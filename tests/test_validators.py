"""Unit tests for the 15-validator Validation Firewall."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR
from src.core.models import Candidate, CandidateResult, Detection, DetectorName
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.validators.blacklist_validator import BlacklistValidator
from src.validation.validators.campaign_validator import CampaignValidator
from src.validation.validators.context_validator import ContextValidator
from src.validation.validators.grammar_validator import GrammarValidator
from src.validation.validators.initial_validator import InitialValidator
from src.validation.validators.language_validator import LanguageValidator
from src.validation.validators.length_validator import LengthValidator
from src.validation.validators.location_validator import LocationValidator
from src.validation.validators.organization_validator import OrganizationValidator
from src.validation.validators.punctuation_validator import PunctuationValidator
from src.validation.validators.repetition_validator import RepetitionValidator
from src.validation.validators.stopword_validator import StopwordValidator
from src.validation.validators.structure_validator import StructureValidator
from src.validation.validators.system_log_validator import SystemLogValidator
from src.validation.validators.title_validator import TitleValidator

KB = KnowledgeBase.load(BASE_DIR / "assets")


def _candidate(text: str, pattern: str = "bare") -> CandidateResult:
    detection = Detection(
        text=text, start=0, end=len(text), page_index=0,
        detector=DetectorName.REGEX, confidence=0.9, metadata={"pattern": pattern},
    )
    candidate = Candidate.new(text, text, 0, len(text), 0, (detection,))
    return CandidateResult(candidate=candidate)


def _validate(validator, text: str, document_text: str | None = None, pattern: str = "bare"):
    cand = _candidate(text, pattern=pattern)
    return validator.validate(cand, KB, document_text if document_text is not None else text)


def test_length_validator_rejects_too_short():
    assert _validate(LengthValidator(), "Al").passed is False


def test_length_validator_accepts_normal_name():
    assert _validate(LengthValidator(), "John Smith").passed is True


def test_structure_validator_rejects_lowercase_token():
    assert _validate(StructureValidator(), "john smith").passed is False


def test_structure_validator_rejects_acronym():
    assert _validate(StructureValidator(), "CCTV Footage").passed is False


def test_structure_validator_rejects_hyphenated_acronym_fragment():
    assert _validate(StructureValidator(), "IDs TXN").passed is False


def test_punctuation_validator_rejects_digits():
    assert _validate(PunctuationValidator(), "John5 Smith").passed is False


def test_repetition_validator_rejects_repeated_tokens():
    assert _validate(RepetitionValidator(), "John John").passed is False


def test_initial_validator_rejects_all_initials():
    assert _validate(InitialValidator(), "J. R.").passed is False


def test_initial_validator_accepts_initial_plus_full_token():
    assert _validate(InitialValidator(), "J. Smith").passed is True


def test_blacklist_validator_rejects_known_entry():
    assert _validate(BlacklistValidator(), "New York").passed is False


def test_blacklist_validator_rejects_phrase_with_leading_article():
    result = _validate(BlacklistValidator(), "The United Nations")
    assert result.passed is False


def test_blacklist_validator_accepts_normal_name():
    assert _validate(BlacklistValidator(), "John Smith").passed is True


def test_campaign_validator_rejects_known_slogan():
    assert _validate(CampaignValidator(), "Jai Telangana").passed is False


def test_campaign_validator_rejects_novel_slogan_via_marker_word():
    assert _validate(CampaignValidator(), "Bharat Mata Ki Jai").passed is False


def test_campaign_validator_accepts_normal_name():
    assert _validate(CampaignValidator(), "Suresh Kumar").passed is True


def test_organization_validator_rejects_business_name():
    assert _validate(OrganizationValidator(), "Sri Lakshmi Ganesh Traders").passed is False


def test_organization_validator_accepts_normal_name():
    assert _validate(OrganizationValidator(), "Suresh Kumar").passed is True


def test_location_validator_rejects_known_city():
    assert _validate(LocationValidator(), "Hyderabad").passed is False


def test_location_validator_accepts_normal_name():
    assert _validate(LocationValidator(), "Suresh Kumar").passed is True


def test_stopword_validator_rejects_any_stopword_token():
    assert _validate(StopwordValidator(), "The CCTV").passed is False


def test_language_validator_rejects_non_latin_script():
    assert _validate(LanguageValidator(), "\u0930\u093e \u0905\u0928\u094d\u0928\u093e").passed is False


def test_language_validator_accepts_normal_name():
    assert _validate(LanguageValidator(), "Suresh Kumar").passed is True


def test_title_validator_boosts_titled_pattern():
    result = _validate(TitleValidator(), "Suresh Kumar", pattern="titled")
    assert result.score > 0


def test_grammar_validator_rejects_sentence_initial_unsupported_single_token():
    doc = "The meeting ended. Nonsense was reported by nobody."
    cand = _candidate("Nonsense")
    # override candidate span to point at the real sentence-initial position
    from dataclasses import replace
    start = doc.index("Nonsense")
    cand.candidate = replace(cand.candidate, start=start, end=start + len("Nonsense"), source_detections=())
    result = GrammarValidator().validate(cand, KB, doc)
    assert result.passed is False


def test_grammar_validator_exempts_dictionary_confirmed_single_token():
    doc = "Suresh claimed he heard shouting outside."
    detection = Detection(text="Suresh", start=0, end=6, page_index=0,
                           detector=DetectorName.DICTIONARY, confidence=0.4,
                           metadata={"pattern": "dictionary_single_token"})
    candidate = Candidate.new("Suresh", "Suresh", 0, 6, 0, (detection,))
    cand = CandidateResult(candidate=candidate)
    result = GrammarValidator().validate(cand, KB, doc)
    assert result.passed is True


def test_context_validator_rejects_url_adjacency():
    doc = "Visit https://Example.com for details."
    start = doc.index("Example")
    cand = _candidate("Example")
    from dataclasses import replace
    cand.candidate = replace(cand.candidate, start=start, end=start + len("Example"))
    result = ContextValidator().validate(cand, KB, doc)
    assert result.passed is False


def test_context_validator_boosts_occupation_precedence():
    doc = "The engineer Suresh reviewed the design."
    start = doc.index("Suresh")
    cand = _candidate("Suresh")
    from dataclasses import replace
    cand.candidate = replace(cand.candidate, start=start, end=start + len("Suresh"))
    result = ContextValidator().validate(cand, KB, doc)
    assert result.passed is True and result.score > 0


def test_system_log_validator_rejects_dumpsys_package_identifier_line():
    # Regression test for a real production false positive (2026-09-17):
    # after the global name-dictionary expansion, "READ" (also a genuine
    # entry in last_names.txt in some locale) got auto-ACCEPTED 1,068
    # times from a real dumpsys export because its line
    # ("requiredPermission=com.samsung.cmh.data.READ") has no logcat
    # timestamp prefix to key off, but does contain a reverse-DNS-style
    # Android package identifier.
    doc = "      requiredPermission=com.samsung.cmh.data.READ\n"
    start = doc.index("READ")
    from dataclasses import replace
    cand = _candidate("READ")
    cand.candidate = replace(cand.candidate, start=start, end=start + len("READ"))
    result = SystemLogValidator().validate(cand, KB, doc)
    assert result.passed is False


def test_system_log_validator_rejects_android_permission_constant_line():
    # The 4-segment package-identifier check alone missed this: standard
    # Android permission constants ("android.permission.READ_CALENDAR")
    # are only 3 segments, and were the single largest real contributor
    # to the "READ" false positive (1,068 raw occurrences in one real
    # dumpsys export) - caught here via the ALL-CAPS-WITH-UNDERSCORES
    # last-segment convention instead of segment count.
    doc = "      requiredPermission=android.permission.READ_CALENDAR\n"
    start = doc.index("READ_CALENDAR")
    from dataclasses import replace
    cand = _candidate("READ")
    cand.candidate = replace(cand.candidate, start=start, end=start + len("READ"))
    result = SystemLogValidator().validate(cand, KB, doc)
    assert result.passed is False


def test_system_log_validator_rejects_uid_pkg_table_row():
    # A dumpsys UID/package/policy table row can have a package name as
    # short as 2 segments ("com.truecaller"), structurally identical to
    # a plain domain, so the package-identifier regex alone can't safely
    # catch it. The "-Uid <n>-Pkg " line prefix is the safe signal.
    doc = "-Uid    10294-Pkg com.truecaller-POL (8)\n"
    start = doc.index("POL")
    from dataclasses import replace
    cand = _candidate("POL")
    cand.candidate = replace(cand.candidate, start=start, end=start + len("POL"))
    result = SystemLogValidator().validate(cand, KB, doc)
    assert result.passed is False


def test_system_log_validator_rejects_package_identifier_deep_in_long_line():
    # Regression test for a real production false positive (2026-09-17):
    # a dumpsys ANR report concatenates an entire call stack onto one very
    # long line. "Handler" (also a genuine last-name entry after the
    # dictionary expansion) sat ~350 chars into its line, past the
    # original line-START-anchored 300-char scan window, even though it's
    # literally part of "android.os.Handler.handleCallback" right next to
    # it. Measured: 83 real accepted occurrences before this fix. The
    # window must be centered on the candidate's own position, not the
    # line start, to catch this.
    padding = "x" * 340
    doc = f"{padding} android.os.Handler.handleCallback:958\n"
    start = doc.index("Handler")
    from dataclasses import replace
    cand = _candidate("Handler")
    cand.candidate = replace(cand.candidate, start=start, end=start + len("Handler"))
    result = SystemLogValidator().validate(cand, KB, doc)
    assert result.passed is False


def test_system_log_validator_accepts_url_on_same_line_as_real_name():
    # Control case: a genuine name sharing a line with an ordinary
    # 2-3-segment URL/domain (realistic in chat/email content) must NOT
    # be rejected - only 4+-segment reverse-DNS-style identifiers should
    # trip this, not everyday domains.
    doc = "Contact Suresh at www.example.com for details.\n"
    start = doc.index("Suresh")
    from dataclasses import replace
    cand = _candidate("Suresh")
    cand.candidate = replace(cand.candidate, start=start, end=start + len("Suresh"))
    result = SystemLogValidator().validate(cand, KB, doc)
    assert result.passed is True


def test_system_log_validator_accepts_name_after_missing_space_domain_sentence():
    # Regression test for a real production false NEGATIVE (2026-09-23,
    # manual audit of real case's CASEID_chat.txt): a
    # WhatsApp message reading "...founder of Giftly.co.in. At GFT, we
    # create..." lost the space after the sentence-ending period during
    # export, producing "Giftly.co.in.At". The package-identifier regex
    # restarted its match at "co.in.At" - structurally identical to a
    # genuine "android.os.Handler" identifier - and hard-rejected "Rehan"
    # (a real self-identified sender name earlier on the same line) as
    # system-log noise. A genuine Android/Java identifier never starts
    # immediately after another dotted chain's "." - only after
    # whitespace/"="/line-start - so this must NOT trip the validator.
    doc = "Hi Team,I am Rehan , founder of Giftly.co.in.At GFT, we create corporate gifting experience.\n"
    start = doc.index("Rehan")
    from dataclasses import replace
    cand = _candidate("Rehan")
    cand.candidate = replace(cand.candidate, start=start, end=start + len("Rehan"))
    result = SystemLogValidator().validate(cand, KB, doc)
    assert result.passed is True


def _validate_on_line(name: str, doc: str):
    from dataclasses import replace
    start = doc.index(name)
    cand = _candidate(name)
    cand.candidate = replace(cand.candidate, start=start, end=start + len(name))
    return SystemLogValidator().validate(cand, KB, doc)


def test_system_log_validator_accepts_name_beside_four_label_hostname():
    # Regression test (2026-09-25): a 4+-label hostname - a country-code
    # suffix ("ac.in", "co.uk") or a subdomain - matched the 4-segment
    # package-identifier form, so a real name on the same line was
    # hard-rejected. Hostnames end in their TLD; packages start with one.
    assert _validate_on_line("Suresh Kumar", "Please contact Suresh Kumar at www.bvrit.ac.in for the forms.\n").passed
    assert _validate_on_line("Priya Raman", "From: Priya Raman <priya@cse.bvrit.ac.in>\n").passed
    assert _validate_on_line("Arjun Mehta", "Arjun Mehta (mail.company.co.uk) replied.\n").passed


def test_system_log_validator_still_rejects_lowercase_package_identifiers():
    # Control: all-lowercase 4+-segment identifiers are still packages when
    # they start with a reverse-DNS root, or don't end in a hostname suffix -
    # even when the last segment looks like a country code ("io").
    assert not _validate_on_line("Handler", "at org.apache.commons.io Handler\n").passed
    assert not _validate_on_line("Handler", "vendor.qti.hardware.radio.ims Handler\n").passed


def test_structure_validator_accepts_accented_latin_names():
    """Every accented name used to be hard-rejected here as a malformed
    token, even though RegexDetector found it correctly."""
    for name in ("José García", "François Dubois", "Björn Lindqvist", "Łukasz Kowalski",
                 "Ștefan Popescu", "Šimun Babić", "Nguyễn Văn"):
        assert _validate(StructureValidator(), name).passed, name


def test_structure_validator_still_rejects_non_latin_and_lowercase_start():
    assert not _validate(StructureValidator(), "élan Vital").passed
    assert not _validate(StructureValidator(), "Иван Петров").passed


def test_structure_validator_rejects_accented_acronyms_like_ascii_ones():
    """'ÑËÙ' used to pass because the acronym check only knew A-Z."""
    assert not _validate(StructureValidator(), "ÑËÙ").passed
    assert not _validate(StructureValidator(), "ÉCOLE").passed


def test_structure_validator_rejects_machine_identifiers_and_keeps_prefixed_names():
    """From a real Android case scan: mixed-case identifier fragments and
    CamelCase compounds were ACCEPTED as names."""
    for garbage in ("ToMs", "KeR", "LUt", "DeX", "DisplayType", "KeyCharacterMapFile"):
        assert not _validate(StructureValidator(), garbage).passed, garbage
    for name in ("McDonald", "MacArthur", "DeShawn", "LaToya", "JoAnn", "O'Brien", "Smith-Jones"):
        assert _validate(StructureValidator(), name).passed, name


def test_structure_validator_rejects_utf8_read_as_latin1():
    """'TomÃ¡s' is 'Tomás' misdecoded - the candidate 'TomÃ' is garbage."""
    document = "Contacts: TomÃ¡s and others"
    start = document.index("TomÃ")
    detection = Detection("TomÃ", start, start + 4, 0, DetectorName.REGEX, 0.55, {"pattern": "bare"})
    cand = CandidateResult(candidate=Candidate.new("TomÃ", "TomÃ", start, start + 4, 0, (detection,)))
    assert not StructureValidator().validate(cand, KB, document).passed
