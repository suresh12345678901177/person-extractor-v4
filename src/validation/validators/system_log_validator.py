"""
src.validation.validators.system_log_validator
==================================================
Hard-rejects candidates sitting on an Android/system logcat-format
line (timestamp + PID/TID + log level + tag: message) - this is
always machine-generated diagnostic output, never narrative text, so
a repeated internal component/library/network name in these lines
(e.g. "OneDS Library", "Aircel North") satisfies the repetition-based
corroboration signal (see CorroborationValidator) exactly as well as a
real recurring person name would, producing confident, high-occurrence
false positives.

Confirmed directly on a real ProDiscover case export: Android app/
telemetry logs ("INFO/App: Starting initialization of OneDS Library...")
and radio/connectivity logs ("E/CscConnection: ... [nwkname : Aircel
North] ...") both repeat internal component names hundreds of times
per file, each occurrence individually indistinguishable from a real
title-case name by shape alone.

Checked only against a short, bounded window at the start of the
candidate's own line (never the whole document) - an unbounded scan
here would repeat the exact class of performance bug already found and
fixed in grammar_validator.py/segmenter.py on these same large log
files.

Second pattern, added 2026-09-17 after a real production regression: the
2026-09-16/17 global name-dictionary expansion (see README) gave common
Android/Java internals - "READ", "HANDLER", "KNOX", "WILD", "POL" - a
dictionary hit for the first time (they're also, coincidentally, real
personal names/surnames in some locale), and `dumpsys`-style structured
dumps repeat each of these hundreds to a thousand-plus times (measured:
"READ" 1,068 times, "HANDLER" 274, "KNOX" 66 in one real case export),
easily clearing CorroborationValidator's repetition bar. Unlike the
logcat format above, `dumpsys` output has no timestamp/log-level prefix
to key off - but it does reliably contain reverse-DNS-style Android/Java
package or class identifiers ("com.samsung.cmh.data.READ",
"android.os.Handler", "com.android.server.am.ActivityManagerService") on
the same line as the noise token, confirmed directly against this
case's real dumpsys/pm_debug_info exports. Requiring 4+ dot-separated
segments (not 2-3) deliberately keeps this from firing on an ordinary
URL or email domain that might legitimately share a line with a real
name in chat/email content (e.g. "www.example.com" is only 3 segments);
longer hostnames ("www.bvrit.ac.in") are recognized and exempted - see
_is_hostname. This check scans the candidate's own full line (bounded, same
performance rationale as the logcat check above), not just a fixed
prefix, since the identifier can appear anywhere on a `dumpsys`
attribute line.
"""

from __future__ import annotations

import re

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator

# Matches the line-start prefix of the Android/system log formats
# observed in real casework:
#   2024-09-25 22:40:47,136 INFO/App: ...
#   2023-12-02 17:17:38.677+05:30 E/CscConnection: ...
#   11-28 10:30:15.477     0  6531   104 V CpEventLog(u0): ...
#   	2024-09-25 22:40:47	136 INFO/App: ...   (tab-separated ms variant -
#     same underlying log, seen re-exported/re-carved this way elsewhere
#     in the same case; the [.,\t ]+ separator class covers it)
# date (YYYY-MM-DD or MM-DD) + time with fractional seconds (+
# optional timezone) + zero or more numeric fields (PID/TID/UID) + a
# log level (single letter or full word) + optional /-or-space-
# separated tag + ':'. Deliberately narrow: real chat/email export
# headers ("Timestamp: 26-11-2024 15:32:10(UTC+0)") don't match this
# shape (label comes first, no log-level token) - verified directly
# against real WhatsApp export header lines during development.
_LOG_LINE_RE = re.compile(
    r"^\s*(?:\d{4}-\d{2}-\d{2}|\d{2}-\d{2})[ T]\d{2}:\d{2}:\d{2}[.,\t ]+\d+"
    r"(?:[+-]\d{2}:?\d{2})?\s+(?:\d+\s+)*"
    r"(?:[VDIWEF]|VERBOSE|DEBUG|INFO|WARN|ERROR|FATAL)(?:[/\s]\S+)?:"
)
_LINE_PREFIX_WINDOW = 100

# A reverse-DNS-style Android/Java package or class identifier. Two
# forms, both verified directly against this case's real dumpsys export:
#  1. 4+ total dot-separated segments, e.g.
#     "com.samsung.cmh.data.READ" or
#     "com.android.server.am.ActivityManagerService". The 4-segment
#     floor (not 2 or 3) is deliberate: a simple URL/domain a real chat
#     or email might legitimately share a line with a genuine name
#     ("www.example.com", "mail.company.com") is only 2-3 segments and
#     must NOT trip this.
#  2. The standard Android CONSTANT_NAME convention - 3+ segments where
#     the LAST is ALL-CAPS-WITH-UNDERSCORES, e.g.
#     "android.permission.READ_CALENDAR" or
#     "com.samsung.knox.KNOX_MMS_CONTROL". This is only 3 segments (below
#     the (1) floor) but was the single largest source of a real 2026-
#     09-17 false-positive ("READ", 1,068 raw occurrences - almost all
#     "android.permission.READ_*" constants) that (1) alone missed - a
#     real domain/URL can never contain an ALL-CAPS-WITH-UNDERSCORES
#     segment, so this stays safe against the same false-trigger risk.
#  3. Any 2+-segment lowercase dotted identifier that contains an
#     underscore ANYWHERE, e.g. "com.example.first_responder" (an
#     Android package name - real Java/Android package/class segments
#     routinely use snake_case; a real DNS hostname can never contain an
#     underscore at all, so this is also safe against the URL/domain
#     false-trigger risk (1) and (2) were designed to avoid).
#  4. Added 2026-09-18 after a real production regression (this case's
#     `dumpsys_ANR_WindowManager.txt`, 10.3MB): a 3-segment package +
#     PascalCase-class reference, e.g. "android.os.Handler" or
#     "android.os.Binder@10fc77e" or "com.android.server.wm.
#     DisplayContent$RemoteInsetsControlTarget" - structurally identical
#     to (1) except the final segment is a normal PascalCase class name,
#     not ALL-CAPS-WITH-UNDERSCORES. "Handler" and "Binder" (both real,
#     if unusual, last names) were repeating 60+ times each in exactly
#     this shape and neither existing pattern covered it: (1) requires
#     an ALL-CAPS final segment, (2) requires 4+ total segments, (3)
#     requires an underscore. A lowercase-first-segment, PascalCase-
#     final-segment dotted identifier is exactly as safe against the
#     URL/domain false-trigger risk as (1) - real prose/chat/email
#     content never writes a dotted phrase this way.
#  5. Added 2026-09-23 after a real production false-NEGATIVE (a
#     real WhatsApp chat export): a missing space after a sentence-
#     ending period ("...founder of Giftly.co.in.At GFT...", names changed) let the
#     regex restart its match mid-domain, at "co.in.At" - structurally
#     identical to (4)'s "android.os.Handler" shape (lowercase start,
#     dotted middle segment, PascalCase final segment) purely because
#     "co"/"in" happen to look like package segments. A genuine Android/
#     Java identifier is never embedded like this - it starts at its own
#     word boundary (after whitespace, "=", a line start, etc.), never
#     immediately after another dotted chain's "." - so requiring the
#     match not be preceded by "." closes this hole without affecting any
#     of the confirmed real cases above (all of which start after
#     whitespace or "="). Confirmed directly against this case's real
#     WhatsApp export text and all four existing regression tests below.
_PACKAGE_IDENTIFIER_RE = re.compile(
    r"\b(?<!\.)[a-z][a-z0-9_]*(?:\.[a-z0-9_]+){1,}\.[A-Z][A-Z0-9_]*\b"
    r"|\b(?<!\.)[a-z][a-z0-9_]*(?:\.[A-Za-z0-9_]+){3,}\b"
    r"|\b(?<!\.)(?=[a-z0-9_.]*_)[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+\b"
    r"|\b(?<!\.)[a-z][a-z0-9_]*(?:\.[a-z0-9_]+){1,}\.[A-Z][a-zA-Z0-9_]*\b"
)
_LINE_SCAN_WINDOW = 300

# Hostnames, added 2026-09-25: form (1) above - 4+ segments - also matches
# an ordinary hostname with a country-code suffix or a subdomain
# ("www.bvrit.ac.in", "priya@cse.bvrit.ac.in", "mail.company.co.uk"), so a
# real name sharing a line with one was hard-rejected without ever reaching
# REVIEW (verified: "Contact Suresh Kumar at www.bvrit.ac.in" -> rejected,
# the same sentence with "www.bvrit.com" -> accepted). A hostname ENDS in
# its TLD; a Java/Android package STARTS with one (reverse DNS). So an
# all-lowercase, underscore-free match is exempt only when it reads as a
# hostname and doesn't start like a package - every other identifier on the
# line is still checked.
_PACKAGE_ROOTS = frozenset({
    "com", "org", "net", "edu", "gov", "io", "android", "androidx",
    "java", "javax", "kotlin", "kotlinx", "dalvik", "sun",
})
_GENERIC_TLDS = frozenset({"com", "org", "net", "edu", "gov", "mil", "int", "info", "biz"})
# Second labels of two-level country suffixes: co.in, ac.in, gov.in, co.uk, com.au, ...
_SECOND_LEVEL_SUFFIX_LABELS = frozenset({
    "co", "ac", "gov", "org", "net", "edu", "com", "nic", "res", "mil", "ltd", "plc", "sch", "nhs",
})


def _is_hostname(match: re.Match, window: str) -> bool:
    identifier = match.group()
    if identifier != identifier.lower() or "_" in identifier:
        return False
    labels = identifier.split(".")
    if labels[0] in _PACKAGE_ROOTS or len(labels[0]) == 2:
        return False  # reverse-DNS roots, incl. country-code ones ("de.", "ro.")
    if labels[0] == "www" or window[:match.start()].endswith(("@", "://")):
        return True
    return labels[-1] in _GENERIC_TLDS or (
        len(labels[-1]) == 2 and labels[-1].isalpha() and labels[-2] in _SECOND_LEVEL_SUFFIX_LABELS
    )


# Radius (chars) scanned on EACH SIDE of the candidate itself for the
# package-identifier check, added 2026-09-17 after a real regression: a
# `dumpsys` ANR report can concatenate an entire call stack onto one very
# long line ("...ExternalSyntheticLambda0.run:2 android.os.Handler.
# handleCallback:958 android.os.Handler.dispatchMessage:99 ..."), so a
# candidate token ("Handler") can sit 300+ chars into its own line - past
# a line-START-anchored window - even though the package identifier it's
# literally part of ("android.os.Handler.handleCallback") is right next
# to it. Confirmed directly: "Handler" was still being accepted 83 times
# after the original _LINE_SCAN_WINDOW fix because of exactly this.
# Centering the window on the candidate's own position instead keeps the
# same bounded-cost guarantee (still a small constant per candidate) while
# actually covering the text immediately around it, regardless of how far
# into a long line that text sits.
_LINE_SCAN_RADIUS = 150

# dumpsys's per-app UID/package/policy table row format, e.g.:
#   -Uid    10294-Pkg com.truecaller-POL (8)
#   -Uid    10306-Pkg com.google.android.apps.docs-POL (8)
# Package names here can be as short as 2 segments ("com.truecaller"),
# which (1)-(3) above deliberately can't safely catch (structurally
# identical to a plain domain). The "-Uid <digits>-Pkg " line prefix
# itself is the safe signal instead - real narrative text never starts a
# line this way. Narrower and more specific than the general package-
# identifier check by design, so it only fires on this exact table shape.
_UID_PKG_TABLE_ROW_RE = re.compile(r"^-Uid\s+\d+-Pkg\s")


class SystemLogValidator(BaseValidator):
    name = "system_log"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        start = candidate.candidate.start
        end = candidate.candidate.end
        line_start = document_text.rfind("\n", 0, start) + 1
        prefix = document_text[line_start:line_start + _LINE_PREFIX_WINDOW]
        if _LOG_LINE_RE.match(prefix):
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message="Sits on an Android/system log line (timestamp + log level + tag) - "
                        "machine-generated diagnostic output, not narrative text",
            )

        true_line_end = document_text.find("\n", start)
        if true_line_end == -1:
            true_line_end = len(document_text)

        # UID/Pkg table rows are always short and line-initial, so this
        # check stays anchored to the line's own start.
        table_row_end = min(true_line_end, line_start + _LINE_SCAN_WINDOW)
        if _UID_PKG_TABLE_ROW_RE.match(document_text[line_start:table_row_end]):
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message="Sits on a dumpsys UID/package/policy table row "
                        "(\"-Uid <n>-Pkg ...\") - machine-generated diagnostic output, "
                        "not narrative text",
            )

        # Package-identifier scan is centered on the candidate itself (see
        # _LINE_SCAN_RADIUS above), not the line start, so it still finds
        # an identifier the candidate is part of even on a very long line.
        window_start = max(line_start, start - _LINE_SCAN_RADIUS)
        window_end = min(true_line_end, end + _LINE_SCAN_RADIUS)
        window = document_text[window_start:window_end]
        if any(not _is_hostname(m, window) for m in _PACKAGE_IDENTIFIER_RE.finditer(window)):
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message="Sits on a line containing an Android/Java reverse-DNS package or "
                        "class identifier (e.g. \"com.samsung.cmh.data.READ\") - "
                        "machine-generated dumpsys/stack-trace output, not narrative text",
            )

        return ValidationResult(self.name, passed=True, score=0.0, message="Not a system log line")
