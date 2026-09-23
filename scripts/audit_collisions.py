"""
scripts/audit_collisions.py
==============================
One-off audit: finds tokens that appear in BOTH assets/first_names.txt or
assets/last_names.txt AND a "non-person" list (titles, honorifics,
occupations, organizations, locations, stopwords, blacklist, campaigns,
common_words).

common_words was added 2026-09-18 after a real case scan found "Max",
"June", "Server", "Sales", "Read", "Key", "Asia", "Binder" and others all
had a genuine first/last-name dictionary hit AND were ordinary English
vocabulary, but were never checked here (common_words is a soft ML-guard
list, not one of the original hard-reject lists this script was built
against) - so none of them were in ambiguous_first_names.txt and a bare
single-token occurrence of any of them corroborated on the dictionary hit
alone, the exact same failure mode as the original "Major" bug, just
against a list this script didn't examine yet.

Why this matters: a single-token dictionary hit against first_names.txt
or last_names.txt is treated by CorroborationValidator/DecisionEngine as
"knowledge corroboration" strong enough to auto-accept a bare candidate
(see src/decision/decision_engine.py's ML_WEAK_DICTIONARY_VETO_FLOOR
path). If that same token is ALSO a rank/title/occupation/place/org
word, a bare non-name occurrence of it (e.g. "Major" in an Android
dumpsys dump) can slide through as a false positive purely because it
happens to also be a rare real first/last name. Real example found via
production use: "Major" is in both first_names.txt and titles.txt and
was wrongly ACCEPTED 63 times in a real case scan.

This script only reports; it does not edit any asset file automatically
- collisions need a human judgment call (some, like "Major", clearly
belong in assets/ambiguous_words/ambiguous_first_names.txt; others may
be false alarms depending on list intent).

Usage:
    python scripts/audit_collisions.py
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS = BASE_DIR / "assets"


def load(path: Path) -> set[str]:
    entries: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(line.lower())
    return entries


def report(name_a: str, set_a: set[str], name_b: str, set_b: set[str], ambiguous: set[str]) -> None:
    overlap = set_a & set_b
    if not overlap:
        return
    mitigated = sorted(w for w in overlap if w in ambiguous)
    unmitigated = sorted(w for w in overlap if w not in ambiguous)
    print(f"{name_a} x {name_b} ({len(overlap)}):")
    if unmitigated:
        print(f"  NOT YET in ambiguous_first_names.txt: {unmitigated}")
    if mitigated:
        print(f"  already gated via ambiguous_first_names.txt: {mitigated}")


def main() -> None:
    first = load(ASSETS / "first_names" / "first_names.txt")
    last = load(ASSETS / "last_names" / "last_names.txt")
    titles = load(ASSETS / "titles" / "titles.txt")
    honorifics = load(ASSETS / "honorifics" / "honorifics.txt")
    occupations = load(ASSETS / "occupations" / "occupations.txt")
    organizations = load(ASSETS / "organizations" / "organizations.txt")
    locations = load(ASSETS / "locations" / "locations.txt")
    stopwords = load(ASSETS / "stopwords" / "stopwords.txt")
    blacklist = load(ASSETS / "blacklist" / "blacklist.txt")
    campaigns = load(ASSETS / "campaigns" / "campaigns.txt")
    common_words = load(ASSETS / "common_words" / "common_english_words.txt")
    ambiguous = load(ASSETS / "ambiguous_words" / "ambiguous_first_names.txt")

    # NOTE: "already gated" only protects a BARE single-token occurrence
    # from auto-corroborating (see corroboration_validator.py) - it does
    # NOT protect against LocationValidator/OrganizationValidator/etc.
    # hard-rejecting a genuine name that happens to also be a known
    # place/org string. Overlaps with locations/organizations/blacklist
    # are a DIFFERENT failure direction (false reject, not false accept)
    # and this list can't fix those - flag them for a human look instead.
    print("=== FIRST NAMES collisions (single-token dictionary hit risk) ===")
    report("first_names", first, "titles", titles, ambiguous)
    report("first_names", first, "honorifics", honorifics, ambiguous)
    report("first_names", first, "occupations", occupations, ambiguous)
    report("first_names", first, "organizations", organizations, ambiguous)
    report("first_names", first, "locations", locations, ambiguous)
    report("first_names", first, "stopwords", stopwords, ambiguous)
    report("first_names", first, "blacklist", blacklist, ambiguous)
    report("first_names", first, "campaigns", campaigns, ambiguous)
    report("first_names", first, "common_words", common_words, ambiguous)

    print()
    print("=== LAST NAMES collisions ===")
    report("last_names", last, "titles", titles, ambiguous)
    report("last_names", last, "honorifics", honorifics, ambiguous)
    report("last_names", last, "occupations", occupations, ambiguous)
    report("last_names", last, "organizations", organizations, ambiguous)
    report("last_names", last, "locations", locations, ambiguous)
    report("last_names", last, "stopwords", stopwords, ambiguous)
    report("last_names", last, "blacklist", blacklist, ambiguous)
    report("last_names", last, "campaigns", campaigns, ambiguous)
    report("last_names", last, "common_words", common_words, ambiguous)


if __name__ == "__main__":
    main()
