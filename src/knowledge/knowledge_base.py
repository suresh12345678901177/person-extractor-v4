"""
src.knowledge.knowledge_base
==============================
Offline knowledge base. Loads reference lists (first names, last names,
stopwords, blacklist, titles, honorifics, organizations, locations,
occupations, campaigns, language markers) from plain-text files in
`assets/`. No list is ever hardcoded in code - everything is
configurable by editing a text file, per the blueprint's "Fully Offline"
and "Plugin Architecture" design principles.

File format: one entry per line, '#' starts a comment, blank lines
ignored. Lookups are case-insensitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.preprocessing.latin import fold_accents
from src.utils.logger import get_logger

logger = get_logger("knowledge.knowledge_base")


def _load_set(path: Path) -> frozenset[str]:
    if not path.exists():
        logger.warning("Knowledge file not found, treating as empty: %s", path)
        return frozenset()

    entries: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(line.lower())
    return frozenset(entries)


@dataclass(slots=True)
class KnowledgeBase:
    """Immutable, eagerly-loaded set of offline reference lists."""

    assets_dir: Path
    first_names: frozenset[str] = field(default_factory=frozenset)
    last_names: frozenset[str] = field(default_factory=frozenset)
    stopwords: frozenset[str] = field(default_factory=frozenset)
    blacklist: frozenset[str] = field(default_factory=frozenset)
    titles: frozenset[str] = field(default_factory=frozenset)
    honorifics: frozenset[str] = field(default_factory=frozenset)
    organizations: frozenset[str] = field(default_factory=frozenset)
    locations: frozenset[str] = field(default_factory=frozenset)
    occupations: frozenset[str] = field(default_factory=frozenset)
    campaigns: frozenset[str] = field(default_factory=frozenset)
    non_english_markers: frozenset[str] = field(default_factory=frozenset)
    ambiguous_first_names: frozenset[str] = field(default_factory=frozenset)
    common_words: frozenset[str] = field(default_factory=frozenset)
    calendar_words: frozenset[str] = field(default_factory=frozenset)
    language_region_names: frozenset[str] = field(default_factory=frozenset)
    # Accent-folded copies of the name lists ("stefan", "nguyen"), used
    # ONLY as a fallback for tokens containing non-ASCII letters - see
    # is_known_first_name(). Built in load().
    first_names_folded: frozenset[str] = field(default_factory=frozenset)
    last_names_folded: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def load(cls, assets_dir: str | Path) -> "KnowledgeBase":
        assets_dir = Path(assets_dir)
        kb = cls(
            assets_dir=assets_dir,
            first_names=_load_set(assets_dir / "first_names" / "first_names.txt"),
            last_names=_load_set(assets_dir / "last_names" / "last_names.txt"),
            stopwords=_load_set(assets_dir / "stopwords" / "stopwords.txt"),
            blacklist=_load_set(assets_dir / "blacklist" / "blacklist.txt"),
            titles=_load_set(assets_dir / "titles" / "titles.txt"),
            honorifics=_load_set(assets_dir / "honorifics" / "honorifics.txt"),
            organizations=_load_set(assets_dir / "organizations" / "organizations.txt"),
            locations=_load_set(assets_dir / "locations" / "locations.txt"),
            occupations=_load_set(assets_dir / "occupations" / "occupations.txt"),
            campaigns=_load_set(assets_dir / "campaigns" / "campaigns.txt"),
            non_english_markers=_load_set(assets_dir / "languages" / "non_english_markers.txt"),
            ambiguous_first_names=_load_set(assets_dir / "ambiguous_words" / "ambiguous_first_names.txt"),
            common_words=_load_set(assets_dir / "common_words" / "common_english_words.txt"),
            calendar_words=_load_set(assets_dir / "common_words" / "multilingual_calendar_words.txt"),
            language_region_names=_load_set(assets_dir / "languages" / "language_and_region_names.txt"),
        )
        kb.first_names_folded = frozenset(fold_accents(n) for n in kb.first_names)
        kb.last_names_folded = frozenset(fold_accents(n) for n in kb.last_names)
        logger.info(
            "KnowledgeBase loaded: %d first names, %d last names, %d organizations, "
            "%d locations, %d campaigns, %d blacklist entries",
            len(kb.first_names), len(kb.last_names), len(kb.organizations),
            len(kb.locations), len(kb.campaigns), len(kb.blacklist),
        )
        return kb

    def is_known_first_name(self, token: str) -> bool:
        key = token.lower()
        if key in self.first_names:
            return True
        # Accent-variant fallback ("Ștefan" vs listed "ştefan"/"stefan",
        # "Nguyễn" vs "nguyen") - ONLY for tokens that themselves contain
        # non-ASCII letters, so plain-ASCII text is looked up exactly as
        # before. The reverse (ASCII "Francois" matching listed
        # "françois") was measured and deliberately NOT added: it would
        # add 7,416 new ASCII keys, 43 of them ordinary English words
        # ("back", "come", "lower", "magic", ...) - a direct false-
        # positive source.
        return not key.isascii() and fold_accents(key) in self.first_names_folded

    def is_known_last_name(self, token: str) -> bool:
        key = token.lower()
        if key in self.last_names:
            return True
        return not key.isascii() and fold_accents(key) in self.last_names_folded

    def is_stopword(self, token: str) -> bool:
        return token.lower() in self.stopwords

    def is_blacklisted(self, text: str) -> bool:
        return text.lower() in self.blacklist

    def is_title(self, token: str) -> bool:
        return token.lower().rstrip(".") in self.titles

    def is_honorific(self, token: str) -> bool:
        return token.lower().rstrip(".") in self.honorifics

    def is_organization(self, text: str) -> bool:
        return text.lower() in self.organizations

    def is_location(self, text: str) -> bool:
        return text.lower() in self.locations

    def is_occupation(self, token: str) -> bool:
        return token.lower() in self.occupations

    def is_ambiguous_first_name(self, token: str) -> bool:
        return token.lower() in self.ambiguous_first_names

    def is_common_word(self, token: str) -> bool:
        return token.lower() in self.common_words

    def is_calendar_word(self, token: str) -> bool:
        return token.lower() in self.calendar_words

    def is_language_or_region_name(self, token: str) -> bool:
        """'Español', 'Telugu', 'Schweiz' - see
        assets/languages/language_and_region_names.txt."""
        return token.lower() in self.language_region_names
