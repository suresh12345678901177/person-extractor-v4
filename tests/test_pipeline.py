"""End-to-end integration tests for the full V4 pipeline, run against
real TXT files on disk - proves every stage wires together correctly."""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR, DEFAULT_CONFIG


def _test_config():
    """A config copy with feedback logging disabled, so tests never
    write into the real project's datasets/feedback/ directory."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["feedback"]["enabled"] = False
    return cfg
from src.core.models import Decision
from src.export.csv_exporter import CsvExporter
from src.export.json_exporter import JsonExporter
from src.export.report_exporter import ReportExporter
from src.pipeline.orchestrator import Pipeline

SAMPLE_TEXT = (
    "Dr. Suresh Kumar met with Mrs. Jennifer Wilson yesterday in New York.\n"
    "The United Nations released a report. John Smith was also present.\n"
    "The meeting was long. Xzqlt Vbnmp did not attend.\n"
    "\"Jai Telangana! Vande Mataram!\" played from a nearby loudspeaker.\n"
)


def _write_sample(tmp_path: Path) -> Path:
    sample = tmp_path / "sample.txt"
    sample.write_text(SAMPLE_TEXT, encoding="utf-8")
    return sample


def test_pipeline_end_to_end(tmp_path):
    sample_path = _write_sample(tmp_path)
    pipeline = Pipeline(config=_test_config(), base_dir=BASE_DIR)
    result = pipeline.run(sample_path)

    assert result.success is True
    assert result.statistics.candidates_generated > 0
    assert result.statistics.total_seconds > 0
    assert len(result.statistics.stage_timings) >= 8  # every named stage recorded

    accepted_names = {p.display_text for p in result.persons}
    assert any("Suresh Kumar" in n for n in accepted_names)
    assert any("Jennifer Wilson" in n for n in accepted_names)

    # Known false-positive strings must never be accepted
    for p in result.persons:
        assert "New York" not in p.display_text
        assert "United Nations" not in p.display_text
        assert "Jai" not in p.display_text
        assert "Vande" not in p.display_text

    for person in result.persons:
        assert person.decision == Decision.ACCEPTED
        assert person.occurrence_count >= 1
        assert len(person.locations) == person.occurrence_count


def test_pipeline_repeated_mentions_are_aggregated_with_correct_count(tmp_path):
    text = "Suresh spoke first. Later, Suresh spoke again. Finally, Suresh left.\n"
    sample = tmp_path / "repeated.txt"
    sample.write_text(text, encoding="utf-8")

    pipeline = Pipeline(config=_test_config(), base_dir=BASE_DIR)
    result = pipeline.run(sample)

    all_people = list(result.persons) + list(result.review_persons)
    suresh_entries = [p for p in all_people if p.display_text == "Suresh"]
    assert len(suresh_entries) == 1, "Suresh should be aggregated into exactly one entry"
    assert suresh_entries[0].occurrence_count == 3
    assert len(suresh_entries[0].locations) == 3


def test_pipeline_rejects_fabricated_name_with_zero_dictionary_support(tmp_path):
    # Regression test for a real bug found via adversarial testing: a
    # fabricated two-word capitalized string with no dictionary/title
    # support was wrongly ACCEPTED (score 1.25) purely because bare
    # regex + spaCy agreed on its shape. The knowledge-corroboration
    # gate in DecisionEngine must prevent this - shape agreement alone
    # is capped at REVIEW, never ACCEPTED.
    text = "The meeting was long. Xzqlt Vbnmp did not attend.\n"
    sample = tmp_path / "fabricated.txt"
    sample.write_text(text, encoding="utf-8")

    pipeline = Pipeline(config=_test_config(), base_dir=BASE_DIR)
    result = pipeline.run(sample)

    accepted_names = {p.display_text for p in result.persons}
    assert "Xzqlt Vbnmp" not in accepted_names, (
        "Fabricated name with zero dictionary/title support must never be ACCEPTED"
    )


def test_pipeline_rejects_ambiguous_single_token_dictionary_word_without_repetition(tmp_path):
    # Regression test for a real production false positive (2026-09-16,
    # real case scan): "Major" is a genuine (if rare) entry
    # in assets/first_names/first_names.txt, but it is also an ordinary
    # English rank/role word. A bare, low-occurrence "Major" with no
    # other corroborating evidence (title, spaCy support beyond the
    # collision, or a full first+last dictionary window match) was
    # wrongly ACCEPTED 63 times across unrelated real files purely on the
    # strength of a single-token dictionary hit. It must now require the
    # same repetition-or-title corroboration any other ambiguous word
    # needs (see assets/ambiguous_words/ambiguous_first_names.txt and
    # CorroborationValidator's fourth refinement).
    text = (
        "The Major update was delayed until Friday.\n"
        "According to the report, the Major issue was resolved quickly.\n"
    )
    sample = tmp_path / "ambiguous_word.txt"
    sample.write_text(text, encoding="utf-8")

    pipeline = Pipeline(config=_test_config(), base_dir=BASE_DIR)
    result = pipeline.run(sample)

    accepted_names = {p.display_text for p in result.persons}
    assert "Major" not in accepted_names, (
        "A low-occurrence ambiguous word ('Major') with only a single-token "
        "dictionary hit must never be ACCEPTED without repetition/title corroboration"
    )


def test_pipeline_rejects_repeated_calendar_word_even_with_high_repetition(tmp_path):
    # Regression test for a real production false positive (2026-09-17,
    # real case scan): a single Android locale/calendar-
    # picker resource-dump file got "Jan"/"Sep" (and similar day/month
    # names in other languages) ACCEPTED 10x/8x each - they are genuine
    # dictionary first names in some locale, and unlike an ordinary
    # ambiguous word, heavy repetition in a resource file reflects the
    # file's fixed vocabulary, not a real recurring person. Calendar
    # words must be rejected even when repeated well past
    # CorroborationValidator.MIN_REPETITION_FOR_CORROBORATION (4).
    text = "Jan\n" * 10
    sample = tmp_path / "calendar_dump.txt"
    sample.write_text(text, encoding="utf-8")

    pipeline = Pipeline(config=_test_config(), base_dir=BASE_DIR)
    result = pipeline.run(sample)

    accepted_names = {p.display_text for p in result.persons}
    assert "Jan" not in accepted_names, (
        "A repeated calendar word ('Jan') must never be ACCEPTED on repetition alone"
    )


def test_pipeline_rejects_allcaps_acronym_matching_a_last_name(tmp_path):
    # Regression test for a real production false positive (2026-09-17,
    # real case scan, eula_12.txt): "LAW" is ALL-CAPS and
    # acronym-shaped, but "Law" is also a genuine (rare) entry in
    # last_names.txt, so StructureValidator's acronym-rejection escape
    # hatch let it through, and it was wrongly ACCEPTED from repeated
    # "GOVERNING LAW" EULA section headers (5 occurrences in one file
    # alone - the same class as "READ"/"HANDLER"/"KNOX"/"POL"/"FOA"/
    # "UUS"/"SHA"/"WILD" found earlier this session in dumpsys/log text).
    text = "GOVERNING LAW\n" * 5 + "This agreement is governed by LAW in this jurisdiction.\n"
    sample = tmp_path / "eula_style.txt"
    sample.write_text(text, encoding="utf-8")

    pipeline = Pipeline(config=_test_config(), base_dir=BASE_DIR)
    result = pipeline.run(sample)

    accepted_names = {p.display_text for p in result.persons}
    assert "LAW" not in accepted_names, (
        "An ALL-CAPS acronym-shaped word ('LAW') that only passes structure "
        "validation via a coincidental last-name match must never be ACCEPTED "
        "on repetition alone"
    )


def test_pipeline_missing_file_returns_structured_failure():
    pipeline = Pipeline(config=_test_config(), base_dir=BASE_DIR)
    result = pipeline.run("/nonexistent/path/does_not_exist.txt")
    assert result.success is False
    assert result.error is not None


def test_pipeline_output_is_exportable(tmp_path):
    sample_path = _write_sample(tmp_path)
    pipeline = Pipeline(config=_test_config(), base_dir=BASE_DIR)
    result = pipeline.run(sample_path)

    csv_path = CsvExporter().export(
        list(result.persons), list(result.review_persons), list(result.rejected),
        tmp_path / "out.csv", str(sample_path),
    )
    assert csv_path.exists() and csv_path.stat().st_size > 0

    json_path = JsonExporter().export(
        list(result.persons), list(result.review_persons), list(result.rejected),
        tmp_path / "out.json", str(sample_path),
    )
    assert json_path.exists() and json_path.stat().st_size > 0

    report_path = ReportExporter().export(result, tmp_path / "report.txt")
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "TIME TAKEN" in report_text
    assert "OCCURRENCES" in report_text
    assert "LOCATIONS" in report_text


def test_pipeline_accepts_middle_initial_names_from_an_officer_table(tmp_path):
    # Regression (2026-09-28, a real SEC 10-K officer table): "Craig J.
    # Mundie" and "Lisa E. Brummel" were rejected (collision first names)
    # and "Kurt D. DelBene" was cut to a rejected "Kurt".
    text = ("Name Age Position\n"
            "Craig J. Mundie 62 Chief Research and Strategy Officer\n"
            "Lisa E. Brummel 52 Senior Vice President, Human Resources\n"
            "Kurt D. DelBene 51 President, Office Division\n")
    sample = tmp_path / "officers.txt"
    sample.write_text(text, encoding="utf-8")
    accepted = {p.display_text for p in Pipeline(config=_test_config(), base_dir=BASE_DIR).run(sample).persons}
    assert {"Craig J. Mundie", "Lisa E. Brummel", "Kurt D. DelBene"} <= accepted, accepted


def test_pipeline_rejects_language_menu_and_news_header_words(tmp_path):
    # Regression (2026-09-28, exp07): a language/region menu and a "Telugu
    # News" site header got "Español" and "Telugu" ACCEPTED as people.
    # (Surrounded by English prose, as on the real pages - a menu alone is
    # skipped by the non-English language filter.)
    text = ("Welcome to the help centre. Choose the language you would like to read this page in, "
            "and the page will reload.\n"
            "Language: English | Español (Latinoamérica) | Français | Português\n"
            "You can change the language at any time from the menu at the bottom of every page.\n"
            "Language: English | Español (Latinoamérica) | Français | Português\n"
            "Online Telugu News Today - Telugu Breaking News and the latest headlines from the state.\n"
            "Ramesh Kumar reads Telugu News every morning before he leaves for the office.\n")
    sample = tmp_path / "menu.txt"
    sample.write_text(text, encoding="utf-8")
    accepted = {p.display_text for p in Pipeline(config=_test_config(), base_dir=BASE_DIR).run(sample).persons}
    assert "Español" not in accepted and "Telugu" not in accepted, accepted
    assert "Ramesh Kumar" in accepted, accepted
