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
