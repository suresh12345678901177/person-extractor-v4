"""Tests for the standard-library format readers added 2026-09-24:
CSV/TSV, JSON/JSONL, HTML, XML, DOCX, EML (src/io/)."""
import copy
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.models import SourceFormat
from src.io.dispatcher import ReaderDispatcher


def _read(path: Path):
    result = ReaderDispatcher().dispatch(path)
    assert result.success, result.error
    return result.document


def test_dispatcher_lists_every_new_format():
    extensions = set(ReaderDispatcher().supported_extensions())
    assert {".txt", ".pdf", ".csv", ".tsv", ".json", ".jsonl", ".html", ".htm", ".xml", ".docx", ".eml"} <= extensions


def test_csv_one_record_per_line_and_cells_never_merge(tmp_path):
    path = tmp_path / "contacts.csv"
    path.write_text('Name;Phone;Notes\nAndre Silva;+1404;"met twice\nin Lagos"\nNadia;+1405;\n', encoding="utf-8")
    doc = _read(path)
    assert doc.source_format == SourceFormat.CSV
    lines = doc.full_text.split("\n")
    assert lines == ["Name | Phone | Notes", "Andre Silva | +1404 | met twice in Lagos", "Nadia | +1405 | "]


def test_tsv_uses_tabs(tmp_path):
    path = tmp_path / "calls.tsv"
    path.write_text("Caller\tDuration\nKatya Sokolova\t00:03\n", encoding="utf-8")
    assert _read(path).full_text.split("\n")[1] == "Katya Sokolova | 00:03"


def test_json_keeps_keys_skips_non_strings_in_document_order(tmp_path):
    path = tmp_path / "chat.json"
    path.write_text(json.dumps({"messages": [
        {"sender": "Andre Silva", "text": "Did Katya confirm?", "ts": 1727000000, "read": True},
        {"sender": "Katya Sokolova", "text": "Yes.", "attachment": None},
    ]}), encoding="utf-8")
    assert _read(path).full_text.split("\n") == [
        "sender: Andre Silva", "text: Did Katya confirm?", "sender: Katya Sokolova", "text: Yes.",
    ]


def test_jsonl_and_invalid_json_are_still_read(tmp_path):
    jsonl = tmp_path / "events.jsonl"
    jsonl.write_text('{"user": "Bruno Alves"}\nnot json at all\n\n{"user": "Milo Reyes"}\n', encoding="utf-8")
    assert _read(jsonl).full_text.split("\n") == ["user: Bruno Alves", "not json at all", "user: Milo Reyes"]

    broken = tmp_path / "broken.json"
    broken.write_text('{"user": "Bruno Alves", ', encoding="utf-8")
    doc = _read(broken)
    assert "Bruno Alves" in doc.full_text and doc.source_format == SourceFormat.TXT


def test_html_keeps_visible_text_only_with_cell_and_block_boundaries(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><script>var n = 'Fake Person';</script><style>p{}</style></head><body>"
        "<p>Chaired by <b>Dr. Lalita Kher</b>.</p><table><tr><td>Kavya</td><td>Bhandari</td></tr></table>"
        "<p>Tom &amp; Jerry</p></body></html>", encoding="utf-8")
    text = _read(path).full_text
    assert "Fake Person" not in text
    assert "Chaired by Dr. Lalita Kher." in text.split("\n")
    assert "Kavya | Bhandari" in text.split("\n")
    assert "Tom & Jerry" in text


def test_xml_reads_attributes_and_text_and_survives_malformed_input(tmp_path):
    good = tmp_path / "contacts.xml"
    good.write_text('<contacts><contact name="Trevor Higgins" phone="+1404"/>'
                    '<contact><note>Brother of Marcus Delaney</note></contact></contacts>', encoding="utf-8")
    assert _read(good).full_text.split("\n") == ["name: Trevor Higgins", "Brother of Marcus Delaney"]

    bad = tmp_path / "carved.xml"
    bad.write_text("<contacts><contact>Shawn Delaney</contact><contact>unterminated", encoding="utf-8")
    assert "Shawn Delaney" in _read(bad).full_text


def _write_docx(path: Path) -> None:
    w = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'

    def p(text: str) -> str:
        return f'<w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'

    document = (f"<w:document {w}><w:body>{p('Internal memo')}{p('Prepared for Mrs. Jennifer Wilson.')}"
                f"<w:tbl><w:tr><w:tc>{p('Geetha Rao')}</w:tc><w:tc>{p('Legal')}</w:tc></w:tr></w:tbl>"
                f"<w:p><w:r><w:t>Approved by</w:t><w:br/><w:t>Dr. Suresh Kumar.</w:t></w:r></w:p></w:body></w:document>")
    comments = f'<w:comments {w}><w:comment w:id="0" w:author="Prasad Menon">{p("Check totals.")}</w:comment></w:comments>'
    core = ('<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:creator>Naresh Varma</dc:creator></cp:coreProperties>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("word/comments.xml", comments)
        archive.writestr("docProps/core.xml", core)


def test_docx_paragraph_per_line_tables_comments_and_author(tmp_path):
    path = tmp_path / "memo.docx"
    _write_docx(path)
    lines = _read(path).full_text.split("\n")
    assert lines[:4] == ["Internal memo", "Prepared for Mrs. Jennifer Wilson.", "Geetha Rao | Legal",
                         "Approved by Dr. Suresh Kumar."]
    assert "Comment author: Prasad Menon" in lines
    assert "Document author: Naresh Varma" in lines


def test_eml_headers_and_html_body(tmp_path):
    path = tmp_path / "mail.eml"
    path.write_bytes(b"From: Priyanka Deshmukh <p.d@example.com>\r\nTo: Rohan Vaidya <r.v@example.com>\r\n"
                     b"Subject: Invoice\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
                     b"<p>Hi Rohan,</p><p>Ashwin Ranade signed off.</p>\r\n")
    lines = _read(path).full_text.split("\n")
    assert lines[:3] == ["From: Priyanka Deshmukh <p.d@example.com>", "To: Rohan Vaidya <r.v@example.com>",
                         "Subject: Invoice"]
    assert "Ashwin Ranade signed off." in lines


def test_contacts_csv_is_not_skipped_by_the_prose_language_gate(tmp_path):
    """A contact list is names and numbers with almost no English
    stopwords - the prose gate would skip it wholesale if applied."""
    from config import BASE_DIR, DEFAULT_CONFIG
    from src.pipeline.orchestrator import Pipeline

    people = ["Andre Silva", "Priya Ramanathan", "Marcus Delaney", "Farah Nasser", "Ibrahim Coulibaly",
              "Renee Fontaine", "Geetha Rao", "Suresh Kumar", "Lalita Kher", "Ashwin Ranade", "Amit Ranade"]
    rows = ["Name,Phone,City"] + [f"{name},+1404555{i:04d},Atlanta" for i, name in enumerate(people)]
    path = tmp_path / "contacts.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["feedback"]["enabled"] = False
    result = Pipeline(config=config, base_dir=BASE_DIR).run(path)
    assert result.model_info.get("skipped_reason") is None
    assert len({p.display_text for p in result.persons} & set(people)) >= 8
    first = next(p for p in result.persons if p.display_text == "Andre Silva")
    assert first.mentions[0].candidate.location.line_number == 2  # record 2 (header is 1)


def test_binary_xml_is_flagged_as_binary_not_decoded_into_words(tmp_path):
    """Android 12+ stores many system XML files in binary form ('ABX\0');
    decoding their bytes produced accepted garbage 'names' like 'ÑËÙ' on a
    real case scan."""
    abx = tmp_path / "appops_accesses.xml"
    abx.write_bytes(b"ABX\x00\x102\xff\xff\x00\x07app-ops\x00\xd1\xcb\xd9\x00\xd1\xd8I")
    compiled = tmp_path / "AndroidManifest.xml"
    compiled.write_bytes(b"\x03\x00\x08\x00\x10\x0e\x00\x00\x01\x00\x1c\x00manifest")
    for path in (abx, compiled):
        doc = _read(path)
        assert doc.full_text == ""
        assert doc.metadata["skipped_reason"] == "binary_content"


def test_utf16_xml_is_text_not_binary(tmp_path):
    path = tmp_path / "contacts_utf16.xml"
    path.write_bytes('<contacts><contact name="Trevor Higgins"/></contacts>'.encode("utf-16"))
    assert _read(path).full_text == "name: Trevor Higgins"


def test_binary_file_is_reported_as_skipped_by_the_pipeline(tmp_path):
    from config import BASE_DIR, DEFAULT_CONFIG
    from src.pipeline.orchestrator import Pipeline

    path = tmp_path / "prefs.xml"
    path.write_bytes(b"ABX\x00" + bytes(range(256)))
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["feedback"]["enabled"] = False
    result = Pipeline(config=config, base_dir=BASE_DIR).run(path)
    assert result.success and not result.persons
    assert result.model_info["skipped_reason"] == "binary_content"


def test_lone_single_word_values_in_structured_data_go_to_review(tmp_path):
    """Android settings JSON/XML is full of values that happen to be list
    names ('Edit', 'Read', 'Block'); a single word there is held for
    review, while a full name - and a first name the same file's accepted
    full name vouches for - is still accepted."""
    from config import BASE_DIR, DEFAULT_CONFIG
    from src.pipeline.orchestrator import Pipeline

    path = tmp_path / "prefs.json"
    path.write_text(json.dumps({"menu": ["Edit", "Block", "Read"], "owner": "Dr. Suresh Kumar",
                                "note": "Suresh approved the layout."}), encoding="utf-8")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["feedback"]["enabled"] = False
    result = Pipeline(config=config, base_dir=BASE_DIR).run(path)
    accepted = {p.display_text for p in result.persons}
    assert "Dr. Suresh Kumar" in accepted
    assert not accepted & {"Edit", "Block", "Read"}
