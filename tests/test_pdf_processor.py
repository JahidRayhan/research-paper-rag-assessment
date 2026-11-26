import pytest

from src.services.pdf_processor import (PDFProcessingError, chunk_blocks, match_heading, parse_pdf,
                                        split_sentences)
from tests.conftest import paragraph

SECTIONS = [("Abstract", [paragraph("attention")]),
            ("1 Introduction", [paragraph("sequence modelling")]),
            ("3 Model Architecture", [paragraph("self-attention"), paragraph("encoder stacks")]),
            ("5 Results", [paragraph("translation quality")]),
            ("6 Conclusion", [paragraph("future directions")]),
            ("References", [paragraph("citations")])]


@pytest.mark.parametrize("line,expected", [
    ("Abstract", "Abstract"), ("1 Introduction", "Introduction"), ("I. INTRODUCTION", "Introduction"),
    ("3 Model Architecture", "Methodology"), ("2.1 Methods", "Methodology"), ("5 Results", "Results"),
    ("Conclusions and Future Work", "Conclusion"), ("References", "References"),
    ("3.1 Encoder and Decoder Stacks", None), ("The results were surprising.", None)])
def test_match_heading(line, expected):
    assert match_heading(line) == expected


def test_parse_sections_metadata(pdf_factory):
    path = pdf_factory(title="Attention Mechanisms for Testing", authors="Alice Smith, Bob Jones and Carol White",
                       sections=SECTIONS)
    parsed = parse_pdf(path)
    assert parsed.title == "Attention Mechanisms for Testing"
    assert parsed.authors[:3] == ["Alice Smith", "Bob Jones", "Carol White"]
    assert parsed.year == 2017
    assert parsed.sections == ["Abstract", "Introduction", "Methodology", "Results", "Conclusion", "References"]
    assert {b.section for b in parsed.blocks} >= {"Abstract", "Methodology", "References"}


def test_chunks_respect_sections_pages_and_size(pdf_factory):
    path = pdf_factory(title="Chunking Test Paper", authors="Alice Smith", sections=SECTIONS * 2)
    parsed = parse_pdf(path)
    chunks = chunk_blocks(parsed.blocks, target_words=40, max_words=80, overlap=1, skip_sections=("References",))
    assert chunks and all(c.section != "References" for c in chunks)
    assert all(c.page_start >= 1 and c.page_end >= c.page_start for c in chunks)
    assert all(len(c.text.split()) <= 80 + 30 for c in chunks)  # max + overlap sentence slack
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert max(c.page_end for c in chunks) >= 2  # multi-page PDF keeps page numbers


def test_metadata_title_preferred(pdf_factory):
    path = pdf_factory(title="Layout Title", authors="Alice Smith", sections=SECTIONS, meta_title="Real Metadata Title")
    assert parse_pdf(path).title == "Real Metadata Title"


def test_sentence_split_protects_abbreviations():
    s = split_sentences("Vaswani et al. introduced attention. It works well (see Fig. 2). Next one!")
    assert len(s) == 3 and s[0].startswith("Vaswani et al.")


def test_bad_pdf_raises(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(PDFProcessingError):
        parse_pdf(str(p))
