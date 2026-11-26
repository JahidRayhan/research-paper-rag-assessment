import httpx
import pytest

from src.config import Settings
from src.services import biblio, discovery
from src.services.discovery import Candidate
from tests.conftest import FakeEmbedder

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
 <entry><id>http://arxiv.org/abs/1706.03762v7</id><published>2017-06-12T17:57:34Z</published>
  <title>Attention Is All
   You Need</title><summary>The dominant sequence transduction models use attention.</summary>
  <author><name>Ashish Vaswani</name></author><author><name>Noam Shazeer</name></author></entry>
</feed>"""

OA_JSON = {"results": [{
    "id": "https://openalex.org/W2626778328", "doi": "https://doi.org/10.48550/arxiv.1706.03762",
    "title": "Attention is All you Need", "publication_year": 2017, "cited_by_count": 90000,
    "abstract_inverted_index": {"attention": [0, 3], "is": [1], "all": [2]},
    "authorships": [{"author": {"display_name": "Ashish Vaswani"}}],
    "primary_location": {"source": {"display_name": "NeurIPS"}, "landing_page_url": "https://x.org/p"},
    "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/1706.03762", "landing_page_url": "https://arxiv.org/abs/1706.03762"}}]}


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_parse_arxiv_feed():
    [c] = discovery.parse_arxiv_feed(ARXIV_XML)
    assert (c.title, c.arxiv_id, c.year) == ("Attention Is All You Need", "1706.03762", 2017)
    assert c.pdf_url == "https://arxiv.org/pdf/1706.03762" and c.authors == ["Ashish Vaswani", "Noam Shazeer"]


def test_openalex_candidate_and_abstract():
    c = discovery.oa_to_candidate(OA_JSON["results"][0])
    assert c.openalex_id == "W2626778328" and c.arxiv_id == "1706.03762" and c.abstract == "attention is all attention"
    assert c.venue == "NeurIPS" and c.citation_count == 90000


def test_merge_dedupes_across_sources_and_fills_gaps():
    a = discovery.parse_arxiv_feed(ARXIV_XML)[0]
    b = discovery.oa_to_candidate(OA_JSON["results"][0])
    [m] = discovery.merge([a, b])
    assert m.source == "arxiv+openalex" and m.openalex_id == "W2626778328" and m.citation_count == 90000
    assert m.venue == "arXiv preprint"  # first non-empty value wins; gaps are filled


def test_search_all_survives_one_failing_source(monkeypatch):
    monkeypatch.setattr(discovery, "ARXIV_API", "https://export.arxiv.org/api/query")

    def handler(req):
        if "arxiv" in req.url.host:
            return httpx.Response(200, text=ARXIV_XML)
        return httpx.Response(500)

    s = Settings(arxiv_interval_s=0, openalex_api_key="k")
    cands, warnings = discovery.search_all("attention transformer", 5, ["arxiv", "openalex"], s, FakeEmbedder(),
                                           client=_client(handler))
    assert len(cands) == 1 and cands[0].relevance > 0
    assert any("openalex failed" in w for w in warnings)


def test_expand_references_ranks_by_how_many_papers_cite_them():
    def handler(req):
        path = req.url.path
        if path.startswith("/works/W1"):
            return httpx.Response(200, json={"id": "W1", "referenced_works": ["https://openalex.org/W9", "https://openalex.org/W8"]})
        if path.startswith("/works/W2"):
            return httpx.Response(200, json={"id": "W2", "referenced_works": ["https://openalex.org/W9"]})
        works = [{"id": f"https://openalex.org/{i}", "title": f"Paper {i}", "cited_by_count": 5} for i in ("W9", "W8")]
        return httpx.Response(200, json={"results": works})

    out = discovery.expand(_client(handler), ["W1", "W2"], "references", 10, "k")
    assert [c.openalex_id for c in out] == ["W9", "W8"] and "2 of your papers" in out[0].found_via


def test_fetch_pdf_rejects_landing_pages_and_falls_back():
    s = Settings(arxiv_interval_s=0)
    c = Candidate(title="T", pdf_url="https://pub.org/paper", arxiv_id="1706.03762")

    def handler(req):
        if req.url.host == "pub.org":
            return httpx.Response(200, text="<html>paywall</html>")
        return httpx.Response(200, content=b"%PDF-1.5 data")

    assert discovery.fetch_pdf(c, s, _client(handler)).startswith(b"%PDF")
    with pytest.raises(discovery.DiscoveryError):
        discovery.fetch_pdf(Candidate(title="T", pdf_url="https://pub.org/paper"), s, _client(handler))


def test_bibtex_and_markdown():
    p = {"title": "Attention Is All You Need", "authors": ["Ashish Vaswani", "Noam Shazeer"], "year": 2017,
         "venue": "NeurIPS", "doi": "10.5555/x", "arxiv_id": "1706.03762", "url": None, "sections_used": ["Methodology p.3"]}
    [a, b] = biblio.assign_keys([dict(p), dict(p)])
    assert (a["key"], b["key"]) == ("vaswani2017attention", "vaswani2017attentionb")
    bib = biblio.to_bibtex(a)
    assert bib.startswith("@article{vaswani2017attention,") and "author = {Ashish Vaswani and Noam Shazeer}" in bib
    assert "eprint = {1706.03762}" in bib
    assert "(2017). Attention Is All You Need." in biblio.to_markdown(a, 1) and "Methodology p.3" in biblio.to_markdown(a, 1)
