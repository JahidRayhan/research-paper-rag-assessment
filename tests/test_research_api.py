"""Topic workflow end-to-end: search -> import -> exact sections -> references -> summary -> export."""
import pytest

from src.services import discovery
from src.services.discovery import Candidate
from tests.conftest import make_pdf, paragraph


def _pdf_bytes(tmp_path, name, title, topic):
    path = tmp_path / name
    make_pdf(str(path), title=title, authors="Alice Smith, Bob Jones", sections=[
        ("Abstract", [paragraph(topic)]), ("1 Introduction", [paragraph("background")]),
        ("3 Methods", [paragraph(topic + " method"), paragraph(topic + " method details")]),
        ("5 Results", [paragraph("benchmark numbers")]), ("References", [paragraph("refs")])])
    return path.read_bytes()


@pytest.fixture
def fake_web(tmp_path, monkeypatch):
    pdfs = {"1": _pdf_bytes(tmp_path, "a.pdf", "Self-Attention Networks for Sequences", "self-attention"),
            "2": _pdf_bytes(tmp_path, "b.pdf", "Convolutional Sequence Models", "convolution"),
            "3": None}
    cands = [Candidate(title="Self-Attention Networks for Sequences", authors=["Alice Smith", "Bob Jones"], year=2017,
                       arxiv_id="1", doi="10.1/a", url="https://arxiv.org/abs/1", pdf_url="u1", source="arxiv", relevance=0.9),
             Candidate(title="Convolutional Sequence Models", authors=["Carol White"], year=2018, arxiv_id="2",
                       pdf_url="u2", source="arxiv", relevance=0.6, venue="ICML"),
             Candidate(title="Paywalled Work", authors=["Dan Gray"], year=2019, doi="10.1/c", pdf_url="u3",
                       source="openalex", relevance=0.4)]
    monkeypatch.setattr(discovery, "search_all", lambda q, limit, sources, s, emb, client=None: (cands, ["note: test"]))

    def fake_fetch(c, settings, client):
        data = pdfs.get(c.pdf_url[1:])
        if data is None:
            raise discovery.DiscoveryError("No open-access PDF could be downloaded")
        return data
    monkeypatch.setattr(discovery, "fetch_pdf", fake_fetch)
    monkeypatch.setattr(discovery, "resolve_openalex_id", lambda *a, **k: "W1")
    monkeypatch.setattr(discovery, "expand", lambda client, ids, direction, limit, key: [
        Candidate(title="Older Foundational Paper", year=2010, openalex_id="W9", source="openalex",
                  found_via="referenced by 2 of your papers")])


def test_topic_workflow(client, fake_web):
    # discovery (stateless)
    d = client.post("/api/discover", json={"query": "self-attention sequences"}).json()
    assert len(d["candidates"]) == 3 and d["warnings"] == ["note: test"] and d["candidates"][0]["status"] == "new"

    t = client.post("/api/topics", json={"name": "Attention study", "query": "self-attention sequence models"}).json()
    tid = t["id"]
    s = client.post(f"/api/topics/{tid}/search", json={"limit": 10}).json()
    assert [c["index"] for c in s["candidates"]] == [0, 1, 2]

    # import: 2 succeed, 1 has no open-access PDF; nothing crashes
    imp = client.post(f"/api/topics/{tid}/import", json={"top_n": 3}).json()
    assert [r["status"] for r in imp] == ["imported", "imported", "no_pdf"]
    assert client.get(f"/api/topics/{tid}").json()["num_papers"] == 2
    again = client.post(f"/api/topics/{tid}/import", json={"indexes": [0]}).json()
    assert again[0]["status"] == "indexed"  # never downloads twice
    assert client.get(f"/api/papers/{imp[0]['paper_id']}").json()["title"] == "Self-Attention Networks for Sequences"

    # exact sections: whole Methods section, both paragraphs, with pages
    ps = client.post(f"/api/topics/{tid}/sections", json={"query": "self-attention method", "per_paper": 1}).json()
    top = ps[0]
    assert top["paper_id"] == imp[0]["paper_id"] and top["section"] == "Methodology" and top["page_start"] >= 1
    assert top["text"].count("Sentence 0") >= 2 and not top["truncated"]   # both Methods paragraphs present
    assert top["excerpt_id"]
    ex = client.get(f"/api/topics/{tid}/excerpts").json()
    assert len(ex) == len(ps)
    assert client.patch(f"/api/excerpts/{ex[0]['id']}", json={"note": "cite this"}).json()["note"] == "cite this"
    # re-running updates instead of duplicating
    client.post(f"/api/topics/{tid}/sections", json={"query": "self-attention method", "per_paper": 1})
    assert len(client.get(f"/api/topics/{tid}/excerpts").json()) == len(ps)

    # references
    refs = client.get(f"/api/topics/{tid}/references").json()
    assert {r["key"] for r in refs} == {"smith2017self", "white2018convolutional"}
    assert any(r["sections_used"] for r in refs)
    bib = client.get(f"/api/topics/{tid}/references?format=bibtex").text
    assert "@misc{smith2017self" in bib and "@article{white2018convolutional" in bib
    assert client.get(f"/api/topics/{tid}/references?format=markdown").text.startswith("[1]")

    # snowballing appends new candidates
    ex_resp = client.post(f"/api/topics/{tid}/expand", json={"direction": "references"}).json()
    assert ex_resp["candidates"][0]["title"] == "Older Foundational Paper" and ex_resp["candidates"][0]["index"] == 3

    # summary: invalid [7] citation stripped, references appended, sections recorded
    sm = client.post(f"/api/topics/{tid}/summary", json={"focus": "self-attention method", "max_papers": 2}).json()
    assert "[7]" not in sm["markdown"] and "[1]" in sm["markdown"] and "## References" in sm["markdown"]
    assert sm["llm_provider"] == "fake" and len(sm["paper_ids"]) >= 1
    assert len(client.get(f"/api/topics/{tid}/summaries").json()) == 1

    md = client.get(f"/api/topics/{tid}/export").text
    assert "## Summary" in md and "## Saved excerpts" in md and "Note: cite this" in md and "```bibtex" in md

    # deleting a paper cleans up links/excerpts; deleting the topic keeps papers
    pid = imp[1]["paper_id"]
    assert client.delete(f"/api/papers/{pid}").status_code == 200
    assert client.get(f"/api/topics/{tid}").json()["num_papers"] == 1
    assert client.delete(f"/api/topics/{tid}").status_code == 200
    assert client.get(f"/api/topics/{tid}").status_code == 404
    assert client.get(f"/api/papers/{imp[0]['paper_id']}").status_code == 200


def test_topic_edge_cases(client, fake_web):
    tid = client.post("/api/topics", json={"name": "Empty"}).json()["id"]
    assert client.post(f"/api/topics/{tid}/sections", json={}).status_code == 409
    assert client.post(f"/api/topics/{tid}/import", json={"top_n": 2}).status_code == 409
    assert client.post(f"/api/topics/{tid}/summary", json={}).status_code == 409
    assert client.post("/api/topics/999/search", json={}).status_code == 404
    client.post(f"/api/topics/{tid}/search", json={})
    assert client.post(f"/api/topics/{tid}/import", json={"indexes": [42]}).status_code == 422
    assert client.post("/api/discover", json={"query": "x"}).status_code == 422
