"""End-to-end API test with an in-memory Qdrant, fake embedder and fake LLM (no network/models needed)."""
from tests.conftest import paragraph


def _upload(client, pdf_factory, name, title, topic):
    path = pdf_factory(name, title=title, authors="Alice Smith, Bob Jones",
                       sections=[("Abstract", [paragraph(topic)]), ("1 Introduction", [paragraph(topic)]),
                                 ("3 Methods", [paragraph(topic + " method")]), ("References", [paragraph("refs")])])
    with open(path, "rb") as f:
        data = f.read()
    _upload.last_bytes = data
    return client.post("/api/papers/upload", files={"file": (name, data, "application/pdf")})


def test_full_flow(client, pdf_factory):
    r1 = _upload(client, pdf_factory, "paper_3.pdf", "Attention Mechanisms for Transformers", "self-attention transformer")
    r2 = _upload(client, pdf_factory, "paper_5.pdf", "Reward Optimisation in Agents", "reinforcement reward policy")
    assert r1.status_code == r2.status_code == 201
    p1 = r1.json()
    assert p1["num_chunks"] > 0 and "Methodology" in p1["sections"] and p1["year"] == 2017
    assert len(client.get("/api/papers").json()) == 2

    # duplicate upload, wrong type, empty library handling
    first_bytes = _upload.last_bytes  # bytes of paper_5 (the last upload)
    dup = client.post("/api/papers/upload", files={"file": ("again.pdf", first_bytes, "application/pdf")})
    assert dup.status_code == 409 and dup.json()["detail"]["paper_id"] == r2.json()["id"]
    assert client.post("/api/papers/upload", files={"file": ("a.txt", b"hi", "text/plain")}).status_code == 415

    q = client.post("/api/query", json={"question": "How does self-attention work in the transformer?", "top_k": 3})
    assert q.status_code == 200, q.text
    body = q.json()
    assert body["citations"][0]["paper_id"] == p1["id"]
    assert set(body) >= {"answer", "citations", "sources_used", "confidence", "query_id"}
    assert body["sources_used"] == ["paper_3.pdf"] and 0 <= body["confidence"] <= 1
    assert {"paper_title", "section", "page", "relevance_score"} <= set(body["citations"][0])

    # cache + scoping + validation
    assert client.post("/api/query", json={"question": "How does self-attention work in the transformer?",
                                           "top_k": 3}).json()["cached"] is True
    assert client.post("/api/query", json={"question": "reward policy?", "paper_ids": [999]}).status_code == 404
    assert client.post("/api/query", json={"question": "hi"}).status_code == 422

    hist = client.get("/api/queries/history").json()
    assert len(hist) == 2 and hist[0]["question"].startswith("How does")
    assert client.post(f"/api/queries/{body['query_id']}/rating", json={"rating": 5}).status_code == 200
    pop = client.get("/api/analytics/popular").json()
    assert pop["total_queries"] == 2 and pop["popular_topics"] and pop["avg_rating"] == 5

    assert client.get(f"/api/papers/{p1['id']}").status_code == 200
    assert client.get(f"/api/papers/{p1['id']}/download").status_code == 200
    stats = client.get(f"/api/papers/{p1['id']}/stats").json()
    assert (stats["view_count"], stats["download_count"], stats["times_cited"]) == (1, 1, 2)

    d = client.delete(f"/api/papers/{p1['id']}")
    assert d.status_code == 200 and client.get(f"/api/papers/{p1['id']}").status_code == 404
    after = client.post("/api/query", json={"question": "How does self-attention work in the transformer?",
                                            "paper_ids": [r2.json()["id"]]}).json()
    assert all(c["paper_id"] != p1["id"] for c in after["citations"])  # vectors really deleted


def test_query_with_no_papers(client):
    assert client.post("/api/query", json={"question": "anything at all"}).status_code == 409
