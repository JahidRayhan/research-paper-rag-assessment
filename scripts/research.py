"""One-command literature workflow.

  python scripts/research.py "self-attention for long sequences" --papers 5 --out review.md
  python scripts/research.py "graph neural networks" --focus "training stability" --style compare

Creates a topic, searches arXiv/OpenAlex, imports the top open-access papers, extracts the matching
sections, writes a cited summary and saves everything (excerpts, references, BibTeX) to a markdown file.
"""
import argparse
import sys

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("topic")
ap.add_argument("--focus", help="what to extract/summarise (defaults to the topic)")
ap.add_argument("--papers", type=int, default=5, help="how many papers to import")
ap.add_argument("--style", default="overview", choices=["overview", "compare", "brief"])
ap.add_argument("--out", default="review.md")
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--api-key", default=None)
a = ap.parse_args()

c = httpx.Client(base_url=a.url, timeout=600, headers={"X-API-Key": a.api_key} if a.api_key else {})


def call(method, path, **kw):
    r = c.request(method, path, **kw)
    if r.status_code >= 400:
        sys.exit(f"{method} {path} failed: {r.status_code} {r.text}")
    return r


tid = call("POST", "/api/topics", json={"name": a.topic, "query": a.topic}).json()["id"]
found = call("POST", f"/api/topics/{tid}/search", json={"limit": max(15, a.papers * 3)}).json()
for w in found["warnings"]:
    print("warning:", w)
print(f"Found {len(found['candidates'])} candidates; importing up to {a.papers}...")
got = 0
for cand in found["candidates"]:
    if got >= a.papers:
        break
    if cand["status"] not in ("new", "indexed"):
        continue
    r = call("POST", f"/api/topics/{tid}/import", json={"indexes": [cand["index"]]}).json()[0]
    print(f"  [{r['status']:<8}] {r['title'][:70]}" + (f"  ({r['detail']})" if r["detail"] else ""))
    got += r["status"] in ("imported", "indexed")
if not got:
    sys.exit("No papers could be imported (no open-access PDFs?). Try a broader topic.")
focus = a.focus or a.topic
call("POST", f"/api/topics/{tid}/sections", json={"query": focus, "per_paper": 2})
s = call("POST", f"/api/topics/{tid}/summary", json={"focus": focus, "style": a.style, "max_papers": a.papers}).json()
open(a.out, "w", encoding="utf-8").write(call("GET", f"/api/topics/{tid}/export").text)
print(f"\nTopic {tid} saved. Summary by {s['llm_provider']}; full notes + references + BibTeX in {a.out}")
