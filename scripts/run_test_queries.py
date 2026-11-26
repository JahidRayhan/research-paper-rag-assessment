"""Run test_queries.json against the API and report which expected papers were cited.

  python scripts/run_test_queries.py [--url ...] [--file test_queries.json]

Expected papers are matched by number ("paper_3.pdf" == "paper3_nlp_transformers.pdf"), so
name your files paperN*.pdf. Writes results/test_run.json for manual answer review.
"""
import argparse
import json
import pathlib
import re
import time

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--file", default="test_queries.json")
ap.add_argument("--api-key", default=None)
ap.add_argument("--top-k", type=int, default=6)
a = ap.parse_args()
headers = {"X-API-Key": a.api_key} if a.api_key else {}


def num(name: str):
    m = re.search(r"paper[_\-]?(\d+)", name.lower())
    return m.group(1) if m else name.lower()


results, recalls = [], []
for q in json.loads(pathlib.Path(a.file).read_text()):
    t0 = time.time()
    r = httpx.post(f"{a.url}/api/query", json={"question": q["question"], "top_k": a.top_k},
                   headers=headers, timeout=180)
    if r.status_code != 200:
        print(f"#{q['id']:>2} ERROR {r.status_code}: {r.text[:120]}")
        results.append({**q, "error": r.text})
        continue
    j = r.json()
    cited, expected = {num(s) for s in j["sources_used"]}, {num(e) for e in q["expected_papers"]}
    recall = len(cited & expected) / len(expected)
    recalls.append(recall)
    print(f"#{q['id']:>2} [{q['difficulty']:<6}] recall={recall:.2f} conf={j['confidence']:.2f} "
          f"{time.time() - t0:5.1f}s cited={sorted(cited)} expected={sorted(expected)}")
    results.append({**q, "answer": j["answer"], "sources_used": j["sources_used"], "confidence": j["confidence"],
                    "citations": j["citations"], "recall": recall})
if recalls:
    print(f"\nMean source recall: {sum(recalls) / len(recalls):.2f} over {len(recalls)} queries")
pathlib.Path("results").mkdir(exist_ok=True)
pathlib.Path("results/test_run.json").write_text(json.dumps(results, indent=2))
print("Full answers saved to results/test_run.json (read them - recall alone doesn't prove correctness).")
