"""Upload every PDF in a folder:  python scripts/ingest_samples.py [folder] [--url http://localhost:8000]"""
import argparse
import pathlib
import time

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("folder", nargs="?", default="sample_papers")
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--api-key", default=None)
a = ap.parse_args()
headers = {"X-API-Key": a.api_key} if a.api_key else {}

pdfs = sorted(pathlib.Path(a.folder).glob("*.pdf"))
if not pdfs:
    raise SystemExit(f"No PDFs found in {a.folder}/")
t0 = time.time()
for p in pdfs:
    r = httpx.post(f"{a.url}/api/papers/upload", files={"file": (p.name, p.read_bytes(), "application/pdf")},
                   headers=headers, timeout=300)
    if r.status_code == 201:
        j = r.json()
        print(f"OK   {p.name}: id={j['id']} '{j['title']}' chunks={j['num_chunks']} sections={j['sections']}")
    else:
        print(f"FAIL {p.name}: {r.status_code} {r.text}")
print(f"Done in {time.time() - t0:.1f}s")
