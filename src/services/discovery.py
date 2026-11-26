"""Find papers on a topic from free scholarly sources and fetch their open-access PDFs.

Sources: arXiv (no key; 1 request / 3 s) and OpenAlex (free API key required since Feb 2026).
Only legally open-access PDFs are downloaded (arXiv, OpenAlex's best OA location, optional Unpaywall).
Every network helper takes an httpx client so tests can inject a mock transport.
"""
import logging
import math
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, field, fields

import httpx

from src.services.analytics import STOP

log = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX = "https://api.openalex.org"
OA_SELECT = ("id,doi,title,publication_year,authorships,cited_by_count,abstract_inverted_index,"
             "primary_location,best_oa_location")
_NS = {"a": "http://www.w3.org/2005/Atom", "ax": "http://arxiv.org/schemas/atom"}

_arxiv_lock = threading.Lock()
_arxiv_last = 0.0


class DiscoveryError(RuntimeError):
    pass


@dataclass
class Candidate:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    venue: str | None = None
    citation_count: int | None = None
    source: str = ""
    relevance: float = 0.0
    found_via: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def make_client(settings) -> httpx.Client:
    ua = "ResearchPaperAssistant/2.0" + (f" (mailto:{settings.contact_email})" if settings.contact_email else "")
    return httpx.Client(timeout=settings.discovery_timeout_s, follow_redirects=True, headers={"User-Agent": ua})


def _throttle_arxiv(interval: float) -> None:
    global _arxiv_last
    wait = interval - (time.monotonic() - _arxiv_last)
    if wait > 0:
        time.sleep(wait)
    _arxiv_last = time.monotonic()


# ------------------------------------------------------------------ arXiv
def _strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def parse_arxiv_feed(xml_text: str) -> list[Candidate]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", _NS):
        raw_id = (e.findtext("a:id", "", _NS) or "").strip()
        aid = _strip_version(re.sub(r"^https?://arxiv\.org/abs/", "", raw_id))
        title = " ".join((e.findtext("a:title", "", _NS) or "").split())
        if not title or not aid:
            continue
        published = e.findtext("a:published", "", _NS) or ""
        out.append(Candidate(
            title=title, authors=[(a.findtext("a:name", "", _NS) or "").strip() for a in e.findall("a:author", _NS)],
            year=int(published[:4]) if published[:4].isdigit() else None,
            abstract=" ".join((e.findtext("a:summary", "", _NS) or "").split()) or None,
            doi=e.findtext("ax:doi", None, _NS) or f"10.48550/arXiv.{aid}", arxiv_id=aid,
            url=f"https://arxiv.org/abs/{aid}", pdf_url=f"https://arxiv.org/pdf/{aid}",
            venue=(e.findtext("ax:journal_ref", None, _NS) or "arXiv preprint"), source="arxiv"))
    return out


def search_arxiv(client: httpx.Client, query: str, limit: int, interval: float) -> list[Candidate]:
    terms = [t for t in re.findall(r"[A-Za-z0-9\-]+", query) if t.lower() not in STOP][:7] or query.split()[:5]
    q = " AND ".join(f"all:{t}" for t in terms)
    _throttle_arxiv(interval)
    r = client.get(ARXIV_API, params={"search_query": q, "start": 0, "max_results": limit, "sortBy": "relevance"})
    r.raise_for_status()
    return parse_arxiv_feed(r.text)


# ------------------------------------------------------------------ OpenAlex
def _oa_headers(key: str | None) -> dict:
    return {"Authorization": f"Bearer {key}"} if key else {}


def _oa_get(client: httpx.Client, path: str, params: dict, key: str | None) -> dict:
    r = client.get(f"{OPENALEX}{path}", params=params, headers=_oa_headers(key))
    r.raise_for_status()
    return r.json()


def _rebuild_abstract(inv: dict | None) -> str | None:
    if not inv:
        return None
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos)) or None


def oa_to_candidate(w: dict, source: str = "openalex") -> Candidate | None:
    title = (w.get("title") or "").strip()
    if not title:
        return None
    best, primary = w.get("best_oa_location") or {}, w.get("primary_location") or {}
    pdf = best.get("pdf_url")
    landing = best.get("landing_page_url") or primary.get("landing_page_url") or w.get("doi")
    blob = " ".join(filter(None, [pdf, landing]))
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:\.pdf)?(?:v\d+)?(?:$|\s)", blob)
    doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
    return Candidate(
        title=title, year=w.get("publication_year"), abstract=_rebuild_abstract(w.get("abstract_inverted_index")),
        authors=[(a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or []][:12],
        doi=doi, arxiv_id=_strip_version(m.group(1)) if m else None,
        openalex_id=(w.get("id") or "").rsplit("/", 1)[-1] or None, url=landing, pdf_url=pdf,
        venue=((primary.get("source") or {}).get("display_name")), citation_count=w.get("cited_by_count"),
        source=source)


def search_openalex(client: httpx.Client, query: str, limit: int, key: str | None) -> list[Candidate]:
    data = _oa_get(client, "/works", {"search": query, "per_page": min(limit, 50), "select": OA_SELECT}, key)
    return [c for w in data.get("results", []) if (c := oa_to_candidate(w))]


def resolve_openalex_id(client, key, *, doi=None, arxiv_id=None, title=None) -> str | None:
    for d in [x for x in (doi, f"10.48550/arXiv.{arxiv_id}" if arxiv_id else None) if x]:
        try:
            return _oa_get(client, f"/works/doi:{d}", {"select": "id"}, key)["id"].rsplit("/", 1)[-1]
        except (httpx.HTTPError, KeyError):
            continue
    if title:
        try:
            for w in _oa_get(client, "/works", {"search": title, "per_page": 3, "select": "id,title"}, key)["results"]:
                if norm_title(w.get("title")) == norm_title(title):
                    return w["id"].rsplit("/", 1)[-1]
        except (httpx.HTTPError, KeyError):
            pass
    return None


def expand(client, openalex_ids: list[str], direction: str, limit: int, key: str | None) -> list[Candidate]:
    """Snowballing: 'references' = works your papers cite; 'citations' = works that cite your papers."""
    found: dict[str, Candidate] = {}
    if direction == "references":
        counts: Counter = Counter()
        for wid in openalex_ids:
            w = _oa_get(client, f"/works/{wid}", {"select": "id,referenced_works"}, key)
            counts.update(r.rsplit("/", 1)[-1] for r in w.get("referenced_works") or [])
        top = [r for r, _ in counts.most_common(100)]
        for i in range(0, len(top), 50):
            batch = top[i:i + 50]
            data = _oa_get(client, "/works", {"filter": "openalex:" + "|".join(batch), "per_page": len(batch),
                                              "select": OA_SELECT}, key)
            for w in data.get("results", []):
                c = oa_to_candidate(w)
                if c and c.openalex_id:
                    c.found_via = f"referenced by {counts[c.openalex_id]} of your papers"
                    found[c.openalex_id] = c
        ranked = sorted(found.values(), key=lambda c: (-counts[c.openalex_id or ""], -(c.citation_count or 0)))
    else:
        for wid in openalex_ids:
            data = _oa_get(client, "/works", {"filter": f"cites:{wid}", "sort": "cited_by_count:desc",
                                              "per_page": min(limit, 50), "select": OA_SELECT}, key)
            for w in data.get("results", []):
                c = oa_to_candidate(w)
                if c and c.openalex_id and c.openalex_id not in found:
                    c.found_via, found[c.openalex_id] = "cites one of your papers", c
        ranked = sorted(found.values(), key=lambda c: -(c.citation_count or 0))
    return ranked[:limit]


# ------------------------------------------------------------------ merge + rank
def _keys(c: Candidate) -> list[str]:
    ks = [f"t:{norm_title(c.title)}"]
    if c.doi:
        ks.append("d:" + c.doi.lower())
    if c.arxiv_id:
        ks.append("a:" + c.arxiv_id.lower())
    return ks


def merge(cands: list[Candidate]) -> list[Candidate]:
    index: dict[str, Candidate] = {}
    out: list[Candidate] = []
    for c in cands:
        hit = next((index[k] for k in _keys(c) if k in index), None)
        if hit is None:
            out.append(c)
            hit = c
        else:
            for f in fields(Candidate):
                if f.name in ("relevance", "found_via"):
                    continue
                if getattr(hit, f.name) in (None, "", []) and getattr(c, f.name) not in (None, "", []):
                    setattr(hit, f.name, getattr(c, f.name))
            hit.citation_count = max(hit.citation_count or 0, c.citation_count or 0) or None
            if c.source not in hit.source:
                hit.source += f"+{c.source}"
        for k in _keys(hit):
            index[k] = hit
    return out


def rank(cands: list[Candidate], query: str, embedder) -> list[Candidate]:
    """Order by semantic similarity of title+abstract to the topic, with a small citation bonus."""
    if not cands:
        return cands
    vecs = embedder.embed_documents([f"{c.title}. {(c.abstract or '')[:600]}" for c in cands])
    q = embedder.embed_query(query)
    for c, v in zip(cands, vecs):
        c.relevance = round(sum(a * b for a, b in zip(q, v)), 3)
    return sorted(cands, key=lambda c: -(c.relevance + 0.02 * min(math.log10(1 + (c.citation_count or 0)), 4)))


def search_all(query: str, limit: int, sources: list[str], settings, embedder, client=None):
    """Returns (ranked candidates, warnings). One failing source never fails the whole search."""
    own = client is None
    client = client or make_client(settings)
    cands: list[Candidate] = []
    warnings: list[str] = []
    try:
        for src in sources:
            try:
                if src == "arxiv":
                    cands += search_arxiv(client, query, limit, settings.arxiv_interval_s)
                elif src == "openalex":
                    if not settings.openalex_api_key:
                        warnings.append("OPENALEX_API_KEY not set: OpenAlex runs on a tiny keyless daily budget "
                                        "(free key: https://openalex.org/settings/api)")
                    cands += search_openalex(client, query, limit, settings.openalex_api_key)
            except Exception as exc:  # noqa: BLE001
                log.warning("Discovery source %s failed: %s", src, exc)
                warnings.append(f"{src} failed: {exc}")
    finally:
        if own:
            client.close()
    return rank(merge(cands), query, embedder)[:limit], warnings


# ------------------------------------------------------------------ PDF download (open access only)
def _download(client: httpx.Client, url: str, settings) -> bytes:
    if "arxiv.org" in url:
        _throttle_arxiv(settings.arxiv_interval_s)
    r = client.get(url)
    r.raise_for_status()
    data = r.content
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise DiscoveryError("PDF too large")
    if not data.startswith(b"%PDF"):
        raise DiscoveryError("URL did not return a PDF (landing page or paywall)")
    return data


def fetch_pdf(c: Candidate, settings, client: httpx.Client) -> bytes:
    urls = [u for u in (c.pdf_url, f"https://arxiv.org/pdf/{c.arxiv_id}" if c.arxiv_id else None) if u]
    if c.doi and settings.unpaywall_email:
        try:
            r = client.get(f"https://api.unpaywall.org/v2/{c.doi}", params={"email": settings.unpaywall_email})
            if r.status_code == 200:
                loc = (r.json().get("best_oa_location") or {})
                if loc.get("url_for_pdf"):
                    urls.append(loc["url_for_pdf"])
        except httpx.HTTPError:
            pass
    errors = []
    for u in dict.fromkeys(urls):
        try:
            return _download(client, u, settings)
        except (httpx.HTTPError, DiscoveryError) as exc:
            errors.append(f"{u}: {exc}")
    raise DiscoveryError("No open-access PDF could be downloaded" + (f" ({'; '.join(errors)})" if errors else ""))
