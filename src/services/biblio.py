"""Reference formatting (BibTeX / Markdown) from paper metadata dicts."""
import re
import unicodedata

_TEX = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_"}


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _tex(s: str) -> str:
    return "".join(_TEX.get(ch, ch) for ch in s or "")


def bibtex_key(p: dict) -> str:
    last = _ascii((p.get("authors") or ["anon"])[0].split()[-1]).lower()
    word = next((w for w in re.findall(r"[a-z]+", _ascii(p.get("title", "")).lower())
                 if w not in {"a", "an", "the", "on", "of", "for", "and", "in", "to"}), "paper")
    return f"{re.sub(r'[^a-z]', '', last) or 'anon'}{p.get('year') or 'nd'}{word}"


def assign_keys(papers: list[dict]) -> list[dict]:
    seen: dict[str, int] = {}
    for p in papers:
        k = bibtex_key(p)
        seen[k] = seen.get(k, 0) + 1
        p["key"] = k if seen[k] == 1 else k + chr(96 + seen[k])
    return papers


def to_bibtex(p: dict) -> str:
    kind = "article" if p.get("venue") and "arxiv" not in (p.get("venue") or "").lower() else "misc"
    f = {"title": "{" + _tex(p["title"]) + "}", "author": " and ".join(_tex(a) for a in p.get("authors") or []),
         "year": p.get("year"), "doi": p.get("doi"), "url": p.get("url")}
    if kind == "article":
        f["journal"] = _tex(p["venue"])
    if p.get("arxiv_id"):
        f.update(eprint=p["arxiv_id"], archivePrefix="arXiv")
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items() if v)
    return f"@{kind}{{{p['key']},\n{body}\n}}"


def to_markdown(p: dict, n: int | None = None) -> str:
    authors = p.get("authors") or []
    a = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if authors else "Unknown authors"
    tail = f" {p['venue']}." if p.get("venue") else ""
    link = f" https://doi.org/{p['doi']}" if p.get("doi") else (f" {p['url']}" if p.get("url") else "")
    sec = ""
    if p.get("sections_used"):
        sec = " _(used: " + "; ".join(p["sections_used"]) + ")_"
    return f"{f'[{n}] ' if n else ''}{a} ({p.get('year') or 'n.d.'}). {p['title']}.{tail}{link}{sec}"
