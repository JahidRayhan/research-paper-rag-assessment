"""PDF parsing: text extraction, section detection, metadata guessing and chunking.

Section detection is heuristic (heading regexes over PyMuPDF text blocks), which works
well for typical IEEE/ACM/arXiv layouts. Papers with no recognisable headings fall back
to a single "Full Text" section so they are still searchable.
"""
import itertools
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import pymupdf as fitz  # PyMuPDF

log = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    pass


@dataclass
class Block:
    section: str
    page: int
    text: str


@dataclass
class Chunk:
    index: int
    section: str
    page_start: int
    page_end: int
    text: str


@dataclass
class ParsedPaper:
    title: str
    authors: list[str]
    year: int | None
    num_pages: int
    blocks: list[Block]
    sections: list[str]


# ---------------------------------------------------------------- headings
_SECTIONS = [
    ("Abstract", r"abstract"),
    ("Introduction", r"introduction"),
    ("Related Work", r"related\s+work|background|literature\s+review|preliminaries"),
    ("Methodology", r"methods?|methodology|approach|(?:proposed\s+)?model(?:\s+architecture)?"
                    r"|materials\s+and\s+methods|algorithms?"),
    ("Experiments", r"experiments?|experimental\s+(?:setup|results)|evaluation|datasets?"),
    ("Results", r"results?(?:\s+and\s+discussion)?|findings"),
    ("Discussion", r"discussion|analysis|limitations"),
    ("Conclusion", r"conclusions?(?:\s+and\s+future\s+work)?|future\s+work|concluding\s+remarks"),
    ("References", r"references|bibliography"),
    ("Acknowledgements", r"acknowledge?ments?"),
    ("Appendix", r"appendix(?:\s+[A-Z])?(?:[:.].*)?"),
]
_NUM = r"(?:(?:\d{1,2}|[IVX]{1,4})(?:\.\d{1,2})*\.?\s+)?"
_HEADING_RES = [(n, re.compile(rf"^\s*{_NUM}(?:{p})\s*[:.]?\s*$", re.I)) for n, p in _SECTIONS]
_INLINE_ABSTRACT = re.compile(r"^\s*abstract\s*[-—–:.]*\s*(.+)$", re.I | re.S)


def match_heading(line: str) -> str | None:
    line = line.strip()
    if not line or len(line) > 60:
        return None
    for name, rx in _HEADING_RES:
        if rx.match(line):
            return name
    return None


# ---------------------------------------------------------------- extraction
def _clean(text: str) -> str:
    text = re.sub(r"-\s*\n\s*(?=[a-z])", "", text)  # undo end-of-line hyphenation
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _extract_pages(doc) -> list[list[str]]:
    pages = []
    for page in doc:
        raw = [b[4].strip() for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        pages.append(raw)

    norm = lambda t: re.sub(r"\d+", "#", re.sub(r"\s+", " ", t.lower()))[:120]  # noqa: E731
    counts: Counter = Counter()
    for raw in pages:
        counts.update({norm(t) for t in raw})
    min_rep = max(3, int(0.4 * len(pages)))
    page_no = re.compile(r"\s*(page\s*)?\d{1,4}(\s*of\s*\d+)?\s*", re.I)

    cleaned = []
    for raw in pages:
        keep = []
        for t in raw:
            if page_no.fullmatch(t):
                continue
            if len(pages) >= 4 and counts[norm(t)] >= min_rep and len(t) < 150:
                continue  # running header / footer
            keep.append(t)
        cleaned.append(keep)
    return cleaned


_AFFIL = re.compile(r"universit|institut|department|dept|laborator|research|google|microsoft|openai|"
                    r"deepmind|inc\.|corp|school|college|brain|facebook|meta ai|\d{4}|@|conference|"
                    r"arxiv|journal|proceedings", re.I)
_NAME = re.compile(r"^[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){1,3}$")


def _guess_authors(lines: list[str]) -> list[str]:
    authors: list[str] = []
    for line in lines:
        if _AFFIL.search(line) or len(line) > 160:
            continue
        for part in re.split(r",|\band\b|;|&|∗|†|‡|\*", line):
            part = re.sub(r"[\d\W_]+$", "", part.strip())
            if _NAME.match(part) and part not in authors:
                authors.append(part)
    return authors[:10]


def _extract_title_authors(doc) -> tuple[str, list[str]]:
    meta_title = ((doc.metadata or {}).get("title") or "").strip()
    page = doc[0]
    lines = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for l in b["lines"]:
            txt = "".join(s["text"] for s in l["spans"]).strip()
            if txt and l["spans"]:
                lines.append((l["bbox"][1], max(s["size"] for s in l["spans"]), txt))
    lines.sort()
    top = [x for x in lines if x[0] < page.rect.height * 0.45]
    layout_title, authors = "", []
    if top:
        max_size = max(s for _, s, _ in top)
        title_lines = [x for x in top if x[1] >= max_size - 0.6]
        layout_title = " ".join(t for _, _, t in title_lines)
        last_y = max(y for y, _, _ in title_lines)
        cand = []
        for y, _, txt in lines:
            if y <= last_y:
                continue
            if match_heading(txt) or txt.lower().startswith("abstract"):
                break
            cand.append(txt)
        authors = _guess_authors(cand)
    junk = re.compile(r"untitled|\.pdf$|^microsoft word|^\s*$", re.I)
    if len(meta_title) >= 10 and not junk.search(meta_title):
        return meta_title[:300], authors
    return (layout_title[:300] or "Untitled paper"), authors


def _extract_year(doc) -> int | None:
    text = doc[0].get_text()
    this_year = datetime.now().year
    years = [int(y) for y in re.findall(r"\b(19[5-9]\d|20[0-3]\d)\b", text) if int(y) <= this_year + 1]
    if years:
        counts = Counter(years)
        top = max(counts.values())
        return min(y for y, c in counts.items() if c == top)
    m = re.match(r"D:(\d{4})", (doc.metadata or {}).get("creationDate") or "")
    return int(m.group(1)) if m else None


def parse_pdf(path: str) -> ParsedPaper:
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise PDFProcessingError(f"Cannot open PDF: {exc}") from exc
    with doc:
        if doc.needs_pass:
            raise PDFProcessingError("PDF is password protected")
        if doc.page_count == 0:
            raise PDFProcessingError("PDF has no pages")
        pages = _extract_pages(doc)
        title, authors = _extract_title_authors(doc)
        year = _extract_year(doc)
        num_pages = doc.page_count

    blocks: list[Block] = []
    section, seen = "Front Matter", []
    for pno, raw in enumerate(pages, start=1):
        for text in raw:
            lines = text.split("\n")
            head = match_heading(lines[0])
            if head:
                section = head
                if head not in seen:
                    seen.append(head)
                rest = _clean(" ".join(lines[1:]))
                if rest:
                    blocks.append(Block(section, pno, rest))
                continue
            m = _INLINE_ABSTRACT.match(text)
            if m and len(m.group(1)) > 40:
                section = "Abstract"
                if "Abstract" not in seen:
                    seen.append("Abstract")
                blocks.append(Block(section, pno, _clean(m.group(1))))
                continue
            cleaned = _clean(text)
            if cleaned:
                blocks.append(Block(section, pno, cleaned))

    if not blocks or sum(len(b.text) for b in blocks) < 200:
        raise PDFProcessingError("No extractable text found (scanned PDF? OCR is not supported)")
    if not seen:
        for b in blocks:
            b.section = "Full Text"
        seen = ["Full Text"]
    log.info("Parsed %d pages, %d blocks, sections=%s", num_pages, len(blocks), seen)
    return ParsedPaper(title, authors, year, num_pages, blocks, seen)


# ---------------------------------------------------------------- chunking
_PROTECT = ["et al.", "Fig.", "Figs.", "Eq.", "Sec.", "e.g.", "i.e.", "vs.", "cf.", "No."]
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def split_sentences(text: str) -> list[str]:
    for p in _PROTECT:
        text = text.replace(p, p.replace(".", "\u2024"))
    parts = [s.strip().replace("\u2024", ".") for s in _SENT.split(text)]
    return [s for s in parts if s]


def _make(index: int, section: str, sents: list) -> Chunk:
    pages = [s[1] for s in sents]
    return Chunk(index, section, min(pages), max(pages), " ".join(s[0] for s in sents))


def chunk_blocks(blocks: list[Block], target_words: int = 200, max_words: int = 300,
                 overlap: int = 1, skip_sections: tuple[str, ...] = ()) -> list[Chunk]:
    """Sentence-aware chunks that never cross a section boundary.

    Each chunk grows until ~target_words (hard cap max_words) and the next chunk starts
    with `overlap` trailing sentences of the previous one to keep context continuous.
    """
    chunks: list[Chunk] = []
    for section, group in itertools.groupby(blocks, key=lambda b: b.section):
        if section in skip_sections:
            continue
        sents = []  # (text, page, n_words)
        for b in group:
            for s in split_sentences(b.text):
                words = s.split()
                if len(words) > max_words:  # tables / unpunctuated text
                    for i in range(0, len(words), max_words):
                        piece = words[i:i + max_words]
                        sents.append((" ".join(piece), b.page, len(piece)))
                else:
                    sents.append((s, b.page, len(words)))

        cur: list = []
        cur_words = fresh = 0
        for s in sents:
            if cur and fresh and cur_words + s[2] > max_words:
                chunks.append(_make(len(chunks), section, cur))
                cur = cur[-overlap:] if overlap else []
                cur_words, fresh = sum(x[2] for x in cur), 0
            cur.append(s)
            cur_words += s[2]
            fresh += 1
            if cur_words >= target_words:
                chunks.append(_make(len(chunks), section, cur))
                cur = cur[-overlap:] if overlap else []
                cur_words, fresh = sum(x[2] for x in cur), 0
        if fresh:
            tail = cur[-fresh:]
            if sum(x[2] for x in tail) < 30 and chunks and chunks[-1].section == section:
                prev = chunks[-1]  # merge tiny tail into previous chunk
                prev.text += " " + " ".join(x[0] for x in tail)
                prev.page_end = max(prev.page_end, tail[-1][1])
            else:
                chunks.append(_make(len(chunks), section, cur))
    return chunks
