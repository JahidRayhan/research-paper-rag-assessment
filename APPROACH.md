# Design decisions

## Chunking
- PDFs are read block-by-block (PyMuPDF). A line that fully matches a heading pattern (Abstract, Introduction,
  Related Work, Methodology, Experiments, Results, Discussion, Conclusion, References, ...) switches the current
  section; numbered/Roman headings ("3 Model Architecture", "I. INTRODUCTION") are handled. Sub-headings stay in
  their parent section. Papers without recognisable headings fall back to one `Full Text` section.
- Chunks are **sentence-aware, ~200 words (cap 300), 1-sentence overlap, and never cross a section boundary**, so a
  citation's section label is always correct and Methods text is not blended with Results text. Each chunk keeps
  its first/last page for citations. Tiny tails are merged into the previous chunk.
- Running headers/footers and bare page numbers are dropped; references are not indexed by default
  (`INDEX_REFERENCES=true` to include) because they add noise to retrieval.
- Why ~200 words: small enough that one chunk answers one idea (precise citations), large enough to keep a
  method description intact. Both numbers are env-configurable.

## Embeddings
`all-MiniLM-L6-v2` (384-d, cosine): fast on CPU (5 papers index in seconds), small, free, runs locally.
Each chunk is embedded as `"<paper title> | <section>\n<text>"`, so queries like "methodology of the transformer"
match on paper and section, not only body text. A stronger model (e.g. `bge-base`) is a one-line config change
and a re-index.

## Retrieval
Over-fetch `4 x top_k`, drop chunks below `MIN_SCORE`, then **diversify**: any paper whose best chunk scores within
80% of the global best is guaranteed a slot. This makes comparison questions ("CNNs vs Transformers") see both
papers while single-paper questions still get the best chunks. If the question contains a paper's title, retrieval
is automatically scoped to that paper. Optional LLM query rewriting (off by default) uses the cheaper model.

## Prompting
The system prompt forces: answer only from numbered excerpts, cite with `[n]`, compare papers explicitly, refuse with a
fixed sentence when excerpts are insufficient, and treat excerpts as untrusted data (prompt-injection defence).
Citations in the response are built from the `[n]` markers the model actually used, mapped back to chunk metadata,
so every citation is traceable to retrieved text.

## LLM routing (Claude + Gemini)
`LLMRouter` holds an ordered list of providers behind one interface. The first success wins; any exception
(rate limit, timeout, empty/blocked output) falls through to the next. Separate lists for answering and query
rewriting let you put the strongest model on answers and a cheaper one on utility calls.

## Database schema
- `papers`: id, title, authors (JSON), year, filename, `file_hash` (unique, dedupes uploads), stored_path, pages,
  chunks, sections (JSON), view/download/query counters, created_at.
- `query_logs`: id, question, answer, cited paper ids + sources (JSON), top_k, confidence, response_time_ms,
  llm_provider, cached, rating (nullable), created_at (indexed).
Qdrant payload: paper_id (indexed), title, filename, section, page_start/end, chunk_index, text.
Delete order is vectors -> DB row -> file, so a failure never leaves vectors pointing at a missing paper.

## Confidence
Heuristic, **not calibrated**: mean of the top-3 cited chunks' cosine scores, rescaled to 0-1, reduced if the
model cited nothing, fixed low value when it says the papers don't contain the answer.

## Trade-offs and limitations
- Section/title/author/year detection is heuristic. Unusual layouts can mislabel sections; upload accepts
  `title`/`authors`/`year` overrides for those cases. Multi-column reading order relies on PyMuPDF block order.
- Scanned PDFs are rejected (no OCR). Tables and equations extract as plain text.
- Ingestion is synchronous inside the request (simple, fine for a handful of papers); large batches would
  need a background queue.
- Cache is in-memory and per-process; analytics topics are keyword counts, not semantic clustering.
- Evaluation is limited to citation recall against `test_queries.json` plus manual reading; there is no
  automated answer-correctness metric.

## Topic research (v2)
- **Discovery:** arXiv (Atom API, serialised to 1 request / 3.1 s as its terms require) and OpenAlex (needs a free key;
  also gives citation counts, OA PDF links and the citation graph). Results are merged by DOI / arXiv id / normalised
  title, then ranked by embedding similarity of title+abstract to the topic plus a small log-citation bonus. A failing
  source is reported as a warning and never fails the search.
- **Import:** only open-access PDFs; the response must really start with `%PDF` (landing pages and paywalls are rejected),
  with fallbacks arXiv -> OpenAlex OA location -> optional Unpaywall. The same ingestion code path as manual upload is used,
  so imported papers get identical section detection, chunking and vectors; bibliographic metadata from the source
  overrides the PDF heuristics for title/authors/year.
- **Exact sections:** retrieval is done per paper (so one dominant paper cannot crowd out the rest), chunk hits are grouped
  by section, sections are ranked by best score plus a small bonus for several matching chunks, and the whole section is
  rebuilt from Qdrant in chunk order (overlap sentence removed). Excerpts are upserted per (topic, paper, section).
- **References:** the bibliography is generated from stored metadata (never by the LLM) and records which sections you used;
  summaries append the same reference list, and invalid `[n]` markers produced by the model are stripped.
- **Citation snowballing** uses OpenAlex: works referenced by several of your papers rank first; citing works are ranked by
  citation count.
- **Limits:** free OpenAlex/arXiv quotas; non-OA papers cannot be fetched; relevance of discovery uses abstracts only;
  summaries are only as good as the extracted sections (tables/figures are not read) and should be checked against the
  excerpts, which are saved next to them for that purpose.
