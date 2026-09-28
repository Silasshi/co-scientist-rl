"""Paper retrieval via OpenAlex (forward search) and Semantic Scholar (backward refs).

Forward search: the policy model generates search queries, executed against OpenAlex.
Backward search (D5): given an arxiv paper, fetch its full reference list (papers
that paper CITES) via S2 API for grounding oracle abstraction.

OpenAlex: free, no API key required, 250M+ works, 10 req/s with mailto.
S2 (Semantic Scholar): free, no auth, rate-limited 100 req/5min.
"""

import logging
import re
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org"
OPENALEX_TIMEOUT = 15.0
OPENALEX_MAX_RETRIES = 3
OPENALEX_BASE_DELAY = 2.0
OPENALEX_MAX_DELAY = 30.0
OPENALEX_RETRYABLE_STATUS = {429, 500, 502, 503}
OPENALEX_RATE_DELAY = 0.12
MAX_QUERIES = 5

S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_TIMEOUT = 30.0
S2_MAX_RETRIES = 5
S2_BASE_DELAY = 5.0
S2_MAX_DELAY = 60.0
S2_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# S2 unauthenticated rate limit: 100 req per 5 min = 1.2 req/s; we use 3.5s/req for safety
S2_RATE_DELAY = 3.5


@dataclass
class PaperResult:
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    citation_count: int
    venue: str
    doi: str | None = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract[:500],
            "citation_count": self.citation_count,
            "venue": self.venue,
            "doi": self.doi,
        }


@dataclass
class RetrievalResult:
    papers: list[PaperResult]
    queries_used: list[str]
    n_api_calls: int
    latency_s: float
    errors: list[str] = field(default_factory=list)


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as inverted index {word: [positions]}."""
    if not inverted_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


class PaperRetriever:
    """Synchronous OpenAlex client with rate limiting and deduplication."""

    def __init__(self, mailto: str = "your-email@example.com"):
        self._mailto = mailto
        self._client = httpx.Client(
            timeout=httpx.Timeout(OPENALEX_TIMEOUT, connect=5.0),
            headers={"User-Agent": f"CoScientist/1.0 (mailto:{mailto})"},
        )
        self._last_call_time = 0.0
        self._total_calls = 0

    def _rate_limit(self):
        now = time.monotonic()
        wait = OPENALEX_RATE_DELAY - (now - self._last_call_time)
        if wait > 0:
            time.sleep(wait)
        self._last_call_time = time.monotonic()

    def search_papers(self, query: str, limit: int = 5) -> list[PaperResult]:
        if not query.strip():
            return []

        self._rate_limit()
        self._total_calls += 1

        params = {
            "search": query.strip(),
            "per_page": min(limit, 25),
            "mailto": self._mailto,
            "select": "title,publication_year,authorships,cited_by_count,primary_location,abstract_inverted_index,doi",
        }

        delay = OPENALEX_BASE_DELAY
        for attempt in range(OPENALEX_MAX_RETRIES + 1):
            try:
                resp = self._client.get(f"{OPENALEX_BASE_URL}/works", params=params)

                if resp.status_code in OPENALEX_RETRYABLE_STATUS:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else delay
                    wait = min(wait, OPENALEX_MAX_DELAY)
                    logger.warning(
                        "OpenAlex %d on query %r (attempt %d/%d), retrying in %.1fs",
                        resp.status_code, query[:50], attempt + 1, OPENALEX_MAX_RETRIES + 1, wait,
                    )
                    time.sleep(wait)
                    delay = min(delay * 2, OPENALEX_MAX_DELAY)
                    continue

                resp.raise_for_status()
                data = resp.json()
                raw_papers = data.get("results", [])
                results = []
                for p in raw_papers:
                    title = (p.get("title") or "").strip()
                    if not title:
                        continue
                    authorships = p.get("authorships") or []
                    author_names = []
                    for a in authorships[:3]:
                        raw_name = (a.get("author") or {}).get("display_name", "")
                        if raw_name:
                            author_names.append(raw_name.split()[-1])

                    abstract = _reconstruct_abstract(p.get("abstract_inverted_index"))

                    location = p.get("primary_location") or {}
                    source = location.get("source") or {}
                    venue = (source.get("display_name") or "").strip()

                    results.append(PaperResult(
                        title=title,
                        authors=author_names,
                        year=p.get("publication_year"),
                        abstract=abstract[:800],
                        citation_count=p.get("cited_by_count") or 0,
                        venue=venue,
                        doi=p.get("doi"),
                    ))
                return results

            except httpx.ConnectError:
                logger.warning("OpenAlex connection failed for query %r (attempt %d)", query[:50], attempt + 1)
                if attempt < OPENALEX_MAX_RETRIES:
                    time.sleep(delay)
                    delay = min(delay * 2, OPENALEX_MAX_DELAY)
            except Exception as e:
                logger.warning("OpenAlex search error for query %r: %s: %s", query[:50], type(e).__name__, e)
                return []

        logger.warning("OpenAlex search exhausted retries for query %r", query[:50])
        return []

    def search_multiple(
        self, queries: list[str], limit_per_query: int = 5, dedup: bool = True,
    ) -> RetrievalResult:
        t0 = time.monotonic()
        all_papers: list[PaperResult] = []
        errors: list[str] = []
        n_calls = 0

        for q in queries[:MAX_QUERIES]:
            try:
                papers = self.search_papers(q, limit=limit_per_query)
                all_papers.extend(papers)
                n_calls += 1
            except Exception as e:
                errors.append(f"{q}: {e}")

        if dedup:
            seen_titles: set[str] = set()
            unique: list[PaperResult] = []
            for p in all_papers:
                key = p.title.lower().strip()
                if key not in seen_titles:
                    seen_titles.add(key)
                    unique.append(p)
            all_papers = unique

        return RetrievalResult(
            papers=all_papers,
            queries_used=queries[:MAX_QUERIES],
            n_api_calls=n_calls,
            latency_s=time.monotonic() - t0,
            errors=errors,
        )

    @property
    def total_api_calls(self) -> int:
        return self._total_calls

    def close(self):
        self._client.close()


# =============================================================================
# Semantic Scholar backward-reference retrieval (added for D5 oracle build)
# =============================================================================


@dataclass
class S2Reference:
    """One paper cited by another. Subset of S2 schema."""

    paper_id: str            # S2 paperId
    title: str
    authors: list[str]       # last names only
    year: int | None
    abstract: str            # may be empty if S2 doesn't have it
    tldr: str                # S2 TLDR if available
    citation_count: int
    arxiv_id: str | None     # if externalIds.ArXiv exists
    doi: str | None          # if externalIds.DOI exists
    venue: str
    publication_date: str | None = None  # ISO YYYY-MM-DD if S2 has it

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "tldr": self.tldr,
            "citation_count": self.citation_count,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "venue": self.venue,
            "publication_date": self.publication_date,
        }


class S2BackwardRetriever:
    """Synchronous Semantic Scholar client for fetching backward references.

    Given an arxiv paper, retrieves all papers it cites with metadata.
    Used for D5 oracle abstraction build (Phase 0b).

    No authentication required, but rate-limited 100 req/5min unauth.
    """

    def __init__(self):
        self._client = httpx.Client(
            timeout=httpx.Timeout(S2_TIMEOUT, connect=10.0),
            headers={"User-Agent": "CoScientist/1.0 (D5 oracle build)"},
        )
        self._last_call_time = 0.0
        self._total_calls = 0

    def _rate_limit(self):
        now = time.monotonic()
        wait = S2_RATE_DELAY - (now - self._last_call_time)
        if wait > 0:
            time.sleep(wait)
        self._last_call_time = time.monotonic()

    def _get_with_retry(self, url: str, params: dict | None = None) -> dict | None:
        """GET with rate limiting + exponential backoff on retryable status."""
        self._rate_limit()
        self._total_calls += 1
        delay = S2_BASE_DELAY
        for attempt in range(S2_MAX_RETRIES + 1):
            try:
                resp = self._client.get(url, params=params)
                if resp.status_code in S2_RETRYABLE_STATUS:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else delay
                    wait = min(wait, S2_MAX_DELAY)
                    logger.warning(
                        "S2 %d at %s (attempt %d/%d), retrying in %.1fs",
                        resp.status_code, url[:80], attempt + 1, S2_MAX_RETRIES + 1, wait,
                    )
                    time.sleep(wait)
                    delay = min(delay * 2, S2_MAX_DELAY)
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.ConnectError:
                logger.warning("S2 connection failed at %s (attempt %d)", url[:80], attempt + 1)
                if attempt < S2_MAX_RETRIES:
                    time.sleep(delay)
                    delay = min(delay * 2, S2_MAX_DELAY)
            except Exception as e:
                logger.warning("S2 error at %s: %s: %s", url[:80], type(e).__name__, e)
                return None
        logger.warning("S2 retries exhausted for %s", url[:80])
        return None

    def fetch_batch(self, ids: list[str]) -> dict[str, S2Reference | None]:
        """Batch fetch papers by S2 paperId or 'arXiv:XXXX' ID.

        Uses S2 /paper/batch POST endpoint — much more efficient than per-paper GET.
        Up to 500 IDs per call.
        """
        if not ids:
            return {}
        url = f"{S2_BASE_URL}/paper/batch"
        params = {
            "fields": "paperId,title,authors,year,abstract,tldr,citationCount,externalIds,venue",
        }
        delay = S2_BASE_DELAY
        self._rate_limit()
        self._total_calls += 1
        for attempt in range(S2_MAX_RETRIES + 1):
            try:
                resp = self._client.post(url, params=params, json={"ids": ids[:500]})
                if resp.status_code in S2_RETRYABLE_STATUS:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else delay
                    wait = min(wait, S2_MAX_DELAY)
                    logger.warning(
                        "S2 batch %d (attempt %d/%d), retrying in %.1fs",
                        resp.status_code, attempt + 1, S2_MAX_RETRIES + 1, wait,
                    )
                    time.sleep(wait)
                    delay = min(delay * 2, S2_MAX_DELAY)
                    continue
                resp.raise_for_status()
                data = resp.json()  # list aligned with input ids; null for missing
                out: dict[str, S2Reference | None] = {}
                for in_id, rec in zip(ids[:500], data):
                    if rec is None:
                        out[in_id] = None
                    else:
                        out[in_id] = self._parse_paper_record(rec)
                return out
            except Exception as e:
                logger.warning("S2 batch error: %s: %s", type(e).__name__, e)
                return {i: None for i in ids[:500]}
        return {i: None for i in ids[:500]}

    def search_by_title(self, title: str, year_hint: int | None = None) -> S2Reference | None:
        """Resolve a paper by title (best-match search).

        Used as fallback when arxiv ID isn't available in the reference text.
        S2 graph endpoint /paper/search returns top matches.
        """
        if not title.strip():
            return None
        url = f"{S2_BASE_URL}/paper/search"
        params = {
            "query": title.strip()[:300],
            "limit": 1,
            "fields": "paperId,title,authors,year,abstract,tldr,citationCount,externalIds,venue",
        }
        data = self._get_with_retry(url, params)
        if not data or not data.get("data"):
            return None
        return self._parse_paper_record(data["data"][0])

    def fetch_paper_by_id(self, paper_id_or_arxiv: str) -> S2Reference | None:
        """Fetch a single paper's metadata by paperId or 'arXiv:XXXX.XXXXX' format."""
        url = f"{S2_BASE_URL}/paper/{paper_id_or_arxiv}"
        params = {
            "fields": "paperId,title,authors,year,abstract,tldr,citationCount,externalIds,venue",
        }
        data = self._get_with_retry(url, params)
        if not data:
            return None
        return self._parse_paper_record(data)

    def _parse_paper_record(self, rec: dict) -> S2Reference | None:
        """Convert a S2 paper record into S2Reference."""
        if not rec.get("paperId"):
            return None
        external = rec.get("externalIds") or {}
        tldr_obj = rec.get("tldr") or {}
        authors = rec.get("authors") or []
        author_lastnames = []
        for a in authors[:3]:
            name = (a.get("name") or "").strip()
            if name:
                author_lastnames.append(name.split()[-1])
        return S2Reference(
            paper_id=rec["paperId"],
            title=(rec.get("title") or "").strip(),
            authors=author_lastnames,
            year=rec.get("year"),
            abstract=(rec.get("abstract") or "").strip(),
            tldr=(tldr_obj.get("text") or "").strip() if isinstance(tldr_obj, dict) else "",
            citation_count=rec.get("citationCount") or 0,
            arxiv_id=external.get("ArXiv"),
            doi=external.get("DOI"),
            venue=(rec.get("venue") or "").strip(),
            publication_date=rec.get("publicationDate"),
        )

    def fetch_paper_bibliography(self, arxiv_id: str, limit: int = 100) -> list[S2Reference]:
        """Fetch all papers that the arxiv paper cites (backward references).

        Args:
            arxiv_id: arxiv ID like "2601.16175" (no version suffix needed)
            limit: max refs to return (S2 caps at 1000; D5 expects ~30-60)

        Returns:
            List of S2Reference. Empty list on failure.
        """
        url = f"{S2_BASE_URL}/paper/arXiv:{arxiv_id}/references"
        params = {
            "limit": min(limit, 1000),
            "fields": "citedPaper.paperId,citedPaper.title,citedPaper.authors,citedPaper.year,"
                       "citedPaper.abstract,citedPaper.tldr,citedPaper.citationCount,"
                       "citedPaper.externalIds,citedPaper.venue,citedPaper.publicationDate",
        }
        data = self._get_with_retry(url, params)
        if not data:
            return []

        refs = self._parse_paper_list(data.get("data", []), nested_key="citedPaper")
        logger.info("S2 fetched %d references for arxiv:%s", len(refs), arxiv_id)
        return refs

    def fetch_paper_incoming_citations(
        self, paper_id_or_arxiv: str, limit: int = 200
    ) -> list[S2Reference]:
        """Fetch all papers that CITE this paper (forward citations / follow-ups).

        Args:
            paper_id_or_arxiv: S2 paperId or "arXiv:XXXX.XXXXX" format. Bare
                arxiv IDs (e.g. "2601.16175") are auto-prefixed with "arXiv:".
            limit: max citers to return (S2 caps at 1000; for follow-up
                discovery typically 100-200 is enough)

        Returns:
            List of S2Reference, one per citing paper. Empty list on failure.
        """
        if paper_id_or_arxiv and "/" not in paper_id_or_arxiv and ":" not in paper_id_or_arxiv:
            paper_id_or_arxiv = f"arXiv:{paper_id_or_arxiv}"
        url = f"{S2_BASE_URL}/paper/{paper_id_or_arxiv}/citations"
        params = {
            "limit": min(limit, 1000),
            # S2 /citations does NOT support citingPaper.tldr (unlike citedPaper.tldr in /references)
            "fields": "citingPaper.paperId,citingPaper.title,citingPaper.authors,citingPaper.year,"
                       "citingPaper.abstract,citingPaper.citationCount,"
                       "citingPaper.externalIds,citingPaper.venue,citingPaper.publicationDate",
        }
        data = self._get_with_retry(url, params)
        if not data:
            return []

        refs = self._parse_paper_list(data.get("data", []), nested_key="citingPaper")
        logger.info("S2 fetched %d incoming citations for %s", len(refs), paper_id_or_arxiv)
        return refs

    def _parse_paper_list(self, entries: list[dict], nested_key: str) -> list[S2Reference]:
        """Shared parser for /references and /citations response shapes.

        Both endpoints return data[]. each entry has either citedPaper (refs)
        or citingPaper (citations) as nested key.
        """
        out: list[S2Reference] = []
        for entry in entries:
            paper = entry.get(nested_key) or {}
            if not paper.get("paperId"):
                continue
            external = paper.get("externalIds") or {}
            tldr_obj = paper.get("tldr") or {}
            authors = paper.get("authors") or []
            author_lastnames = []
            for a in authors[:3]:
                name = (a.get("name") or "").strip()
                if name:
                    author_lastnames.append(name.split()[-1])
            out.append(S2Reference(
                paper_id=paper["paperId"],
                title=(paper.get("title") or "").strip(),
                authors=author_lastnames,
                year=paper.get("year"),
                abstract=(paper.get("abstract") or "").strip(),
                tldr=(tldr_obj.get("text") or "").strip() if isinstance(tldr_obj, dict) else "",
                citation_count=paper.get("citationCount") or 0,
                arxiv_id=external.get("ArXiv"),
                doi=external.get("DOI"),
                venue=(paper.get("venue") or "").strip(),
                publication_date=paper.get("publicationDate"),
            ))
        return out

    @property
    def total_api_calls(self) -> int:
        return self._total_calls

    def close(self):
        self._client.close()


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for with on at by from as into "
    "through during before after above below between under and but or nor not "
    "no so yet both either neither each every all any few more most other some "
    "such than too very that this these those it its they them their he she his "
    "her we our you your which what who whom how when where why if then also "
    "about up out just only already still even back well much now new also".split()
)


_BROAD_TERMS = frozenset(
    "model models neural network networks learning training data dataset "
    "method methods approach approaches system systems framework algorithm "
    "research work paper study performance results based using propose proposed "
    "show shown recent existing current modern practice practitioners "
    "artificial intelligence machine deep".split()
)


def goal_to_queries(goal: str, n_queries: int = 3) -> list[str]:
    """Extract keyword-based search queries from goal text (no model needed).

    Strategy: find cited papers + extract technical noun phrases.
    """
    queries: list[str] = []

    # Strategy 1: extract inline citations with surrounding context
    cite_re = re.compile(r'([A-Z][a-z]+(?:\s+et\s+al\.?)?)[,\s]+(?:[A-Z]+\s+)?(\d{4})')
    for m in cite_re.finditer(goal):
        author, year = m.group(1).strip().rstrip(","), m.group(2)
        start = max(0, m.start() - 80)
        context_words = re.findall(r'[a-zA-Z]{4,}', goal[start:m.start()].lower())
        context_kw = [w for w in context_words if w not in _STOPWORDS and w not in _BROAD_TERMS][-3:]
        queries.append(f"{author} {year} {' '.join(context_kw)}".strip())

    # Strategy 2: extract parenthetical technical terms  (term1, term2, ...)
    paren_re = re.compile(r'\(([^)]{10,80})\)')
    for m in paren_re.finditer(goal):
        content = m.group(1)
        if re.search(r'\d{4}', content):
            continue
        words = re.findall(r'[a-zA-Z]{3,}', content.lower())
        technical = [w for w in words if w not in _STOPWORDS and w not in _BROAD_TERMS]
        if len(technical) >= 2:
            queries.append(" ".join(technical[:5]))

    # Strategy 3: first sentence topic (usually states core problem)
    if len(queries) < n_queries:
        first_sent = re.split(r'(?<=[.!?])\s+', goal.strip())[0]
        words = re.findall(r'[a-zA-Z]{3,}', first_sent.lower())
        technical = [w for w in words if w not in _STOPWORDS and w not in _BROAD_TERMS]
        if len(technical) >= 2:
            queries.append(" ".join(technical[:5]))

    return queries[:n_queries]


def citation_match_rate(plan_text: str, papers: list[PaperResult]) -> float:
    """Fraction of retrieved papers cited in the plan (by surname + year)."""
    if not papers or not plan_text:
        return 0.0
    text_lower = plan_text.lower()
    matched = 0
    for p in papers:
        if not p.authors or not p.year:
            continue
        surname = p.authors[0].lower()
        year_str = str(p.year)
        if surname in text_lower and year_str in text_lower:
            matched += 1
    return matched / len(papers)


def extract_search_queries(model_output: str, max_queries: int = 3) -> list[str]:
    # Search the FULL output first (Qwen3 may put <search> inside <think>)
    matches = _SEARCH_RE.findall(model_output)
    if not matches:
        text = _THINK_RE.sub("", model_output)
        matches = _SEARCH_RE.findall(text)
    queries = [m.strip() for m in matches if m.strip()]
    return queries[:max_queries]


def format_papers_for_prompt(
    papers: list[PaperResult],
    max_papers: int = 10,
    max_chars: int = 2000,
) -> str:
    if not papers:
        return ""

    sorted_papers = sorted(papers, key=lambda p: p.citation_count, reverse=True)

    lines = []
    total_chars = 0
    for p in sorted_papers[:max_papers]:
        author_str = ", ".join(p.authors[:3])
        if len(p.authors) > 3:
            author_str += " et al."
        year_str = f" ({p.year})" if p.year else ""
        venue_str = f". {p.venue}" if p.venue else ""
        cite_str = f" [{p.citation_count} citations]" if p.citation_count else ""

        abstract_sentences = _first_sentences(p.abstract, max_sentences=2)

        entry = (
            f"- {author_str}{year_str}. \"{p.title}\"{venue_str}.{cite_str}\n"
            f"  Abstract: {abstract_sentences}"
        )

        if total_chars + len(entry) > max_chars:
            break
        lines.append(entry)
        total_chars += len(entry)

    return "\n\n".join(lines)


def _first_sentences(text: str, max_sentences: int = 2) -> str:
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = " ".join(sentences[:max_sentences])
    if len(result) > 400:
        result = result[:397] + "..."
    return result


QUERY_GENERATION_PROMPT = """\
# Research Goal
{goal}

# Task
You are preparing to write a research proposal for the goal above. \
Before writing, you need to find relevant published papers to ground \
your proposal with real citations and evidence.

Generate 2-3 search queries that would retrieve the most relevant \
academic papers. Each query should target a different aspect:
1. Core technical problem or methodology
2. Key prior work or baselines in this area
3. Application domain or related empirical findings

Wrap each query in <search> tags. Queries should be 3-8 words, \
suitable for an academic paper search engine.

<search>your first query</search>
<search>your second query</search>
<search>your third query</search>
/no_think"""


REVISION_QUERY_GENERATION_PROMPT = """\
# Research Goal
{goal}

# Current Plan Weaknesses
The plan scored {g3_score}/5 on Technical Evidence (citations and \
specific evidence supporting claims).

Grader feedback: {g3_critique}

# Task
Generate 2-3 search queries to find specific papers that would \
address the evidence gaps above. Focus on finding:
1. Papers providing the missing citations
2. Papers with quantitative results relevant to the claimed approach
3. Papers establishing baseline methods mentioned

<search>your first query</search>
<search>your second query</search>
<search>your third query</search>
/no_think"""
