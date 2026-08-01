import re
import asyncio
from typing import List
from schemas.schema import SearchQuery, SearchResult
from models.web_client import client as tavily_client

TAVILY_SEMAPHORE = asyncio.Semaphore(5)


_MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\([^)]*\)")

_PDF_ARTIFACT_MARKERS = (
    "endobj", "endstream", "<rdf:rdf>", "xmpmeta",
    "<pdf:producer>", "extensisfontsense", "<xmp:createdate>",
)


def _is_low_quality_content(
    content: str,
    min_length: int = 200,
    max_link_coverage: float = 0.3,
    min_alpha_ratio: float = 0.6,
) -> bool:
    """
    Flags content that's unusable for downstream synthesis. Catches three
    distinct failure modes seen in real Tavily output:

      1. Too short to carry information.
      2. Mostly markdown link/image syntax -- aggregator sign-in prompts,
         footer recirculation widgets, image-fetch URL walls (Medium,
         Substack, etc.). Measured as the FRACTION of total characters
         consumed by markdown link/image patterns, not how often a link
         marker appears -- a handful of very long URLs can dominate content
         length without producing many marker occurrences.
      3. Garbled non-prose content from failed PDF text extraction -- raw
         PDF internal structure (XMP metadata, font tables, binary stream
         bytes) instead of the actual article text. Caught two ways: a
         marker check for PDF-internal tokens, and an alphabetic-character
         ratio check as a general-purpose backstop for other binary-
         extraction failures that don't contain these exact markers.
    """
    if not content or len(content) < min_length:
        return True

    lowered = content.lower()
    if any(marker in lowered for marker in _PDF_ARTIFACT_MARKERS):
        return True

    link_matches = _MARKDOWN_LINK_PATTERN.findall(content)
    link_chars = sum(len(m) for m in link_matches)
    if link_chars / len(content) > max_link_coverage:
        return True

    alpha_chars = sum(1 for c in content if c.isalpha())
    if alpha_chars / len(content) < min_alpha_ratio:
        return True

    return False


async def _execute_single_query(query_string: str, max_results: int = 5) -> dict:
    """
    Calls Tavily for a single query string and returns the raw response dict.
    """
    async with TAVILY_SEMAPHORE:
        return await tavily_client.search(query=query_string, max_results=max_results)


async def execute_and_clean_searches(
    search_queries: List[SearchQuery],
    max_results: int = 5,
) -> List[SearchResult]:
    """
    Main entry point. For each SearchQuery:
      1. Calls Tavily with its query_string
      2. Filters out low-quality/junk results
      3. Maps surviving results into SearchResult objects
      4. Updates SearchQuery.status to "executed" or "failed" in place
    """
    results: List[SearchResult] = []

    async def fetch_and_process(sq: SearchQuery):
        try:
            response = await _execute_single_query(sq.query_string, max_results=max_results)
        except Exception as e:
            print(f"[search_executor] Tavily search failed for '{sq.query_string}': {e}")
            sq.status = "failed"
            return []

        raw_results = response.get("results", [])

        if not raw_results:
            sq.status = "failed"
            return []

        survived_count = 0
        local_results = []
        for item in raw_results:
            content = item.get("content", "") or ""

            if _is_low_quality_content(content):
                continue

            survived_count += 1
            local_results.append(
                SearchResult(
                    search_query_id=sq.id,
                    sub_question_id=sq.sub_question_id,
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    content=content,
                    score=item.get("score", 0.0),
                )
            )

        sq.status = "executed" if survived_count > 0 else "failed"
        return local_results

    # Run all searches concurrently
    all_local_results = await asyncio.gather(*(fetch_and_process(sq) for sq in search_queries))
    
    for r in all_local_results:
        results.extend(r)

    return results