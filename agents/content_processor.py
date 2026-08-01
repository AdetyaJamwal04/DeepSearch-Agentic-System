import asyncio
from typing import List, Dict, Tuple
from pydantic import BaseModel
from schemas.schema import SubQuestion, SearchResult, Evidence
from models.model import chat


class _RawEvidence(BaseModel):
    source_id: str
    claim: str


class EvidenceBatch(BaseModel):
    """
    Container schema. Same reason as SubQuestionList and SearchQueryBatch --
    binding directly to _RawEvidence would constrain the LLM to returning
    exactly one claim for the entire batch instead of one per relevant source.
    """
    evidence: List[_RawEvidence]


SYSTEM_PROMPT = """You are an evidence extraction specialist. You will be given a sub-question
and a batch of source documents, each labeled with a short source id.

Your job is to extract the specific factual claims from these sources that
are relevant to answering the sub-question.

Rules:
- Extract claims only from sources that actually contain relevant
  information. If a source has nothing relevant to the sub-question
  (e.g., institutional boilerplate, marketing copy, navigation text, an
  unrelated topic), contribute ZERO claims from that source. Do not force
  an extraction just because the source is present in the batch.
- Each claim must be a concise, self-contained factual statement in your
  own words -- not a verbatim quote from the source.
- A single source may yield zero, one, or multiple claims, depending on how
  much relevant information it actually contains.
- Do not infer, speculate, or add information not present in the source
  content.
- Do not merge claims from different sources into one claim -- each claim
  must be traceable to exactly one source.
- Reference each source using its exact source id as given (e.g. "src_1").
  Do not alter, abbreviate, or invent source ids.
"""

USER_PROMPT_TEMPLATE = """Sub-question: {sub_question_text}

Extract relevant claims from the following sources:

{sources_block}
"""


def _dedupe_by_url(search_results: List[SearchResult]) -> List[SearchResult]:
    """
    Removes duplicate sources within a single sub-question's group. The same
    URL can legitimately appear under different sub-questions (a source
    relevant to two facets), but within ONE sub-question's batch, the same
    URL showing up multiple times (e.g. because two of its search queries
    both surfaced it) just means the LLM sees it under two different src_N
    labels and extracts the same fact twice -- sometimes worded slightly
    differently each time, which makes it hard to catch as a duplicate later.
    Deduping here, before the prompt is built, is the only place this is
    cheap and reliable to fix.
    """
    seen = set()
    deduped = []
    for sr in search_results:
        if sr.url in seen:
            continue
        seen.add(sr.url)
        deduped.append(sr)
    return deduped


def _build_sources_block(
    search_results: List[SearchResult],
) -> Tuple[str, Dict[str, SearchResult]]:
    """
    Labels each SearchResult with a short source id (src_1, src_2, ...) and
    returns both the formatted prompt block and a mapping from source_id back
    to the original SearchResult -- so the LLM never has to handle/reproduce
    actual URLs, and we look up the real url/title programmatically after
    parsing instead.
    """
    blocks = []
    source_map: Dict[str, SearchResult] = {}

    for i, sr in enumerate(search_results, start=1):
        source_id = f"src_{i}"
        source_map[source_id] = sr
        blocks.append(f"[{source_id}] {sr.title}\n{sr.content}")

    return "\n\n".join(blocks), source_map


async def _call_llm_for_evidence_extraction(
    sub_question_text: str,
    search_results: List[SearchResult],
) -> Tuple[List[_RawEvidence], Dict[str, SearchResult]]:
    sources_block, source_map = _build_sources_block(search_results)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        sub_question_text=sub_question_text,
        sources_block=sources_block,
    )

    for attempt in range(5):
        try:
            result = await chat.with_structured_output(EvidenceBatch).ainvoke(
                [
                    ("system", SYSTEM_PROMPT),
                    ("user", user_prompt),
                ]
            )
            return result.evidence, source_map
        except Exception as e:
            print(f"[content_processor] LLM evidence extraction failed (attempt {attempt+1}/5): {e}")
            if attempt < 4:
                await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s backoff
            else:
                return [], source_map
    
    return [], source_map


async def extract_evidence(
    sub_questions: List[SubQuestion],
    search_results: List[SearchResult],
) -> List[Evidence]:
    """
    Main entry point. Groups search_results by sub_question_id and runs one
    batched extraction call per sub-question (not per individual SearchResult
    -- same cost/latency reasoning as the search query generator).

    Sub-questions with zero surviving SearchResults are skipped entirely --
    no LLM call wasted on an empty batch.
    """
    grouped: Dict[str, List[SearchResult]] = {sq.id: [] for sq in sub_questions}
    for sr in search_results:
        grouped.setdefault(sr.sub_question_id, []).append(sr)

    async def _extract_for_sq(sq: SubQuestion) -> List[Evidence]:
        sq_results = grouped.get(sq.id, [])
        sq_results = _dedupe_by_url(sq_results)
        if not sq_results:
            return []

        raw_evidence, source_map = await _call_llm_for_evidence_extraction(sq.text, sq_results)

        evidence_items = []
        for item in raw_evidence:
            source = source_map.get(item.source_id)
            if source is None:
                # LLM referenced a source id that doesn't exist in this batch -- skip it
                continue

            evidence_items.append(
                Evidence(
                    sub_question_id=sq.id,
                    claim=item.claim,
                    source_url=source.url,
                    source_title=source.title,
                )
            )
        return evidence_items

    # Run all sub-question extractions concurrently
    results = await asyncio.gather(*(_extract_for_sq(sq) for sq in sub_questions))

    all_evidence: List[Evidence] = []
    for batch in results:
        all_evidence.extend(batch)

    return all_evidence