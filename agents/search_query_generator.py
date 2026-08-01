import asyncio
from typing import List, Dict, Optional, Set
from pydantic import BaseModel
from schemas.schema import SubQuestion, SearchQuery
from models.model import chat


class _RawSearchQuery(BaseModel):
    sub_question_id: str
    query_string: str
    angle: str


class SearchQueryBatch(BaseModel):
    """
    Container schema. Same reason as SubQuestionList -- binding directly to
    _RawSearchQuery would constrain the LLM to returning exactly one query
    for the entire batch instead of several per sub-question.
    """
    queries: List[_RawSearchQuery]


SYSTEM_PROMPT = """You are a search query generation specialist. You will be given a batch of
sub-questions, each belonging to a research objective with a known intent.
Your job is to generate diverse search engine query strings for EACH
sub-question so that a search API can retrieve good source material.

For every sub-question, generate 2 to 4 search query strings. Each query
must explore a different angle, not just be a rephrasing of another query
for the same sub-question.

Choose angles based on the sub-question's intent:
- explanation: intuitive, mathematical/technical, visual/diagrammatic
- comparison: benchmark/quantitative, technical/architectural mechanism, practical/use-case
- trend_analysis: historical baseline, recent developments, expert outlook
- evaluation: criteria-based, critique/limitations, real-world case study
- causal: direct mechanism, contributing factors, edge case/exception

Rules:
- Every sub_question_id provided MUST receive at least 2 queries. Do not skip any.
- Do not output the sub-question text verbatim as a query. Rephrase into
  concise, keyword-rich search strings a person would actually type into a
  search engine.
- Avoid generating near-duplicate queries across DIFFERENT sub-questions in
  this batch. If two sub-questions would naturally produce a similar query,
  adjust phrasing so each stays distinct to its own facet.
- Do not answer the sub-questions. Only generate queries.
"""

USER_PROMPT_TEMPLATE = """Generate search queries for the following sub-questions:

{sub_questions_block}
"""

RESEARCH_CONTEXT_TEMPLATE = """
Additional context for this round of searching:

The following search strings have ALREADY been used in previous rounds.
Do NOT generate any query that is identical or near-identical to these:
{excluded_queries_block}

The reflection agent identified these gaps and suggested these search angles.
Prioritize generating queries that target these specific angles:
{suggested_angles_block}
"""


def _build_sub_questions_block(sub_questions: List[SubQuestion]) -> str:
    blocks = [
        f"[{sq.id}] intent: {sq.parent_intent}\nText: {sq.text}"
        for sq in sub_questions
    ]
    return "\n\n".join(blocks)


async def _call_llm_for_search_queries(
    sub_questions: List[SubQuestion],
    suggested_angles: Optional[Dict[str, List[str]]] = None,
    exclude_queries: Optional[Set[str]] = None,
) -> List[_RawSearchQuery]:
    sub_questions_block = _build_sub_questions_block(sub_questions)
    user_prompt = USER_PROMPT_TEMPLATE.format(sub_questions_block=sub_questions_block)

    # On round 2+, append re-search context to the prompt
    if suggested_angles or exclude_queries:
        angles_block = "\n".join(
            f"  [{sq_id}] {', '.join(angles)}"
            for sq_id, angles in (suggested_angles or {}).items()
            if angles
        ) or "  (none)"

        excluded_block = "\n".join(
            f"  - {q}" for q in sorted(exclude_queries or set())
        ) or "  (none)"

        user_prompt += RESEARCH_CONTEXT_TEMPLATE.format(
            excluded_queries_block=excluded_block,
            suggested_angles_block=angles_block,
        )

    for attempt in range(5):
        try:
            result = await chat.with_structured_output(SearchQueryBatch).ainvoke(
                [
                    ("system", SYSTEM_PROMPT),
                    ("user", user_prompt),
                ]
            )
            return result.queries
        except Exception as e:
            print(f"[search_query_generator] LLM call failed (attempt {attempt+1}/5): {e}")
            if attempt < 4:
                await asyncio.sleep(2 ** attempt)
            else:
                return []
    
    return []


def _coverage_check_and_fill(
    sub_questions: List[SubQuestion],
    raw_queries: List[_RawSearchQuery],
) -> List[_RawSearchQuery]:
    """
    Ensures every sub_question_id has at least one query. If the LLM dropped
    a sub-question entirely, fill it with a single fallback query built
    directly from the sub-question's own text -- no extra LLM call.
    """
    covered_ids = {q.sub_question_id for q in raw_queries}
    filled = list(raw_queries)

    for sq in sub_questions:
        if sq.id not in covered_ids:
            filled.append(
                _RawSearchQuery(
                    sub_question_id=sq.id,
                    query_string=sq.text,
                    angle="fallback",
                )
            )

    return filled

async def generate_search_queries(
    sub_questions: List[SubQuestion],
    round_num: int = 1,
    suggested_angles: Optional[Dict[str, List[str]]] = None,
    exclude_queries: Optional[Set[str]] = None,
) -> List[SearchQuery]:
    """
    Main entry point. Takes the full list of sub-questions and returns a
    flat list of SearchQuery objects, generated via a single batched LLM call.

    On round 2+, suggested_angles and exclude_queries guide the LLM to
    generate new queries targeting identified gaps while avoiding repeats.
    """
    raw_queries = await _call_llm_for_search_queries(
        sub_questions,
        suggested_angles=suggested_angles,
        exclude_queries=exclude_queries,
    )
    raw_queries = _coverage_check_and_fill(sub_questions, raw_queries)

    # Group raw queries by sub_question_id, preserving the input sub_questions
    # order (i.e. priority order) rather than whatever order the LLM emitted them in.
    grouped: Dict[str, List[_RawSearchQuery]] = {sq.id: [] for sq in sub_questions}
    for item in raw_queries:
        grouped.setdefault(item.sub_question_id, []).append(item)

    search_queries: List[SearchQuery] = []
    for sq in sub_questions:
        for idx, item in enumerate(grouped.get(sq.id, []), start=1):
            search_queries.append(
                SearchQuery(
                    id=f"{sq.id}_r{round_num}_q{idx}",
                    sub_question_id=item.sub_question_id,
                    query_string=item.query_string,
                    angle=item.angle,
                    round=round_num,
                )
            )

    return search_queries