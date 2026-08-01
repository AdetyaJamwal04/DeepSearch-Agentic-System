import asyncio
from typing import List
from pydantic import BaseModel
from schemas.schema import SubQuestionVerdict
from models.model import chat
from agents.knowledge_store import KnowledgeStore


class VerdictBatch(BaseModel):
    """
    Container schema for structured output. Same pattern as SubQuestionList
    and SearchQueryBatch — binding to the list container rather than a single
    verdict so the LLM returns one verdict per sub-question.
    """
    verdicts: List[SubQuestionVerdict]


SYSTEM_PROMPT = """You are a research coverage evaluator. You will be given a set of
sub-questions along with the evidence that has been collected so far for each.

Your job is to evaluate whether the evidence is SUFFICIENT to answer each
sub-question, or whether more searching is needed.

For each sub-question, produce a verdict:

- "sufficient" if:
  - The evidence contains enough factual claims to construct a meaningful
    answer to the sub-question.
  - The evidence comes from more than one source (not single-source reliance).
  - The claims include specific details (numbers, mechanisms, names) rather
    than only vague or surface-level statements.

- "needs_more" if ANY of the following are true:
  - Zero or very few evidence claims (< 2).
  - All claims come from a single source.
  - The claims are shallow — they state a conclusion without supporting data,
    mechanisms, or specifics.
  - Important aspects of the sub-question are not addressed by any claim.
  - Claims contradict each other and more sources are needed to resolve the
    conflict.

When the verdict is "needs_more":
- In "gap", describe concisely what is missing or weak.
- In "suggested_angles", provide 1 to 3 specific search angles that would
  fill the gap. These should be concrete search directions, not vague
  restatements of the sub-question.

When the verdict is "sufficient":
- Set "gap" to an empty string.
- Set "suggested_angles" to an empty list.

Rules:
- Evaluate each sub-question independently.
- Do not be overly generous — "sufficient" means genuinely answerable, not
  just "something was found."
- Do not be overly strict — if there are 4+ specific claims from 2+ sources,
  that is usually sufficient for a single sub-question.
- Every sub_question_id in the input MUST appear in your output. Do not skip any.
"""

USER_PROMPT_TEMPLATE = """Evaluate the evidence coverage for each sub-question below.

{coverage_block}
"""


def _build_coverage_block(store: KnowledgeStore) -> str:
    """
    Builds the prompt block that shows each sub-question and its collected
    evidence to the reflection LLM.
    """
    blocks = []
    for sq in store.sub_questions:
        evidence_text = store.evidence_summary_for(sq.id)
        count = store.evidence_count(sq.id)
        queries_used = len(store.queries_for(sq.id))

        blocks.append(
            f"[{sq.id}] Sub-question: {sq.text}\n"
            f"Queries used so far: {queries_used}\n"
            f"Evidence claims ({count}):\n"
            f"{evidence_text}"
        )

    return "\n\n".join(blocks)


async def reflect(store: KnowledgeStore) -> List[SubQuestionVerdict]:
    """
    Main entry point. Takes the knowledge store and returns a verdict
    per sub-question via a single batched LLM call.
    """
    coverage_block = _build_coverage_block(store)
    user_prompt = USER_PROMPT_TEMPLATE.format(coverage_block=coverage_block)

    for attempt in range(5):
        try:
            result = await chat.with_structured_output(VerdictBatch).ainvoke(
                [
                    ("system", SYSTEM_PROMPT),
                    ("user", user_prompt),
                ]
            )
            return result.verdicts
        except Exception as e:
            print(f"[reflection] LLM call failed (attempt {attempt+1}/5): {e}")
            if attempt < 4:
                await asyncio.sleep(2 ** attempt)
            else:
                # On failure, mark everything as sufficient to avoid infinite loops
                return [
                    SubQuestionVerdict(
                        sub_question_id=sq.id,
                        verdict="sufficient",
                        gap="",
                        suggested_angles=[],
                    )
                    for sq in store.sub_questions
                ]
    
    # Fallback
    return [
        SubQuestionVerdict(
            sub_question_id=sq.id,
            verdict="sufficient",
            gap="",
            suggested_angles=[],
        )
        for sq in store.sub_questions
    ]
