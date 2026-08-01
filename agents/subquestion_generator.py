import asyncio
from typing import List
from models.model import chat
from pydantic import BaseModel
from schemas.schema import QuerySynthesis, SubQuestion

class _RawSubQuestion(BaseModel):
    text: str
    priority: int

class SubQuestionList(BaseModel):
    """
    Container schema. with_structured_output needs a schema that actually
    describes a LIST -- binding directly to _RawSubQuestion constrains the
    LLM to returning exactly one object, no matter what the prompt asks for.
    """
    sub_questions: List[_RawSubQuestion]


SYSTEM_PROMPT = """You are a research strategist decomposing a complex question into distinct,
independently researchable sub-questions.

Your job is NOT to answer the question. Your job is to break it into the
smallest set of genuinely distinct facets that, together, would let a
researcher fully address the objective.

Rules:
- Each sub-question must investigate a different facet. Do not produce
  rephrasings or near-duplicates of another sub-question.
- Each sub-question should be answerable through its own independent search
  - avoid sub-questions that depend on the answer to another sub-question
  first, unless the dependency is unavoidable.
- Order sub-questions by research priority: foundational/prerequisite facets
  first, narrower or derived facets later.
- Produce as many sub-questions as the objective genuinely requires -
  typically 3 to 6. Do not pad the list to hit a target count, and do not
  force decomposition where a facet doesn't naturally exist.
- Do not answer, explain, or add information. Only generate questions.
- Do NOT generate a sub-question that asks how the other facets relate to,
  influence, shape, or impact one another (e.g., "how have X and Y affected
  Z", "what is the relationship between A and B", "how do these factors
  shape the current state of..."). Synthesizing connections across facets
  is the job of the final report, not a research sub-question. If a
  candidate sub-question's main verb is "impacted", "shaped", "influenced",
  "affected", or similar, and its object is the conclusion of other facets
  rather than a new fact to look up, drop it.
  
Example of what NOT to produce (synthesis disguised as research):
  "How have past trade deals and historical tensions impacted the current
  state of India-USA relations?"
  -- this depends on the answers to the other sub-questions and doesn't
  introduce a new fact to search for. Do not generate sub-questions like this.

Example of what a genuine, independent facet looks like:
  "What are the primary economic sectors involved in the trade relationship
  between India and the USA?"
  -- this is answerable on its own, with its own dedicated search.
"""

USER_PROMPT_TEMPLATE = """Objective: {objective}
Primary intent: {intent}
Key entities: {entities}

Decompose this objective into distinct sub-questions following the rules above.

Decomposition pattern by intent (use the one matching the primary intent above,
or the closest match if intent doesn't appear here):
- comparison: per-entity characteristics first, then explicit comparison
  dimensions (performance, cost, limitations, use case fit)
- trend_analysis: historical baseline -> recent developments -> drivers of
  change -> outlook
- explanation: mechanism/definition -> causes -> evidence/examples ->
  implications
- evaluation: criteria for judgment -> evidence per criterion ->
  counterarguments/limitations
- causal: direct mechanism -> contributing factors -> conditions under which
  it doesn't hold
"""

async def _call_llm_for_decomposition(synthesis: QuerySynthesis) -> List[_RawSubQuestion]:
    """
    Calls the LLM to decompose a multi-question objective into sub-questions.

    Uses LangChain's with_structured_output, bound to SubQuestionList (a
    container schema) rather than _RawSubQuestion directly -- binding to the
    bare item schema would constrain the model to returning exactly one
    sub-question regardless of how many the objective actually needs.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        objective=synthesis.objective,
        intent=synthesis.intent,
        entities=", ".join(synthesis.entities) if synthesis.entities else "none provided",
    )

    for attempt in range(5):
        try:
            result = await chat.with_structured_output(SubQuestionList).ainvoke(
                [
                    ("system", SYSTEM_PROMPT),
                    ("user", user_prompt),
                ]
            )
            return result.sub_questions
        except Exception as e:
            print(f"[subquestion_generator] LLM call failed (attempt {attempt+1}/5): {e}")
            if attempt < 4:
                await asyncio.sleep(2 ** attempt)
            else:
                return []
    
    return []

async def generate_sub_questions(synthesis: QuerySynthesis) -> list[SubQuestion]:
    """
    Main entry point. Branches on synthesis.query_architecture.
    """
    if synthesis.query_architecture == "single_question":
        return [
            SubQuestion(
                id="sq_1",
                text=synthesis.objective,
                priority=1,
                parent_intent=synthesis.intent,
                parent_entities=synthesis.entities,
            )
        ]

    # multi_question branch
    raw_items = await _call_llm_for_decomposition(synthesis)

    if not raw_items:
        return [
            SubQuestion(
                id="sq_1",
                text=synthesis.objective,
                priority=1,
                parent_intent=synthesis.intent,
                parent_entities=synthesis.entities,
            )
        ]

    ordered = sorted(raw_items, key=lambda x: x.priority)
    return [
        SubQuestion(
            id=f"sq_{i + 1}",
            text=item.text,
            priority=item.priority,
            parent_intent=synthesis.intent,
            parent_entities=synthesis.entities,
        )
        for i, item in enumerate(ordered)
    ]