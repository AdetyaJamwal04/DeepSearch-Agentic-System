import asyncio
from typing import Optional
from models.model import chat
from schemas.schema import QuerySynthesis


async def synthesize_query(query: str) -> Optional[QuerySynthesis]:

    prompt = f"""
You are an expert research query analyst.

Your task is to analyze the user's query and extract structured information.

User Query:
{query}

Extract:

1. original_query
   - The user's query verbatim.

2. intent
   - The primary intent behind the query.
   - Examples:
     - explanation
     - comparison
     - analysis
     - recommendation
     - factual_lookup
     - troubleshooting
     - research

3. objective
   - What the user ultimately wants to learn, understand, compare, or achieve.

4. query_architecture
   - multi_question: the objective requires researching 2 or more genuinely
     independent facets to be fully answered. This includes comparisons
     between 2+ entities, trend analyses, evaluations against multiple
     criteria, or any objective with multiple distinct sub-parts.
   - single_question: the objective is one coherent question that does not
     naturally split into independent research threads (e.g., "why does X
     happen", definitional lookups, narrow factual questions).
   - Default rule: if intent is "comparison", "evaluation", or
     "trend_analysis" and involves 2 or more named entities, criteria, or
     time periods, classify as multi_question — unless the comparison is a
     single trivial fact (e.g., "is X faster than Y?").

5. scope
   - One of:
     - narrow
     - moderate
     - broad

6. entities
   - Important entities, concepts, technologies, organizations, people, products, or topics mentioned.

7. ambiguity
   - True if the query lacks sufficient context or can reasonably be interpreted in multiple ways.
   - Otherwise False.

8. ambiguity_reason
   - Explain the ambiguity if present.
   - Empty string if not ambiguous.

Return only the structured output matching the provided schema.
"""

    for attempt in range(5):
        try:
            result = await chat.with_structured_output(QuerySynthesis).ainvoke(
                [
                    ("system", prompt),
                    ("user", "start extracting structured output.")
                ]
            )
            return result
        except Exception as e:
            print(f"[query_synthesizer] LLM call failed (attempt {attempt+1}/5): {e}")
            if attempt < 4:
                await asyncio.sleep(2 ** attempt)
            else:
                return None

    return None


if __name__ == "__main__":
    import asyncio as _asyncio

    async def _test():
        result = await synthesize_query("why does gradient descent zig-zag?")
        print(result)

    _asyncio.run(_test())
