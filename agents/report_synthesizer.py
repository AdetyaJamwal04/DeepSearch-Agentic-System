from typing import Dict, Tuple
import asyncio
from models.model import chat
from agents.knowledge_store import KnowledgeStore

SYSTEM_PROMPT = """You are an expert research analyst. Your job is to synthesize raw evidence claims into a detailed, highly structured, and strictly factual research report.

You will be given the original research query, the structural sub-questions that guided the research, and the extracted evidence claims. Each claim is labeled with a source ID (e.g., [1], [2]).

Report Requirements:
1. Structure:
   - Title (derived from the query)
   - Executive Summary (high-level synthesis of the findings)
   - Detailed thematic sections (synthesize the sub-questions into flowing analytical prose, don't just list them as Q&A)
   - Conclusion
   - References (A numbered list of the sources used, matching the provided Source Mapping)
2. Tone: Strictly factual, objective, and analytical. Do not add fluff, speculation, or commentary not supported by the evidence.
3. Citations: You MUST use inline citations (e.g., [1], [2]) whenever you state a fact derived from the evidence. Every claim must be tied to its source.
4. Format: Use proper Markdown formatting (headers, bullet points, bold text).
"""

def _build_evidence_block(store: KnowledgeStore) -> Tuple[str, Dict[str, int]]:
    """
    Builds the formatted evidence block for the prompt and generates a 
    consistent mapping of URLs to citation IDs (e.g., [1]).
    """
    url_to_id = {}
    next_id = 1
    
    blocks = []
    for sq in store.sub_questions:
        claims = store.evidence_for(sq.id)
        if not claims:
            continue
            
        blocks.append(f"### Theme: {sq.text}")
        for claim in claims:
            url = claim.source_url
            if url not in url_to_id:
                url_to_id[url] = next_id
                next_id += 1
            
            citation_id = url_to_id[url]
            blocks.append(f"- {claim.claim} [{citation_id}]")
        blocks.append("")
        
    return "\n".join(blocks), url_to_id

async def synthesize_report(query: str, store: KnowledgeStore) -> str:
    """
    Takes the accumulated knowledge and generates a final markdown report.
    """
    evidence_block, url_to_id = _build_evidence_block(store)
    
    sorted_urls = sorted(url_to_id.items(), key=lambda x: x[1])
    mapping_block = "\n".join([f"[{cit_id}] {url}" for url, cit_id in sorted_urls])
    
    user_prompt = f"""Original Query: {query}

Evidence Collected:
{evidence_block}

Source Mapping:
{mapping_block}

Please generate the detailed research report now.
"""

    for attempt in range(5):
        try:
            response = await chat.ainvoke(
                [
                    ("system", SYSTEM_PROMPT),
                    ("user", user_prompt),
                ]
            )
            return response.content
        except Exception as e:
            print(f"[report_synthesizer] LLM call failed (attempt {attempt+1}/5): {e}")
            if attempt < 4:
                await asyncio.sleep(2 ** attempt)
    
    return f"# Error generating report\n\nFailed to synthesize report after multiple attempts."
