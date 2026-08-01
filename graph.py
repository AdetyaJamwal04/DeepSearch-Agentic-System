import sys
import os
from datetime import datetime
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import asyncio

sys.stdout.reconfigure(encoding='utf-8')

from schemas.schema import QuerySynthesis, SubQuestion, SearchQuery, SearchResult, SubQuestionVerdict
from agents.knowledge_store import KnowledgeStore
from agents.query_synthesizer import synthesize_query
from agents.subquestion_generator import generate_sub_questions
from agents.search_query_generator import generate_search_queries
from agents.search_executor import execute_and_clean_searches
from agents.content_processor import extract_evidence
from agents.reflection import reflect
from agents.report_synthesizer import synthesize_report

MAX_ROUNDS = 3

class ResearchState(TypedDict):
    query: str
    synthesis: QuerySynthesis
    sub_questions: List[SubQuestion]
    search_queries: List[SearchQuery]
    search_results: List[SearchResult]
    store: KnowledgeStore
    verdicts: List[SubQuestionVerdict]
    round_num: int
    report_markdown: str

async def synthesize_node(state: ResearchState) -> dict:
    print("\n[Node: Synthesize] Analyzing original query...")
    synthesis = await synthesize_query(state["query"])
    if synthesis is None:
        raise ValueError("Query synthesis failed after all retries. Cannot proceed.")
    print(f"  -> Intent: {synthesis.intent}, Architecture: {synthesis.query_architecture}")
    return {"synthesis": synthesis}

async def decompose_node(state: ResearchState) -> dict:
    print("\n[Node: Decompose] Breaking down query into sub-questions...")
    sqs = await generate_sub_questions(state["synthesis"])
    store = KnowledgeStore(sub_questions=sqs)
    print(f"  -> Generated {len(sqs)} sub-questions.")
    return {"sub_questions": sqs, "store": store, "round_num": 1}

async def generate_queries_node(state: ResearchState) -> dict:
    round_num = state.get("round_num", 1)
    print(f"\n============================================================")
    print(f"  ROUND {round_num} of {MAX_ROUNDS}")
    print(f"============================================================")
    print("[Node: Generate Queries] Pivoting and creating search strings...")
    
    verdicts = state.get("verdicts", [])
    store = state["store"]
    
    if round_num == 1:
        target_sqs = state["sub_questions"]
        search_queries = await generate_search_queries(target_sqs, round_num=1)
    else:
        weak_sqs = [sq for sq in state["sub_questions"] if any(v.sub_question_id == sq.id and v.verdict == "needs_more" for v in verdicts)]
        
        suggested_angles = {v.sub_question_id: v.suggested_angles for v in verdicts if v.verdict == "needs_more"}
        excluded = store.previously_searched_strings()
        
        search_queries = await generate_search_queries(
            weak_sqs, 
            round_num=round_num, 
            suggested_angles=suggested_angles, 
            exclude_queries=excluded
        )
        
    store.search_queries.extend(search_queries)
    print(f"  -> Generated {len(search_queries)} queries.")
    return {"search_queries": search_queries}

async def execute_searches_node(state: ResearchState) -> dict:
    print(f"\n[Node: Search] Hitting web search API for {len(state['search_queries'])} queries...")
    results = await execute_and_clean_searches(state["search_queries"])
    executed = sum(1 for q in state['search_queries'] if getattr(q, 'status', 'failed') == 'executed')
    print(f"  -> Returned {len(results)} clean pages ({executed} queries successfully executed).")
    return {"search_results": results}

async def extract_claims_node(state: ResearchState) -> dict:
    print(f"\n[Node: Extract] Parsing pages and extracting evidence claims...")
    evidence = await extract_evidence(state["sub_questions"], state["search_results"])
    store = state["store"]
    store.evidence.extend(evidence)
    print(f"  -> Extracted {len(evidence)} new claims.")
    return {"store": store}

async def reflect_node(state: ResearchState) -> dict:
    print(f"\n[Node: Reflect] Evaluating Knowledge Store coverage...")
    verdicts = await reflect(state["store"])
    weak = [v for v in verdicts if v.verdict == "needs_more"]
    if weak:
        print(f"  -> Identified {len(weak)} sub-questions lacking sufficient evidence.")
    else:
        print(f"  -> All sub-questions have sufficient evidence!")
    return {"verdicts": verdicts, "round_num": state["round_num"] + 1}

def should_continue(state: ResearchState) -> str:
    if state["round_num"] > MAX_ROUNDS:
        return "report"
    
    all_sufficient = all(v.verdict == "sufficient" for v in state.get("verdicts", []))
    if all_sufficient:
        return "report"
        
    return "generate_queries"

async def report_node(state: ResearchState) -> dict:
    print(f"\n============================================================")
    print(f"  Final Report Synthesis")
    print(f"============================================================")
    print(f"[Node: Report] Generating detailed markdown report from {len(state['store'].evidence)} claims...")
    report = await synthesize_report(state["query"], state["store"])
    print(f"\n✅ DeepSearch complete!")
    return {"report_markdown": report}

# Graph Construction
workflow = StateGraph(ResearchState)

workflow.add_node("synthesize", synthesize_node)
workflow.add_node("decompose", decompose_node)
workflow.add_node("generate_queries", generate_queries_node)
workflow.add_node("execute_searches", execute_searches_node)
workflow.add_node("extract_claims", extract_claims_node)
workflow.add_node("reflect", reflect_node)
workflow.add_node("report", report_node)

workflow.set_entry_point("synthesize")
workflow.add_edge("synthesize", "decompose")
workflow.add_edge("decompose", "generate_queries")
workflow.add_edge("generate_queries", "execute_searches")
workflow.add_edge("execute_searches", "extract_claims")
workflow.add_edge("extract_claims", "reflect")

workflow.add_conditional_edges(
    "reflect",
    should_continue,
    {
        "report": "report",
        "generate_queries": "generate_queries"
    }
)

workflow.add_edge("report", END)

app = workflow.compile()

if __name__ == "__main__":
    async def run_research():
        test_query = "Impact of mephentermin on digestion, hormonal health and overall body functioning. Describe in detail the health complications it might lead to over a perdio of continuous use."
        print(f"Starting LangGraph run for query: '{test_query}'")

        inputs = {"query": test_query}
        final_state = None
        async for state in app.astream(inputs, stream_mode="values"):
            final_state = state

        # Save report to disk only when running as CLI
        report = final_state.get("report_markdown", "") if final_state else ""
        if report:
            os.makedirs("reports", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/research_report_langgraph_{timestamp}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Report saved to: {filename}")

    asyncio.run(run_research())
