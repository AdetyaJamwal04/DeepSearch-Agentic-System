import sys
import os
import asyncio
from datetime import datetime
sys.stdout.reconfigure(encoding='utf-8')

from agents.query_synthesizer import synthesize_query
from agents.subquestion_generator import generate_sub_questions
from agents.search_query_generator import generate_search_queries
from agents.search_executor import execute_and_clean_searches
from agents.content_processor import extract_evidence
from agents.knowledge_store import KnowledgeStore
from agents.reflection import reflect
from agents.report_synthesizer import synthesize_report

MAX_ROUNDS = 3


async def main():
    query = "Write a detailed report on diplomatic ties of India and the USA, taking into consideration the historical angles, and the ongoing exchanges. Also, explore the angle of how trade affecting the relationship between the two."

    # ── Stage 1: Query Synthesis ─────────────────────────────────────
    synth = await synthesize_query(query)
    if synth is None:
        print("Query synthesis failed. Exiting.")
        return

    print(f"[Stage 1] Synthesis complete — intent: {synth.intent}, architecture: {synth.query_architecture}")

    # ── Stage 2: Sub-question Decomposition ──────────────────────────
    sub_questions = await generate_sub_questions(synth)
    print(f"[Stage 2] Generated {len(sub_questions)} sub-question(s)")
    for sq in sub_questions:
        print(f"  [{sq.id}] {sq.text}")

    # ── Initialize Knowledge Store ───────────────────────────────────
    store = KnowledgeStore(sub_questions)

    # ── Search → Extract → Reflect Loop ─────────────────────────────
    # On round 1, all sub-questions are searched.
    # On round 2+, only sub-questions that need more evidence are searched.
    active_sub_questions = list(sub_questions)
    suggested_angles = None

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_num} of {MAX_ROUNDS} — searching {len(active_sub_questions)} sub-question(s)")
        print(f"{'='*60}")

        # ── Stage 3: Search Query Generation ─────────────────────────
        search_queries = await generate_search_queries(
            active_sub_questions,
            round_num=round_num,
            suggested_angles=suggested_angles,
            exclude_queries=store.previously_searched_strings() if round_num > 1 else None,
        )
        print(f"[Stage 3] Generated {len(search_queries)} search queries")
        store.add_queries(search_queries)

        # ── Stage 4: Search Execution ────────────────────────────────
        search_results = await execute_and_clean_searches(search_queries)
        executed = sum(1 for sq in search_queries if sq.status == "executed")
        failed = sum(1 for sq in search_queries if sq.status == "failed")
        print(f"[Stage 4] Search complete — {len(search_results)} results ({executed} executed, {failed} failed)")

        # ── Stage 5: Evidence Extraction ─────────────────────────────
        new_evidence = await extract_evidence(active_sub_questions, search_results)
        print(f"[Stage 5] Extracted {len(new_evidence)} evidence claims")
        store.add_evidence(new_evidence)

        # ── Stage 6: Knowledge Store Summary ─────────────────────────
        print(f"\n[Stage 6] Knowledge Store state:")
        print(store.coverage_summary())

        # ── Stage 7: Reflection ──────────────────────────────────────
        if round_num == MAX_ROUNDS:
            print(f"\n[Stage 7] Max rounds reached — accepting current evidence.")
            break

        verdicts = await reflect(store)
        needs_more = [v for v in verdicts if v.verdict == "needs_more"]

        if not needs_more:
            print(f"\n[Stage 7] All sub-questions have sufficient evidence.")
            break

        print(f"\n[Stage 7] Reflection — {len(needs_more)} sub-question(s) need more evidence:")
        for v in needs_more:
            print(f"  [{v.sub_question_id}] Gap: {v.gap}")
            if v.suggested_angles:
                print(f"    Suggested angles: {', '.join(v.suggested_angles)}")

        # Narrow the active set to only sub-questions that need more
        needs_more_ids = {v.sub_question_id for v in needs_more}
        active_sub_questions = [sq for sq in sub_questions if sq.id in needs_more_ids]

        # Build the suggested_angles dict for the search query generator
        suggested_angles = {
            v.sub_question_id: v.suggested_angles
            for v in needs_more
            if v.suggested_angles
        }

    # ── Final Output ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    total = len(store.evidence)
    zero_coverage = store.has_zero_evidence()
    print(f"Total evidence claims: {total}")
    if zero_coverage:
        print(f"Sub-questions with zero evidence: {[sq.id for sq in zero_coverage]}")
    print()

    # ── Stage 8: Final Report Synthesis ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Stage 8: Synthesizing Final Report")
    print(f"{'='*60}")

    report_markdown = await synthesize_report(query, store)

    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/research_report_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_markdown)

    print(f"\n✅ DeepSearch complete! Final report saved to: {filename}")


if __name__ == "__main__":
    asyncio.run(main())