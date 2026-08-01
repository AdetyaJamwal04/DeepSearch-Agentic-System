from typing import List, Set
from schemas.schema import SubQuestion, SearchQuery, Evidence


class KnowledgeStore:
    """
    Per-query in-memory store that accumulates evidence, search queries,
    and sub-questions across multiple search rounds. Not a database — lives
    for the duration of a single research run.

    Design: flat lists with smart accessors. Grouping happens at query time,
    not at insertion time.
    """

    def __init__(self, sub_questions: List[SubQuestion]):
        self.sub_questions: List[SubQuestion] = list(sub_questions)
        self.search_queries: List[SearchQuery] = []
        self.evidence: List[Evidence] = []

    # ── Accessors ────────────────────────────────────────────────

    def evidence_for(self, sub_question_id: str) -> List[Evidence]:
        """All evidence claims belonging to a specific sub-question."""
        return [e for e in self.evidence if e.sub_question_id == sub_question_id]

    def evidence_count(self, sub_question_id: str) -> int:
        """Number of evidence claims for a specific sub-question."""
        return sum(1 for e in self.evidence if e.sub_question_id == sub_question_id)

    def has_zero_evidence(self) -> List[SubQuestion]:
        """Sub-questions that have no evidence at all after searching."""
        return [sq for sq in self.sub_questions if self.evidence_count(sq.id) == 0]

    def queries_for(self, sub_question_id: str) -> List[SearchQuery]:
        """All search queries that were generated for a specific sub-question."""
        return [q for q in self.search_queries if q.sub_question_id == sub_question_id]

    def previously_searched_strings(self) -> Set[str]:
        """
        All query strings that have been sent to the search engine so far.
        Used by the search query generator to avoid regenerating the same
        queries on subsequent rounds, and by the reflection agent for
        stagnation detection.
        """
        return {q.query_string for q in self.search_queries}

    # ── Mutators ─────────────────────────────────────────────────

    def add_queries(self, queries: List[SearchQuery]) -> None:
        """Append a batch of search queries (from one round)."""
        self.search_queries.extend(queries)

    def add_evidence(self, evidence: List[Evidence]) -> None:
        """Append a batch of evidence claims (from one round)."""
        self.evidence.extend(evidence)

    # ── Summaries ────────────────────────────────────────────────

    def coverage_summary(self) -> str:
        """
        Human-readable summary of evidence coverage per sub-question.
        Used for logging and as input to the reflection agent.
        """
        lines = []
        for sq in self.sub_questions:
            count = self.evidence_count(sq.id)
            queries = self.queries_for(sq.id)
            executed = sum(1 for q in queries if q.status == "executed")
            failed = sum(1 for q in queries if q.status == "failed")

            sources = set()
            for e in self.evidence_for(sq.id):
                sources.add(e.source_url)

            lines.append(
                f"[{sq.id}] \"{sq.text}\"\n"
                f"  Evidence claims: {count} | Unique sources: {len(sources)} | "
                f"Queries: {len(queries)} (executed: {executed}, failed: {failed})"
            )

        return "\n".join(lines)

    def evidence_summary_for(self, sub_question_id: str) -> str:
        """
        Returns a text block listing all evidence claims for a sub-question.
        Used by the reflection agent to assess depth and quality.
        """
        claims = self.evidence_for(sub_question_id)
        if not claims:
            return "No evidence collected."

        lines = []
        for i, e in enumerate(claims, start=1):
            lines.append(f"  {i}. {e.claim} [source: {e.source_title}]")
        return "\n".join(lines)
