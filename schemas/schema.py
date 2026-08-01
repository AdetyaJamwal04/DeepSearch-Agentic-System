from pydantic import BaseModel, Field
from typing import List, Literal

class QuerySynthesis(BaseModel):
    original_query: str

    intent: str
    objective: str

    query_architecture: Literal[
        "single_question",
        "multi_question"
    ]

    scope: Literal[
        "narrow",
        "moderate",
        "broad"
    ]

    entities: list[str]

    ambiguity: bool
    ambiguity_reason: str

class SubQuestion(BaseModel):
    id: str
    text: str
    priority: int
    parent_intent: str
    parent_entities: list[str] = Field(default_factory=list)

class SearchQuery(BaseModel):
    id: str                                          # e.g. "sq_1_q1"
    sub_question_id: str                             # FK back to its SubQuestion
    query_string: str
    angle: str
    status: Literal["pending", "executed", "failed"] = "pending"
    round: int = 1
    
class SearchResult(BaseModel):
    search_query_id: str
    sub_question_id: str
    url: str
    title: str
    content: str
    score: float = 0.0
    
class Evidence(BaseModel):
    sub_question_id: str
    claim: str
    source_url: str
    source_title: str

class SubQuestionVerdict(BaseModel):
    sub_question_id: str
    verdict: Literal["sufficient", "needs_more"]
    gap: str                                        # what's missing (empty if sufficient)
    suggested_angles: list[str] = []                # new search angles to try (empty if sufficient)
