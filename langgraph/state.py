"""
The state object that flows through every node in the graph.

LangGraph passes this dict from node to node. Each node receives the full state
and returns a partial dict with the keys it wants to update. LangGraph merges
the updates back in. This is the same pattern as React's useReducer or Redux —
explicit state transitions, no hidden side effects.
"""

from typing import TypedDict, Optional, Any


class ReviewGuardState(TypedDict):
    # Inputs (set once at the start)
    review_id: str
    reviewer_id: str
    product: str
    rating: int
    text: str

    # Filled by the retrieve node
    similar_reviews: Optional[list[dict]]

    # Filled by the history node
    reviewer_history: Optional[dict]

    # Filled by the decide node
    label: Optional[str]          # "fake" | "genuine" | "suspicious"
    confidence: Optional[float]
    reasoning: Optional[str]

    # Filled by the escalate node (only if confidence is low)
    escalated: Optional[bool]