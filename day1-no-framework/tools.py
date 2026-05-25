"""
Tools the agent can call. Each tool has two parts:
  1. A JSON schema (TOOL_SCHEMAS) — this is what Claude sees. Claude reads
     the description and decides when to call the tool.
  2. A Python function — this is what actually runs when Claude requests the tool.

The agent loop in agent.py connects these: when Claude says "call retrieve_similar_reviews
with these args", the loop looks up the function by name and runs it.
"""

import pandas as pd
from sample_reviews import REVIEWER_HISTORY  # keeping mock reviewer history for Day 2
from retriever import CorpusRetriever

# Load the retriever once at import time. The agent calls search() many times,
# we don't want to reload the index on every call.
_retriever = CorpusRetriever()

# Load real reviewer history from the eval set + corpus, replacing the tiny mock dict.
# Each row's behavioral metadata becomes a profile entry keyed by reviewer_id.
_corpus_df = pd.read_csv("data/reviews_corpus.csv")
_eval_df = pd.read_csv("data/eval_set.csv")
_all_profiles = pd.concat([_corpus_df, _eval_df])
REAL_REVIEWER_HISTORY = {
    row["reviewer_id"]: {
        "total_reviews": int(row["total_reviews"]),
        "avg_rating": float(row["avg_rating"]),
        "account_age_days": int(row["account_age_days"]),
        "all_5_star_pct": float(row["all_5_star_pct"]),
    }
    for _, row in _all_profiles.iterrows()
}


# ---------- Tool 1: Retrieve similar reviews (now REAL RAG) ----------
def retrieve_similar_reviews(query_text: str, top_k: int = 3) -> dict:
    """
    Real semantic search over 200 indexed reviews.
    Returns top-k matches with their text, label, and similarity score.
    """
    hits = _retriever.search(query_text, k=top_k)
    return {"query": query_text, "similar_reviews": hits}


# ---------- Tool 2: Check reviewer history (now real profiles) ----------
def check_reviewer_history(reviewer_id: str) -> dict:
    """
    Looks up behavioral signals from the synthesized reviewer profile database.
    """
    history = REAL_REVIEWER_HISTORY.get(reviewer_id)
    if history is None:
        return {"error": f"No history found for reviewer {reviewer_id}"}
    return {"reviewer_id": reviewer_id, **history}


# ---------- Tool 3: Flag for human review ----------
def flag_for_human(review_id: str, reason: str, confidence: float) -> dict:
    """
    Human-in-the-loop escalation. In production this would write to a queue.
    Here we just print and return a confirmation.
    """
    print(f"\n  🚩 FLAGGED FOR HUMAN: {review_id}")
    print(f"     Reason: {reason}")
    print(f"     Confidence: {confidence}")
    return {"status": "flagged", "review_id": review_id}


# ---------- The schemas Claude sees ----------
# Each schema tells Claude: name, what it does, what arguments to pass.
# Good descriptions here = good tool selection by the agent. This is the
# single highest-leverage thing in agent design.

TOOL_SCHEMAS = [
    {
        "name": "retrieve_similar_reviews",
        "description": (
            "Retrieve reviews from the corpus that are textually similar to a query. "
            "Use this to check if the review under analysis matches patterns of known "
            "spam or known genuine reviews."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": "The review text to find similar examples for.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many similar reviews to return. Default 3.",
                },
            },
            "required": ["query_text"],
        },
    },
    {
        "name": "check_reviewer_history",
        "description": (
            "Look up behavioral signals for a reviewer: account age, total reviews, "
            "average rating, percentage of 5-star reviews. Critical for detecting "
            "fake accounts (typically: new account, many reviews, near-perfect ratings)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reviewer_id": {
                    "type": "string",
                    "description": "The reviewer's unique ID, e.g. 'U101'.",
                },
            },
            "required": ["reviewer_id"],
        },
    },
    {
        "name": "flag_for_human",
        "description": (
            "Escalate a review to a human moderator when confidence is low or the "
            "case is ambiguous. Use this when text and behavioral signals disagree, "
            "or when confidence in a fake/genuine call is below 0.7."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "review_id": {"type": "string"},
                "reason": {"type": "string", "description": "Why human review is needed."},
                "confidence": {
                    "type": "number",
                    "description": "Your confidence in the call, 0.0 to 1.0.",
                },
            },
            "required": ["review_id", "reason", "confidence"],
        },
    },
]


# Map tool names to functions, so the agent loop can dispatch.
TOOL_FUNCTIONS = {
    "retrieve_similar_reviews": retrieve_similar_reviews,
    "check_reviewer_history": check_reviewer_history,
    "flag_for_human": flag_for_human,
}