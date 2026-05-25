import json
import pandas as pd
from crewai.tools import tool
from dotenv import load_dotenv
from retriever import CorpusRetriever

load_dotenv()

_retriever = CorpusRetriever()

_corpus_df = pd.read_csv("data/reviews_corpus.csv")
_eval_df = pd.read_csv("data/eval_set.csv")
_history_db = {
    row["reviewer_id"]: {
        "total_reviews": int(row["total_reviews"]),
        "avg_rating": float(row["avg_rating"]),
        "account_age_days": int(row["account_age_days"]),
        "all_5_star_pct": float(row["all_5_star_pct"]),
    }
    for _, row in pd.concat([_corpus_df, _eval_df]).iterrows()
}


@tool("retrieve_similar_reviews")
def retrieve_similar_reviews(query_text: str) -> str:
    """
    Search the review corpus for reviews similar to the query text.
    Returns the top 3 most similar reviews with their labels and similarity scores.
    Use this to find patterns matching known fake or genuine reviews.
    """
    hits = _retriever.search(query_text, k=3)
    return json.dumps(hits, indent=2)


@tool("check_reviewer_history")
def check_reviewer_history(reviewer_id: str) -> str:
    """
    Look up behavioral signals for a reviewer by their ID.
    Returns account age, total reviews, average rating, and 5-star percentage.
    Red flags: new account under 60 days, over 50 reviews, above 90% 5-star rate.
    """
    profile = _history_db.get(reviewer_id, {"error": "reviewer not found"})
    return json.dumps(profile, indent=2)