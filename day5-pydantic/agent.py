import json
import os
import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from models import ReviewVerdict
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

reviewguard_agent = Agent(
    model=AnthropicModel("claude-opus-4-5"),
    output_type=ReviewVerdict,
    system_prompt="""You are ReviewGuard, a fake review detection system.

You will receive a review along with two signals:
1. Similar reviews from a corpus with their known labels
2. The reviewer's behavioral history

Behavioral red flags: account under 60 days old, more than 2 reviews per day,
above 90% 5-star rate, average rating above 4.8.

Weigh both signals together. If they conflict or confidence is low,
set label to suspicious and flagged_for_human to true.

Return a structured verdict. Be concise in reasoning — max 2 sentences."""
)


def run_review(review: dict) -> ReviewVerdict:
    similar = _retriever.search(review["text"], k=3)
    history = _history_db.get(review["reviewer_id"], {"error": "not found"})

    prompt = f"""Review to analyze:
product: {review['product']}
rating: {review['rating']}
text: {review['text']}

Similar reviews from corpus:
{json.dumps(similar, indent=2)}

Reviewer behavioral history:
{json.dumps(history, indent=2)}"""

    result = reviewguard_agent.run_sync(prompt)
    return result.output


if __name__ == "__main__":
    row = pd.read_csv("data/eval_set.csv").iloc[0]
    review = {
        "review_id": row["review_id"],
        "reviewer_id": row["reviewer_id"],
        "product": row["product"],
        "rating": int(row["rating"]),
        "text": row["text"],
    }
    print(f"\nAnalyzing {review['review_id']}...")
    verdict = run_review(review)
    print(f"\n=== verdict ===")
    print(f"  label:             {verdict.label}")
    print(f"  confidence:        {verdict.confidence}")
    print(f"  reasoning:         {verdict.reasoning}")
    print(f"  flagged_for_human: {verdict.flagged_for_human}")
    print(f"\nType: {type(verdict)}")