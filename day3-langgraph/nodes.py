import json
import re
import pandas as pd
from langchain_anthropic import ChatAnthropic
from retriever import CorpusRetriever
from state import ReviewGuardState
from dotenv import load_dotenv

load_dotenv()

# load shared resources once at import time
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

_llm = ChatAnthropic(model="claude-opus-4-5", max_tokens=1024, temperature=0.0)


def retrieve_node(state: ReviewGuardState) -> dict:
    hits = _retriever.search(state["text"], k=3)
    return {"similar_reviews": hits}


def history_node(state: ReviewGuardState) -> dict:
    profile = _history_db.get(state["reviewer_id"], {"error": "not found"})
    return {"reviewer_history": profile}


DECIDE_PROMPT = """You are ReviewGuard. Given a review and two signals, return a verdict.

REVIEW:
  product: {product}
  rating: {rating}
  text: {text}

SIMILAR REVIEWS FROM CORPUS (with known labels):
{similar_reviews}

REVIEWER BEHAVIORAL HISTORY:
{reviewer_history}

Behavioral red flags: new accounts, high review volume, near-100% 5-star ratings.
If text and behavior conflict and confidence is low, use label="suspicious".

Respond with ONLY a JSON object, no extra text:
{{"label": "fake|genuine|suspicious", "confidence": 0.0-1.0, "reasoning": "brief reason"}}"""


def decide_node(state: ReviewGuardState) -> dict:
    prompt = DECIDE_PROMPT.format(
        product=state["product"],
        rating=state["rating"],
        text=state["text"],
        similar_reviews=json.dumps(state["similar_reviews"], indent=2),
        reviewer_history=json.dumps(state["reviewer_history"], indent=2),
    )
    response = _llm.invoke(prompt)
    text = response.content if isinstance(response.content, str) else str(response.content)

    match = re.search(r'\{.*?"label".*?\}', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return {
                "label": parsed.get("label", "parse_error"),
                "confidence": float(parsed.get("confidence", 0.0)),
                "reasoning": parsed.get("reasoning", "")[:500],
            }
        except (json.JSONDecodeError, ValueError):
            pass
    return {"label": "parse_error", "confidence": 0.0, "reasoning": text[:200]}


def escalate_node(state: ReviewGuardState) -> dict:
    print(f"  🚩 FLAGGED: {state['review_id']} — confidence {state['confidence']:.2f}")
    return {"escalated": True}