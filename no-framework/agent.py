"""
The ReviewGuard agent loop. This is the most important file of the entire week.
Once you understand this loop, every framework after this is just a wrapper.
"""

import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv

from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS
import pandas as pd

VERBOSE = False  # set True for per-review tool-call traces
# Load 5 real reviews from the eval set instead of hardcoded samples
_eval = pd.read_csv("data/eval_set.csv").head(5)
SAMPLE_REVIEWS = [
    {
        "review_id": row["review_id"],
        "reviewer_id": row["reviewer_id"],
        "product": row["product"],
        "rating": int(row["rating"]),
        "text": row["text"],
        "expected_label": row["label"],
    }
    for _, row in _eval.iterrows()
]

load_dotenv()
client = Anthropic()  # picks up ANTHROPIC_API_KEY from .env automatically

MODEL = "claude-opus-4-5"  # use opus for reasoning quality on Day 1
MAX_TURNS = 6  # safety cap so the loop can't run forever


SYSTEM_PROMPT = """You are ReviewGuard, an AI system that detects fake product reviews.

For each review you receive, you must:
1. Use the retrieve_similar_reviews tool to find textually similar reviews in the corpus.
2. Use the check_reviewer_history tool to inspect the reviewer's behavioral signals.
3. Reason about both signals together. Behavioral red flags include: new accounts,
   very high review counts, near-100% 5-star ratings.
4. If the case is ambiguous (text seems okay but behavior is suspicious, or vice versa),
   call flag_for_human.
5. Return a final verdict in this exact JSON format:
   {"label": "genuine" | "suspicious" | "fake", "confidence": 0.0-1.0, "reasoning": "..."}

Always use both tools before deciding. Never decide on text alone — your training
showed text-only detection fails on sophisticated fakes."""


def run_agent(review: dict) -> dict:
    """
    Run the agent on a single review. Returns the final verdict.

    This is THE agent loop. Read it carefully — the rest of the week builds on it.
    """
    # Initial user message: hand the review to the agent.
    user_message = (
        f"Analyze this review:\n"
        f"  review_id: {review['review_id']}\n"
        f"  reviewer_id: {review['reviewer_id']}\n"
        f"  product: {review['product']}\n"
        f"  rating: {review['rating']}\n"
        f"  text: {review['text']}\n"
    )
    messages = [{"role": "user", "content": user_message}]

    print(f"  → Tool call: ...")
    print(f"Analyzing {review['review_id']} (expected: {review['expected_label']})")
    print(f"{'='*60}")

    # The loop. Each iteration is one "turn" of the agent.
    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        # The response has a stop_reason. Two cases matter:
        #   "tool_use" → Claude wants to call a tool, we run it and continue
        #   "end_turn" → Claude is done, we return the final answer

        if response.stop_reason == "end_turn":
            # Extract the text content from the final response.
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            print(f"\n  ✓ Final verdict: {final_text}")
            return {"review_id": review["review_id"], "verdict": final_text}

        if response.stop_reason == "tool_use":
            # Append Claude's response (with tool_use blocks) to messages.
            messages.append({"role": "assistant", "content": response.content})

            # Find every tool_use block in the response and execute each one.
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_args = block.input
                    print(f"  → Tool call: {tool_name}({json.dumps(tool_args)})")

                    # Dispatch to the right Python function.
                    fn = TOOL_FUNCTIONS[tool_name]
                    result = fn(**tool_args)
                    print(f"    ← Result: {json.dumps(result)[:120]}...")

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )

            # Send all tool results back as a single user message, then loop.
            messages.append({"role": "user", "content": tool_results})
            continue

        # Any other stop_reason — shouldn't happen on Day 1, but bail safely.
        print(f"  ! Unexpected stop_reason: {response.stop_reason}")
        break

    return {"review_id": review["review_id"], "verdict": "ERROR: max turns exceeded"}


def main():
    results = []
    for review in SAMPLE_REVIEWS:
        result = run_agent(review)
        results.append(result)

    print(f"\n\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['review_id']}: {r['verdict'][:100]}")


if __name__ == "__main__":
    main()