import json
import re
import os
import pandas as pd
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from tools_crewai import retrieve_similar_reviews, check_reviewer_history

load_dotenv()




# ---------- the three agents ----------

retriever_agent = Agent(
    role="Review Retrieval Specialist",
    goal="Find reviews from the corpus that are most similar to the review under analysis",
    backstory="""You are an expert at semantic search and pattern matching in review data.
    You know that fake reviews often share textual patterns — generic praise, excessive
    punctuation, lack of specific details. You surface the most relevant comparison cases.""",
    tools=[retrieve_similar_reviews],
    llm="claude-opus-4-5",
    verbose=False,
)

analyst_agent = Agent(
    role="Behavioral Analyst",
    goal="Analyze reviewer behavioral signals to identify suspicious account patterns",
    backstory="""You are an expert in detecting fake reviewer accounts through behavioral
    signals. You know the red flags: new accounts under 60 days old, posting more than
    2 reviews per day, maintaining above 90% 5-star ratings, near-perfect average ratings.
    You provide a clear behavioral risk assessment.""",
    tools=[check_reviewer_history],
    llm="claude-opus-4-5",
    verbose=False,
)

moderator_agent = Agent(
    role="Review Moderator",
    goal="Make a final fake/genuine/suspicious determination based on text and behavioral evidence",
    backstory="""You are the final decision maker for review authenticity. You receive
    reports from the retrieval specialist and behavioral analyst and weigh both signals.
    You know that text alone is insufficient — behavioral signals are critical.
    When signals conflict and confidence is low, you mark as suspicious.""",
    tools=[],
    llm="claude-opus-4-5",
    verbose=False,
)


# ---------- run a single review ----------

def run_review(review: dict) -> dict:
    # Task 1: retrieve similar reviews
    retrieve_task = Task(
        description=f"""Search for reviews similar to this one:
        
review_id: {review['review_id']}
product: {review['product']}
rating: {review['rating']}
text: {review['text']}

Use the retrieve_similar_reviews tool with the review text as the query.
Report what you found including similarity scores and labels.""",
        expected_output="A summary of the 3 most similar reviews including their labels and similarity scores.",
        agent=retriever_agent,
    )

    # Task 2: check reviewer history
    history_task = Task(
        description=f"""Analyze the behavioral profile of reviewer {review['reviewer_id']}.

Use the check_reviewer_history tool to look up their profile.
Assess whether the behavioral signals suggest a fake or genuine reviewer account.
Flag if: account under 60 days old, more than 2 reviews/day, above 90% 5-star rate.""",
        expected_output="A behavioral risk assessment with specific signal values and a risk level: low, medium, or high.",
        agent=analyst_agent,
    )

    # Task 3: make the final decision
    moderator_task = Task(
        description=f"""Make a final authenticity determination for this review:

review_id: {review['review_id']}
product: {review['product']}  
rating: {review['rating']}
text: {review['text']}

You have received:
- A retrieval report showing similar reviews and their labels
- A behavioral analysis of the reviewer account

Weigh both signals. If they conflict and you are not confident, use suspicious.

Respond with ONLY this JSON:
{{"label": "fake|genuine|suspicious", "confidence": 0.0-1.0, "reasoning": "brief reason"}}""",
        expected_output='JSON object with label, confidence, and reasoning fields.',
        agent=moderator_agent,
        context=[retrieve_task, history_task],
    )

    # assemble the crew and run
    crew = Crew(
        agents=[retriever_agent, analyst_agent, moderator_agent],
        tasks=[retrieve_task, history_task, moderator_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    raw_output = str(result)

    # parse the verdict
    match = re.search(r'\{.*?"label".*?\}', raw_output, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return {
                "review_id": review["review_id"],
                "label": parsed.get("label", "parse_error"),
                "confidence": float(parsed.get("confidence", 0.0)),
                "reasoning": parsed.get("reasoning", "")[:500],
            }
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        "review_id": review["review_id"],
        "label": "parse_error",
        "confidence": 0.0,
        "reasoning": raw_output[:300],
    }


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
    result = run_review(review)
    print(f"\n=== result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")