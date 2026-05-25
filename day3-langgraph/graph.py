from langgraph.graph import StateGraph, START, END
from state import ReviewGuardState
from nodes import retrieve_node, history_node, decide_node, escalate_node


def should_escalate(state: ReviewGuardState) -> str:
    """conditional edge: route to escalate if confidence is low"""
    if state.get("confidence", 1.0) < 0.7:
        return "escalate"
    return "end"


def build_graph():
    g = StateGraph(ReviewGuardState)

    # add the four nodes
    g.add_node("retrieve", retrieve_node)
    g.add_node("history", history_node)
    g.add_node("decide", decide_node)
    g.add_node("escalate", escalate_node)

    # retrieve and history both start from START — runs them in parallel
    g.add_edge(START, "retrieve")
    g.add_edge(START, "history")

    # both feed into decide — LangGraph waits for both before running decide
    g.add_edge("retrieve", "decide")
    g.add_edge("history", "decide")

    # conditional: low confidence → escalate, otherwise end
    g.add_conditional_edges(
        "decide",
        should_escalate,
        {"escalate": "escalate", "end": END}
    )
    g.add_edge("escalate", END)

    return g.compile()


graph = build_graph()


def run_review(review: dict) -> dict:
    """run a single review through the graph, return final state"""
    result = graph.invoke({
        "review_id": review["review_id"],
        "reviewer_id": review["reviewer_id"],
        "product": review["product"],
        "rating": int(review["rating"]),
        "text": review["text"],
    })
    return result


if __name__ == "__main__":
    import pandas as pd
    row = pd.read_csv("data/eval_set.csv").iloc[0]
    review = {
        "review_id": row["review_id"],
        "reviewer_id": row["reviewer_id"],
        "product": row["product"],
        "rating": row["rating"],
        "text": row["text"],
    }
    print(f"\nAnalyzing {review['review_id']}...")
    result = run_review(review)
    print("\n=== final state ===")
    for k, v in result.items():
        if k == "similar_reviews":
            print(f"  similar_reviews: {len(v)} hits")
        elif k == "reviewer_history":
            print(f"  reviewer_history: {v}")
        else:
            print(f"  {k}: {v}")