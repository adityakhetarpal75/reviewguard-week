"""
Evaluation harness for ReviewGuard.

Runs the agent on the full eval set, parses verdicts, and computes
precision/recall/F1. The output CSV is the canonical artifact this week —
LangGraph, CrewAI, and Pydantic AI ports all get measured against the
same eval set so the framework comparison is honest.

Usage: python eval.py
Output: data/eval_results_day2.csv  +  console summary
"""

import json
import re
import time
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from agent import run_agent  # reuse Day 1's agent loop unchanged

EVAL_CSV = "data/eval_set.csv"
RESULTS_CSV = "data/eval_results_day2.csv"


def parse_verdict(verdict_text: str) -> dict:
    """
    Extract structured fields from the agent's final text output.

    The agent is instructed to return JSON, but it sometimes wraps it in
    markdown fences or adds prose around it. We extract the first JSON
    object we can find. If parsing fails, we mark it as 'parse_error'
    rather than silently misclassifying.
    """
    # Try to find a JSON object — handles ```json ... ``` and bare JSON
    json_match = re.search(r'\{[^{}]*"label"[^{}]*\}', verdict_text, re.DOTALL)
    if not json_match:
        # Try with nested braces (reasoning fields can contain quotes/braces)
        json_match = re.search(r'\{.*?"label".*?\}', verdict_text, re.DOTALL)

    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return {
                "predicted_label": parsed.get("label", "parse_error"),
                "confidence": float(parsed.get("confidence", 0.0)),
                "reasoning": parsed.get("reasoning", "")[:500],
            }
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        "predicted_label": "parse_error",
        "confidence": 0.0,
        "reasoning": verdict_text[:500],
    }


def main():
    print(f"Loading eval set from {EVAL_CSV}...")
    eval_df = pd.read_csv(EVAL_CSV)
    print(f"  {len(eval_df)} reviews to evaluate\n")

    results = []
    start_time = time.time()

    for i, row in eval_df.iterrows():
        review = {
            "review_id": row["review_id"],
            "reviewer_id": row["reviewer_id"],
            "product": row["product"],
            "rating": int(row["rating"]),
            "text": row["text"],
            "expected_label": row["label"],
        }

        # Time each agent call individually for latency stats
        t0 = time.time()
        try:
            agent_result = run_agent(review)
            verdict_text = agent_result["verdict"]
            parsed = parse_verdict(verdict_text)
            error = ""
        except Exception as e:
            verdict_text = ""
            parsed = {"predicted_label": "error", "confidence": 0.0, "reasoning": ""}
            error = str(e)[:200]

        latency = time.time() - t0

        # 'suspicious' label means the agent flagged for human — count separately
        flagged = parsed["predicted_label"] == "suspicious"

        results.append({
            "review_id": row["review_id"],
            "expected_label": row["label"],
            "predicted_label": parsed["predicted_label"],
            "confidence": parsed["confidence"],
            "flagged_for_human": flagged,
            "latency_seconds": round(latency, 2),
            "error": error,
            "reasoning_snippet": parsed["reasoning"][:200],
        })

        # Live progress so you don't wonder if it's hanging
        status = "✓" if parsed["predicted_label"] == row["label"] else "✗"
        print(
            f"  [{i+1:2d}/{len(eval_df)}] {row['review_id']} "
            f"expected={row['label']:8s} predicted={parsed['predicted_label']:12s} "
            f"conf={parsed['confidence']:.2f} {status}"
        )

    total_time = time.time() - start_time
    print(f"\n  Total runtime: {total_time:.1f}s ({total_time/len(eval_df):.1f}s per review)")

    # Save raw results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"  Saved to {RESULTS_CSV}")

    # ---------- Compute metrics ----------
    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)

    # Drop parse_errors and exceptions for the metrics computation,
    # but report them separately so failures aren't silently hidden.
    valid = results_df[
        results_df["predicted_label"].isin(["fake", "genuine", "suspicious"])
    ]
    n_invalid = len(results_df) - len(valid)
    if n_invalid > 0:
        print(f"\n⚠  {n_invalid} reviews had parse_error or exception — excluded from metrics.")

    # For precision/recall we treat 'suspicious' (flagged) as a third class.
    # If you care about a binary fake-vs-not-fake metric, see the binary block below.
    print("\nPer-label classification report (3-class: fake / genuine / suspicious):")
    labels = ["fake", "genuine", "suspicious"]
    p, r, f, support = precision_recall_fscore_support(
        valid["expected_label"], valid["predicted_label"],
        labels=labels, zero_division=0,
    )
    print(f"\n  {'label':12s}  {'precision':>10s}  {'recall':>10s}  {'f1':>10s}  {'support':>8s}")
    for lbl, pi, ri, fi, si in zip(labels, p, r, f, support):
        print(f"  {lbl:12s}  {pi:>10.3f}  {ri:>10.3f}  {fi:>10.3f}  {si:>8d}")

    # Binary view: did the agent catch the fakes? (suspicious counts as 'flagged')
    print("\nBinary view — fake detection (suspicious treated as 'caught'):")
    binary_pred = valid["predicted_label"].apply(
        lambda x: "fake_or_flagged" if x in ("fake", "suspicious") else "genuine"
    )
    binary_true = valid["expected_label"].apply(
        lambda x: "fake_or_flagged" if x == "fake" else "genuine"
    )
    bp, br, bf, bs = precision_recall_fscore_support(
        binary_true, binary_pred, labels=["fake_or_flagged", "genuine"], zero_division=0,
    )
    print(f"  {'label':18s}  {'precision':>10s}  {'recall':>10s}  {'f1':>10s}  {'support':>8s}")
    for lbl, pi, ri, fi, si in zip(["fake_or_flagged", "genuine"], bp, br, bf, bs):
        print(f"  {lbl:18s}  {pi:>10.3f}  {ri:>10.3f}  {fi:>10.3f}  {si:>8d}")

    # Confusion matrix (3-class)
    print("\nConfusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(valid["expected_label"], valid["predicted_label"], labels=labels)
    print(f"           {'  '.join(f'{l:>10s}' for l in labels)}")
    for lbl, row in zip(labels, cm):
        print(f"  {lbl:8s}  {'  '.join(f'{v:>10d}' for v in row)}")

    # HIL stats
    flagged_count = valid["flagged_for_human"].sum()
    print(f"\nHIL escalation rate: {flagged_count}/{len(valid)} = {flagged_count/len(valid):.1%}")

    # Latency stats
    print(f"\nLatency: mean={results_df['latency_seconds'].mean():.1f}s "
          f"p50={results_df['latency_seconds'].median():.1f}s "
          f"p95={results_df['latency_seconds'].quantile(0.95):.1f}s")


if __name__ == "__main__":
    main()