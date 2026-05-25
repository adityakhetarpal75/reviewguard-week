import time
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from crew import run_review

EVAL_CSV = "data/eval_set.csv"
RESULTS_CSV = "data/eval_results_day4.csv"


def main():
    print("Loading eval set...")
    eval_df = pd.read_csv(EVAL_CSV)
    print(f"  {len(eval_df)} reviews\n")

    results = []

    for i, row in eval_df.iterrows():
        review = {
            "review_id": row["review_id"],
            "reviewer_id": row["reviewer_id"],
            "product": row["product"],
            "rating": int(row["rating"]),
            "text": row["text"],
        }

        t0 = time.time()
        try:
            result = run_review(review)
            label = result.get("label", "error")
            confidence = result.get("confidence", 0.0)
            error = ""
        except Exception as e:
            label, confidence, error = "error", 0.0, str(e)[:200]

        latency = round(time.time() - t0, 2)
        status = "✓" if label == row["label"] else "✗"

        print(
            f"  [{i+1:2d}/50] {row['review_id']} "
            f"expected={row['label']:8s} predicted={label:12s} "
            f"conf={confidence:.2f} {status} {latency:.1f}s"
        )

        results.append({
            "review_id": row["review_id"],
            "expected_label": row["label"],
            "predicted_label": label,
            "confidence": confidence,
            "flagged_for_human": label == "suspicious",
            "latency_seconds": latency,
            "error": error,
        })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n  Saved to {RESULTS_CSV}")

    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)

    valid = df[df["predicted_label"].isin(["fake", "genuine", "suspicious"])]
    n_invalid = len(df) - len(valid)
    if n_invalid:
        print(f"\n⚠  {n_invalid} errors excluded")

    labels = ["fake", "genuine", "suspicious"]
    p, r, f, support = precision_recall_fscore_support(
        valid["expected_label"], valid["predicted_label"],
        labels=labels, zero_division=0
    )
    print(f"\n  {'label':12s}  {'precision':>10s}  {'recall':>10s}  {'f1':>10s}  {'support':>8s}")
    for lbl, pi, ri, fi, si in zip(labels, p, r, f, support):
        print(f"  {lbl:12s}  {pi:>10.3f}  {ri:>10.3f}  {fi:>10.3f}  {si:>8d}")

    binary_pred = valid["predicted_label"].apply(
        lambda x: "fake_or_flagged" if x in ("fake", "suspicious") else "genuine"
    )
    binary_true = valid["expected_label"].apply(
        lambda x: "fake_or_flagged" if x == "fake" else "genuine"
    )
    bp, br, bf, bs = precision_recall_fscore_support(
        binary_true, binary_pred,
        labels=["fake_or_flagged", "genuine"], zero_division=0
    )
    print(f"\nBinary view:")
    print(f"  {'label':18s}  {'precision':>10s}  {'recall':>10s}  {'f1':>10s}  {'support':>8s}")
    for lbl, pi, ri, fi, si in zip(["fake_or_flagged", "genuine"], bp, br, bf, bs):
        print(f"  {lbl:18s}  {pi:>10.3f}  {ri:>10.3f}  {fi:>10.3f}  {si:>8d}")

    cm = confusion_matrix(
        valid["expected_label"], valid["predicted_label"], labels=labels
    )
    print(f"\nConfusion matrix:")
    print(f"           {'  '.join(f'{l:>10s}' for l in labels)}")
    for lbl, row in zip(labels, cm):
        print(f"  {lbl:8s}  {'  '.join(f'{v:>10d}' for v in row)}")

    flagged = valid["flagged_for_human"].sum()
    print(f"\nHIL rate: {flagged}/{len(valid)} = {flagged/len(valid):.1%}")
    print(
        f"Latency: mean={df['latency_seconds'].mean():.1f}s  "
        f"p50={df['latency_seconds'].median():.1f}s  "
        f"p95={df['latency_seconds'].quantile(0.95):.1f}s"
    )

    # comparison table
    print(f"\n--- vs previous days ---")
    try:
        d2 = pd.read_csv("../day1-no-framework/data/eval_results_day2.csv")
        d2v = d2[d2["predicted_label"].isin(["fake", "genuine", "suspicious"])]
        _, _, d2f, _ = precision_recall_fscore_support(
            d2v["expected_label"], d2v["predicted_label"],
            labels=["fake", "genuine", "suspicious"], zero_division=0
        )
        print(f"  Day2 binary F1: {sum(d2f[:2])/2:.3f}  latency p50: {d2['latency_seconds'].median():.1f}s")
    except FileNotFoundError:
        pass
    try:
        d3 = pd.read_csv("../day3-langgraph/data/eval_results_day3.csv")
        d3v = d3[d3["predicted_label"].isin(["fake", "genuine", "suspicious"])]
        _, _, d3f, _ = precision_recall_fscore_support(
            d3v["expected_label"], d3v["predicted_label"],
            labels=["fake", "genuine", "suspicious"], zero_division=0
        )
        print(f"  Day3 binary F1: {sum(d3f[:2])/2:.3f}  latency p50: {d3['latency_seconds'].median():.1f}s")
    except FileNotFoundError:
        pass
    print(f"  Day4 binary F1: {sum(f[:2])/2:.3f}  latency p50: {df['latency_seconds'].median():.1f}s")


if __name__ == "__main__":
    main()