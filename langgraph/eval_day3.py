import time
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from graph import run_review

EVAL_CSV = "data/eval_set.csv"
RESULTS_CSV = "data/eval_results_day3.csv"


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
            state = run_review(review)
            label = state.get("label", "error")
            confidence = state.get("confidence", 0.0)
            escalated = state.get("escalated", False)
            error = ""
        except Exception as e:
            label, confidence, escalated, error = "error", 0.0, False, str(e)[:200]

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
            "flagged_for_human": escalated,
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
        print(f"\n⚠  {n_invalid} parse errors excluded")

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
        binary_true, binary_pred, labels=["fake_or_flagged", "genuine"], zero_division=0
    )
    print(f"\nBinary view:")
    print(f"  {'label':18s}  {'precision':>10s}  {'recall':>10s}  {'f1':>10s}  {'support':>8s}")
    for lbl, pi, ri, fi, si in zip(["fake_or_flagged", "genuine"], bp, br, bf, bs):
        print(f"  {lbl:18s}  {pi:>10.3f}  {ri:>10.3f}  {fi:>10.3f}  {si:>8d}")

    cm = confusion_matrix(valid["expected_label"], valid["predicted_label"], labels=labels)
    print(f"\nConfusion matrix:")
    print(f"           {'  '.join(f'{l:>10s}' for l in labels)}")
    for lbl, row in zip(labels, cm):
        print(f"  {lbl:8s}  {'  '.join(f'{v:>10d}' for v in row)}")

    flagged = valid["flagged_for_human"].sum()
    print(f"\nHIL rate: {flagged}/{len(valid)} = {flagged/len(valid):.1%}")
    print(f"Latency: mean={df['latency_seconds'].mean():.1f}s  p50={df['latency_seconds'].median():.1f}s  p95={df['latency_seconds'].quantile(0.95):.1f}s")

    # Day 2 comparison
    try:
        d2 = pd.read_csv("../day1-no-framework/data/eval_results_day2.csv")
        print(f"\n--- vs Day 2 ---")
        print(f"  latency p50:  Day2={d2['latency_seconds'].median():.1f}s  Day3={df['latency_seconds'].median():.1f}s")
        d2_acc = (d2["expected_label"] == d2["predicted_label"]).mean()
        d3_acc = (valid["expected_label"] == valid["predicted_label"]).mean()
        print(f"  3-class acc:  Day2={d2_acc:.2%}  Day3={d3_acc:.2%}")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()