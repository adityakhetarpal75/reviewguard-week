import time
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from agent import run_review

EVAL_CSV = "data/eval_set.csv"
RESULTS_CSV = "data/eval_results_day5.csv"


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
            verdict = run_review(review)
            label = verdict.label
            confidence = verdict.confidence
            flagged = verdict.flagged_for_human
            error = ""
        except Exception as e:
            label, confidence, flagged, error = "error", 0.0, False, str(e)[:200]

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
            "flagged_for_human": flagged,
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
    else:
        print(f"\n✅ Zero parse errors")

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

    print(f"\n{'='*60}")
    print("FULL WEEK COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Day':6s}  {'Framework':12s}  {'F1':>8s}  {'Latency p50':>12s}  {'Errors':>8s}")

    days = [
        ("Day2", "Raw SDK", "../day1-no-framework/data/eval_results_day2.csv"),
        ("Day3", "LangGraph", "../day3-langgraph/data/eval_results_day3.csv"),
        ("Day4", "CrewAI", "../day4-crewai/data/eval_results_day4.csv"),
    ]
    for day, fw, path in days:
        try:
            d = pd.read_csv(path)
            dv = d[d["predicted_label"].isin(["fake", "genuine", "suspicious"])]
            _, _, df2, _ = precision_recall_fscore_support(
                dv["expected_label"], dv["predicted_label"],
                labels=["fake", "genuine", "suspicious"], zero_division=0
            )
            errors = len(d) - len(dv)
            print(f"  {day:6s}  {fw:12s}  {sum(df2[:2])/2:>8.3f}  {d['latency_seconds'].median():>12.1f}s  {errors:>8d}")
        except FileNotFoundError:
            pass

    d5_f = sum(f[:2]) / 2
    print(f"  {'Day5':6s}  {'Pydantic AI':12s}  {d5_f:>8.3f}  {df['latency_seconds'].median():>12.1f}s  {n_invalid:>8d}")


if __name__ == "__main__":
    main()