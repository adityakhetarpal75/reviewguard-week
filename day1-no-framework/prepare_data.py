"""
Prepare ReviewGuard data from the Amazon fake reviews dataset.

Outputs:
  data/reviews_corpus.csv  - 200 reviews for the RAG corpus
  data/eval_set.csv        - 50 reviews for evaluation (no overlap with corpus)

Both files have schema: review_id, reviewer_id, product, rating, text, label
                       + reviewer behavioral metadata for check_reviewer_history

Note: The original CSV has no reviewer info, so we synthesize reviewer profiles
that probabilistically correlate with the true label (70% aligned, 30% inverted)
to keep the agent's two-signal design meaningful.
"""

import pandas as pd
import numpy as np
import random

# Reproducibility — use the same seed every run so corpus and eval are stable
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INPUT_FILE = "data/fake reviews dataset.csv"
CORPUS_OUT = "data/reviews_corpus.csv"
EVAL_OUT = "data/eval_set.csv"

CORPUS_SIZE = 200  # 100 fake + 100 genuine
EVAL_SIZE = 50     # 25 fake + 25 genuine


def synthesize_reviewer_profile(true_label: str, idx: int) -> dict:
    """
    Generate a plausible reviewer profile.

    70% of profiles match the label's typical signature.
    30% are 'inverted' — fake review from a normal-looking account, or
    genuine review from a suspicious-looking account. This is realistic
    (sophisticated fakers age accounts; new users post genuine reviews)
    and forces the agent to weigh signals rather than match on one.
    """
    aligned = random.random() < 0.7
    looks_fake = (true_label == "fake") if aligned else (true_label == "genuine")

    if looks_fake:
        # Fake-ish profile: new account, high volume, near-perfect ratings
        account_age_days = random.randint(5, 60)
        total_reviews = random.randint(20, 150)
        all_5_star_pct = round(random.uniform(0.85, 0.99), 2)
        avg_rating = round(random.uniform(4.7, 4.99), 2)
    else:
        # Genuine-ish profile: established account, balanced ratings
        account_age_days = random.randint(180, 2500)
        total_reviews = random.randint(5, 80)
        all_5_star_pct = round(random.uniform(0.15, 0.55), 2)
        avg_rating = round(random.uniform(3.2, 4.4), 2)

    return {
        "reviewer_id": f"U{idx:04d}",
        "account_age_days": account_age_days,
        "total_reviews": total_reviews,
        "avg_rating": avg_rating,
        "all_5_star_pct": all_5_star_pct,
    }


def main():
    print("Loading Amazon fake reviews dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df)} rows")

    # Clean & rename columns
    df = df.rename(columns={"text_": "text", "category": "product"})
    df["label"] = df["label"].map({"CG": "fake", "OR": "genuine"})
    # Drop any rows with missing text (defensive)
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    # Sample balanced sets — corpus and eval, no overlap
    fakes = df[df["label"] == "fake"].sample(
        n=(CORPUS_SIZE // 2 + EVAL_SIZE // 2), random_state=SEED
    )
    genuines = df[df["label"] == "genuine"].sample(
        n=(CORPUS_SIZE // 2 + EVAL_SIZE // 2), random_state=SEED
    )

    # Split: first half goes to corpus, second to eval
    corpus_fakes = fakes.iloc[: CORPUS_SIZE // 2]
    eval_fakes = fakes.iloc[CORPUS_SIZE // 2 :]
    corpus_genuines = genuines.iloc[: CORPUS_SIZE // 2]
    eval_genuines = genuines.iloc[CORPUS_SIZE // 2 :]

    corpus = pd.concat([corpus_fakes, corpus_genuines]).sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)
    eval_set = pd.concat([eval_fakes, eval_genuines]).sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)

    # Add review_id and synthesized reviewer profiles
    def enrich(df_subset, prefix: str) -> pd.DataFrame:
        rows = []
        for i, row in df_subset.iterrows():
            profile = synthesize_reviewer_profile(row["label"], i)
            rows.append(
                {
                    "review_id": f"{prefix}{i:04d}",
                    "reviewer_id": profile["reviewer_id"],
                    "product": row["product"],
                    "rating": int(row["rating"]),
                    "text": row["text"],
                    "label": row["label"],
                    "account_age_days": profile["account_age_days"],
                    "total_reviews": profile["total_reviews"],
                    "avg_rating": profile["avg_rating"],
                    "all_5_star_pct": profile["all_5_star_pct"],
                }
            )
        return pd.DataFrame(rows)

    corpus_final = enrich(corpus, "C")
    eval_final = enrich(eval_set, "E")

    corpus_final.to_csv(CORPUS_OUT, index=False)
    eval_final.to_csv(EVAL_OUT, index=False)

    print(f"\n  Wrote {len(corpus_final)} rows to {CORPUS_OUT}")
    print(f"  Wrote {len(eval_final)} rows to {EVAL_OUT}")
    print(f"\n  Corpus label distribution:\n{corpus_final['label'].value_counts()}")
    print(f"\n  Eval label distribution:\n{eval_final['label'].value_counts()}")
    print(f"\n  Sample corpus row:\n{corpus_final.iloc[0].to_dict()}")


if __name__ == "__main__":
    main()