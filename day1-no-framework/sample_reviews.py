# 5 reviews to test the agent on. Mix of obvious fakes, obvious genuine,
# and one ambiguous case where reviewer history matters.

SAMPLE_REVIEWS = [
    {
        "review_id": "R001",
        "reviewer_id": "U101",
        "product": "Wireless Headphones",
        "rating": 5,
        "text": "Best product ever!!! Amazing quality!!! Buy now!!! Five stars!!!",
        "expected_label": "fake",  # for your eyeballing later
    },
    {
        "review_id": "R002",
        "reviewer_id": "U102",
        "product": "Wireless Headphones",
        "rating": 4,
        "text": "Sound quality is great, but the ear cushions feel a bit tight after 2 hours. Battery lasts about 25 hours which matches the spec sheet. Worth the price.",
        "expected_label": "genuine",
    },
    {
        "review_id": "R003",
        "reviewer_id": "U103",
        "product": "Coffee Maker",
        "rating": 5,
        "text": "I was skeptical but this changed my life. My mornings are transformed. Highly recommend to everyone.",
        "expected_label": "suspicious",  # ambiguous, history will tell
    },
    {
        "review_id": "R004",
        "reviewer_id": "U104",
        "product": "Running Shoes",
        "rating": 1,
        "text": "Broke after one run. Cheap material. Avoid.",
        "expected_label": "genuine",
    },
    {
        "review_id": "R005",
        "reviewer_id": "U105",
        "product": "Coffee Maker",
        "rating": 5,
        "text": "Excellent excellent excellent product. Five stars five stars. Buy this immediately.",
        "expected_label": "fake",
    },
]

# Mock reviewer history database. In Day 2 this becomes a real lookup.
REVIEWER_HISTORY = {
    "U101": {"total_reviews": 47, "avg_rating": 4.98, "account_age_days": 12, "all_5_star_pct": 0.96},
    "U102": {"total_reviews": 23, "avg_rating": 3.7, "account_age_days": 890, "all_5_star_pct": 0.35},
    "U103": {"total_reviews": 3, "avg_rating": 5.0, "account_age_days": 8, "all_5_star_pct": 1.0},
    "U104": {"total_reviews": 156, "avg_rating": 3.4, "account_age_days": 1840, "all_5_star_pct": 0.22},
    "U105": {"total_reviews": 89, "avg_rating": 4.95, "account_age_days": 19, "all_5_star_pct": 0.94},
}

# Mock "similar reviews" corpus. Day 2 this becomes a real vector store.
SIMILAR_REVIEWS_CORPUS = [
    "Best product ever amazing quality five stars",
    "Sound is decent for the price point",
    "Changed my life highly recommend to everyone",
    "Broke quickly poor build quality",
    "Excellent excellent buy immediately",
    "Battery life matches the spec",
    "Comfortable for long sessions",
    "Cheap plastic feel",
]