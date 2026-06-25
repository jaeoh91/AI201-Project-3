"""
Merges labeled joke_or_meta candidates into the main dataset.

Reads all supplement CSVs (pass 1 + pass 2) and pulls joke_or_meta rows
until TARGET_MINIMUM is reached.

Usage:
  1. Label data/hn_jokes_candidates.csv and data/hn_jokes_candidates_p2.csv
  2. Run: uv run python scripts/merge_joke_supplement.py

Output: data/hn_dataset_balanced.csv
"""

import csv
from collections import Counter

MAIN_CSV = "data/hn_dataset_prelabeled.csv"
SUPPLEMENT_CSVS = [
    "data/hn_jokes_candidates.csv",
    "data/hn_jokes_candidates_p2.csv",
]
OUTPUT_CSV = "data/hn_dataset_balanced.csv"
TARGET_CLASS = "joke_or_meta"
TARGET_MINIMUM = 50  # raised now that pass 2 gives us more jokes to choose from
MAX_OPINION = 122    # cap opinion_or_critique to bring technical_insight to ~20%


def read_csv(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def main():
    main_rows = read_csv(MAIN_CSV)

    existing_jokes = [r for r in main_rows if r["label"] == TARGET_CLASS]
    print(f"Existing '{TARGET_CLASS}' in main dataset: {len(existing_jokes)}")

    # Collect all labeled jokes across both supplement passes
    all_new_jokes = []
    for path in SUPPLEMENT_CSVS:
        rows = read_csv(path)
        jokes = [r for r in rows if r.get("label", "").strip() == TARGET_CLASS]
        print(f"  {path}: {len(jokes)} joke_or_meta rows")
        all_new_jokes.extend(jokes)

    print(f"Total new '{TARGET_CLASS}' available: {len(all_new_jokes)}")

    needed = max(0, TARGET_MINIMUM - len(existing_jokes))
    to_add = all_new_jokes[:needed]
    print(f"Adding {len(to_add)} rows (need {needed} to reach {TARGET_MINIMUM} total)")

    combined = main_rows + to_add

    # Cap majority class
    opinion_rows = [r for r in combined if r["label"] == "opinion_or_critique"]
    other_rows   = [r for r in combined if r["label"] != "opinion_or_critique"]
    if len(opinion_rows) > MAX_OPINION:
        print(f"Capping opinion_or_critique: {len(opinion_rows)} → {MAX_OPINION}")
        opinion_rows = opinion_rows[:MAX_OPINION]
    combined = opinion_rows + other_rows

    # Print class distribution
    dist = Counter(r["label"] for r in combined)
    total = len(combined)
    print(f"\nNew distribution ({total} total):")
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {label:<25} {count:>4}  ({100*count/total:.1f}%)")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["story_title", "story_url", "text", "label", "notes"]
        )
        writer.writeheader()
        writer.writerows(combined)

    print(f"\nSaved to {OUTPUT_CSV}")
    print(f"Update CSV_PATH in the notebook to: ../data/hn_dataset_balanced.csv")


if __name__ == "__main__":
    main()
