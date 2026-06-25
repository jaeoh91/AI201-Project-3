"""
Apply hardcoded labels to data/hn_jokes_candidates_p2.csv.

Labels determined by manual review of all 360 rows.
Run: uv run python scripts/label_jokes_p2.py
"""

import csv

INPUT_CSV = "data/hn_jokes_candidates_p2.csv"

# 1-indexed positions of non-opinion_or_critique rows
JOKE_OR_META = {
    11, 14, 33, 103, 129, 135, 159, 183, 192, 195,
    209, 214, 217, 220, 226, 252, 260, 271, 299, 317,
    318, 326, 332, 338, 341, 350, 352, 353, 356,
}
TECHNICAL_INSIGHT = {
    17, 55, 57, 62, 76, 125, 130, 164, 170, 218,
    243, 248, 254, 256, 266, 287, 297, 331, 357,
}


def get_label(one_indexed_row):
    if one_indexed_row in JOKE_OR_META:
        return "joke_or_meta"
    if one_indexed_row in TECHNICAL_INSIGHT:
        return "technical_insight"
    return "opinion_or_critique"


def main():
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows from {INPUT_CSV}")

    for i, row in enumerate(rows, start=1):
        row["label"] = get_label(i)

    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<25} {count:>4}  ({100*count/len(rows):.1f}%)")

    with open(INPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["story_title", "story_url", "text", "label", "notes"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nLabels written to {INPUT_CSV}")


if __name__ == "__main__":
    main()
