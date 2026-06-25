"""
Targeted scraper for joke_or_meta HN comments.

Pass 2 expands on Pass 1 by:
- Pulling from all four feeds: ask, show, top, new (vs. just ask+show)
- Pre-seeding seen_texts from existing CSVs to avoid duplicates
- Target 360 candidates (3x the original 120)

The original scrape_hn.py biases that suppressed joke_or_meta:
1. 100-char minimum — jokes are often 40–90 chars; lowered to 40 here
2. top+new stories only — ask/show threads skew more sarcastic/meta

Run: uv run python scripts/scrape_hn_jokes.py
Output: data/hn_jokes_candidates_p2.csv
Then label it and run: uv run python scripts/merge_joke_supplement.py
"""

import urllib.request
import json
import csv
import time
import re
TARGET_COUNT = 360
MIN_LENGTH = 40

EXISTING_CSVS = [
    "data/hn_dataset_prelabeled.csv",
    "data/hn_jokes_candidates.csv",   # pass 1 (already labeled)
]

def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def clean_html(raw_html):
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", raw_html)
    return (
        cleantext
        .replace("&#x27;", "'")
        .replace("&quot;", '"')
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&#x2F;", "/")
        .strip()
    )

def load_seen_texts():
    seen = set()
    for path in EXISTING_CSVS:
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    t = row.get("text", "").strip()
                    if t:
                        seen.add(t)
        except FileNotFoundError:
            pass
    print(f"Pre-seeded {len(seen)} already-seen texts from existing CSVs")
    return seen

def fetch_story_ids():
    feeds = {
        "ask":  "https://hacker-news.firebaseio.com/v0/askstories.json",
        "show": "https://hacker-news.firebaseio.com/v0/showstories.json",
        "top":  "https://hacker-news.firebaseio.com/v0/topstories.json",
        "new":  "https://hacker-news.firebaseio.com/v0/newstories.json",
    }
    results = {name: get_json(url) or [] for name, url in feeds.items()}
    for name, ids in results.items():
        print(f"  {name}: {len(ids)} stories")

    # Interleave ask/show first (higher joke density), then top/new
    ask, show = results["ask"], results["show"]
    combined = []
    for a, s in zip(ask, show):
        combined.extend([a, s])
    combined.extend(ask[len(show):])
    combined.extend(show[len(ask):])
    for t, n in zip(results["top"], results["new"]):
        combined.extend([t, n])
    combined.extend(results["top"][len(results["new"]):])
    combined.extend(results["new"][len(results["top"]):])
    return list(dict.fromkeys(combined))

def main():
    print("Fetching story IDs from all four feeds...")
    story_ids = fetch_story_ids()
    print(f"Total unique stories: {len(story_ids)}")

    seen_texts = load_seen_texts()
    collected = []

    for story_id in story_ids:
        if len(collected) >= TARGET_COUNT:
            break

        story = get_json(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
        if not story or story.get("type") != "story" or not story.get("title"):
            continue

        title = story.get("title", "")
        url = story.get("url", f"https://news.ycombinator.com/item?id={story_id}")
        kids = story.get("kids", [])
        if not kids:
            continue

        print(f"Checking: {title[:70]}")

        for comment_id in kids[:15]:
            if len(collected) >= TARGET_COUNT:
                break

            comment = get_json(f"https://hacker-news.firebaseio.com/v0/item/{comment_id}.json")
            if not comment:
                continue
            if comment.get("type") != "comment":
                continue
            if comment.get("deleted") or comment.get("dead"):
                continue
            raw = comment.get("text", "")
            if not raw:
                continue

            text = clean_html(raw)
            if len(text) < MIN_LENGTH:
                continue
            if text in seen_texts:
                continue

            seen_texts.add(text)
            collected.append({
                "story_title": title,
                "story_url": url,
                "text": text,
                "label": "",
                "notes": "targeted-joke-pass-2",
            })
            print(f"  Collected {len(collected)}/{TARGET_COUNT}")
            time.sleep(0.05)

    out_path = "data/hn_jokes_candidates_p2.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["story_title", "story_url", "text", "label", "notes"]
        )
        writer.writeheader()
        writer.writerows(collected)

    print(f"\nSaved {len(collected)} candidates to {out_path}")
    print("Label the CSV, then run: uv run python scripts/merge_joke_supplement.py")

if __name__ == "__main__":
    main()
