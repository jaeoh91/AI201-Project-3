import urllib.request
import json
import csv
import time
import re

def get_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.replace("&#x27;", "'").replace("&quot;", '"').replace("&gt;", ">").replace("&amp;", "&").replace("&#x2F;", "/")

def main():
    print("Fetching top and new stories...")
    top_stories = get_json("https://hacker-news.firebaseio.com/v0/topstories.json") or []
    new_stories = get_json("https://hacker-news.firebaseio.com/v0/newstories.json") or []
    
    all_stories = list(dict.fromkeys(top_stories + new_stories)) # remove duplicates
    
    comments_collected = []
    
    print(f"Checking {len(all_stories)} stories for comments...")
    for story_id in all_stories:
        if len(comments_collected) >= 200:
            break
            
        story = get_json(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
        if not story or story.get("type") != "story" or not story.get("title"):
            continue
            
        story_title = story.get("title", "")
        # Use provided URL if available, otherwise link to the HN comments page
        story_url = story.get("url", f"https://news.ycombinator.com/item?id={story_id}")
        
        print(f"Checking story: {story_title}")
        kids = story.get("kids", [])
        # take up to 10 top-level comments per story to ensure variety
        for comment_id in kids[:10]: 
            if len(comments_collected) >= 200:
                break
            
            comment = get_json(f"https://hacker-news.firebaseio.com/v0/item/{comment_id}.json")
            if comment and comment.get("type") == "comment" and comment.get("text") and not comment.get("deleted") and not comment.get("dead"):
                text = clean_html(comment.get("text"))
                # We want substantive takes, not just "agreed"
                if len(text) > 100: 
                    comments_collected.append({
                        "story_title": story_title,
                        "story_url": story_url,
                        "text": text,
                        "label": "",
                        "notes": ""
                    })
                    print(f"  Collected comment. Total: {len(comments_collected)}/200")
            time.sleep(0.05)
                
    print(f"Finished collecting {len(comments_collected)} comments.")
    
    with open("hn_dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["story_title", "story_url", "text", "label", "notes"])
        writer.writeheader()
        writer.writerows(comments_collected)
        
    print("Saved to hn_dataset.csv")

if __name__ == "__main__":
    main()
