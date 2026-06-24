# TakeMeter: Hacker News Discourse Classifier

## 1. Community
**What community did you choose and why?**
I chose Hacker News (`news.ycombinator.com`). This community is an excellent fit for a classification task because it is highly active, entirely text-based, and features a massive variance in discourse quality. While the community prides itself on deep technical expertise and thoughtful analysis, comment threads frequently devolve into cynical complaining, low-effort dismissals, or inside jokes (like "rewrite it in Rust"). This makes distinguishing between genuine technical insight and unbacked opinion an interesting and highly relevant classification challenge.

## 2. Labels
1. **`technical_insight`**: The commenter explains how something works under the hood, shares a specific engineering experience, or corrects a technical misconception with concrete details.
   - *Example 1:* "Actually, the reason this database is slow is because of how Postgres handles MVCC. When you have high update churn, the vacuum process can't keep up..."
   - *Example 2:* "I built a similar system using Rust, and the main bottleneck we found was lock contention on the async runtime, not the network I/O."
2. **`opinion_or_critique`**: The commenter shares a strong personal opinion, complains about a company/technology, or debates the merits of the post without providing a deep technical explanation or evidence.
   - *Example 1:* "I'm so tired of every app requiring a subscription now. I miss when you could just buy software once and own it."
   - *Example 2:* "This entire startup is just a thin wrapper around OpenAI's API. There is absolutely no moat here."
3. **`joke_or_meta`**: The comment is pure sarcasm, a meme, or just a meta-comment about Hacker News itself.
   - *Example 1:* "Can't wait to rewrite this in Rust next week."
   - *Example 2:* "Of course this is on the front page, HN loves anything tangentially related to obscure Lisp dialects."

## 3. Hard Edge Cases
**The Anticipated Edge Case:** The "Buzzword Complaint." 
* *Example Post:* "PostgreSQL is completely unusable for modern apps because of MVCC."
* *The Dilemma:* This borders between `opinion_or_critique` (it's a complaint) and `technical_insight` (it drops a specific database architecture term).
* *Decision Rule:* If the comment drops a technical term but does not actually *explain* the mechanism or share a concrete, detailed engineering anecdote, label it `opinion_or_critique`. The comment must contain an actual explanation or substantive new information to earn the `technical_insight` label.

## 4. Data Collection Plan
**Where will you collect examples? How many per label?**
I wrote a Python script (`scrape_hn.py`) that uses the official Hacker News Firebase API to scrape the top and newest stories. It extracts the top 10 substantive comments (length > 100 characters) from each thread until 200 comments are collected. The data is saved to `hn_dataset.csv`. I will manually annotate these 200 examples. If any label (especially `joke_or_meta`) is underrepresented (less than 20% of the dataset), I will manually scrape highly upvoted comment threads known for meta-commentary to balance the dataset.

## 5. Evaluation Metrics
**Which metrics will you use to evaluate your model and why?**
I will report Overall Accuracy, but my primary evaluation metric will be the **Per-Class F1-Score**, particularly for the `joke_or_meta` and `technical_insight` classes. Accuracy alone is insufficient because HN comments are heavily skewed towards opinions; if the model simply guesses `opinion_or_critique` for every post, it might achieve a high accuracy while completely failing to identify the minority classes (jokes and deep technical insights). F1 ensures the model is penalized for both false positives and false negatives in those critical minority classes.

## 6. Definition of Success
**What performance would make this classifier genuinely useful?**
A success threshold for this model is **75% to 80% accuracy** with an F1 score above **0.70** for all classes. Human language—especially sarcasm and cynicism—is highly subjective. Even two humans reading a Hacker News thread might disagree 20% of the time on whether a comment is a "joke" or a "genuine critique." If the model hits 80%, it is roughly matching human-level agreement for this ambiguous task, which would make it "good enough" to deploy as an automated HN thread-filtering tool.

## 7. AI Tool Plan

### Planned
* **Label stress-testing:** Before I finish my 200 annotations, I will give Claude/ChatGPT my 3 label definitions and ask it to generate 5 ambiguous Hacker News comments. I will test my decision rules against these generated posts to ensure my boundaries hold up.
* **Annotation assistance:** I will use an LLM for pre-labeling.
* **Failure analysis:** After testing the fine-tuned model, I will take all the misclassified comments from the confusion matrix, feed them to an LLM, and prompt it: *"Here are the comments my model got wrong. Identify any systematic patterns (e.g., are they all short? Do they all rely on sarcasm?)."* I will verify these patterns myself and write them into the final evaluation report.

### Actual (updated as work progresses)
* **Annotation assistance (done):** Claude was prompted with the three label definitions and the "buzzword complaint" decision rule and asked to pre-label all 200 scraped comments. Output saved to `data/hn_dataset_prelabeled.csv` and reviewed manually. Actual distribution: `opinion_or_critique` 146, `technical_insight` 43, `joke_or_meta` 11. The `joke_or_meta` class fell to 5.5% — below the 20% rebalancing threshold in the plan — but manual rebalancing was not done due to time constraints.
* **Notebook setup (done):** Claude Code filled in the `LABEL_MAP`, `CSV_PATH`, and `SYSTEM_PROMPT` cells in the starter notebook. The system prompt was derived directly from the label definitions in this file.
* **Label stress-testing:** TODO
* **Failure analysis:** TODO — run after fine-tuning completes.
