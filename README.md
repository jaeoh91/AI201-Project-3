# TakeMeter: Hacker News Discourse Classifier

A fine-tuned DistilBERT model that classifies Hacker News comments into three discourse categories: genuine technical insight, opinion/critique, or joke/meta.

---

## Community

**Hacker News** (`news.ycombinator.com`) was chosen because it is highly active, entirely text-based, and features an enormous variance in discourse quality. The community prides itself on deep technical expertise, but threads frequently devolve into cynical complaining, low-effort dismissals, or inside jokes ("rewrite it in Rust"). This makes distinguishing genuine technical insight from unbacked opinion an interesting and relevant classification challenge — one where a working model could realistically filter thread quality.

---

## Label Taxonomy

### `technical_insight`
The commenter explains how something works under the hood, shares a specific engineering experience, or corrects a technical misconception with concrete details. The comment must contain an actual explanation or substantive new information — dropping a technical term alone is insufficient.

- **Example 1:** "Actually, the reason this database is slow is because of how Postgres handles MVCC. When you have high update churn, the vacuum process can't keep up, leading to table bloat and sequential scans."
- **Example 2:** "I built a similar system using Rust, and the main bottleneck we found was lock contention on the async runtime, not the network I/O."

### `opinion_or_critique`
The commenter shares a strong personal opinion, complains about a company/technology, or debates the merits of the post without providing a deep technical explanation or evidence.

- **Example 1:** "I'm so tired of every app requiring a subscription now. I miss when you could just buy software once and own it."
- **Example 2:** "This entire startup is just a thin wrapper around OpenAI's API. There is absolutely no moat here."

### `joke_or_meta`
The comment is pure sarcasm, a meme, or a meta-comment about Hacker News itself.

- **Example 1:** "Can't wait to rewrite this in Rust next week."
- **Example 2:** "Of course this is on the front page, HN loves anything tangentially related to obscure Lisp dialects."

---

## Data Collection

**Source:** Official Hacker News Firebase API (`https://hacker-news.firebaseio.com/v0/`)

**Collection script:** `scripts/scrape_hn.py` fetches the union of `topstories` and `newstories`, takes up to 10 top-level comments per thread, filters to comments longer than 100 characters (to exclude low-effort "Agreed"-type responses), and stops at 200 examples. Stories without a title or URL are skipped.

**Labeling process:** Comments were pre-labeled using an LLM (Claude) prompted with the three label definitions and the decision rule for the "buzzword complaint" edge case from `planning.md`. Labels were then reviewed manually. The final annotated file is `data/hn_dataset_prelabeled.csv`.

**Label distribution:**

| Label | Count | % |
|---|---|---|
| `opinion_or_critique` | 146 | 73.0% |
| `technical_insight` | 43 | 21.5% |
| `joke_or_meta` | 11 | 5.5% |
| **Total** | **200** | |

The heavy skew toward `opinion_or_critique` reflects the actual distribution on HN — most comments are opinions, not deep technical explanations or sarcastic one-liners.

**Three difficult-to-label examples:**

> TODO: Pick 3 specific rows from `data/hn_dataset_prelabeled.csv` where the label decision was non-obvious. For each, quote the comment text, state the label assigned, and explain which decision rule from `planning.md` resolved the ambiguity.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace Transformers)

**Dataset split:** 70% train / 15% validation / 15% test, stratified by label.

| Split | Examples |
|---|---|
| Train | 140 |
| Validation | 30 |
| Test | 30 |

**Training setup:**

> TODO: Fill in after running Section 3 of the notebook. Include: number of epochs, batch size, learning rate, weight decay, and whether early stopping was used.

**Hyperparameter decision:**

> TODO: Describe at least one deliberate hyperparameter choice and why (e.g., "Reduced epochs from 5 to 3 to avoid overfitting on the 8 `joke_or_meta` training examples").

---

## Baseline

**Model:** LLaMA-3.3-70B-Versatile (via Groq API, `temperature=0`)

**Method:** Zero-shot classification. Each test comment was passed to the model with the following system prompt:

```
You are classifying Hacker News comments into exactly one of three categories.

technical_insight: The commenter explains how something works under the hood, shares a specific engineering experience, or corrects a technical misconception with concrete details. The comment must contain an actual explanation or substantive new information — dropping a technical term alone is not sufficient.
Example: "Actually, the reason this database is slow is because of how Postgres handles MVCC. When you have high update churn, the vacuum process can't keep up, leading to table bloat and sequential scans."

opinion_or_critique: The commenter shares a strong personal opinion, complains about a company or technology, or debates the merits of the post without providing a deep technical explanation or evidence.
Example: "This entire startup is just a thin wrapper around OpenAI's API. There is absolutely no moat here."

joke_or_meta: The comment is pure sarcasm, a meme, or a meta-comment about Hacker News itself.
Example: "Can't wait to rewrite this in Rust next week."

Respond with ONLY the label name, exactly as written below. Do not explain your reasoning.

Valid labels:
technical_insight
opinion_or_critique
joke_or_meta
```

**Collection:** Run via Section 5 of `notebooks/ai201_project3_takemeter_starter_clean.ipynb` on the locked 30-example test set. All 30 responses were parseable (0% parse failure rate).

---

## Evaluation

### Baseline — LLaMA-3.3-70B (zero-shot)

**Accuracy: 86.7%** (26/30 correct, evaluated on 30/30 parseable responses)

| Label | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `technical_insight` | 0.75 | 0.86 | 0.80 | 7 |
| `opinion_or_critique` | 0.91 | 0.91 | 0.91 | 22 |
| `joke_or_meta` | 0.00 | 0.00 | 0.00 | 1 |
| **macro avg** | 0.55 | 0.59 | 0.57 | 30 |
| **weighted avg** | 0.84 | 0.87 | 0.85 | 30 |

**Confusion matrix:**

| Actual \ Predicted | `technical_insight` | `opinion_or_critique` | `joke_or_meta` |
|---|---|---|---|
| `technical_insight` | 6 | 1 | 0 |
| `opinion_or_critique` | 2 | 20 | 0 |
| `joke_or_meta` | 0 | 1 | 0 |

Key findings:
- The model is well-calibrated on `opinion_or_critique` (the 73% majority class) but the high overall accuracy is largely a function of that dominance.
- 2 `opinion_or_critique` comments were misclassified as `technical_insight` — the "buzzword complaint" edge case: comments that reference a technical term without explaining it.
- The single `joke_or_meta` test example was absorbed into `opinion_or_critique` — zero-shot LLaMA cannot distinguish sarcasm from genuine criticism.
- The macro avg F1 (0.57) vs weighted avg (0.85) gap reveals the model is essentially ignoring the minority class.

---

### Fine-Tuned Model — DistilBERT

> TODO: Run Sections 3 and 4 of the notebook, then fill in:
> - Overall accuracy
> - Per-class precision, recall, F1 table (same format as above)
> - Confusion matrix as markdown table

---

### Model Comparison

> TODO: Add a two-row summary table comparing baseline vs. fine-tuned on accuracy and macro F1.

---

### Error Analysis: 3 Wrong Predictions

> TODO: After fine-tuning, pick 3 specific misclassified examples from the Section 4 wrong-predictions output. For each, include: the comment text, the true label, the predicted label, and a hypothesis for why the model got it wrong.

---

### Sample Classifications

> TODO: Table of 3–5 examples from the test set with: comment text (truncated), true label, predicted label, and confidence score. Include at least one correct `technical_insight` prediction and explain why it was clearly identifiable.

---

## Reflection

> TODO: After fine-tuning, write 2–3 sentences on:
> - What pattern the fine-tuned model learned that the baseline did not
> - Whether fine-tuning improved macro F1 on the minority class (`joke_or_meta`)
> - Any unexpected behavior (e.g., regression on `opinion_or_critique` accuracy)

---

## Spec Reflection

> TODO:
> - **One way the spec helped:** Describe one specific decision from `planning.md` that paid off during implementation (e.g., the "buzzword complaint" decision rule making ambiguous labels consistent).
> - **One way implementation diverged:** Describe one place where what you planned and what you actually did differed, and why (e.g., the `joke_or_meta` class ended up with only 11 examples despite the plan to balance it, because suitable meta-commentary threads were hard to find in the scrape window).

---

## AI Usage

1. **Pre-labeling (annotation assistance):** Claude was prompted with the three label definitions and the "buzzword complaint" decision rule from `planning.md` and asked to label all 200 scraped comments. The instruction was to assign `opinion_or_critique` to any comment that uses a technical term without actually explaining the mechanism.
   > TODO: Describe what you reviewed, overrode, or corrected in the AI's labels. Were there systematic errors? Did you flip any labels and why?

2. **Notebook setup:** Claude Code was used to fill in the `LABEL_MAP`, `CSV_PATH`, and `SYSTEM_PROMPT` cells in the starter notebook during this session. The system prompt was generated from the planning.md definitions and verified before running the baseline. No changes were made to the generated prompt before running.

   > TODO: Add at least one more specific AI-assisted step (e.g., label stress-testing, failure analysis) and describe what you directed the AI to do and what you revised or overrode.
