# TakeMeter: Hacker News Discourse Classifier

A fine-tuned DistilBERT model that classifies Hacker News comments into three discourse categories: genuine technical insight, opinion/critique, or joke/meta.

> **Reviewer note:** This README is the final report and is self-contained. Working notes and the full decision history live in [`planning.md`](planning.md); the label rubric and the v3 dataset audit live in [`LABELING_RUBRIC.md`](LABELING_RUBRIC.md).

---

## Deployed Interface

[`app.py`](app.py) loads the fine-tuned model and classifies any comment you paste in, showing the predicted label and the confidence for **all three** classes.

**One-time setup.** The trained weights (~770 MB) are *not* committed. Produce them by running Sections 1–3 of the [notebook](notebooks/ai201_project3_takemeter_starter_clean.ipynb) — they save to `notebooks/takemeter-model/`, which `app.py` discovers automatically. (Optional, for a stable path: run `trainer.save_model("../model"); tokenizer.save_pretrained("../model")` in the notebook; the app prefers a top-level `model/` dir if present.)

**Web UI** (Gradio — use this for the demo video):
```bash
uv run python app.py
```
Opens a browser app: type or paste a comment → see the top label and a confidence bar for each class.

**Command line** (no browser, handy for quick checks):
```bash
uv run python app.py "Can't wait to rewrite this in Rust next week."
```
```
Prediction: joke_or_meta  (53% confidence)
  joke_or_meta           52.8%  ███████████████
  opinion_or_critique    28.6%  ████████
  technical_insight      18.7%  █████
```

Point the app at a specific model with `MODEL_DIR=/path/to/model uv run python app.py`.

---

## Community

**Hacker News** (`news.ycombinator.com`) was chosen because it is highly active, entirely text-based, and features an enormous variance in discourse quality. The community prides itself on deep technical expertise, but threads frequently devolve into cynical complaining, low-effort dismissals, or inside jokes ("rewrite it in Rust"). This makes distinguishing genuine technical insight from unbacked opinion an interesting and relevant classification challenge — one where a working model could realistically filter thread quality.

---

## Label Taxonomy

The decisive question for the hard boundary is **"does the comment *explain a mechanism*, or *express a view about one*?"** Citing data, using jargon, or writing analytically does **not** make a comment technical — those are surface features. The full rubric (v3) is in [`LABELING_RUBRIC.md`](LABELING_RUBRIC.md).

### `technical_insight`
Explains *how* or *why* something works, or states a verifiable mechanism. Referencing a technical topic, asking a technical question, or describing a feature is **not** enough — there must be an actual explanation.
- *"Actually, the reason this database is slow is because of how Postgres handles MVCC. When you have high update churn, the vacuum process can't keep up, leading to table bloat and sequential scans."*

### `opinion_or_critique`
A stance, preference, or critique — **even if it is analytical, cites data, or uses heavy technical vocabulary.**
- *"This entire startup is just a thin wrapper around OpenAI's API. There is absolutely no moat here."*

### `joke_or_meta`
Pure sarcasm, a meme, or a meta-comment about HN itself. (A genuine critique with a sarcastic tone is still opinion.)
- *"Can't wait to rewrite this in Rust next week."*

---

## Data Collection

**Source:** Official Hacker News Firebase API (`https://hacker-news.firebaseio.com/v0/`), via `scripts/scrape_hn.py` (top-level comments, length-filtered).

**Labeling:** Comments were pre-labeled by an LLM (Claude) prompted with the three definitions and the "buzzword complaint" decision rule from `planning.md`, then reviewed manually. (Disclosed in [AI Usage](#ai-usage).)

**Dataset evolution.** The dataset went through three stages — full history in `planning.md` / `LABELING_RUBRIC.md`:

1. **Original (200):** heavily skewed — `opinion_or_critique` 73%, `joke_or_meta` only 5.5%. Jokes were being lost to the scraper's 100-char minimum (jokes are short).
2. **Balanced (215):** targeted re-scrapes of sarcastic feeds raised jokes to 50; `opinion_or_critique` was capped at 122. All classes ≥ 20%.
3. **v3 audit (224, used for this report):** after the first fine-tuning runs revealed an opinion↔technical confusion, I re-audited the `opinion_or_critique` class against the tightened rubric. **Finding: the labels were mostly correct — the confusion was a model problem, not a label problem.** Seven comments that fit *no* class (pure link/resource dumps, a bare feature description, a one-line skeptical question) were dropped, and **16 synthetic boundary examples** were authored to sharpen the failing distinction (tech-heavy *opinions*, true *mechanism* explanations, clean jokes). The synthetic examples are disclosed below and tagged `synthetic` in the `notes` column.

**Final dataset (`data/hn_dataset_v3_full.csv`): 224 examples**

| Label | Count | % |
|---|---|---|
| `opinion_or_critique` | 121 | 54.0% |
| `joke_or_meta` | 52 | 23.2% |
| `technical_insight` | 51 | 22.8% |
| **Total** | **224** | |

**Three difficult-to-label examples** (and the rule that resolved each):

1. *"But can it bypass the magic performed by the SSD controller? In particular, can it be certain that a flush is really a flush?"* — A **technical question**, not an explanation. Rule: questions aren't `technical_insight`, and this isn't really a stance either → poor fit for all three classes; **dropped** in the v3 audit.
2. *"Europeans don't get scolded enough for their resistance to air conditioning… Greece has 2x more heat-related deaths per capita than Mississippi has gun deaths…"* — Cites statistics, but its purpose is to **argue a position**. Rule: a data-backed argument is `opinion_or_critique`, not `technical_insight`.
3. *"Ah, I love CircuitiTikZ. Only way to do simple text-based circuit diagrams… Some years ago I wired it up with asciidoctor…"* — Opens with pure opinion ("I love"), then a concrete engineering anecdote. Rule: labeled `technical_insight` for the anecdote, but genuinely mixed — and the model flips it (see Error Analysis #3).

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace Transformers).

**Split:** 70 / 15 / 15, stratified by label.

| Split | technical | opinion | joke | Total |
|---|---|---|---|---|
| Train | 37 | 86 | 37 | 160 |
| Validation | 7 | 19 | 8 | 34 |
| Test | 7 | 16 | 7 | 30 |

*Train includes the 16 synthetic boundary examples (144 real + 16 synthetic); validation and test are 100% real held-out HN comments.*

**Training setup:** 6 epochs · batch size 16 · learning rate 2e-5 · weight decay 0.01 · `warmup_ratio=0.1` · `load_best_model_at_end=True` with **`metric_for_best_model="f1_macro"`** · class-weighted cross-entropy loss (see below).

**Hyperparameter decisions** (deviations from the starter defaults, with reasoning):

- **#1 — `warmup_steps=50` → `warmup_ratio=0.1`.** With ~160 train examples and batch 16, a run is only ~60 optimization steps; the original 50-step warmup meant the learning rate spent almost the entire run *ramping up* and never decayed. A ratio scales warmup to the actual step count.
- **#2 — epochs 3 → 6.** The "3 epochs" default comes from BERT benchmarks on datasets of thousands. For a few-hundred-example dataset, more passes help (Zhang et al. 2021, *Revisiting Few-sample BERT Fine-tuning*); `load_best_model_at_end` keeps overfitting in check by retaining the best validation checkpoint.
- **#3 — model selection `accuracy` → macro-F1.** With a 54% majority class, accuracy rewards a model that nails `opinion_or_critique` and ignores the minorities. Macro-F1 weights all three classes equally.
- **#4 — class-weighted loss, strength tuned by a β-sweep.** An unweighted model collapsed to the majority class (`joke`/`technical` F1 ≈ 0). I added inverse-frequency class weights, then swept the weighting strength β ∈ {0, 0.3, 0.5, 0.7, 1.0} and selected on **validation** macro-F1. The weight for class *i* is `(N/n_i)^β`, normalized to mean 1. β = 1.0 won decisively on validation (see Reflection for the catch).

---

## Baseline

**Model:** LLaMA-3.3-70B-Versatile (via Groq API, `temperature=0`), zero-shot. Each test comment is classified with this system prompt:

```
You are classifying Hacker News comments into exactly one of three categories.

technical_insight: The commenter explains how something works under the hood, shares a specific
engineering experience, or corrects a technical misconception with concrete details. The comment
must contain an actual explanation or substantive new information — dropping a technical term alone
is not sufficient.
Example: "Actually, the reason this database is slow is because of how Postgres handles MVCC..."

opinion_or_critique: The commenter shares a strong personal opinion, complains about a company or
technology, or debates the merits of the post without providing a deep technical explanation or evidence.
Example: "This entire startup is just a thin wrapper around OpenAI's API. There is absolutely no moat here."

joke_or_meta: The comment is pure sarcasm, a meme, or a meta-comment about Hacker News itself.
Example: "Can't wait to rewrite this in Rust next week."

Respond with ONLY the label name. [valid labels listed]
```

Evaluated on the same locked 30-example test set as the fine-tuned model. All responses were parseable (0% parse-failure rate).

---

## Evaluation

Both models evaluated on the **identical 30-example test set** (7 technical / 16 opinion / 7 joke) — **100% real HN comments**; the synthetic boundary examples are confined to the training split (see Section 2).

### Baseline — LLaMA-3.3-70B (zero-shot)

**Accuracy: 0.633** (19/30) · **Macro-F1: 0.626**


| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `technical_insight` | 0.50 | 0.86 | 0.63 | 7 |
| `opinion_or_critique` | 0.67 | 0.62 | 0.65 | 16 |
| `joke_or_meta` | 1.00 | 0.43 | 0.60 | 7 |
| **macro avg** | 0.72 | 0.64 | 0.63 | 30 |

### Fine-Tuned Model — DistilBERT

**Accuracy: 0.633** (19/30) · **Macro-F1: 0.646**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `technical_insight` | 0.71 | 0.71 | 0.71 | 7 |
| `opinion_or_critique` | 0.73 | 0.50 | 0.59 | 16 |
| `joke_or_meta` | 0.50 | 0.86 | 0.63 | 7 |
| **macro avg** | 0.65 | 0.69 | 0.65 | 30 |
| **weighted avg** | 0.67 | 0.63 | 0.63 | 30 |

**Confusion matrix (fine-tuned model):**

| Actual ↓ \ Predicted → | `technical_insight` | `opinion_or_critique` | `joke_or_meta` |
|---|---|---|---|
| **`technical_insight`** | **5** | 2 | 0 |
| **`opinion_or_critique`** | 2 | **8** | 6 |
| **`joke_or_meta`** | 0 | 1 | **6** |

### Model Comparison

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Zero-shot baseline (LLaMA-3.3-70B) | 0.633 | 0.626 |
| Fine-tuned DistilBERT (66M params) | 0.633 | 0.646 |

**The accuracy tie hides that the two models are doing very different things.** A 66M fine-tuned model matches a 1000× larger zero-shot model on *both* accuracy (0.633) and macro-F1 (0.646 vs 0.626 — within noise on 30 examples). But they agree on only **13 of 30** predictions:

| Outcome | Count |
|---|---|
| Both correct | 12 |
| Only fine-tuned correct | 7 |
| Only baseline correct | 7 |
| Both wrong | 4 |

The models are **complementary, not redundant** — each uniquely classifies 7 examples the other misses, and only 4 defeat both. An oracle picking the better model per example would reach **26/30 (87%)**. The clearest divergence is the `joke_or_meta` row: the zero-shot 70B is *conservative* (precision 1.00, recall 0.43 — it rarely says "joke" and misses 4 of 7), while the fine-tuned model is *aggressive* (recall 0.86, precision 0.50). **Fine-tuning's distinctive contribution is sarcasm/joke detection** — the +3 jokes it catches exactly offset the baseline's stronger opinion/technical recall, which is mechanically why the totals tie at 19/30.

---

### Error Analysis

**AI-assisted pattern surfacing.** Per the spec, I pasted all 11 misclassified test examples into Claude and asked it to identify systematic patterns, then verified each by re-reading. Claude surfaced three patterns, all of which held up:

1. **One dominant directional error: `opinion_or_critique` → `joke_or_meta` (6 of 11 errors).** The model over-fires `joke` (precision 0.50, recall 0.86 — it catches 6 of 7 real jokes but misfiles 6 opinions as jokes). This is the direct footprint of the β=1.0 class weighting, which up-weights `joke` hardest. The original opinion→technical confusion from earlier runs is essentially gone (down to 2).
2. **The model is pervasively uncertain.** All 11 errors are ≤ 0.49 confidence, and the model never exceeds 0.51 *except* on two correct `technical_insight` predictions (0.65). But most *correct* predictions also sit in the 0.38–0.51 band, barely above the 0.33 chance floor — so confidence isn't a clean error detector here, it just shows the model rarely commits. That fits a small dataset with genuinely fuzzy boundaries.
3. **A pattern I had to verify and partly *correct*:** several "errors" are defensible given borderline labels — including some `technical_insight` gold labels that are arguably `opinion`. We audited the *opinion* class but never the *technical* class, which appears to have its own mirror-image label noise (see #2 below).

**Three failures analyzed in depth:**

**Failure 1 — sarcasm read as a joke** *(the dominant error type)*
> *"Great timing given I just installed SwiftKey since Microsoft has started embedding ads and dark pattern redirects to Bing!"*
> **True:** `opinion_or_critique` · **Predicted:** `joke_or_meta` (confidence 0.49)

- **Which boundary:** opinion ↔ joke — the dominant 6-error cluster.
- **Why it's hard:** "Great timing" is sarcasm, and the comment *is* sarcastic — but it's sarcasm in service of a genuine critique of Microsoft, which makes it opinion. Sarcastic tone and joke are correlated but not identical, and the model keyed on the tone.
- **Labeling or data?** A **data/model** problem, not a labeling one — this is consistently labeled opinion. The β=1.0 weighting (plus a couple of sarcastic synthetic jokes) taught the model "sarcastic register → joke."
- **What would fix it:** more sarcastic-but-critical *opinion* examples so the model learns that sarcasm alone isn't a joke; and/or a lower β (β=0.7 over-fires `joke` less).

**Failure 2 — a data-backed argument labeled `technical_insight`** *(the model is arguably right)*
> *"Data shows that introduction of iPhones in 2007 is a better explanation for the increase in pedestrian deaths than heavier trucks and SUVs: [link]"*
> **True:** `technical_insight` · **Predicted:** `opinion_or_critique` (confidence 0.41)

- **Which boundary:** technical ↔ opinion (reverse direction).
- **Why it's hard:** it cites data, but it's *arguing a causal position* — which, by my own rubric, is `opinion_or_critique`. The model's prediction is arguably **more correct than the gold label**.
- **Labeling or data?** A **labeling** problem in the `technical_insight` class. The v3 audit only cleaned the opinion class; this is evidence the technical class contains data-driven *arguments* mislabeled as mechanism explanations.
- **What would fix it:** audit `technical_insight` with the same rubric (explain vs. argue) and relabel data-arguments to opinion — in the **training** data, then retrain.

**Failure 3 — a technical help-request labeled opinion** *(a no-fit example)*
> *"To all experts in video cards reading this: Do you know how to initialize a NVidia card in the RISC-V system… to the basic VGA mode 3 (text mode, 80x25)?"*
> **True:** `opinion_or_critique` · **Predicted:** `technical_insight` (confidence 0.43)

- **Which boundary:** opinion ↔ technical.
- **Why it's hard:** it's a **help-request dense with technical terms** — not an explanation (so not `technical_insight` by the rubric) and not a stance (so not really `opinion` either). It fits no class cleanly; the model latched onto the technical vocabulary.
- **Labeling or data?** A **labeling** problem — a few question/help-request comments survived the opinion-class audit that should have been dropped like the SSD-flush example.
- **What would fix it:** finish the audit by removing or recategorizing help-requests; they belong to a fourth category the taxonomy doesn't have.

---

### Sample Classifications

| Comment (truncated) | True | Predicted | Confidence | ✓/✗ |
|---|---|---|---|---|
| "the actual base64 email itself is an HTML document, with a bunch of filler…" | technical | technical | 0.65 | ✓ |
| "Only thing left is to make a Kanban out of it…" | joke | joke | 0.41 | ✓ |
| "Sounds good but the big problem I have with futo keyboard is that it can o…" | opinion | opinion | 0.42 | ✓ |
| "Great timing given I just installed SwiftKey since Microsoft…" | opinion | joke | 0.49 | ✗ |
| "Data shows that introduction of iPhones in 2007 is a better explanation…" | technical | opinion | 0.41 | ✗ |
| "To all experts in video cards reading this: Do you know how to initialize…" | opinion | technical | 0.43 | ✗ |

**Why a correct prediction is reasonable:** the base64/HTML comment was classified `technical_insight` at **0.65 — the model's single highest confidence on the test set** (every error sits at ≤ 0.49). That comment genuinely *explains a mechanism* (the email payload is an HTML document padded with filler text), which is exactly the class definition — so the model is confident for the right reason, and its most-confident prediction aligns with real explanatory content rather than surface vocabulary.

---

## Reflection: Intended vs. Learned

My labels were designed to capture a comment's **function** — *is it explaining, evaluating, or joking?* What the model actually learned, on 160 training examples, was closer to **surface register** — *what does the comment sound like?*

- **What it overfit to:** stylistic proxies. Early runs equated "technical vocabulary / analytical tone / statistics" with `technical_insight`; the final run equates "sarcastic / informal / short" with `joke_or_meta` (joke recall 0.86 but precision only 0.50 — it fires `joke` for anything that *sounds* like a quip). These are correlations with the true labels, but they aren't the definitions.
- **What it missed:** the explain-vs-evaluate distinction that *defines* the technical↔opinion boundary. It cannot tell a data-backed *argument* (opinion) from a *mechanism explanation* (technical), because both "sound technical" — exactly the gap visible in Failure 2.
- **The gap, stated plainly:** 160 examples is enough to learn *style* but not *function*. Closing it needs either many more examples per class or labels that are clean enough that style and function stop being confounded — not more hyperparameter tuning.

This is also why I stopped tuning. With a 30-example test set, run-to-run variance was ~1–2 examples (the same configuration produced 0.647, 0.676, and 0.633 across runs — partly Apple-Silicon/MPS non-determinism), and the β-sweep's best validation config (β=1.0, val macro-F1 ≈ 0.79) lands at 0.65 macro-F1 on test — a val→test gap inside the noise of a 30-example measurement. The model's behavior is understandable and diagnosable, which the spec rightly values over an un-interpretable higher score.

A higher-level observation from the baseline comparison: **what the model *captured* is almost exactly what the zero-shot 70B *missed*, and vice versa.** It learned sarcasm/joke detection (which the 70B can't do — that model's joke recall is 0.43) but not the world-knowledge-dependent opinion↔technical nuance (which the 70B handles). The two models agree on only 13/30 predictions despite identical accuracy: the fine-tuned model didn't learn a *superset* of the baseline, it learned a *different* function of similar overall quality.

---

## Spec Reflection

- **One way the spec helped:** the `planning.md` "buzzword complaint" decision rule (label by whether the comment *explains* a mechanism, not whether it *mentions* one) was the single most useful artifact. It gave a consistent tie-breaker for the hardest boundary and became the backbone of the v3 rubric and the entire error analysis.
- **One way the implementation diverged:** the plan was to reach quality purely by *collecting and balancing* real data. In practice, balancing wasn't enough — the model still couldn't learn the opinion↔technical boundary from the real examples alone. So I diverged by (a) *auditing and dropping* poor-fit examples and (b) *authoring synthetic contrastive examples*, a data-quality intervention not in the original plan. I also overrode the starter notebook's "3 epochs is a good default," which was miscalibrated for a dataset this small.

---

## Known Limitations
- **Tiny evaluation set.** 30 test / ~32 val examples — every example is ~3% of accuracy, so single-run differences below ~3 examples are noise. A more rigorous estimate would use k-fold cross-validation or report mean ± std over several seeds.

---

## AI Usage

AI tools (Claude, via Claude Code) were used throughout. Specific instances:

1. **Hyperparameter audit.** *Directed:* "analyze the training arguments and tell me if anything should change." *Produced:* identified that `warmup_steps=50` exceeded the ~60 total training steps, so the learning rate never finished warming up; recommended `warmup_ratio`, more epochs, and macro-F1 model selection. *I changed/overrode:* verified the step-count math myself before accepting it, and adopted Changes #1–#3.
2. **Class-imbalance fix + β-sweep.** *Directed:* "the model collapsed to the majority class — how do I fix it?" *Produced:* a class-weighted `Trainer` subclass and, when I asked to tune it, a β-sweep that selects weighting strength on validation. *I changed/overrode:* replaced the initial hardcoded weights (which were computed for an earlier dataset) with sweep-selected weights, and added the "select the least-aggressive β within noise" discussion.
3. **Label audit (opinion class).** *Directed:* "re-audit the opinion class against the tightened rubric." *Produced:* a finding that the labels were *mostly correct* (correcting an earlier, looser estimate that "~half were mislabeled"), a list of 7 no-fit examples to drop, and 16 synthetic boundary examples. *I changed/overrode:* kept the synthetic examples in a **separate** file and disclosed them rather than silently merging; reviewed each drop.
4. **Failure-analysis pattern surfacing.** *Directed:* pasted the 11 misclassified examples and asked for systematic patterns. *Produced:* the opinion→joke directional pattern, the all-errors-low-confidence pattern, and the technical-class-label-noise observation. *I verified:* re-read all 11 examples to confirm each pattern before writing it up.
5. **Annotation assistance.** Claude pre-labeled the scraped comments from the three definitions + the buzzword-complaint rule; labels were reviewed manually before use.
