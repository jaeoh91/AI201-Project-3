# Labeling Rubric & Opinion-Class Audit

## Tightened rubric (v3)

Three mutually exclusive classes. When a comment plausibly fits more than one,
**label by its primary purpose** (the tie-breaker rule).

| Label | Definition | NOT this label |
|---|---|---|
| **technical_insight** | Explains *how* or *why* something works, or states a verifiable mechanism / causal chain. | Merely *referencing* technical topics, *asking* a technical question, or *describing* a feature is **not** enough — there must be an actual explanation. |
| **opinion_or_critique** | A stance, preference, evaluation, or critique — **even if it is analytical, cites data, or uses heavy technical vocabulary.** | Pure mechanism explanations with no judgment; pure jokes. |
| **joke_or_meta** | Sarcasm, a meme, or a meta-comment about HN / the discussion itself. | A genuine critique that merely has a sarcastic tone (that's opinion). |

**The decisive question for the opinion↔technical boundary:**
*Does the comment EXPLAIN a mechanism, or EXPRESS a view about one?*
Citing statistics, using jargon, or writing in an analytical register does **not**
make a comment `technical_insight` — those are surface features. A data-backed
argument is still `opinion_or_critique`.

## Audit finding (the important part)

I re-read all 122 `opinion_or_critique` examples against this rubric.

**The opinion labels are mostly correct.** Almost none of them actually belong in
`technical_insight`; they are evaluative comments that merely *use* technical
vocabulary. This means the model's dominant error — predicting `technical_insight`
for true opinions (8 of 19 in the test set) — is **a model problem, not a label
problem.** The model learned the surface heuristic "tech words / statistics /
analytical tone ⇒ technical" instead of the real distinction (explain vs. evaluate).

> This corrects an earlier, looser estimate that "~half the errors are bad labels."
> A strict pass shows the labels are sound; the fix belongs on the model/data-balance
> side (loss weighting + contrastive examples), not in mass relabeling.

What the audit *did* surface: a small set of comments that fit **none** of the three
classes (link/resource dumps, a bare feature description, a one-line skeptical
question). These add label noise and were removed.

## Changelog: `hn_dataset_balanced.csv` (215) → `hn_dataset_v3.csv` (208)

7 `opinion_or_critique` rows dropped as poor fits for *any* class (not relabeled —
none of them are mechanism explanations, so moving them to `technical_insight`
would violate the rubric):

| Orig. row | Reason | Excerpt |
|---|---|---|
| 54 | resource pointer, no stance | "You can read about Shrubbery … [link]" |
| 57 | "related submissions" link dump | "Related. Others? … [HN links]" |
| 59 | feature description, no stance/mechanism | "Rhombus is designed to be approachable …" |
| 95 | bare fact + link | "previously featured on a video by Real Engineering [link]" |
| 96 | link dump | "Fwiw, I'll share some surfing: [links]" |
| 99 | one-line skeptical *question* (not an explanation) | "can it be certain that a flush is really a flush?" |
| 102 | resource pointer | "Here's another 8-page zine maker … [link]" |

> Rows **59** and **99** are the two flagged earlier as "relabel #10 / #7." Under the
> strict rubric the correct action is to **drop** them, not move them to
> `technical_insight`: a feature description and a bare question are not mechanism
> explanations. Relabeling them would have re-introduced the exact confusion we're
> trying to remove.

The original `hn_dataset_balanced.csv` is unchanged.

## Synthetic prototypes (opt-in): `prototype_examples_synthetic.csv`

16 **authored** (not scraped) examples designed to sharpen the failing boundary:

- **6 opinions** that are heavy in tech vocab / statistics but are clearly a *stance*
  — direct counter-examples to the "tech words ⇒ technical" heuristic.
- **8 technical_insight** examples that actually explain a mechanism (Postgres plan
  flips, TCP slow start, generational GC, Bloom filters, B-tree write cost, float
  representation, TLS SNI, hash-map rehashing) — to teach what *real* technical
  content looks like. (`technical_insight` is also the rarest class, so this helps
  balance too.)
- **2 jokes/meta**.

Merging counts: opinion 121 / joke 52 / technical 51 — noticeably more balanced than
the 122 / 50 / 43 you have now.

## How to use

1. In the notebook (Section 1), point `CSV_PATH` at the audited file:
   ```python
   CSV_PATH = "../data/hn_dataset_v3.csv"
   ```
   To also use the synthetic prototypes, concat after loading:
   ```python
   df = pd.read_csv(CSV_PATH)
   df = pd.concat([df, pd.read_csv("../data/prototype_examples_synthetic.csv")], ignore_index=True)
   ```
2. **Re-run the train/val/test split, fine-tuning, AND the Groq baseline (Section 5).**
   Changing the dataset changes the test set, so the old baseline number is no longer
   comparable — both models must be scored on the *same* new test set.
