"""TakeMeter — interface for the fine-tuned HN discourse classifier.

Two ways to run:
  • Web UI:  uv run python app.py            (launches a Gradio app in the browser)
  • CLI:     uv run python app.py "some comment text"   (prints label + confidences)

The trained model is not committed to git (it is ~770 MB). Produce it by running
Sections 1–3 of notebooks/ai201_project3_takemeter_starter_clean.ipynb, which writes
checkpoints under notebooks/takemeter-model/. This script finds that model automatically;
to point at a specific directory, set the MODEL_DIR environment variable.
"""

import glob
import os
import sys

from transformers import pipeline

MAX_LEN = 256  # must match the max_length used during fine-tuning

EXAMPLES = [
    "The reason your Postgres query suddenly got slow is a plan flip: once the table grew "
    "past a threshold the planner switched from an index scan to a sequential scan.",
    "This startup is just a thin wrapper around an LLM API. There's genuinely no moat here.",
    "Can't wait to rewrite this in Rust next week.",
]


def find_model_dir() -> str:
    """Locate a fine-tuned model directory, preferring a stable ./model over a checkpoint."""
    for cand in (os.environ.get("MODEL_DIR"), "model", "notebooks/model"):
        if cand and os.path.exists(os.path.join(cand, "config.json")):
            return cand
    checkpoints = glob.glob("notebooks/takemeter-model/checkpoint-*")
    if checkpoints:
        return max(checkpoints, key=os.path.getmtime)  # newest = best of the latest run
    sys.exit(
        "No fine-tuned model found.\n"
        "Run Sections 1–3 of the notebook to train it, or set MODEL_DIR to a saved "
        "model directory (one containing config.json + model.safetensors)."
    )


MODEL_DIR = find_model_dir()
_clf = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, top_k=None)


def classify(text: str) -> dict:
    """Return {label: probability} over all classes (the shape gr.Label wants)."""
    if not text or not text.strip():
        return {}
    scores = _clf(text, truncation=True, max_length=MAX_LEN)[0]
    return {item["label"]: float(item["score"]) for item in scores}


def _cli(text: str) -> None:
    probs = classify(text)
    if not probs:
        print("Empty input.")
        return
    top = max(probs, key=probs.get)
    print(f"Model:      {MODEL_DIR}")
    print(f"Prediction: {top}  ({probs[top]:.0%} confidence)\n")
    for label, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        bar = "█" * round(p * 30)
        print(f"  {label:<22} {p:5.1%}  {bar}")


def _launch_ui() -> None:
    import gradio as gr

    demo = gr.Interface(
        fn=classify,
        inputs=gr.Textbox(lines=6, label="Hacker News comment",
                          placeholder="Paste a comment to classify…"),
        outputs=gr.Label(num_top_classes=3, label="Prediction (confidence per class)"),
        title="TakeMeter — Hacker News Discourse Classifier",
        description=(
            "Fine-tuned DistilBERT. Classifies a comment as **technical_insight**, "
            "**opinion_or_critique**, or **joke_or_meta**, with confidence for each class."
        ),
        examples=EXAMPLES,
        flagging_mode="never",
    )
    demo.launch()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli(" ".join(sys.argv[1:]))
    else:
        _launch_ui()
