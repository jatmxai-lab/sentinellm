"""Export base DistilBERT (untrained head) to ONNX so the API has something
to serve before you have a real trained model.

The classifier will return ~random scores — useful only for verifying that
the FastAPI / Redis / Postgres / ONNX Runtime stack runs end-to-end.

Replace with a real trained model via `scripts/export_onnx.py` once you've
trained on Colab.

Usage:
    uv run --extra train python scripts/stub_onnx.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "distilbert-base-uncased"
OUT = Path("models/sentinellm.onnx")

OUT.parent.mkdir(parents=True, exist_ok=True)

print(f"Loading {BASE} (head will be randomly initialized)...")
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=2, id2label={0: "clean", 1: "toxic"}, label2id={"clean": 0, "toxic": 1}
)
model.eval()

dummy = tokenizer(["hello world"], return_tensors="pt", padding=True, truncation=True)

print(f"Exporting to {OUT}...")
torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    str(OUT),
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)

size_mb = OUT.stat().st_size / (1024 * 1024)
print(f"OK - wrote {OUT} ({size_mb:.1f} MB)")
print("\nThis is a STUB model. Predictions will be random until you train.")
