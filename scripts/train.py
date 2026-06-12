"""Fine-tune DistilBERT for binary toxicity classification.

Designed to run on Google Colab T4 (free tier). ~3 epochs on ~200k rows
takes 30-45 min with fp16.

Usage (local):
    python scripts/train.py --corpus data/corpus.parquet --out ckpt/

Usage (Colab):
    !pip install -q transformers datasets accelerate scikit-learn pyarrow
    !python scripts/train.py --corpus data/corpus.parquet --push your-user/sentinellm-v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from sentinellm.data.labels import ID_TO_LABEL, LABEL_TO_ID, NUM_LABELS

BASE_MODEL = "distilbert-base-uncased"


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
    }


def load_corpus(path: str, test_size: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    df = pd.read_parquet(path)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    print(f"Loaded {len(df):,} rows; positive rate {df['label'].mean():.2%}")

    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    split = ds.train_test_split(test_size=test_size, seed=seed, stratify_by_column="label")
    return split["train"], split["test"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.parquet")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--train-batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push", default=None, help="HF Hub repo to push best ckpt to")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    args = ap.parse_args()

    train_ds, test_ds = load_corpus(args.corpus, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        logging_steps=100,
        report_to=[],
        seed=args.seed,
        push_to_hub=args.push is not None,
        hub_model_id=args.push,
        hub_strategy="end",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print("\nFinal eval:", eval_metrics)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(
            {
                "model": args.base_model,
                "epochs": args.epochs,
                "lr": args.lr,
                "train_batch": args.train_batch,
                "max_length": args.max_length,
                "metrics": {k: float(v) for k, v in eval_metrics.items()
                            if isinstance(v, (int, float))},
            },
            f,
            indent=2,
        )

    trainer.save_model(str(out / "best"))
    tokenizer.save_pretrained(str(out / "best"))
    if args.push:
        trainer.push_to_hub()
        print(f"Pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
