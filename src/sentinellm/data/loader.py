"""Build the training corpus from public HuggingFace datasets.

Sources:
- `google/civil_comments` — millions of labeled comments with a `toxicity`
  float in [0, 1]. We binarize at 0.5.
- `walledai/AdvBench` — 520 adversarial / jailbreak prompts, all labeled toxic.
- Optional: `data/synthetic.parquet` (from scripts/generate_synthetic.py).

Run: `python -m sentinellm.data.loader`
Output: `data/corpus.parquet` with columns (text, label).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

CIVIL_TOXICITY_THRESHOLD = 0.5
MAX_CIVIL_ROWS = 200_000  # downsample to keep training tractable on Colab T4


def _load_civil_comments(max_rows: int) -> pd.DataFrame:
    print(f"Loading google/civil_comments (cap {max_rows:,})...")
    ds = load_dataset("google/civil_comments", split="train")
    df = ds.to_pandas()[["text", "toxicity"]].copy()
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len().between(1, 2000)]
    df["label"] = (df["toxicity"] >= CIVIL_TOXICITY_THRESHOLD).astype(int)
    df = df[["text", "label"]]

    if len(df) > max_rows:
        # stratified downsample to preserve class ratio
        pos = df[df["label"] == 1]
        neg = df[df["label"] == 0]
        pos_keep = min(len(pos), int(max_rows * (len(pos) / len(df))))
        neg_keep = max_rows - pos_keep
        df = pd.concat(
            [pos.sample(pos_keep, random_state=42), neg.sample(neg_keep, random_state=42)]
        )
    return df.reset_index(drop=True)


def _load_advbench() -> pd.DataFrame:
    print("Loading walledai/AdvBench...")
    ds = load_dataset("walledai/AdvBench", split="train")
    df = ds.to_pandas()
    # AdvBench has a "prompt" column; rename + label as toxic (1).
    text_col = "prompt" if "prompt" in df.columns else df.columns[0]
    df = pd.DataFrame({"text": df[text_col].astype(str).str.strip(), "label": 1})
    df = df[df["text"].str.len().between(1, 2000)]
    return df.reset_index(drop=True)


def _load_synthetic(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Synthetic file not found at {path} — skipping.")
        return None
    print(f"Loading synthetic from {path}...")
    df = pd.read_parquet(path)
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError(f"{path} must have columns (text, label)")
    return df[["text", "label"]]


def build_corpus(
    out_path: str = "data/corpus.parquet",
    max_civil_rows: int = MAX_CIVIL_ROWS,
    synthetic_path: str = "data/synthetic.parquet",
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    parts.append(_load_civil_comments(max_civil_rows))
    parts.append(_load_advbench())
    synth = _load_synthetic(Path(synthetic_path))
    if synth is not None:
        parts.append(synth)

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    pos_rate = df["label"].mean()
    print(f"\nWrote {len(df):,} rows to {out}")
    print(f"Positive rate: {pos_rate:.2%}  (toxic={int(df['label'].sum()):,}, "
          f"clean={int((df['label'] == 0).sum()):,})")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus.parquet")
    ap.add_argument("--max-civil", type=int, default=MAX_CIVIL_ROWS)
    ap.add_argument("--synthetic", default="data/synthetic.parquet")
    args = ap.parse_args()
    build_corpus(args.out, args.max_civil, args.synthetic)


if __name__ == "__main__":
    main()
