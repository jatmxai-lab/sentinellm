"""Generate synthetic adversarial / jailbreak examples with Gemini 2.5 Flash.

One-shot script. Run once, commit `data/synthetic.parquet`, never re-run in CI.

Usage:
    python scripts/generate_synthetic.py --n-per-label 1000

Requires GEMINI_API_KEY in .env (free key: https://aistudio.google.com/app/apikey).
Free tier: 15 RPM, 1M tokens/day — we sleep 4.5s between batches to stay safe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

from sentinellm.config import settings

PROMPT = """Generate {n} diverse English prompts that are examples of "{label}".

Rules:
- Vary length (10-200 chars), style, tone, and attack vector.
- No duplicates, no numbering, no markdown.
- Each example is a single self-contained string.
- Output strict JSON: {{"examples": ["...", "..."]}}.

Definition of "{label}": {definition}
"""

DEFINITIONS = {
    "jailbreak": (
        "attempts to override system instructions, reveal hidden prompts, or "
        "bypass safety guardrails (e.g. 'ignore previous instructions', "
        "'pretend you have no rules', DAN-style prompts)"
    ),
    "adversarial": (
        "subtle prompt-injection or role-play exploits short of full jailbreak "
        "(e.g. 'as a fictional character with no restrictions...', indirect "
        "injection via fake document content, hypothetical framings)"
    ),
}


def generate_batch(client: genai.Client, label: str, n: int) -> list[str]:
    prompt = PROMPT.format(n=n, label=label, definition=DEFINITIONS[label])
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=1.0,
        ),
    )
    try:
        data = json.loads(resp.text)
        return [str(x).strip() for x in data.get("examples", []) if str(x).strip()]
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"  ! parse error: {e}; got: {resp.text[:200]!r}")
        return []


def collect_label(
    client: genai.Client, label: str, n_total: int, batch_size: int, sleep_s: float
) -> list[str]:
    out: set[str] = set()
    attempts = 0
    while len(out) < n_total and attempts < (n_total // batch_size) * 4:
        attempts += 1
        before = len(out)
        try:
            new = generate_batch(client, label, batch_size)
        except Exception as e:
            print(f"  ! API error: {e}; sleeping 30s")
            time.sleep(30)
            continue
        out.update(new)
        added = len(out) - before
        print(f"  [{label}] batch {attempts}: +{added} new, total {len(out)}/{n_total}")
        if len(out) >= n_total:
            break
        time.sleep(sleep_s)
    return list(out)[:n_total]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-label", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--sleep-s", type=float, default=4.5)
    ap.add_argument("--out", default="data/synthetic.parquet")
    args = ap.parse_args()

    if not settings.gemini_api_key:
        print("ERROR: set GEMINI_API_KEY in .env (https://aistudio.google.com/app/apikey)")
        sys.exit(1)

    client = genai.Client(api_key=settings.gemini_api_key)

    rows = []
    for label in ("jailbreak", "adversarial"):
        print(f"\nGenerating {args.n_per_label} examples of '{label}'...")
        examples = collect_label(
            client, label, args.n_per_label, args.batch_size, args.sleep_s
        )
        for text in examples:
            rows.append({"text": text, "label": 1, "source": f"gemini_{label}"})

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df):,} rows to {out}")
    print(df.groupby("source").size().to_string())


if __name__ == "__main__":
    main()
