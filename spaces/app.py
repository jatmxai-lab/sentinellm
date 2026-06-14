"""SentinelLM — HuggingFace Space (Docker SDK).

Self-contained: FastAPI for the JSON API, Gradio for the UI, ONNX Runtime
for inference, in-memory cache, SQLite logging in /tmp. Loads the ONNX
model from the HF Hub at startup so the container stays slim.

Endpoints:
  GET  /                  → Gradio UI
  GET  /v1/health         → JSON health probe
  POST /v1/predict        → JSON predict
  GET  /docs              → OpenAPI / Swagger
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# ---- config (env-overridable) ----
REPO_ID = os.environ.get("HF_MODEL_REPO", "jatmanis1/sentinellm-v1")
ONNX_FILENAME = os.environ.get("ONNX_FILENAME", "sentinellm.onnx")
FLAG_THRESHOLD = float(os.environ.get("FLAG_THRESHOLD", "0.7"))
DB_PATH = os.environ.get("DB_PATH", "/tmp/sentinellm.db")
LABEL_NAMES = ["clean", "toxic"]


# ---- helpers ----
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


# ---- predictor ----
class Predictor:
    def __init__(self, sess: ort.InferenceSession, tok, model_name: str):
        self.sess = sess
        self.tok = tok
        self.model_name = model_name
        self._input_names = {i.name for i in sess.get_inputs()}

    @classmethod
    def from_hf(cls, repo_id: str, onnx_filename: str) -> "Predictor":
        print(f"[startup] downloading {repo_id}:{onnx_filename}...")
        onnx_path = hf_hub_download(repo_id, onnx_filename)
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        tok = AutoTokenizer.from_pretrained(repo_id)
        size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
        print(f"[startup] loaded ONNX ({size_mb:.0f} MB) + tokenizer")
        return cls(sess, tok, model_name=repo_id)

    def predict(self, text: str) -> dict:
        enc = self.tok(text, truncation=True, max_length=256,
                       padding=True, return_tensors="np")
        feeds = {k: v for k, v in enc.items() if k in self._input_names}
        logits = self.sess.run(None, feeds)[0]
        probs = softmax(logits, axis=-1)[0]
        label_id = int(probs.argmax())
        return {
            "label": label_id,
            "label_name": LABEL_NAMES[label_id],
            "score": float(probs[label_id]),
            "probs": {LABEL_NAMES[i]: float(p) for i, p in enumerate(probs)},
        }

    async def predict_async(self, text: str) -> dict:
        return await asyncio.to_thread(self.predict, text)


# ---- in-memory cache (drop-in for ExactCache in this constrained env) ----
class InMemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def get(self, key: str) -> dict[str, Any] | None:
        return self._store.get(key)

    async def set(self, key: str, value: dict[str, Any]) -> None:
        self._store[key] = value


# ---- SQLite logging (replaces Postgres for the demo) ----
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_hash TEXT NOT NULL,
            label INTEGER NOT NULL,
            score REAL NOT NULL,
            latency_ms REAL NOT NULL,
            cache_hit INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def log_prediction(text_hash: str, label: int, score: float,
                   latency_ms: float, cache_hit: bool) -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO prediction_logs (text_hash, label, score, latency_ms, cache_hit) "
            "VALUES (?, ?, ?, ?, ?)",
            (text_hash, label, score, latency_ms, int(cache_hit)),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[warn] sqlite log failed: {e}")


# ---- lifespan: load model once ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.predictor = Predictor.from_hf(REPO_ID, ONNX_FILENAME)
    app.state.cache = InMemoryCache()
    print(f"[startup] ready — model={REPO_ID}")
    yield
    print("[shutdown] bye")


api = FastAPI(
    title="SentinelLM",
    version="1.0",
    description=(
        "Toxicity classifier — DistilBERT fine-tuned on civil_comments, "
        "ONNX Runtime inference, in-memory cache, SQLite logging.\n\n"
        "**Repo:** https://github.com/jatmxai-lab/sentinellm  ·  "
        "**Model:** https://huggingface.co/jatmanis1/sentinellm-v1"
    ),
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    use_cache: bool = True


class PredictResponse(BaseModel):
    label: int
    label_name: str
    score: float
    probs: dict[str, float]
    flagged: bool
    cache_hit: bool
    latency_ms: float
    model_version: str


@api.get("/v1/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok", "model": REPO_ID}


async def _do_predict(text: str, use_cache: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    text_hash = hash_text(text)
    cache_hit = False
    result: dict | None = None
    if use_cache:
        result = await api.state.cache.get(text_hash)
        cache_hit = result is not None
    if result is None:
        result = await api.state.predictor.predict_async(text)
        await api.state.cache.set(text_hash, result)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    label_id = int(result["label"])
    score = float(result["score"])
    label_name = result["label_name"]
    flagged = label_name == "toxic" and score >= FLAG_THRESHOLD
    log_prediction(text_hash, label_id, score, latency_ms, cache_hit)
    return {
        "label": label_id,
        "label_name": label_name,
        "score": score,
        "probs": result["probs"],
        "flagged": flagged,
        "cache_hit": cache_hit,
        "latency_ms": latency_ms,
        "model_version": REPO_ID,
    }


@api.post("/v1/predict", response_model=PredictResponse, tags=["predict"])
async def predict(req: PredictRequest) -> PredictResponse:
    return PredictResponse(**(await _do_predict(req.text, req.use_cache)))


# ---- Gradio UI ----
# Examples chosen to match the model's training scope: civil_comments-style
# toxicity (overt insults, hostility, slurs). The v1 model is NOT trained on
# adversarial/jailbreak prompts (AdvBench was gated mid-training), so we don't
# showcase that capability here.
EXAMPLES = [
    "Thanks for sharing this, it was a really helpful read.",
    "I disagree with the premise but I appreciate the perspective.",
    "you're an absolute idiot and your opinion is worthless",
    "shut up nobody asked for your stupid take",
    "anyone who believes this nonsense is a complete moron",
    "people like you are why this country is going downhill",
]


async def classify_ui(text: str):
    if not text or not text.strip():
        return "—", {}, "enter some text"
    r = await _do_predict(text, use_cache=True)
    verdict = "**\U0001F6A9 FLAGGED**" if r["flagged"] else "**✅ SAFE**"
    meta = (
        f"label: **{r['label_name']}** · "
        f"score: **{r['score']:.3f}** · "
        f"cache: **{'HIT' if r['cache_hit'] else 'MISS'}** · "
        f"latency: **{r['latency_ms']:.0f} ms**"
    )
    return verdict, r["probs"], meta


with gr.Blocks(title="SentinelLM", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# SentinelLM — toxicity classifier\n"
        "DistilBERT fine-tuned on `google/civil_comments` (F1 0.70 on held-out test). "
        "ONNX-accelerated, served via FastAPI. Best on direct insults / hostility — "
        "not trained on adversarial prompts.\n\n"
        "JSON API at [`/v1/predict`](/docs) · "
        "**[GitHub](https://github.com/jatmxai-lab/sentinellm)** · "
        "**[Model card](https://huggingface.co/jatmanis1/sentinellm-v1)**"
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                lines=4, label="Input text",
                placeholder="Type or paste text...",
            )
            btn = gr.Button("Classify", variant="primary")
            gr.Examples(EXAMPLES, inputs=text)
        with gr.Column():
            verdict = gr.Markdown(label="Verdict")
            probs = gr.Label(label="Probabilities", num_top_classes=2)
            meta = gr.Markdown()
    btn.click(classify_ui, inputs=text, outputs=[verdict, probs, meta])
    text.submit(classify_ui, inputs=text, outputs=[verdict, probs, meta])


# Mount Gradio at root; FastAPI keeps /v1/* and /docs
app = gr.mount_gradio_app(api, demo, path="/")
