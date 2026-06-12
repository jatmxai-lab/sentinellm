from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from sentinellm.config import settings
from sentinellm.data.labels import ID_TO_LABEL

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class SentinelPredictor:
    """ONNX Runtime inference wrapper for the toxicity classifier."""

    def __init__(self, session: ort.InferenceSession, tokenizer: PreTrainedTokenizerBase,
                 model_name: str):
        self.session = session
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._input_names = {i.name for i in session.get_inputs()}

    @classmethod
    def from_local(cls, onnx_path: str, tokenizer_repo: str) -> SentinelPredictor:
        path = Path(onnx_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found at {path.resolve()}")
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        tok = AutoTokenizer.from_pretrained(tokenizer_repo)
        return cls(sess, tok, model_name=tokenizer_repo)

    @classmethod
    def from_settings(cls) -> SentinelPredictor:
        onnx_path = settings.onnx_model_path or "models/sentinellm.onnx"
        return cls.from_local(onnx_path, settings.hf_model_repo)

    def predict_sync(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="np",
        )
        feeds = {k: v for k, v in enc.items() if k in self._input_names}
        outputs = self.session.run(None, feeds)
        logits = outputs[0]
        probs = _softmax(logits, axis=-1)[0]
        label_id = int(probs.argmax())
        return {
            "label": label_id,
            "label_name": ID_TO_LABEL[label_id],
            "score": float(probs[label_id]),
            "probs": {ID_TO_LABEL[i]: float(p) for i, p in enumerate(probs)},
        }

    async def predict_async(self, text: str) -> dict:
        return await asyncio.to_thread(self.predict_sync, text)

    def close(self) -> None:
        # ort.InferenceSession has no explicit close in stable API
        pass
