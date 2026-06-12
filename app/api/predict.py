from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.dependencies import Cache, DBSession, Predictor
from sentinellm.config import settings
from sentinellm.data.labels import ID_TO_LABEL
from sentinellm.db.models import PredictionLog
from sentinellm.logging import get_logger
from sentinellm.serving.cache import hash_text

router = APIRouter(prefix="/v1", tags=["predict"])
log = get_logger(__name__)


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


@router.post("/predict", response_model=PredictResponse)
async def predict(
    req: PredictRequest,
    predictor: Predictor,
    cache: Cache,
    session: DBSession,
) -> PredictResponse:
    t0 = time.perf_counter()
    text_hash = hash_text(req.text)
    cache_hit = False
    result: dict | None = None

    if req.use_cache:
        result = await cache.get(text_hash)
        cache_hit = result is not None

    if result is None:
        result = await predictor.predict_async(req.text)
        await cache.set(text_hash, result)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    label_id = int(result["label"])
    score = float(result["score"])
    label_name = result.get("label_name", ID_TO_LABEL.get(label_id, "unknown"))
    flagged = label_name == "toxic" and score >= settings.flag_threshold

    session.add(
        PredictionLog(
            model_version_id=None,
            text_hash=text_hash,
            label=label_id,
            score=score,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        )
    )
    await session.commit()

    log.info(
        "prediction",
        text_hash=text_hash,
        label=label_name,
        score=round(score, 4),
        cache_hit=cache_hit,
        flagged=flagged,
        latency_ms=round(latency_ms, 2),
    )

    return PredictResponse(
        label=label_id,
        label_name=label_name,
        score=score,
        probs={k: float(v) for k, v in result.get("probs", {}).items()},
        flagged=flagged,
        cache_hit=cache_hit,
        latency_ms=latency_ms,
        model_version=predictor.model_name,
    )
