from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, predict
from sentinellm.logging import configure_logging, get_logger
from sentinellm.serving.cache import ExactCache
from sentinellm.serving.predictor import SentinelPredictor

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup_begin")
    app.state.predictor = SentinelPredictor.from_settings()
    app.state.cache = ExactCache.from_url()
    log.info("startup_complete", model=app.state.predictor.model_name)
    try:
        yield
    finally:
        log.info("shutdown_begin")
        await app.state.cache.close()
        app.state.predictor.close()
        log.info("shutdown_complete")


app = FastAPI(
    title="SentinelLM",
    version="1.0",
    description="Toxicity classifier — ONNX Runtime + Redis cache + Postgres logging.",
    lifespan=lifespan,
)
app.include_router(health.router)
app.include_router(predict.router)


@app.get("/", include_in_schema=False)
async def root():
    return {"name": "SentinelLM", "docs": "/docs", "health": "/v1/health"}
