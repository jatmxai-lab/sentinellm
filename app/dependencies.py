from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from sentinellm.db.session import SessionLocal
from sentinellm.serving.cache import ExactCache
from sentinellm.serving.predictor import SentinelPredictor


def get_predictor(request: Request) -> SentinelPredictor:
    return request.app.state.predictor


def get_cache(request: Request) -> ExactCache:
    return request.app.state.cache


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


Predictor = Annotated[SentinelPredictor, Depends(get_predictor)]
Cache = Annotated[ExactCache, Depends(get_cache)]
DBSession = Annotated[AsyncSession, Depends(get_session)]
