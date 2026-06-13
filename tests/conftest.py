from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.dependencies import get_cache, get_predictor, get_session
from app.main import app
from sentinellm.db.base import Base


class FakeRedis:
    """Minimal in-memory replacement for redis.asyncio.Redis."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value

    async def aclose(self) -> None:
        self.store.clear()


class FakeCache:
    """Drop-in for ExactCache that uses FakeRedis."""

    KEY_PREFIX = "sentinellm:pred:"

    def __init__(self) -> None:
        self.client = FakeRedis()

    def _key(self, h: str) -> str:
        return f"{self.KEY_PREFIX}{h}"

    async def get(self, text_hash: str) -> dict[str, Any] | None:
        raw = await self.client.get(self._key(text_hash))
        return json.loads(raw) if raw else None

    async def set(self, text_hash: str, value: dict[str, Any]) -> None:
        await self.client.set(self._key(text_hash), json.dumps(value))

    async def close(self) -> None:
        await self.client.aclose()


class FakePredictor:
    """Deterministic predictor: 'toxic' if 'hate' or 'kill' appears."""

    model_name = "test/fake-predictor"

    @staticmethod
    def _classify(text: str) -> dict[str, Any]:
        t = text.lower()
        toxic = any(w in t for w in ("hate", "kill", "die", "stupid", "trash"))
        score = 0.95 if toxic else 0.92
        label = 1 if toxic else 0
        label_name = "toxic" if toxic else "clean"
        probs = (
            {"clean": 1 - score, "toxic": score}
            if toxic
            else {"clean": score, "toxic": 1 - score}
        )
        return {"label": label, "label_name": label_name, "score": score, "probs": probs}

    def predict_sync(self, text: str) -> dict[str, Any]:
        return self._classify(text)

    async def predict_async(self, text: str) -> dict[str, Any]:
        return self._classify(text)

    def close(self) -> None: ...


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
def fake_predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def fake_cache() -> FakeCache:
    return FakeCache()


@pytest_asyncio.fixture
async def client(fake_predictor, fake_cache, session_factory) -> AsyncIterator[AsyncClient]:
    async def _session():
        async with session_factory() as s:
            yield s

    app.dependency_overrides[get_predictor] = lambda: fake_predictor
    app.dependency_overrides[get_cache] = lambda: fake_cache
    app.dependency_overrides[get_session] = _session

    # don't run lifespan (no real predictor / cache)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    app.dependency_overrides.clear()
