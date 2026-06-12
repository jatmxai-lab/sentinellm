from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as redis

from sentinellm.config import settings


def hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class ExactCache:
    """Redis SHA-256 keyed exact-match cache for prediction results."""

    KEY_PREFIX = "sentinellm:pred:"

    def __init__(self, client: redis.Redis, ttl: int = settings.cache_ttl_seconds):
        self.client = client
        self.ttl = ttl

    @classmethod
    def from_url(cls, url: str | None = None,
                 ttl: int = settings.cache_ttl_seconds) -> ExactCache:
        url = url or settings.redis_url
        return cls(redis.from_url(url, decode_responses=True), ttl=ttl)

    def _key(self, text_hash: str) -> str:
        return f"{self.KEY_PREFIX}{text_hash}"

    async def get(self, text_hash: str) -> dict[str, Any] | None:
        raw = await self.client.get(self._key(text_hash))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set(self, text_hash: str, value: dict[str, Any]) -> None:
        await self.client.set(self._key(text_hash), json.dumps(value), ex=self.ttl)

    async def close(self) -> None:
        await self.client.aclose()
