from sentinellm.serving.cache import hash_text


def test_hash_text_deterministic():
    assert hash_text("hello") == hash_text("hello")


def test_hash_text_strips_whitespace():
    assert hash_text("hello") == hash_text("  hello  ")


def test_hash_text_distinguishes():
    assert hash_text("hello") != hash_text("world")


def test_hash_text_length():
    assert len(hash_text("anything")) == 64


async def test_fake_cache_roundtrip(fake_cache):
    await fake_cache.set("k", {"label": 0, "score": 0.9})
    got = await fake_cache.get("k")
    assert got == {"label": 0, "score": 0.9}


async def test_fake_cache_miss(fake_cache):
    assert await fake_cache.get("nope") is None
