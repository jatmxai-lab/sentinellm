async def test_predict_clean(client):
    r = await client.post("/v1/predict", json={"text": "what a lovely day"})
    assert r.status_code == 200
    body = r.json()
    assert body["label_name"] == "clean"
    assert body["label"] == 0
    assert body["flagged"] is False
    assert body["cache_hit"] is False
    assert body["latency_ms"] >= 0
    assert "clean" in body["probs"] and "toxic" in body["probs"]


async def test_predict_toxic_flagged(client):
    r = await client.post("/v1/predict", json={"text": "I hate this trash"})
    assert r.status_code == 200
    body = r.json()
    assert body["label_name"] == "toxic"
    assert body["label"] == 1
    assert body["flagged"] is True
    assert body["score"] >= 0.7


async def test_predict_cache_hit(client):
    payload = {"text": "deterministic test string"}
    r1 = await client.post("/v1/predict", json=payload)
    r2 = await client.post("/v1/predict", json=payload)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["cache_hit"] is False
    assert r2.json()["cache_hit"] is True
    assert r1.json()["label"] == r2.json()["label"]


async def test_predict_cache_bypass(client):
    payload = {"text": "another deterministic string", "use_cache": False}
    r1 = await client.post("/v1/predict", json=payload)
    r2 = await client.post("/v1/predict", json=payload)
    assert r1.json()["cache_hit"] is False
    assert r2.json()["cache_hit"] is False


async def test_predict_validation_empty(client):
    r = await client.post("/v1/predict", json={"text": ""})
    assert r.status_code == 422


async def test_predict_validation_too_long(client):
    r = await client.post("/v1/predict", json={"text": "x" * 2001})
    assert r.status_code == 422


async def test_predict_persists_log(client, session_factory):
    from sqlalchemy import select

    from sentinellm.db.models import PredictionLog

    await client.post("/v1/predict", json={"text": "persistence check"})

    async with session_factory() as s:
        rows = (await s.execute(select(PredictionLog))).scalars().all()
    assert len(rows) == 1
    assert rows[0].cache_hit is False
    assert rows[0].text_hash
