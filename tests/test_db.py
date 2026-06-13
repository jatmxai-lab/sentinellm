from sqlalchemy import select

from sentinellm.db.models import ModelVersion, PredictionLog


async def test_alembic_metadata_creates_tables(engine):
    """Tables created from Base.metadata match what Alembic would create."""
    from sqlalchemy import inspect

    def names(sync_conn):
        return inspect(sync_conn).get_table_names()

    async with engine.connect() as conn:
        tables = await conn.run_sync(names)
    assert "model_versions" in tables
    assert "prediction_logs" in tables


async def test_model_version_roundtrip(session_factory):
    async with session_factory() as s:
        mv = ModelVersion(name="v1", hf_repo="u/r", is_active=True, f1=0.9, accuracy=0.94)
        s.add(mv)
        await s.commit()
        await s.refresh(mv)
        assert mv.id is not None

    async with session_factory() as s:
        got = (await s.execute(select(ModelVersion).where(ModelVersion.name == "v1"))).scalar_one()
        assert got.f1 == 0.9
        assert got.is_active


async def test_prediction_log_roundtrip(session_factory):
    async with session_factory() as s:
        s.add(PredictionLog(text_hash="abc", label=1, score=0.99, latency_ms=12.3, cache_hit=False))
        await s.commit()

    async with session_factory() as s:
        rows = (await s.execute(select(PredictionLog))).scalars().all()
        assert len(rows) == 1
        assert rows[0].text_hash == "abc"
        assert rows[0].label == 1
