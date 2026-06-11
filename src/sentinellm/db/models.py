from datetime import UTC, datetime

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from sentinellm.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    hf_repo: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(default=False)
    f1: Mapped[float] = mapped_column(default=0.0)
    accuracy: Mapped[float] = mapped_column(default=0.0)
    created_at: Mapped[datetime] = mapped_column(default_factory=_utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), default=None
    )
    text_hash: Mapped[str] = mapped_column(String(64), index=True, default="")
    label: Mapped[int] = mapped_column(default=0)
    score: Mapped[float] = mapped_column(default=0.0)
    latency_ms: Mapped[float] = mapped_column(default=0.0)
    cache_hit: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(default_factory=_utcnow, index=True)
