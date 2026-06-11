"""initial

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-13

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(64), nullable=False, unique=True),
        sa.Column("hf_repo", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("f1", sa.Float(), nullable=False, server_default="0"),
        sa.Column("accuracy", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )

    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("model_version_id", sa.Integer(),
                  sa.ForeignKey("model_versions.id"), nullable=True),
        sa.Column("text_hash", sa.String(64), nullable=False),
        sa.Column("label", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("cache_hit", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("ix_prediction_logs_text_hash", "prediction_logs", ["text_hash"])
    op.create_index("ix_prediction_logs_created_at", "prediction_logs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_prediction_logs_created_at", table_name="prediction_logs")
    op.drop_index("ix_prediction_logs_text_hash", table_name="prediction_logs")
    op.drop_table("prediction_logs")
    op.drop_table("model_versions")
