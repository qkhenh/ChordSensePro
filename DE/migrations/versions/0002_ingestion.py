"""Ingestion: job inputs/title/charge day and the upload-fingerprint cache key.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-26

Hand-written and frozen. Compatible with PostgreSQL 16 and 18.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

S = "app"
SHA256_HEX = r"^[0-9a-f]{64}$"


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("title", sa.String(300), nullable=False, server_default="Untitled"),
        schema=S,
    )
    op.add_column("jobs", sa.Column("input_path", sa.Text()), schema=S)
    op.add_column("jobs", sa.Column("fingerprint", pg.CHAR(64)), schema=S)
    op.add_column("jobs", sa.Column("charged_on", sa.Date()), schema=S)
    op.create_check_constraint(
        "ck_jobs_fingerprint",
        "jobs",
        f"fingerprint IS NULL OR fingerprint ~ '{SHA256_HEX}'",
        schema=S,
    )

    op.create_table(
        "source_fingerprints",
        sa.Column("fingerprint", pg.CHAR(64), primary_key=True),
        sa.Column(
            "content_hash",
            pg.CHAR(64),
            sa.ForeignKey(
                f"{S}.analyses.content_hash",
                ondelete="CASCADE",
                name="fk_source_fingerprints_content_hash_analyses",
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            f"fingerprint ~ '{SHA256_HEX}'", name="ck_source_fingerprints_fingerprint"
        ),
        schema=S,
    )


def downgrade() -> None:
    op.drop_table("source_fingerprints", schema=S)
    op.drop_constraint("ck_jobs_fingerprint", "jobs", schema=S)
    for column in ("charged_on", "fingerprint", "input_path", "title"):
        op.drop_column("jobs", column, schema=S)
