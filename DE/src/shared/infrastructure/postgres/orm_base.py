"""Single SQLAlchemy DeclarativeBase shared across all ORM models.

All ORM models MUST import Base from here — NOT define their own.
This ensures a single metadata registry, required for Alembic migrations
to detect all tables in one pass.

Usage:
    from src.shared.infrastructure.postgres.orm_base import Base

    class MyModelORM(Base):
        __tablename__ = "my_table"
        ...
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base for all SQLAlchemy ORM models."""
    pass
