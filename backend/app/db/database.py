"""
Database models and SQLAlchemy engine setup for BMS prediction history and telemetry storage.
"""

from datetime import datetime, timezone
from typing import AsyncGenerator
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, Text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from app.config import settings

# SQLite / PostgreSQL async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


class PredictionLog(Base):
    """Stores inference execution records for audit trail, telemetry tracking, and degradation history."""
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    battery_id = Column(String(50), index=True, nullable=False)
    endpoint = Column(String(50), index=True, nullable=False)  # "soh", "soc", "rul", "ecm", "fault"
    soh_percent = Column(Float, nullable=True)
    current_soc = Column(Float, nullable=True)
    rul_cycles = Column(Integer, nullable=True)
    prediction_source = Column(String(50), nullable=True)  # "lstm_model", "xgboost_model", "physics_fallback", etc.
    raw_prediction = Column(Float, nullable=True)
    fallback_reason = Column(Text, nullable=True)
    overall_health = Column(String(30), nullable=True)
    ecm_parameters = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


async def init_db():
    """Create database tables if they do not exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
