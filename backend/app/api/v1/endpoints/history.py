"""
Prediction history endpoint for retrieving stored telemetry and diagnostics logs.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel
from datetime import datetime

from app.db.database import get_db, PredictionLog

router = APIRouter()


class PredictionLogResponse(BaseModel):
    id: int
    battery_id: str
    endpoint: str
    soh_percent: Optional[float] = None
    current_soc: Optional[float] = None
    rul_cycles: Optional[int] = None
    prediction_source: Optional[str] = None
    raw_prediction: Optional[float] = None
    fallback_reason: Optional[str] = None
    overall_health: Optional[str] = None
    confidence: Optional[float] = None
    model_version: Optional[str] = None
    timestamp: datetime

    model_config = {"from_attributes": True}


@router.get("/history", response_model=List[PredictionLogResponse])
async def get_history(
    battery_id: Optional[str] = Query(None, description="Filter by battery identifier"),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve recent battery prediction history and audit records.
    """
    stmt = select(PredictionLog).order_by(desc(PredictionLog.timestamp)).limit(limit)
    if battery_id:
        stmt = stmt.where(PredictionLog.battery_id == battery_id)

    result = await db.execute(stmt)
    records = result.scalars().all()
    return records
